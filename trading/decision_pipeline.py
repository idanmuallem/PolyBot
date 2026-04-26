"""Central sequential trading pipeline and decision handlers.

Pipeline order per loop:
1) Hunters gather opportunities.
2) Brains/Parsers evaluate EV from market + live truth.
3) RiskManager and BudgetManager run safety checks.
4) Executor submits approved orders.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from brains import get_brain_for_asset_type
from core.trading_config import TradingConfig
from hunters import PolymarketScannerHunter
from trading.budget_manager import BudgetManager
from trading.executor import RiskConfig, TradeExecutor
from trading.risk_manager import PortfolioManager

ENTRY_PRICE_FLOOR = 0.30
ENTRY_PRICE_CEILING = 0.85


@dataclass
class DecisionContext:
    market: object
    asset_type: str
    token_id: str
    question: str
    signal: object
    model_used: str
    poly_price: float
    actual_bet_usd: float = 0.0
    executed: bool = False
    status: str = "pending"


class DecisionHandler:
    def __init__(self):
        self._next = None

    def set_next(self, handler: "DecisionHandler") -> "DecisionHandler":
        self._next = handler
        return handler

    def handle(self, context: DecisionContext, log_func):
        result = self._process(context, log_func)
        if self._next and result is not None:
            return self._next.handle(result, log_func)
        return result

    def _process(self, context: DecisionContext, log_func):
        raise NotImplementedError


class TradabilityHandler(DecisionHandler):
    def _process(self, context: DecisionContext, log_func):
        if not context.signal.is_tradable:
            context.status = "not_tradable"
            return context
        return context


class BudgetHandler(DecisionHandler):
    def __init__(self, budget_manager):
        super().__init__()
        self.budget_manager = budget_manager

    def _process(self, context: DecisionContext, log_func):
        actual_bet_usd, should_execute = self.budget_manager.check_and_cap_bet(context.signal.kelly_size)
        if not should_execute:
            context.status = "daily_limit"
            return context
        context.actual_bet_usd = actual_bet_usd
        return context


class WatchOnlyHandler(DecisionHandler):
    def __init__(self, budget_manager):
        super().__init__()
        self.budget_manager = budget_manager

    def _process(self, context: DecisionContext, log_func):
        if self.budget_manager.watch_only:
            context.status = "watch_only"
            return context
        return context


class ExecuteHandler(DecisionHandler):
    def __init__(self, executor, budget_manager):
        super().__init__()
        self.executor = executor
        self.budget_manager = budget_manager

    def _process(self, context: DecisionContext, log_func):
        executed = self.executor.evaluate_and_execute(
            market=context.market,
            fair_value=context.signal.fair_value,
            ev=context.signal.expected_value,
            current_poly_price=context.poly_price,
            bet_amount_usd=context.actual_bet_usd,
            side="YES",
            log_func=log_func,
        )
        context.executed = executed
        context.status = "executed" if executed else "execution_failed"
        if executed:
            self.budget_manager.record_trade(context.actual_bet_usd)
        return context


class GuardByStatusHandler(DecisionHandler):
    def __init__(self, terminal_statuses):
        super().__init__()
        self.terminal_statuses = set(terminal_statuses)

    def _process(self, context: DecisionContext, log_func):
        if context.status in self.terminal_statuses:
            return None
        return context


def build_entry_pipeline(executor, budget_manager):
    tradability = TradabilityHandler()
    stop_after_not_tradable = GuardByStatusHandler({"not_tradable"})
    budget = BudgetHandler(budget_manager)
    stop_after_budget = GuardByStatusHandler({"daily_limit"})
    watch_only = WatchOnlyHandler(budget_manager)
    stop_after_watch = GuardByStatusHandler({"watch_only"})
    execute = ExecuteHandler(executor, budget_manager)

    tradability.set_next(stop_after_not_tradable).set_next(budget).set_next(stop_after_budget).set_next(watch_only).set_next(stop_after_watch).set_next(execute)
    return tradability


@dataclass
class CandidateTrade:
    market: object
    token_id: str
    asset_type: str
    question: str
    fair_value: float
    kelly_size: float
    model_used: str
    price_yes: float
    side: str
    ev_yes: float
    ev_no: float
    final_ev: float
    entry_price: float


class SequentialTradingPipeline:
    def __init__(self, bridge, log_func, delay: float | None = None):
        self.bridge = bridge
        self.log_func = log_func
        self.config = TradingConfig.from_env()
        self.loop_delay = float(delay) if delay is not None else float(self.config.loop_delay_seconds)
        self.min_ev_threshold = float(self.config.min_ev)
        self.allocation_fraction = 0.10
        self.max_bet_size_usd = float(self.config.max_bet_size_usd)
        self.safe_minimum = 1.0

        self.executor = TradeExecutor(risk_config=RiskConfig(ev_threshold=self.config.min_ev))
        self.portfolio_manager = PortfolioManager(
            bridge=self.bridge,
            executor=self.executor,
            config=self.config,
        )
        self.hunter = PolymarketScannerHunter(
            bridge=self.bridge,
            executor=self.executor,
            config=self.config,
        )

        self._sync_live_account_state()
        if float(getattr(self.bridge, "starting_balance", 0.0) or 0.0) <= 0.0:
            self.bridge.starting_balance = float(self.bridge.current_balance)

        self.budget_manager = BudgetManager(
            bridge=self.bridge,
            config=self.config,
            initial_balance=float(self.bridge.current_balance),
        )

        self.spent_today = float(getattr(self.bridge, "spent_today", 0.0) or 0.0)
        self.spend_day = datetime.now(timezone.utc).date()
        self.start_of_day_equity = float(getattr(self.bridge, "start_of_day_equity", 0.0) or 0.0)

    def _sync_live_account_state(self):
        """Refresh live positions and collateral balance into bridge state."""
        try:
            positions = self.executor.get_open_positions()
            self.bridge.current_portfolio = positions
            total_open_value = sum(float(getattr(p, "value", 0.0) or 0.0) for p in positions)
            self.bridge.open_position_value = float(total_open_value)
            self.bridge.open_positions_value = float(total_open_value)
            self.bridge.total_pnl = sum(
                (float(getattr(p, "current_price", 0.0) or 0.0) - float(getattr(p, "initial_price", 0.0) or 0.0))
                * float(getattr(p, "shares", 0.0) or 0.0)
                for p in positions
            )
        except Exception as exc:
            self.log_func("SYNC-WARN", "Pipeline", "portfolio", {"reason": "positions_fetch_failed", "error": str(exc)})

        try:
            live_cash = float(self.executor.get_balance())
            self.bridge.current_balance = float(live_cash)
            self.bridge.cash = float(live_cash)
        except Exception as exc:
            self.log_func("SYNC-WARN", "Pipeline", "balance", {"reason": "balance_fetch_failed", "error": str(exc)})

    def _reset_daily_if_needed(self):
        current_day = datetime.now(timezone.utc).date()
        if current_day != self.spend_day:
            self.spent_today = 0.0
            self.spend_day = current_day
            self.start_of_day_equity = 0.0
            self.bridge.spent_today = 0.0
            self.bridge.daily_spend = 0.0
            self.bridge.start_of_day_equity = 0.0
            self.budget_manager.total_spent_today = 0.0

    def _stage_hunt(self):
        market, hunter = self.hunter.get_active_markets(self.log_func)
        if not market or not hunter:
            self.bridge.status = "No markets found. Waiting..."
            return None
        return market, hunter

    def _stage_evaluate_ev(self, market, hunter) -> CandidateTrade | None:
        token_id = str(getattr(market, "market_id", "") or "")
        asset_type = str(getattr(market, "asset_type", "") or "")
        question = str(getattr(market, "market_name", "") or getattr(market, "question", ""))

        # HARD GUARDRAIL: Prevent Spread Suicide (Never buy what we already own)
        # Check the live portfolio to see if we already hold shares in this exact market
        if hasattr(self.bridge, "current_portfolio") and self.bridge.current_portfolio:
            for position in self.bridge.current_portfolio:
                # The position object might use asset_id or token_id depending on the executor
                pos_token = str(getattr(position, "asset_id", getattr(position, "token_id", "")))
                if pos_token == token_id:
                    self.log_func("SCAN-SKIP", asset_type, token_id, {"reason": "already_owned_in_portfolio"})
                    self.hunter.mark_seen(token_id)
                    return None

        # STRICT SANITY CHECK: Prevent BTC/ETH Cross-Contamination
        asset_lower = asset_type.lower()
        q_lower = question.lower()

        if "btc" in asset_lower or "bitcoin" in asset_lower:
            if "eth" in q_lower or "ethereum" in q_lower or "sol" in q_lower or "solana" in q_lower:
                self.log_func("SCAN-SKIP", asset_type, token_id, {"reason": "asset_mismatch_btc_vs_altcoin"})
                self.hunter.mark_seen(token_id)
                return None

        if "eth" in asset_lower or "ethereum" in asset_lower:
            if "btc" in q_lower or "bitcoin" in q_lower or "sol" in q_lower or "solana" in q_lower:
                self.log_func("SCAN-SKIP", asset_type, token_id, {"reason": "asset_mismatch_eth_vs_other"})
                self.hunter.mark_seen(token_id)
                return None

        self.bridge.status = f"Scanning {asset_type}: {question[:60]}..."
        self.bridge.market_question = question
        self.bridge.market_asset_type = asset_type
        self.bridge.current_token_id = token_id

        # Parser output is already embedded in market metadata from the hunter stage.
        poly_price = float(getattr(market, "initial_price", 0.0) or 0.0)
        self.bridge.market_poly = poly_price

        if poly_price < ENTRY_PRICE_FLOOR or poly_price > ENTRY_PRICE_CEILING:
            self.log_func(
                "FILTERED",
                asset_type,
                token_id,
                {
                    "market_name": question,
                    "reason": "entry price out of bounds",
                    "poly_price": round(float(poly_price), 4),
                    "price_floor": ENTRY_PRICE_FLOOR,
                    "price_ceiling": ENTRY_PRICE_CEILING,
                },
            )
            self.hunter.mark_seen(token_id)
            return None

        live_truth = hunter.get_live_truth(market)
        if live_truth is None:
            self.log_func("SCAN-SKIP", asset_type, token_id, {"reason": "live_truth unavailable"})
            return None

        self.bridge.market_actual = live_truth
        brain = get_brain_for_asset_type(asset_type)
        signal = brain.evaluate(market, float(live_truth), min_ev=self.min_ev_threshold)
        model_used = getattr(brain, "last_model_used", "unknown")

        self.bridge.forecast = float(signal.fair_value)
        self.bridge.ev = float(signal.expected_value)

        price_yes = float(poly_price)
        ev_yes = (float(signal.fair_value) / float(price_yes) - 1.0) if price_yes > 0 else -1.0
        price_no = max(1e-9, 1.0 - float(price_yes))
        fair_no = 1.0 - float(signal.fair_value)
        ev_no = (fair_no / price_no - 1.0) if price_no > 0 else -1.0

        side = "YES" if ev_yes > ev_no else "NO"
        final_ev = max(ev_yes, ev_no)
        entry_price = float(price_yes if side == "YES" else price_no)

        diag = (
            f"[EV-MATH] YES(P: {price_yes:.3f}, FV: {float(signal.fair_value):.3f}, EV: {ev_yes:.2f}) | "
            f"NO(P: {price_no:.3f}, FV: {fair_no:.3f}, EV: {ev_no:.2f}) | PICK: {side}"
        )
        print(diag)
        self.bridge.terminal_logs.appendleft(diag)

        return CandidateTrade(
            market=market,
            token_id=token_id,
            asset_type=asset_type,
            question=question,
            fair_value=float(signal.fair_value),
            kelly_size=float(signal.kelly_size),
            model_used=model_used,
            price_yes=price_yes,
            side=side,
            ev_yes=float(ev_yes),
            ev_no=float(ev_no),
            final_ev=float(final_ev),
            entry_price=float(entry_price),
        )

    def _stage_risk_and_budget(self, candidate: CandidateTrade):
        cash_balance = float(self.bridge.current_balance)
        open_positions_value = float(self.bridge.open_position_value)
        total_equity = float(cash_balance) + float(open_positions_value)

        if float(cash_balance) < float(self.safe_minimum):
            self.bridge.status = "Portfolio Management Mode (cash below $1.00)"
            self.log_func(
                "PORTFOLIO-MODE",
                "Pipeline",
                "cash_guard",
                {
                    "reason": "insufficient_cash_for_new_entries",
                    "cash": round(float(cash_balance), 4),
                    "open_positions_value": round(float(open_positions_value), 4),
                    "minimum_required_cash": round(float(self.safe_minimum), 4),
                },
            )
            return 0.0, None

        if float(candidate.final_ev) < float(self.min_ev_threshold):
            self.log_func(
                "REJECTED",
                candidate.asset_type,
                candidate.token_id,
                {
                    "market_name": candidate.question,
                    "reason": "EV below dynamic threshold",
                    "ev_yes": round(float(candidate.ev_yes), 4),
                    "ev_no": round(float(candidate.ev_no), 4),
                    "side": candidate.side,
                    "ev": round(float(candidate.final_ev), 4),
                    "threshold": self.min_ev_threshold,
                },
            )
            self.hunter.mark_seen(candidate.token_id)
            return 0.0, None

        if candidate.entry_price < ENTRY_PRICE_FLOOR or candidate.entry_price > ENTRY_PRICE_CEILING:
            self.log_func(
                "FILTERED",
                candidate.asset_type,
                candidate.token_id,
                {
                    "market_name": candidate.question,
                    "reason": "entry price out of bounds for selected side",
                    "side": candidate.side,
                    "entry_price": round(float(candidate.entry_price), 4),
                    "price_floor": ENTRY_PRICE_FLOOR,
                    "price_ceiling": ENTRY_PRICE_CEILING,
                    "price_yes": round(float(candidate.price_yes), 4),
                    "price_no": round(float(1.0 - candidate.price_yes), 4),
                },
            )
            self.hunter.mark_seen(candidate.token_id)
            return 0.0, None

        target_bet = float(total_equity) * float(self.allocation_fraction)
        desired_bet = min(float(target_bet), float(self.max_bet_size_usd))

        budget_bet, budget_ok = self.budget_manager.check_and_cap_bet(float(candidate.kelly_size))
        if not budget_ok:
            self.log_func(
                "REJECTED",
                candidate.asset_type,
                candidate.token_id,
                {
                    "market_name": candidate.question,
                    "reason": "daily_limit_reached",
                    "kelly_size": round(float(candidate.kelly_size), 4),
                    "daily_limit_usd": round(float(self.budget_manager.daily_limit_usd), 4),
                    "spent_today": round(float(self.budget_manager.total_spent_today), 4),
                },
            )
            self.hunter.mark_seen(candidate.token_id)
            return 0.0, None

        available_cash = float(cash_balance)
        freed_cash = 0.0
        if float(available_cash) < float(desired_bet):
            try:
                freed_cash = float(
                    self.portfolio_manager.optimize_for_candidate(
                        float(candidate.final_ev),
                        min_improvement=0.10,
                        log_func=self.log_func,
                    )
                )
            except Exception as exc:
                print(f"[PIPELINE] Portfolio optimization failed: {exc}")
                freed_cash = 0.0

            available_cash = float(available_cash) + float(freed_cash)
            self.bridge.current_balance = float(available_cash)
            self.bridge.cash = float(available_cash)

        approved_bet = min(float(desired_bet), float(budget_bet), float(available_cash))
        if float(approved_bet) < float(self.safe_minimum):
            self.log_func(
                "REJECTED",
                candidate.asset_type,
                candidate.token_id,
                {
                    "market_name": candidate.question,
                    "reason": "insufficient_cash",
                    "approved_bet": round(float(approved_bet), 4),
                    "available_cash": round(float(available_cash), 4),
                    "freed_cash": round(float(freed_cash), 4),
                    "desired_bet": round(float(desired_bet), 4),
                },
            )
            self.hunter.mark_seen(candidate.token_id)
            return 0.0, None

        if float(approved_bet) < float(desired_bet):
            self.log_func(
                "BET-DOWNSIZE",
                candidate.asset_type,
                candidate.token_id,
                {
                    "market_name": candidate.question,
                    "reason": "using available cash instead of standard bet size",
                    "desired_bet": round(float(desired_bet), 4),
                    "actual_bet": round(float(approved_bet), 4),
                    "available_cash": round(float(available_cash), 4),
                    "budget_bet": round(float(budget_bet), 4),
                },
            )

        return float(approved_bet), {
            "available_cash": float(available_cash),
            "target_bet": float(target_bet),
            "desired_bet": float(desired_bet),
        }

    def _stage_execute(self, candidate: CandidateTrade, approved_bet: float, risk_context: dict):
        executed = self.executor.evaluate_and_execute(
            market=candidate.market,
            fair_value=float(candidate.fair_value),
            ev=float(candidate.final_ev),
            current_poly_price=float(candidate.price_yes),
            bet_amount_usd=float(approved_bet),
            side=candidate.side,
            log_func=self.log_func,
        )

        if executed:
            self.budget_manager.record_trade(float(approved_bet))
            self.spent_today = float(self.budget_manager.total_spent_today)
            self.bridge.spent_today = float(self.spent_today)
            self.bridge.daily_spend = float(self.spent_today)

            self.bridge.current_balance = max(0.0, float(self.bridge.current_balance) - float(approved_bet))
            self.bridge.cash = float(self.bridge.current_balance)

            cash_balance = float(self.bridge.current_balance)
            open_positions_value = float(self.bridge.open_position_value)
            total_equity = float(cash_balance) + float(open_positions_value)

            self.log_func(
                "TRACK",
                candidate.asset_type,
                candidate.token_id,
                {
                    "market_name": candidate.question,
                    "model_used": candidate.model_used,
                    "fair": round(float(candidate.fair_value), 4),
                    "ev": round(float(candidate.final_ev), 4),
                    "ev_yes": round(float(candidate.ev_yes), 4),
                    "ev_no": round(float(candidate.ev_no), 4),
                    "side": candidate.side,
                    "kelly": round(float(candidate.kelly_size), 4),
                    "bet_usd": round(float(approved_bet), 2),
                    "executed": bool(executed),
                    "total_equity": round(float(total_equity), 4),
                    "allocation_fraction": self.allocation_fraction,
                    "max_bet_size_usd": round(float(self.max_bet_size_usd), 2),
                    "target_bet_unclamped": round(float(risk_context.get("target_bet", 0.0)), 2),
                    "target_bet_usd": round(float(approved_bet), 2),
                    "available_cash": round(float(risk_context.get("available_cash", 0.0)), 2),
                    "spent_today": round(float(self.spent_today), 2),
                },
            )

        self.hunter.mark_seen(candidate.token_id)

    async def run_forever(self):
        while True:
            await asyncio.sleep(self.loop_delay)
            self._reset_daily_if_needed()

            requested_live = bool(getattr(self.bridge, "live_trading", False))
            self.executor.dry_run = not requested_live

            self._sync_live_account_state()
            self.portfolio_manager.manage_portfolio(self.log_func)
            self._sync_live_account_state()

            stage1 = self._stage_hunt()
            if not stage1:
                continue

            market, hunter = stage1
            candidate = self._stage_evaluate_ev(market, hunter)
            if candidate is None:
                continue

            approved_bet, risk_context = self._stage_risk_and_budget(candidate)
            if approved_bet <= 0.0 or risk_context is None:
                continue

            self._stage_execute(candidate, approved_bet, risk_context)


async def run_market_monitor(bridge, log_func, delay: float | None = None):
    """Canonical entrypoint for the trading monitor loop."""
    pipeline = SequentialTradingPipeline(bridge=bridge, log_func=log_func, delay=delay)
    await pipeline.run_forever()


__all__ = [
    "DecisionContext",
    "DecisionHandler",
    "TradabilityHandler",
    "BudgetHandler",
    "WatchOnlyHandler",
    "ExecuteHandler",
    "GuardByStatusHandler",
    "build_entry_pipeline",
    "CandidateTrade",
    "SequentialTradingPipeline",
    "run_market_monitor",
]

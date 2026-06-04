"""Central sequential trading pipeline.

Pipeline order per loop:
1. Hunt   — find active markets via PolymarketScannerHunter
2. Evaluate — compute fair value, EV, side (YES/NO)
3. Risk & Budget — position sizing, portfolio optimization, cash guards
4. Execute — submit order via TradeExecutor
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from brains import get_brain_for_asset_type
from core.models import PRICE_FLOOR, PRICE_CEILING
from core.trading_config import TradingConfig
from polymarket import PolymarketScannerHunter
from trading.budget_manager import BudgetManager
from trading.executor import TradeExecutor
from trading.risk_manager import PortfolioManager


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
        self.max_bet_size_usd = float(self.config.max_bet_size_usd)
        self.safe_minimum = 1.0

        self.executor = TradeExecutor()
        self.hunter = PolymarketScannerHunter(
            bridge=self.bridge,
            executor=self.executor,
            config=self.config,
        )
        self.portfolio_manager = PortfolioManager(
            bridge=self.bridge,
            executor=self.executor,
            config=self.config,
            hunter=self.hunter,
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

    # ------------------------------------------------------------------
    # Bridge helpers — keep related fields in sync
    # ------------------------------------------------------------------

    def _set_cash(self, amount: float):
        self.bridge.current_balance = float(amount)
        self.bridge.cash = float(amount)

    def _set_spend(self, amount: float):
        self.bridge.spent_today = float(amount)
        self.bridge.daily_spend = float(amount)

    def _total_equity(self) -> float:
        return float(self.bridge.current_balance) + float(self.bridge.open_position_value)

    def _reject(self, level: str, candidate: CandidateTrade, payload: dict):
        """Log a rejection, cool down the market, and signal the pipeline to skip."""
        self.log_func(level, candidate.asset_type, candidate.token_id, payload)
        self.hunter.add_to_cooldown(candidate.token_id)
        return 0.0, None

    # ------------------------------------------------------------------

    def _sync_live_account_state(self):
        """Refresh live positions and collateral balance into bridge state."""
        try:
            self.portfolio_manager._refresh_portfolio()
        except Exception as exc:
            self.log_func("SYNC-WARN", "Pipeline", "portfolio", {"reason": "positions_fetch_failed", "error": str(exc)})

        try:
            self._set_cash(float(self.executor.get_balance()))
        except Exception as exc:
            self.log_func("SYNC-WARN", "Pipeline", "balance", {"reason": "balance_fetch_failed", "error": str(exc)})

    def _reset_daily_if_needed(self):
        current_day = datetime.now(timezone.utc).date()
        if current_day != self.spend_day:
            self.spent_today = 0.0
            self.spend_day = current_day
            self.start_of_day_equity = 0.0
            self._set_spend(0.0)
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

        # Prevent buying a market already in the portfolio.
        if hasattr(self.bridge, "current_portfolio") and self.bridge.current_portfolio:
            for position in self.bridge.current_portfolio:
                pos_token = str(getattr(position, "asset_id", getattr(position, "token_id", "")))
                if pos_token == token_id:
                    self.log_func("SCAN-SKIP", asset_type, token_id, {"reason": "already_owned_in_portfolio"})
                    self.hunter.add_to_cooldown(token_id)
                    return None

        self.bridge.status = f"Scanning {asset_type}: {question[:60]}..."
        self.bridge.market_question = question
        self.bridge.market_asset_type = asset_type
        self.bridge.current_token_id = token_id

        poly_price = float(getattr(market, "initial_price", 0.0) or 0.0)
        self.bridge.market_poly = poly_price

        if poly_price < PRICE_FLOOR or poly_price > PRICE_CEILING:
            self.log_func("FILTERED", asset_type, token_id, {
                "market_name": question,
                "reason": "entry price out of bounds",
                "poly_price": round(float(poly_price), 4),
                "price_floor": PRICE_FLOOR,
                "price_ceiling": PRICE_CEILING,
            })
            self.hunter.add_to_cooldown(token_id)
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
        total_equity = self._total_equity()

        if cash_balance < self.safe_minimum:
            self.bridge.status = "Portfolio Management Mode (cash below $1.00)"
            self.log_func("PORTFOLIO-MODE", "Pipeline", "cash_guard", {
                "reason": "insufficient_cash_for_new_entries",
                "cash": round(cash_balance, 4),
                "open_positions_value": round(float(self.bridge.open_position_value), 4),
                "minimum_required_cash": round(self.safe_minimum, 4),
            })
            return 0.0, None

        if candidate.final_ev < self.min_ev_threshold:
            return self._reject("REJECTED", candidate, {
                "market_name": candidate.question,
                "reason": "EV below dynamic threshold",
                "ev_yes": round(candidate.ev_yes, 4),
                "ev_no": round(candidate.ev_no, 4),
                "side": candidate.side,
                "ev": round(candidate.final_ev, 4),
                "threshold": self.min_ev_threshold,
            })

        if candidate.entry_price < PRICE_FLOOR or candidate.entry_price > PRICE_CEILING:
            return self._reject("FILTERED", candidate, {
                "market_name": candidate.question,
                "reason": "entry price out of bounds for selected side",
                "side": candidate.side,
                "entry_price": round(candidate.entry_price, 4),
                "price_floor": PRICE_FLOOR,
                "price_ceiling": PRICE_CEILING,
                "price_yes": round(candidate.price_yes, 4),
                "price_no": round(1.0 - candidate.price_yes, 4),
            })

        target_bet = total_equity * candidate.kelly_size
        desired_bet = min(target_bet, self.max_bet_size_usd)

        budget_bet, budget_ok = self.budget_manager.check_and_cap_bet(candidate.kelly_size)
        if not budget_ok:
            return self._reject("REJECTED", candidate, {
                "market_name": candidate.question,
                "reason": "daily_limit_reached",
                "kelly_size": round(candidate.kelly_size, 4),
                "daily_limit_usd": round(float(self.budget_manager.daily_limit_usd), 4),
                "spent_today": round(float(self.budget_manager.total_spent_today), 4),
            })

        available_cash = cash_balance
        freed_cash = 0.0
        if available_cash < desired_bet:
            # First pass: swap out positions with materially worse EV
            try:
                freed_cash = float(self.portfolio_manager.optimize_for_candidate(
                    candidate.final_ev, min_improvement=0.10, log_func=self.log_func,
                ))
            except Exception as exc:
                print(f"[PIPELINE] Portfolio optimization failed: {exc}")
            available_cash += freed_cash

            # Second pass: if still short, sell weakest positions regardless of EV
            if available_cash < desired_bet:
                try:
                    self.portfolio_manager.free_up_capital(desired_bet, self.log_func)
                    available_cash = float(self.bridge.current_balance)
                except Exception as exc:
                    print(f"[PIPELINE] free_up_capital failed: {exc}")

            self._set_cash(available_cash)

        approved_bet = min(desired_bet, budget_bet, available_cash)
        if approved_bet < self.safe_minimum:
            return self._reject("REJECTED", candidate, {
                "market_name": candidate.question,
                "reason": "insufficient_cash",
                "approved_bet": round(approved_bet, 4),
                "available_cash": round(available_cash, 4),
                "freed_cash": round(freed_cash, 4),
                "desired_bet": round(desired_bet, 4),
            })

        if approved_bet < desired_bet:
            self.log_func("BET-DOWNSIZE", candidate.asset_type, candidate.token_id, {
                "market_name": candidate.question,
                "reason": "using available cash instead of standard bet size",
                "desired_bet": round(desired_bet, 4),
                "actual_bet": round(approved_bet, 4),
                "available_cash": round(available_cash, 4),
                "budget_bet": round(budget_bet, 4),
            })

        return approved_bet, {
            "available_cash": available_cash,
            "target_bet": target_bet,
            "desired_bet": desired_bet,
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
            self._set_spend(self.spent_today)
            self._set_cash(max(0.0, float(self.bridge.current_balance) - float(approved_bet)))

            total_equity = self._total_equity()

            self.log_func("TRACK", candidate.asset_type, candidate.token_id, {
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
                "max_bet_size_usd": round(float(self.max_bet_size_usd), 2),
                "target_bet_unclamped": round(float(risk_context.get("target_bet", 0.0)), 2),
                "target_bet_usd": round(float(approved_bet), 2),
                "available_cash": round(float(risk_context.get("available_cash", 0.0)), 2),
                "spent_today": round(float(self.spent_today), 2),
                "strike_price": float(getattr(candidate.market, "strike_price", 0.0) or 0.0),
                "expiry_date": str(getattr(candidate.market, "expiry_date", "") or ""),
            })

        self.hunter.add_to_cooldown(candidate.token_id)

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

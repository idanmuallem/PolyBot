"""Central sequential trading pipeline.

Pipeline order per loop:
1. Hunt   — find active markets via PolymarketScannerHunter
2. Evaluate — compute fair value, EV, side (YES/NO)
3. Risk & Budget — position sizing, portfolio optimization, cash guards
4. Execute — submit order via TradeExecutor
"""

import asyncio
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from brains import get_brain_for_asset_type
from brains.base import calculate_tte
from core.models import PRICE_FLOOR, PRICE_CEILING, MarketData
from core.wallet_context import WalletContext
from polymarket import PolymarketClient
from trading.strategies import EventSumStrategy, Strategy


# Minimum guaranteed edge (as a fraction of the guaranteed payout) an
# arbitrage group's REAL fills must clear after real costs before the
# group is kept, in _execute_strategy_group(). Deliberately hardcoded, not
# a config value: this is a correctness floor for what "genuinely
# profitable" means, not a tunable risk knob like MIN_EV (which is a
# single-market probabilistic-EV threshold — reusing it here, at its
# ~0.35 default, would make real arbitrage opportunities — typically a
# low single-digit percent edge — effectively never fire). 2% matches
# EventSumStrategy's own scan-time min_edge default, so the execution-time
# floor doesn't contradict the strategy's own notion of "worth trying."
_MIN_ARBITRAGE_MARGIN = 0.02


def sync_live_account_state(bridge, executor, portfolio_manager, log_func):
    """Refresh live positions and collateral balance into bridge state.

    Free function (not a pipeline method) so it can run before a
    SequentialTradingPipeline exists — e.g. by a caller wiring up
    WalletContext components, which needs a live balance to construct
    BudgetManager with an accurate initial_balance.
    """
    try:
        portfolio_manager._refresh_portfolio()
    except Exception as exc:
        log_func("SYNC-WARN", "Pipeline", "portfolio", {"reason": "positions_fetch_failed", "error": str(exc)})

    try:
        balance = float(executor.get_balance())
        bridge.current_balance = balance
        bridge.cash = balance
    except Exception as exc:
        log_func("SYNC-WARN", "Pipeline", "balance", {"reason": "balance_fetch_failed", "error": str(exc)})


@dataclass
class CandidateTrade:
    market: MarketData
    token_id: str
    asset_type: str
    question: str
    post_prob: float
    pre_prob: float
    kelly_size: float       # raw Kelly criterion fraction (edge/odds), before confidence/kelly_fraction scaling
    kelly_bet_usd: float    # final dollar bet size — confidence- and kelly_fraction-scaled, clamped to max_bet_size_usd
    model_used: str
    price_yes: float
    side: str
    ev_yes: float
    ev_no: float
    final_ev: float
    entry_price: float
    pricing_mode: str = "wang"
    wang_lambda: Optional[float] = None
    wang_fair_value: Optional[float] = None
    wang_edge: Optional[float] = None
    strategy_type: str = "model"       # "model" for brain-driven trades, "arbitrage" etc. for strategies
    kelly_fraction_used: float = 0.0   # the config.kelly_fraction applied when this bet was sized
    correlation_exposure: float = 0.0  # avg correlation with the currently open book (see trading/correlation.py)


class SequentialTradingPipeline:
    def __init__(self, ctx: WalletContext, log_func, delay: float | None = None):
        if not all([ctx.executor, ctx.scanner, ctx.portfolio_manager, ctx.budget_manager]):
            missing = [name for name, val in [
                ("executor", ctx.executor),
                ("scanner", ctx.scanner),
                ("portfolio_manager", ctx.portfolio_manager),
                ("budget_manager", ctx.budget_manager),
            ] if val is None]
            raise ValueError(f"WalletContext missing required components: {missing}")

        self.ctx = ctx
        self.bridge = ctx.bridge
        self.config = ctx.config
        self.db_path = ctx.db_path
        self.log_func = log_func
        self.loop_delay = float(delay) if delay is not None else float(self.config.loop_delay_seconds)
        self.min_ev_threshold = float(self.config.min_ev)
        self.max_bet_size_usd = float(self.config.max_bet_size_usd)
        self.safe_minimum = 1.0

        # Wang Transform pricing — all of it (Wang Transform, market blending)
        # now lives in BaseBrain.evaluate() (brains/base.py), which this
        # pipeline calls directly rather than re-deriving fair value itself.
        # "legacy" skips both layers for A/B comparison (see _stage_evaluate_ev).
        self.pricing_mode = str(getattr(self.config, "pricing_mode", "wang") or "wang").lower()
        self.wang_min_edge = float(getattr(self.config, "wang_min_edge", 0.05))

        self.executor = ctx.executor
        self.hunter = ctx.scanner

        # Constraint-based strategies (arbitrage) — model-free, run independently
        # of the brain/PricingEngine path above. See _stage_strategy_scan().
        self.polymarket_client = PolymarketClient()
        self.strategies: list[Strategy] = [EventSumStrategy(config=self.config)]
        self.strategy_scan_interval = 30.0  # don't hammer the Gamma API every loop tick
        self._last_strategy_scan = 0.0

        # group_ids rejected this "day" for insufficient_budget,
        # tte_exceeds_max, or daily_trade_limit_would_be_exceeded — none of
        # these conditions change between scans (the per-strategy budget and
        # trade count only reset daily; a leg's expiry date is fixed), so
        # re-evaluating the same groups every 30s was pure waste. Cleared in
        # _reset_daily_if_needed().
        self._exhausted_arb_groups: set[str] = set()

        # token_ids "held" this "day" via a dry-run/paper-trade fill. Neither
        # mode submits a real order, so TradeExecutor.get_open_positions()
        # (which always queries the live Polymarket Data API - see
        # _stage_evaluate_ev's "already_owned_in_portfolio" check) never sees
        # them, and without this the same EV-positive market gets re-bought
        # every hunt cycle. Live mode's real on-chain position already
        # covers this once the order settles; this is purely a simulated-
        # mode backstop. Cleared in _reset_daily_if_needed().
        self._simulated_positions: set[str] = set()

        # group_ids successfully filled this "day" - this applies in EVERY
        # mode, not just simulated ones (unlike _simulated_positions above):
        # the arbitrage path has no "already hold this" check of its own at
        # all, so a group gets rediscovered and re-filled every scan for as
        # long as it stays profitable in the live Gamma API snapshot.
        # Cleared in _reset_daily_if_needed().
        self._filled_arb_groups: set[str] = set()

        self.portfolio_manager = ctx.portfolio_manager

        sync_live_account_state(self.bridge, self.executor, self.portfolio_manager, self.log_func)
        if float(getattr(self.bridge, "starting_balance", 0.0) or 0.0) <= 0.0:
            self.bridge.starting_balance = float(self.bridge.current_balance)

        self.budget_manager = ctx.budget_manager

        self.spent_today = float(getattr(self.bridge, "spent_today", 0.0) or 0.0)
        self.spend_day = datetime.now(timezone.utc).date()
        self.start_of_day_equity = float(getattr(self.bridge, "start_of_day_equity", 0.0) or 0.0)

        # Drawdown circuit breaker — pauses new trade entry (not exits) once
        # equity has fallen more than max_drawdown_pct from its peak.
        self.max_drawdown_pct = float(getattr(self.config, "max_drawdown_pct", 0.20))
        self.peak_equity = self._total_equity()
        self._update_drawdown_guard()

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

    def _update_drawdown_guard(self):
        """Drawdown circuit breaker: pause new trade entry (not position
        management/exits) once equity has fallen more than max_drawdown_pct
        from its peak. Call every loop tick, after balances/positions are synced.

        The peak-tracking/threshold pattern mirrors oracle3's
        StandardRiskManager._check_drawdown() (oracle3/risk/risk_manager.py) -
        https://github.com/YichengYang-Ethan/oracle3, licensed under the
        Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
        """
        current_equity = self._total_equity()
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        drawdown_pct = (
            (self.peak_equity - current_equity) / self.peak_equity
            if self.peak_equity > 0 else 0.0
        )
        should_pause = drawdown_pct >= self.max_drawdown_pct

        was_paused = bool(getattr(self.ctx, "drawdown_paused", False))
        self.ctx.drawdown_paused = should_pause
        self.bridge.drawdown_paused = should_pause
        self.bridge.peak_equity = self.peak_equity
        self.bridge.current_drawdown_pct = drawdown_pct

        if should_pause and not was_paused:
            self.log_func("CIRCUIT-BREAKER", "Pipeline", "drawdown", {
                "reason": "max_drawdown_pct breached — new trade entry paused; "
                          "position management/exits still run",
                "peak_equity": round(self.peak_equity, 4),
                "current_equity": round(current_equity, 4),
                "drawdown_pct": round(drawdown_pct, 4),
                "max_drawdown_pct": self.max_drawdown_pct,
            })
        elif was_paused and not should_pause:
            self.log_func("CIRCUIT-BREAKER-CLEAR", "Pipeline", "drawdown", {
                "reason": "equity recovered above max_drawdown_pct — new trade entry resumed",
                "peak_equity": round(self.peak_equity, 4),
                "current_equity": round(current_equity, 4),
                "drawdown_pct": round(drawdown_pct, 4),
            })

    def _reject(self, level: str, candidate: CandidateTrade, payload: dict):
        """Log a rejection, cool down the market, and signal the pipeline to skip."""
        self.log_func(level, candidate.asset_type, candidate.token_id, payload)
        self.hunter.add_to_cooldown(candidate.token_id)
        return 0.0, None

    @staticmethod
    def _analytics_log_fields(candidate: CandidateTrade) -> dict:
        """Pricing/risk fields for log payloads (Phase 3 Wang fields, Phase 7
        strategy/kelly/correlation fields) — lets us compare raw-brain EV
        against the Wang-adjusted edge, and separate model-driven from
        strategy-driven performance, from the trade history alone."""
        return {
            "pricing_mode": candidate.pricing_mode,
            "pre_prob": round(candidate.pre_prob, 4),
            "wang_lambda": round(candidate.wang_lambda, 4) if candidate.wang_lambda is not None else None,
            "wang_fair_value": round(candidate.wang_fair_value, 4) if candidate.wang_fair_value is not None else None,
            "wang_edge": round(candidate.wang_edge, 4) if candidate.wang_edge is not None else None,
            "strategy_type": candidate.strategy_type,
            "kelly_fraction_used": round(candidate.kelly_fraction_used, 4),
            "correlation_exposure": round(candidate.correlation_exposure, 4),
        }

    # ------------------------------------------------------------------

    def _reset_daily_if_needed(self):
        current_day = datetime.now(timezone.utc).date()
        if current_day != self.spend_day:
            self.spent_today = 0.0
            self.spend_day = current_day
            self.start_of_day_equity = 0.0
            self._set_spend(0.0)
            self.bridge.start_of_day_equity = 0.0
            self.budget_manager.total_spent_today = 0.0
            self.budget_manager.spent_by_strategy = {}
            self.budget_manager.trades_by_strategy = {}
            self.executor.reset_daily_count()
            self._exhausted_arb_groups.clear()
            self._simulated_positions.clear()
            self._filled_arb_groups.clear()

    # ------------------------------------------------------------------
    # Strategy scan — constraint-based arbitrage, independent of the
    # brain/PricingEngine path (no probability estimate needed).
    # ------------------------------------------------------------------

    def _tag_log_func(self, strategy_type: str):
        """Wrap self.log_func so every payload it forwards carries
        strategy_type — strategy-driven trades share the executor's own
        internal log_func calls (AUTO-TRADE, DRY-RUN, ...) with model-driven
        ones, so this is how their P&L stays separable in the log/analytics
        layer.
        """
        base_log_func = self.log_func

        def _tagged(level, asset_type, token_id, payload):
            if isinstance(payload, dict):
                payload = {**payload, "strategy_type": strategy_type}
            else:
                payload = {"message": payload, "strategy_type": strategy_type}
            base_log_func(level, asset_type, token_id, payload)

        return _tagged

    @staticmethod
    def _group_signals(signals: list) -> dict:
        groups: dict = {}
        for signal in signals:
            groups.setdefault(signal.group_id, []).append(signal)
        return groups

    async def _stage_strategy_scan(self):
        """Run every registered Strategy and execute any signals it emits.

        Runs alongside _stage_evaluate_ev (not instead of it): strategies
        find trades directly from market-price arithmetic, so they skip the
        brain/PricingEngine path entirely and go straight to execution.
        """
        if not self.strategies:
            return

        now = time.monotonic()
        if now - self._last_strategy_scan < self.strategy_scan_interval:
            return
        self._last_strategy_scan = now

        try:
            events = self.polymarket_client.get_multi_outcome_events(limit=100)
        except Exception as exc:
            self.log_func("SYNC-WARN", "Strategy", "event_fetch", {
                "reason": "event_fetch_failed", "error": str(exc),
            })
            return

        for strategy in self.strategies:
            try:
                signals = await strategy.scan(events)
            except Exception as exc:
                self.log_func("SYNC-WARN", "Strategy", strategy.strategy_type, {
                    "reason": "scan_failed", "error": str(exc),
                })
                continue

            for group_id, group_signals in self._group_signals(signals).items():
                if group_id in self._filled_arb_groups:
                    continue  # already hold this position, skip silently
                if group_id in self._exhausted_arb_groups:
                    continue
                await self._execute_strategy_group(strategy, group_id, group_signals)

    def _group_has_leg_beyond_max_tte(self, signals: list, group_id: str, asset_type: str, log_func) -> bool:
        """TTE filter: skip the whole arbitrage group if any single leg's
        resolution date is farther out than config.max_tte_days.

        You can't do partial arbitrage with missing legs, so one over-long
        leg voids the whole group — that capital would otherwise sit locked
        for months waiting on just that one outcome to resolve.
        """
        max_tte_days = float(self.config.max_tte_days)
        for signal in signals:
            days_to_expiry = calculate_tte(getattr(signal.market, "expiry_date", None))
            if days_to_expiry > max_tte_days:
                log_func("FILTERED", asset_type, group_id, {
                    "reason": "tte_exceeds_max",
                    "group_id": group_id,
                    "leg_token_id": signal.market.market_id,
                    "leg_expiry": str(getattr(signal.market, "expiry_date", None)),
                    "days_to_expiry": round(days_to_expiry, 2),
                    "max_tte_days": max_tte_days,
                })
                return True
        return False

    @staticmethod
    def _leg_token_id(signal) -> str:
        return (
            signal.market.market_id if signal.side == "YES"
            else str(getattr(signal.market, "no_market_id", None) or signal.market.market_id)
        )

    def _group_already_held(self, signals: list) -> bool:
        """True if any leg of this group is already an open position.

        _filled_arb_groups (the primary re-buy guard) is in-memory and
        resets to empty on every process restart, while the positions it's
        meant to prevent re-buying persist across restarts (paper positions
        on disk, live positions on-chain). Without this check, a redeploy
        wipes the guard but not the holdings, and the very next strategy
        scan re-discovers and re-buys every still-profitable arbitrage group
        it already holds — this is the confirmed root cause of a rapid-fire
        trade/holdings burst observed after redeploys. Checking the actual
        current portfolio (refreshed every loop tick, before this runs) is
        restart-proof because it reflects real holdings, not session state.
        """
        held_tokens = {
            str(getattr(position, "asset_id", getattr(position, "token_id", "")))
            for position in (getattr(self.bridge, "current_portfolio", None) or [])
        }
        for signal in signals:
            token_id = self._leg_token_id(signal)
            if token_id in held_tokens or token_id in self._simulated_positions:
                return True
        return False

    async def _execute_strategy_group(self, strategy: Strategy, group_id: str, signals: list):
        """Execute every leg of one strategy opportunity (e.g. every outcome
        of one event_sum arb) through the same TradeExecutor used by
        model-driven trades, via TradeExecutor.execute_arbitrage_group()
        rather than evaluate_and_execute(): the strategy already vetted the
        trade's profitability itself (its own min_edge, not config.min_ev),
        and its leg prices are expected to sit outside the single-market
        PRICE_FLOOR/PRICE_CEILING band by design.

        Each leg is placed as a limit order and given up to
        config.arbitrage_order_timeout_seconds to fill (see
        TradeExecutor.execute_arbitrage_group). Best-effort, not atomic: if
        every leg gets at least one fill the group is kept; if any leg gets
        zero fills, the rest are cancelled — real atomicity would require a
        settlement layer this pipeline doesn't have.

        Whatever comes back is then re-verified against REAL fill data
        before anything is kept, since neither of the above guarantees a
        genuinely profitable complementary set:
        - Group-level profit check: real per-share cost
          (bet_amount_usd / shares actually filled) vs the guaranteed $1
          payout, required to clear _MIN_ARBITRAGE_MARGIN. If it doesn't,
          everything bought is sold back — the strategy's own scan-time
          edge is a stale pre-execution estimate, not a guarantee.
        - Surplus trim: every leg is sized as a fixed DOLLAR amount, not a
          fixed SHARE count, so thin books on illiquid legs routinely fill
          unevenly for the same spend. Shares beyond arb_sets (the
          smallest leg's fill) aren't part of the guaranteed structure and
          are sold back immediately, even when the matched portion is
          genuinely profitable.
        - Incomplete-group unwind: if even one leg never fills at all,
          arb_sets is 0 (no complementary set exists without every leg) and
          whatever DID fill on the other legs is sold back in full rather
          than held as naked, uncompensated directional exposure.
        """
        if not signals:
            return

        tagged_log = self._tag_log_func(strategy.strategy_type)
        asset_type = signals[0].market.asset_type

        if self._group_already_held(signals):
            tagged_log("SCAN-SKIP", asset_type, group_id, {"reason": "already_owned_in_portfolio"})
            self._filled_arb_groups.add(group_id)
            return

        if self._group_has_leg_beyond_max_tte(signals, group_id, asset_type, tagged_log):
            self._exhausted_arb_groups.add(group_id)
            return

        total_cost = sum(float(s.bet_amount_usd) for s in signals)
        cash_balance = float(self.bridge.current_balance)
        remaining_budget = float(self.budget_manager.get_remaining_budget("arbitrage"))

        # Distinct reject reasons (Point 5): "can't afford it" (wallet cash)
        # vs "budget exhausted" (this strategy's own daily allocation) are
        # different failure modes worth telling apart in the log/analytics
        # layer, so they're no longer folded into one generic reason.
        if total_cost > cash_balance:
            tagged_log("REJECTED", asset_type, group_id, {
                "reason": "insufficient_cash",
                "group_id": group_id,
                "n_legs": len(signals),
                "total_cost": round(total_cost, 4),
                "cash_balance": round(cash_balance, 4),
            })
            return

        if total_cost > remaining_budget:
            tagged_log("REJECTED", asset_type, group_id, {
                "reason": "insufficient_budget",
                "group_id": group_id,
                "n_legs": len(signals),
                "total_cost": round(total_cost, 4),
                "remaining_budget": round(remaining_budget, 4),
            })
            self._exhausted_arb_groups.add(group_id)
            return

        arbitrage_max_daily_trades = self.executor._strategy_max_daily_trades("arbitrage")
        arbitrage_trades_today = self.executor.trades_by_strategy.get("arbitrage", 0)
        if arbitrage_trades_today + len(signals) > arbitrage_max_daily_trades:
            tagged_log("REJECTED", asset_type, group_id, {
                "reason": "daily_trade_limit_would_be_exceeded",
                "group_id": group_id,
                "n_legs": len(signals),
            })
            self._exhausted_arb_groups.add(group_id)
            return

        # Belt-and-suspenders hard ceiling (Point: runaway-trade-rate fix):
        # the check above only compares against this strategy's own
        # allocation (arbitrage_max_daily_trades, e.g. 200) — it says
        # nothing about the account-wide MAX_DAILY_TRADES ceiling a human
        # actually configured. TradingConfig only *warns* if per-strategy
        # caps are set to sum higher than the global one; it never clamps
        # them. This check makes the global cap a real ceiling regardless
        # of how per-strategy limits are configured, so no single strategy
        # (or misconfiguration) can spend the whole account's daily
        # allowance through its own generous per-strategy bucket.
        global_max_daily_trades = int(self.executor.config.max_daily_trades)
        if self.executor.trade_count_today + len(signals) > global_max_daily_trades:
            tagged_log("REJECTED", asset_type, group_id, {
                "reason": "global_daily_trade_limit_would_be_exceeded",
                "group_id": group_id,
                "n_legs": len(signals),
                "trade_count_today": self.executor.trade_count_today,
                "global_max_daily_trades": global_max_daily_trades,
            })
            self._exhausted_arb_groups.add(group_id)
            return

        legs = []
        for signal in signals:
            shares = math.floor((signal.bet_amount_usd / signal.price) * 100.0) / 100.0
            if shares <= 0:
                continue
            legs.append({
                "token_id": self._leg_token_id(signal),
                "price": signal.price,
                "shares": shares,
                "side": signal.side,
                "bet_amount_usd": signal.bet_amount_usd,
                "asset_type": signal.market.asset_type,
                "no_token_id": getattr(signal.market, "no_market_id", None),
                "condition_id": getattr(signal.market, "condition_id", None),
                "slug": getattr(signal.market, "slug", None),
                "group_id": group_id,
            })

        result = await self.executor.execute_arbitrage_group(
            legs=legs,
            timeout_seconds=float(getattr(self.config, "arbitrage_order_timeout_seconds", 60.0)),
            log_func=tagged_log,
            strategy_tag="arbitrage",
        )

        fills: dict = dict(result.get("fills") or {})
        arb_sets = float(result.get("arb_sets", 0.0) or 0.0)

        # Group-level guaranteed-profit re-verification against REAL fills.
        # EventSumStrategy's own scan-time edge check (sum of lastTradePrice
        # snapshots vs a flat fee estimate) is a stale, pre-execution
        # estimate — it never re-confirms against what was actually filled,
        # and every leg is sized as a FIXED DOLLAR AMOUNT
        # (signal.bet_amount_usd), not a fixed share count, so on the thin
        # order books typical of these long-shot outcome markets, the same
        # dollar spend buys unequal share counts per leg. bet_amount_usd/
        # filled_shares is the REAL, fee-and-slippage-inclusive cost per
        # share (no separate estimate needed — it's what was actually
        # spent for what was actually received), so this checks the thing
        # that actually determines whether the trade is locked-in
        # profitable, not a theoretical proxy for it.
        if arb_sets > 0:
            per_leg_unit_cost: dict = {}
            for signal in signals:
                tid = self._leg_token_id(signal)
                filled = fills.get(tid, 0.0)
                if filled > 0:
                    per_leg_unit_cost[tid] = float(signal.bet_amount_usd) / filled

            guaranteed_payout = arb_sets * 1.0
            guaranteed_cost = sum(unit_cost * arb_sets for unit_cost in per_leg_unit_cost.values())
            net_edge = guaranteed_payout - guaranteed_cost

            if net_edge < guaranteed_payout * _MIN_ARBITRAGE_MARGIN:
                # Real fills don't clear a genuine margin after real costs.
                # Keeping this would just be naked directional risk on
                # whichever illiquid long-shot outcomes happened to fill —
                # not arbitrage. Unwind everything bought instead of
                # holding it.
                for signal in signals:
                    tid = self._leg_token_id(signal)
                    filled = fills.get(tid, 0.0)
                    if filled > 0:
                        self.executor.sell_position(tid, filled, signal.price, tagged_log)
                        self.budget_manager.record_trade(float(signal.bet_amount_usd), strategy_tag="arbitrage")
                self.spent_today = float(self.budget_manager.total_spent_today)
                self._set_spend(self.spent_today)

                tagged_log("REJECTED", asset_type, group_id, {
                    "reason": "not_profitable_after_real_fills",
                    "group_id": group_id,
                    "n_legs": len(signals),
                    "arb_sets": arb_sets,
                    "guaranteed_cost": round(guaranteed_cost, 4),
                    "guaranteed_payout": round(guaranteed_payout, 4),
                    "net_edge": round(net_edge, 4),
                    "required_margin": _MIN_ARBITRAGE_MARGIN,
                })
                self._exhausted_arb_groups.add(group_id)
                return

        # Trim every leg down to arb_sets (the guaranteed matched set).
        # Anything beyond it is naked, one-sided exposure — not part of the
        # guaranteed structure (see TradeExecutor.execute_arbitrage_group's
        # docstring) — sold back immediately rather than kept as an
        # unintended directional bet. This single rule covers two distinct
        # cases the same way:
        #   - arb_sets > 0 and the group passed the profitability check
        #     above: trims real surplus from a leg whose illiquid book
        #     happened to fill more for the same fixed dollar spend.
        #   - arb_sets == 0 because at least one leg in the group never
        #     filled at all: every OTHER leg's fill (excess over 0) gets
        #     sold back in full. A complementary-set arb needs every leg —
        #     if one outcome never got bought, whatever filled on the
        #     others is pure directional risk on markets that resolve
        #     however they resolve, not a guaranteed structure, regardless
        #     of how "good" those individual fills looked.
        unwound_incomplete_group = False
        for signal in signals:
            tid = self._leg_token_id(signal)
            filled = fills.get(tid, 0.0)
            excess = round(filled - arb_sets, 4)
            if excess > 0:
                self.executor.sell_position(tid, excess, signal.price, tagged_log)
                if arb_sets <= 0:
                    unwound_incomplete_group = True
                    self.budget_manager.record_trade(float(signal.bet_amount_usd), strategy_tag="arbitrage")
                fills[tid] = arb_sets

        if unwound_incomplete_group:
            self.spent_today = float(self.budget_manager.total_spent_today)
            self._set_spend(self.spent_today)
            tagged_log("REJECTED", asset_type, group_id, {
                "reason": "group_incomplete_unwound_partial_fills",
                "group_id": group_id,
                "n_legs": len(signals),
                "unfilled": result.get("unfilled", []),
            })
            self._exhausted_arb_groups.add(group_id)
            return

        executed_legs = 0
        for signal in signals:
            token_id = self._leg_token_id(signal)
            filled_shares = float(fills.get(token_id, 0.0))
            executed = filled_shares > 0.0

            tagged_log("STRATEGY-LEG", signal.market.asset_type, token_id, {
                "group_id": group_id,
                "market_name": signal.market.market_name,
                "reasoning": signal.reasoning,
                "edge": round(signal.edge, 4),
                "price": round(signal.price, 4),
                "bet_usd": round(signal.bet_amount_usd, 2),
                "shares": filled_shares,
                "side": signal.side,
                "executed": bool(executed),
            })

            if executed:
                executed_legs += 1
                self.budget_manager.record_trade(float(signal.bet_amount_usd), strategy_tag="arbitrage")
                self.spent_today = float(self.budget_manager.total_spent_today)
                self._set_spend(self.spent_today)

        tagged_log("STRATEGY-GROUP", asset_type, group_id, {
            "group_id": group_id,
            "strategy_type": strategy.strategy_type,
            "n_legs": len(signals),
            "executed_legs": executed_legs,
            "success": result["success"],
            "arb_sets": arb_sets,
            "surplus": {},  # any real surplus was trimmed above
            "edge": round(signals[0].edge, 4),
            "total_cost": round(total_cost, 4),
        })

        # We now hold at least part of this group's position - don't
        # rediscover and re-buy it on a future scan (see
        # _filled_arb_groups' definition in __init__). Gated on at least
        # one real fill: a group where every leg got zero fills holds
        # nothing and should stay eligible for a genuine future attempt.
        if executed_legs > 0:
            self._filled_arb_groups.add(group_id)

    async def _stage_hunt(self):
        """Run market discovery off the event loop.

        get_active_markets() does up to 30 synchronous HTTP round-trips
        (see PolymarketScannerHunter.get_active_markets / CryptoHunter.hunt)
        — calling it directly would block this coroutine's single-threaded
        event loop for the whole scan, starving every other stage (including
        the arbitrage strategy scan) until it returns. asyncio.to_thread
        runs it on a worker thread instead, so the loop stays free.
        """
        t0 = time.monotonic()
        market, hunter = await asyncio.to_thread(self.hunter.get_active_markets, self.log_func)
        elapsed = time.monotonic() - t0
        self.log_func("HUNT-TIMING", "Pipeline", "", {"elapsed_s": round(elapsed, 1)})

        if not market or not hunter:
            self.bridge.status = "No markets found. Waiting..."
            return None
        return market, hunter

    def _stage_evaluate_ev(self, market, hunter) -> CandidateTrade | None:
        token_id = str(getattr(market, "market_id", "") or "")
        asset_type = str(getattr(market, "asset_type", "") or "")
        question = str(getattr(market, "market_name", "") or getattr(market, "question", ""))

        # Every market gets one look per cooldown window. Period. Every exit
        # path below used to call add_to_cooldown() individually and at
        # least one (live_truth unavailable) forgot to - doing it once here,
        # unconditionally, before any evaluation begins, replaces all of them.
        self.hunter.add_to_cooldown(token_id)

        # Dry-run/paper-trade equivalent of the real-portfolio check below -
        # see _simulated_positions' definition in __init__. Neither mode
        # submits a real order, so the real-portfolio check has nothing to
        # catch; this is the session-level backstop for those modes.
        if token_id in self._simulated_positions:
            self.log_func("SCAN-SKIP", asset_type, token_id, {"reason": "already_held_simulated"})
            return None

        # Prevent buying a market already in the portfolio.
        if hasattr(self.bridge, "current_portfolio") and self.bridge.current_portfolio:
            for position in self.bridge.current_portfolio:
                pos_token = str(getattr(position, "asset_id", getattr(position, "token_id", "")))
                if pos_token == token_id:
                    self.log_func("SCAN-SKIP", asset_type, token_id, {"reason": "already_owned_in_portfolio"})
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
            return None

        live_truth = hunter.get_live_truth(market)
        if live_truth is None:
            self.log_func("SCAN-SKIP", asset_type, token_id, {"reason": "live_truth unavailable"})
            return None

        # live_truth is a bare float for most hunters, but CryptoHunter now
        # returns CCXTDataClient's enriched data package (dict) — keep the
        # bridge display field numeric either way.
        self.bridge.market_actual = (
            float(live_truth.get("spot_price") or 0.0) if isinstance(live_truth, dict) else float(live_truth)
        )
        brain = get_brain_for_asset_type(asset_type)
        model_used = getattr(brain, "last_model_used", "unknown")

        # Pricing (Wang Transform -> market blend) is all computed inside
        # evaluate() — see brains/base.py. "legacy" mode asks for both layers
        # disabled (lambda=0, full model weight) so it reduces to the brain's
        # raw probability, for A/B comparison against "wang" mode.
        if self.pricing_mode == "legacy":
            signal = brain.evaluate(
                market, live_truth, min_ev=self.min_ev_threshold,
                wang_lambda=0.0, model_weight=1.0,
            )
        else:
            signal = brain.evaluate(
                market, live_truth, min_ev=self.min_ev_threshold,
                wang_lambda=self.config.wang_lambda,
                model_weight=self.config.model_weight,
            )

        pre_prob = signal.pre_prob
        post_prob = signal.post_prob
        confidence = signal.confidence

        if self.pricing_mode == "legacy":
            wang_lambda = wang_fair_value = wang_edge = None
        else:
            wang_lambda = signal.wang_lambda
            wang_fair_value = signal.wang_fair_value
            wang_edge = signal.wang_edge

            if abs(wang_edge) < self.wang_min_edge:
                self.log_func("SCAN-SKIP", asset_type, token_id, {
                    "reason": "wang_edge below minimum",
                    "pre_prob": round(pre_prob, 4),
                    "wang_lambda": round(wang_lambda, 4),
                    "wang_fair_value": round(wang_fair_value, 4),
                    "wang_edge": round(wang_edge, 4),
                    "wang_min_edge": self.wang_min_edge,
                })
                return None

        self.bridge.forecast = float(post_prob)

        price_yes = float(poly_price)
        price_no = max(1e-9, 1.0 - price_yes)
        fair_no = 1.0 - float(post_prob)
        ev_yes = signal.ev_yes
        ev_no = signal.ev_no
        self.bridge.ev = float(ev_yes)

        side = signal.side
        final_ev = signal.expected_value
        entry_price = float(signal.entry_price)

        # Kelly sizing (BudgetManager.compute_kelly_bet_size): kelly_optimal =
        # edge/odds for whichever side was actually picked — YES odds are
        # (1 - price), NO odds are the complementary YES price — scaled by
        # model confidence and kelly_fraction (quarter-Kelly by default),
        # then clamped to max_bet_size_usd as a hard ceiling.
        if side == "YES":
            kelly_edge = float(post_prob) - price_yes
            kelly_odds = 1.0 - price_yes
        else:
            kelly_edge = fair_no - price_no
            kelly_odds = price_yes  # == 1 - price_no

        kelly_raw_fraction = signal.kelly_size
        kelly_bet_usd = self.budget_manager.compute_kelly_bet_size(kelly_edge, kelly_odds, confidence)

        # Correlation exposure (see trading/correlation.py): how correlated
        # this candidate is with the currently open book, on average — logged
        # for analysis, not yet a hard sizing input (that's future work).
        correlation_exposure = self.portfolio_manager.correlation_exposure_for(asset_type)

        diag = (
            f"[EV-MATH] mode={self.pricing_mode} raw_p={pre_prob:.3f} "
            f"YES(P: {price_yes:.3f}, FV: {float(post_prob):.3f}, EV: {ev_yes:.2f}) | "
            f"NO(P: {price_no:.3f}, FV: {fair_no:.3f}, EV: {ev_no:.2f}) | PICK: {side}"
        )
        if wang_edge is not None:
            diag += f" | wang_lambda={wang_lambda:.3f} wang_edge={wang_edge:.3f}"
        print(diag)
        self.bridge.terminal_logs.appendleft(diag)

        return CandidateTrade(
            market=market,
            token_id=token_id,
            asset_type=asset_type,
            question=question,
            post_prob=float(post_prob),
            pre_prob=pre_prob,
            kelly_size=float(kelly_raw_fraction),
            kelly_bet_usd=float(kelly_bet_usd),
            model_used=model_used,
            price_yes=price_yes,
            side=side,
            ev_yes=float(ev_yes),
            ev_no=float(ev_no),
            final_ev=float(final_ev),
            entry_price=float(entry_price),
            pricing_mode=self.pricing_mode,
            wang_lambda=wang_lambda,
            wang_fair_value=wang_fair_value,
            wang_edge=wang_edge,
            strategy_type="model",
            kelly_fraction_used=self.budget_manager.kelly_fraction,
            correlation_exposure=float(correlation_exposure),
        )

    def _stage_risk_and_budget(self, candidate: CandidateTrade):
        cash_balance = float(self.bridge.current_balance)

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
                **self._analytics_log_fields(candidate),
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

        # candidate.kelly_bet_usd is already confidence/kelly_fraction-scaled
        # and clamped to max_bet_size_usd by BudgetManager.compute_kelly_bet_size();
        # re-apply the ceiling explicitly here too, since it must hold as a
        # hard cap regardless of how the bet size was derived upstream.
        target_bet = candidate.kelly_bet_usd
        desired_bet = min(target_bet, self.max_bet_size_usd)

        budget_bet, budget_ok = self.budget_manager.cap_to_remaining_budget(desired_bet, strategy_tag="crypto")
        if not budget_ok:
            return self._reject("REJECTED", candidate, {
                "market_name": candidate.question,
                "reason": "daily_limit_reached",
                "kelly_size": round(candidate.kelly_size, 4),
                "kelly_bet_usd": round(candidate.kelly_bet_usd, 4),
                "strategy_daily_limit_usd": round(float(self.budget_manager._strategy_limit("crypto")), 4),
                "strategy_spent_today": round(float(self.budget_manager.spent_by_strategy.get("crypto", 0.0)), 4),
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
            fair_value=float(candidate.post_prob),
            ev=float(candidate.final_ev),
            current_poly_price=float(candidate.price_yes),
            bet_amount_usd=float(approved_bet),
            side=candidate.side,
            log_func=self.log_func,
            strategy_tag="crypto",
        )

        if executed:
            self.budget_manager.record_trade(float(approved_bet), strategy_tag="crypto")
            self.spent_today = float(self.budget_manager.total_spent_today)
            self._set_spend(self.spent_today)
            if not self.executor.dry_run:
                self._set_cash(max(0.0, float(self.bridge.current_balance) - float(approved_bet)))

            # Record the simulated fill so _stage_evaluate_ev's
            # already_held_simulated check can catch it next cycle - see
            # _simulated_positions' definition in __init__. token_id (the
            # market's canonical YES id, which is what _stage_evaluate_ev
            # always keys its lookup on) is always added; the NO token is
            # added too on a NO-side trade, in case anything else keys off it.
            self._simulated_positions.add(candidate.token_id)
            if candidate.side == "NO" and hasattr(candidate.market, "no_market_id"):
                no_token = str(candidate.market.no_market_id or "")
                if no_token:
                    self._simulated_positions.add(no_token)

            total_equity = self._total_equity()

            self.log_func("TRACK", candidate.asset_type, candidate.token_id, {
                "market_name": candidate.question,
                "model_used": candidate.model_used,
                "post_prob": round(float(candidate.post_prob), 4),
                "ev": round(float(candidate.final_ev), 4),
                "ev_yes": round(float(candidate.ev_yes), 4),
                "ev_no": round(float(candidate.ev_no), 4),
                "side": candidate.side,
                "kelly": round(float(candidate.kelly_size), 4),
                "bet_usd": round(float(approved_bet), 2),
                **self._analytics_log_fields(candidate),
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

            # Periodic Market Resolution Check (Phase 3)
            if getattr(self.executor, "dry_run", True):
                now = time.time()
                if now - getattr(self, "_last_resolve_check_at", 0.0) >= 900.0:  # 15 minutes
                    self._last_resolve_check_at = now
                    paper = getattr(self.executor, "paper", None)
                    if paper is not None:
                        try:
                            # NOT asyncio.to_thread: pm_trader.Engine opens its
                            # sqlite3 connection (check_same_thread defaults to
                            # True, and the installed package exposes no way to
                            # override it) in whatever thread first constructs
                            # PaperAdapter — the polybot-engine thread that also
                            # runs this loop. to_thread hands the call to a
                            # ThreadPoolExecutor worker thread instead, which
                            # sqlite3 then refuses ("SQLite objects created in a
                            # thread can only be used in that same thread"),
                            # failing resolve_all() on every call. Calling it
                            # directly keeps it on the connection's owning
                            # thread; the resulting blocking network calls
                            # (resolve_all() checks each open position's market
                            # via the Gamma API) are accepted here the same way
                            # _get_order_filled_shares()/_cancel_order() accept
                            # blocking calls elsewhere in this sequential loop —
                            # this only runs once per 15 minutes.
                            paper.resolve_closed_markets()
                        except Exception as e:
                            self.log_func("PAPER-ERROR", "Engine", "resolve_all", {"error": str(e)})

            requested_live = bool(getattr(self.bridge, "live_trading", False))
            self.executor.dry_run = not requested_live

            sync_live_account_state(self.bridge, self.executor, self.portfolio_manager, self.log_func)
            self.portfolio_manager.manage_portfolio(self.log_func)  # exits always run, even while paused
            sync_live_account_state(self.bridge, self.executor, self.portfolio_manager, self.log_func)
            self._update_drawdown_guard()

            if self.ctx.drawdown_paused:
                continue  # circuit breaker: skip new-entry scanning/execution below

            await self._stage_strategy_scan()

            stage1 = await self._stage_hunt()
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


def _default_log_func(ctx: WalletContext):
    """Bare log_func that just persists events to this wallet's own DB.

    Callers that want richer behavior (e.g. the dashboard's terminal-log
    formatting) build and pass their own log_func instead — this default only
    exists so a caller like WalletManager can drive a wallet from nothing more
    than its WalletContext.
    """
    import ui.data_manager as data_manager

    def _log(level, asset_type, token_id, payload):
        data_manager.log_event(ctx.bridge, level, asset_type, token_id, payload, db_path=ctx.db_path)

    return _log


async def run_market_monitor(ctx: WalletContext, log_func=None, delay: float | None = None):
    """Canonical entrypoint for the trading monitor loop.

    log_func stays a separate, optional parameter (not a WalletContext field)
    because it's the caller's event-logging closure — it already captures
    ctx.db_path itself when persisting events (see ui/dashboard.py's
    _log_event), so the pipeline never needs to know about database paths
    directly. Omit it to fall back to a bare per-wallet logger.
    """
    if log_func is None:
        log_func = _default_log_func(ctx)
    pipeline = SequentialTradingPipeline(ctx=ctx, log_func=log_func, delay=delay)
    await pipeline.run_forever()

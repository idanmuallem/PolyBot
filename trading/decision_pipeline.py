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
from datetime import datetime, timedelta, timezone
from typing import Optional

from brains import get_brain_for_asset_type
from brains.base import calculate_tte
from core.models import PRICE_FLOOR, PRICE_CEILING, MarketData
from core.wallet_context import WalletContext
from core.polymarket_client import PolymarketClient
from trading.account import Account, Refused
from trading.gate import Skip
from trading.kelly import kelly_bet_size
from trading.strategies import EventSumStrategy, Strategy
from trading.venue import MarketRef, normalize_side
from ui import data_manager


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


@dataclass
class CandidateTrade:
    market: MarketData
    token_id: str
    asset_type: str
    question: str
    post_prob: float
    pre_prob: float
    kelly_size: float       # raw Kelly criterion fraction (edge/odds), before confidence/kelly_fraction scaling
    kelly_bet_usd: float    # Kelly dollar bet size — confidence- and kelly_fraction-scaled, before the Account's caps
    model_used: str
    price_yes: float
    side: str
    ev_yes: float
    ev_no: float
    final_ev: float
    entry_price: float
    pricing_mode: str = "wang"
    wang_lambda: Optional[float] = None  # name kept for continuity; holds entry_k
    # (logit-shrink strength) as of the entry-side redesign, not a Wang
    # probit lambda. See core/models.py TradeSignal.wang_lambda.
    wang_fair_value: Optional[float] = None
    wang_edge: Optional[float] = None
    strategy_type: str = "model"       # "model" for brain-driven trades, "arbitrage" etc. for strategies
    kelly_fraction_used: float = 0.0   # the config.kelly_fraction applied when this bet was sized
    correlation_exposure: float = 0.0  # avg correlation with the currently open book (see trading/correlation.py)
    # Entry-snapshot inputs, logged so forward calibration (see
    # scripts/forward_calibration.py) can bucket resolved trades by moneyness
    # and horizon the way PHASE1_FINDINGS.md did - these cannot be
    # reconstructed after the fact, so capture them at entry time.
    spot: float = 0.0                  # underlying spot price at evaluation
    tte_days: float = 0.0              # time-to-expiry in days at evaluation


class SequentialTradingPipeline:
    def __init__(self, ctx: WalletContext, log_func, delay: float | None = None, clock=None):
        if not all([ctx.executor, ctx.scanner, ctx.portfolio_manager]):
            missing = [name for name, val in [
                ("executor", ctx.executor),
                ("scanner", ctx.scanner),
                ("portfolio_manager", ctx.portfolio_manager),
            ] if val is None]
            raise ValueError(f"WalletContext missing required components: {missing}")

        self.ctx = ctx
        self.bridge = ctx.bridge
        self.config = ctx.config
        self.db_path = ctx.db_path
        self.log_func = log_func
        self.loop_delay = float(delay) if delay is not None else float(self.config.loop_delay_seconds)
        # UTC "now" for the Account's daily budget reset.
        self.clock = clock or (lambda: datetime.now(timezone.utc))
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

        # The pipeline owns the active Venue: every Venue this wallet can
        # trade on, by name, and the one the executor trades on now. A mode
        # change from the dashboard swaps it at a tick boundary (see
        # _apply_mode_request()), so no tick runs half-paper, half-live.
        self.venue = getattr(self.executor, "venue", None)
        self.venues = dict(getattr(ctx, "venues", None) or {})
        if self.venue is not None:
            self.venues.setdefault(self.venue.name, self.venue)
        self._refused_mode: str | None = None
        self._last_resolve_check_at = 0.0
        self._last_drift = None  # last logged Venue/Ledger drift (see _reconcile())
        # Live entries are resting limit orders: a holding the CLOB has never
        # shown is closed as unfilled after a grace period (see
        # _close_unfilled_entries()). Only holdings opened since this start.
        self._started_at = self.clock()
        self._seen_on_venue: set[str] = set()
        # Typed trade record (trading/ledger.py). The engine is its only
        # writer; None (e.g. a bare test context) means not recording.
        self.ledger = getattr(ctx, "ledger", None)

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
        # re-evaluating the same groups every 30s was pure waste. Cleared
        # when the Account starts a new budget day (see tick()).
        self._exhausted_arb_groups: set[str] = set()

        self.portfolio_manager = ctx.portfolio_manager

        # Entry and exit eligibility (already held on either side, post-exit
        # cooldown, exit attempt cap) for model entries, arbitrage baskets
        # and every exit path. One Gate, shared with the PortfolioManager
        # that makes the exits, so an exit's cooldown blocks re-entry here.
        # Holdings come from the Ledger, so the answer survives restarts.
        self.gate = self.portfolio_manager.gate

        # The single writer of cash, equity, the drawdown breaker and the
        # daily budget (spend limits, trade counts, max-bet ceiling) for
        # model and arbitrage entries alike. It reads whichever Venue the
        # pipeline has active, so a mode swap at the tick boundary moves
        # cash with it. Exits that need fresh cash (free_up_capital) read it
        # through the same Account.
        self.account = Account(
            venue=lambda: self.venue,
            bridge=self.bridge,
            ledger=self.ledger,
            max_drawdown_pct=float(getattr(self.config, "max_drawdown_pct", 0.20)),
            log_func=log_func,
            config=self.config,
        )
        self.account.daily_reset(self.clock())
        ctx.account = self.account
        self.portfolio_manager.account = self.account

        self._sync_account()
        if float(getattr(self.bridge, "starting_balance", 0.0) or 0.0) <= 0.0:
            self.bridge.starting_balance = self.account.cash

    def _sync_account(self):
        """Refresh the positions list risk works from, then the Account
        (cash, equity, drawdown breaker) from the active Venue."""
        try:
            self.portfolio_manager._refresh_portfolio()
        except Exception as exc:
            self.log_func("SYNC-WARN", "Pipeline", "portfolio", {"reason": "positions_fetch_failed", "error": str(exc)})
        self.account.refresh()

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
        # Top-level kill switch (config.enable_arbitrage) — checked first,
        # before the scan-interval throttle and before any Gamma API call.
        # self.strategies is arbitrage-only today (EventSumStrategy), so
        # this is the actual entry point for the whole arbitrage path: when
        # disabled, zero event-discovery calls happen and zero strategy.scan()
        # calls happen — not "runs but always rejects downstream". A budget
        # cap like ARBITRAGE_MAX_DAILY_TRADES=0 would still scan/evaluate
        # every cycle for nothing; this early-return skips that work entirely.
        if not self.config.enable_arbitrage:
            return

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
        return MarketRef.from_market(signal.market).token_for(signal.side)

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

        # The Gate refuses a basket that is fully held (or can't be
        # completed: a leg held on its other side, or cooling down after an
        # exit) and otherwise names the legs still missing. Only those are
        # bought, and only new capital goes through the TTE/budget/trade-count
        # gates below. `signals` (the full basket) and `held_legs` are kept
        # for the group-level profitability re-verification further down,
        # which judges completing the basket against its real total cost:
        # the sunk cost of what's already held plus what's newly bought.
        decision = self.gate.can_enter(signals)
        if isinstance(decision, Skip):
            tagged_log("SCAN-SKIP", asset_type, group_id, {"reason": decision.reason, **decision.detail})
            return
        missing_signals = list(decision.legs)
        held_legs = list(decision.held)  # (signal, Ledger Holding)
        held_signals = [signal for signal, _holding in held_legs]

        if self._group_has_leg_beyond_max_tte(missing_signals, group_id, asset_type, tagged_log):
            self._exhausted_arb_groups.add(group_id)
            return

        total_cost = sum(float(s.bet_amount_usd) for s in missing_signals)
        cash_balance = self.account.cash

        # Distinct reject reasons (Point 5): "can't afford it" (wallet cash)
        # vs "budget exhausted" (this strategy's own daily allocation) are
        # different failure modes worth telling apart in the log/analytics
        # layer, so they're no longer folded into one generic reason.
        if total_cost > cash_balance:
            tagged_log("REJECTED", asset_type, group_id, {
                "reason": "insufficient_cash",
                "group_id": group_id,
                "n_legs": len(missing_signals),
                "total_cost": round(total_cost, 4),
                "cash_balance": round(cash_balance, 4),
            })
            return

        # Every leg is one trade: the basket is reserved whole against the
        # same daily spend limits, trade counts and max-bet ceiling as a
        # model entry (Account.reserve()). It is all-or-nothing, so a
        # partial approval is a rejection too.
        budget = self.account.reserve("arbitrage", total_cost, trades=len(missing_signals))
        if isinstance(budget, Refused) or budget.usd < total_cost:
            reason, detail = (
                (budget.reason, budget.detail) if isinstance(budget, Refused)
                else ("insufficient_budget", {"approved_usd": round(budget.usd, 4)})
            )
            tagged_log("REJECTED", asset_type, group_id, {
                "reason": reason,
                "group_id": group_id,
                "n_legs": len(missing_signals),
                "total_cost": round(total_cost, 4),
                **detail,
            })
            self._exhausted_arb_groups.add(group_id)
            return

        legs = []
        for signal in missing_signals:
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

        # Every share that filled is a real entry, whatever happens to the
        # basket below (kept, trimmed or unwound): record each leg's fill
        # in the Ledger now, and each sale back as it happens.
        # Each filled leg is also one trade and its leg's dollars of spend
        # in the Account, and cash is re-read from the Venue.
        for signal in missing_signals:
            filled = float(fills.get(self._leg_token_id(signal), 0.0))
            self._record_arbitrage_entry(signal, group_id, filled)
            if filled > 0:
                self.account.record_fill("arbitrage", float(signal.bet_amount_usd))
        self.account.refresh()

        def _sell_back(signal, tid, shares, reason):
            if self.executor.sell_position(tid, shares, signal.price, tagged_log):
                self._record_arbitrage_sale(signal, tid, shares, reason)

        # Cap the guaranteed-set size by the ALREADY-HELD legs' real share
        # counts too, not just this attempt's fills. A top-up buy is sized
        # to the same fixed trade_size_usd as any other leg, so it can land
        # far more shares per dollar on a newly-added, cheap long-shot leg
        # than the originally-bought legs hold (e.g. a long-shot candidate
        # at $0.02 buys ~50 shares/$1; legs bought earlier around $0.25
        # might only hold ~4 shares each) — the true guaranteed structure
        # is bounded by the SCARCEST leg across the WHOLE basket, held or
        # new, not just what got bought in this one attempt.
        if arb_sets > 0 and held_legs:
            arb_sets = min(arb_sets, min(float(holding.shares) for _s, holding in held_legs))

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
        #
        # held_signals' cost is sunk — already spent whether or not this
        # top-up completes the basket — but still counts toward whether the
        # COMPLETE basket clears the guaranteed-profit margin: ignoring it
        # would let a basket with an expensive already-held leg look
        # "profitable" forever just because today's missing leg is cheap,
        # even though the finished basket, taken as a whole, costs more
        # than its $1 guaranteed payout.
        if arb_sets > 0:
            per_leg_unit_cost: dict = {}
            for signal in missing_signals:
                tid = self._leg_token_id(signal)
                filled = fills.get(tid, 0.0)
                if filled > 0:
                    per_leg_unit_cost[tid] = float(signal.bet_amount_usd) / filled

            held_unit_costs = [float(holding.entry_price) for _s, holding in held_legs]

            guaranteed_payout = arb_sets * 1.0
            guaranteed_cost = (
                sum(unit_cost * arb_sets for unit_cost in per_leg_unit_cost.values())
                + sum(unit_cost * arb_sets for unit_cost in held_unit_costs)
            )
            net_edge = guaranteed_payout - guaranteed_cost

            if net_edge < guaranteed_payout * _MIN_ARBITRAGE_MARGIN:
                # Real fills don't clear a genuine margin after real costs.
                # Keeping this would just be naked directional risk on
                # whichever illiquid long-shot outcomes happened to fill —
                # not arbitrage. Unwind what was newly bought this attempt
                # (held_signals' shares are sunk from an earlier attempt —
                # not this one's to unwind; leaving them held merely
                # reverts to the same partial-basket state this attempt
                # started from, eligible for another top-up try later).
                for signal in missing_signals:
                    tid = self._leg_token_id(signal)
                    filled = fills.get(tid, 0.0)
                    if filled > 0:
                        _sell_back(signal, tid, filled, "ARBITRAGE-UNWIND")

                tagged_log("REJECTED", asset_type, group_id, {
                    "reason": "not_profitable_after_real_fills",
                    "group_id": group_id,
                    "n_legs": len(missing_signals),
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
        for signal in missing_signals:
            tid = self._leg_token_id(signal)
            filled = fills.get(tid, 0.0)
            excess = round(filled - arb_sets, 4)
            if excess > 0:
                _sell_back(signal, tid, excess, "ARBITRAGE-UNWIND" if arb_sets <= 0 else "ARBITRAGE-TRIM")
                if arb_sets <= 0:
                    unwound_incomplete_group = True
                fills[tid] = arb_sets

        if unwound_incomplete_group:
            tagged_log("REJECTED", asset_type, group_id, {
                "reason": "group_incomplete_unwound_partial_fills",
                "group_id": group_id,
                "n_legs": len(missing_signals),
                "unfilled": result.get("unfilled", []),
            })
            self._exhausted_arb_groups.add(group_id)
            return

        executed_legs = 0
        for signal in missing_signals:
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

        tagged_log("STRATEGY-GROUP", asset_type, group_id, {
            "group_id": group_id,
            "strategy_type": strategy.strategy_type,
            "n_legs": len(missing_signals),
            "legs_already_held": len(held_signals),
            "legs_total_in_basket": len(signals),
            "executed_legs": executed_legs,
            "success": result["success"],
            "arb_sets": arb_sets,
            "surplus": {},  # any real surplus was trimmed above
            "edge": round(signals[0].edge, 4),
            "total_cost": round(total_cost, 4),
        })

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

        # Already held (YES or NO side, per the Ledger) or inside the
        # post-exit cooldown: see trading/gate.py.
        decision = self.gate.can_enter(market)
        if isinstance(decision, Skip):
            self.log_func("SCAN-SKIP", asset_type, token_id, {"reason": decision.reason, **decision.detail})
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

        # Pricing (market-anchored interpolation between market and brain) is
        # computed inside evaluate() — see brains/base.py. "legacy" mode uses
        # entry_k=1.0, which under the market-anchored design means "trust the
        # brain fully" (post_prob == raw brain probability, market ignored) —
        # exactly the raw-probability A/B baseline. entry_k=0.0 would be the
        # opposite (defer fully to market).
        if self.pricing_mode == "legacy":
            signal = brain.evaluate(
                market, live_truth, min_ev=self.min_ev_threshold,
                entry_k=1.0,
            )
        else:
            signal = brain.evaluate(
                market, live_truth, min_ev=self.min_ev_threshold,
                entry_k=self.config.entry_k,
            )

        # Read AFTER evaluate() — get_brain_for_asset_type() constructs a
        # fresh brain every call, so last_model_used only reflects this
        # scan's actual choice (short_term/standard_bs/first_passage/...)
        # once evaluate() -> _calculate_probability() has set it. Reading it
        # any earlier just captures __init__'s default, permanently.
        model_used = getattr(brain, "last_model_used", "unknown")

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

        # Kelly sizing (trading/kelly.py): kelly_optimal = edge/odds for
        # whichever side was actually picked — YES odds are (1 - price), NO
        # odds are the complementary YES price — scaled by model confidence
        # and kelly_fraction (quarter-Kelly by default). The max-bet ceiling
        # and daily budget are applied by Account.reserve() at the risk stage.
        if side == "YES":
            kelly_edge = float(post_prob) - price_yes
            kelly_odds = 1.0 - price_yes
        else:
            kelly_edge = fair_no - price_no
            kelly_odds = price_yes  # == 1 - price_no

        kelly_raw_fraction = signal.kelly_size
        kelly_bet_usd = kelly_bet_size(
            kelly_edge, kelly_odds, confidence,
            kelly_fraction=float(self.config.kelly_fraction),
            bankroll_usd=float(self.config.bankroll_usd),
        )

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
            kelly_fraction_used=float(self.config.kelly_fraction),
            correlation_exposure=float(correlation_exposure),
            spot=(
                float(live_truth.get("spot_price") or 0.0)
                if isinstance(live_truth, dict) else float(live_truth)
            ),
            tte_days=float(calculate_tte(getattr(market, "expiry_date", None))),
        )

    def _stage_risk_and_budget(self, candidate: CandidateTrade):
        cash_balance = self.account.cash

        if cash_balance < self.safe_minimum:
            self.bridge.status = "Portfolio Management Mode (cash below $1.00)"
            self.log_func("PORTFOLIO-MODE", "Pipeline", "cash_guard", {
                "reason": "insufficient_cash_for_new_entries",
                "cash": round(cash_balance, 4),
                "open_positions_value": round(self.account.snapshot().open_position_value, 4),
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

        # The Account caps the Kelly size at the max-bet ceiling and what is
        # left of today's budget, or refuses it (budget spent, daily trade
        # count reached): the same rules arbitrage baskets go through.
        target_bet = candidate.kelly_bet_usd
        budget = self.account.reserve("crypto", target_bet)
        if isinstance(budget, Refused):
            return self._reject("REJECTED", candidate, {
                "market_name": candidate.question,
                "reason": budget.reason,
                "kelly_size": round(candidate.kelly_size, 4),
                "kelly_bet_usd": round(candidate.kelly_bet_usd, 4),
                **budget.detail,
            })
        desired_bet = budget.usd

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
                except Exception as exc:
                    print(f"[PIPELINE] free_up_capital failed: {exc}")

            # What the sales actually freed, as the Venue reports it.
            available_cash = self.account.refresh().cash

        approved_bet = min(desired_bet, available_cash)
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
            })

        return approved_bet, {
            "available_cash": available_cash,
            "target_bet": target_bet,
            "desired_bet": desired_bet,
        }

    def _record_model_entry(self, candidate: CandidateTrade, approved_bet: float) -> None:
        """Record one model entry fill in the Ledger, with its entry details.
        Token, price and shares mirror TradeExecutor.evaluate_and_execute():
        a NO-side buy is of the NO token at 1 - YES price."""
        if self.ledger is None:
            return
        side = normalize_side(candidate.side)
        token_id = MarketRef.from_market(candidate.market).token_for(side)
        price = float(candidate.entry_price)
        shares = math.floor((float(approved_bet) / price) * 100.0) / 100.0 if price > 0 else 0.0
        self.ledger.record_entry(
            token_id=token_id,
            condition_id=getattr(candidate.market, "condition_id", None),
            side=side,
            shares=shares,
            price=price,
            venue=self._ledger_venue(),
            strategy="crypto",
            asset_type=candidate.asset_type,
            market_name=candidate.question,
            pre_prob=float(candidate.pre_prob),
            post_prob=float(candidate.post_prob),
            wang_edge=candidate.wang_edge,
        )

    def _ledger_venue(self) -> str:
        """Ledger venue tag for a fill: the name ("paper" / "live") of the
        active Venue."""
        return getattr(self.venue, "name", "paper")

    def _apply_mode_request(self) -> None:
        """Tick boundary: trade on the Venue the dashboard asked for.

        Polled from engine_control (DB) rather than read off self.bridge:
        the dashboard runs as a separate OS process (see run_engine.py).
        Defaults to paper when no control row exists yet (see
        data_manager.read_live_trading_requested()). A mode with no Venue
        (live without a usable CLOB client) is refused, logged once per
        request.
        """
        wanted = "live" if data_manager.read_live_trading_requested(self.db_path) else "paper"
        current = getattr(self.venue, "name", None)
        if wanted == current:
            self._refused_mode = None
            return
        venue = self.venues.get(wanted)
        if venue is None:
            if self._refused_mode != wanted:
                self._refused_mode = wanted
                self.log_func("MODE-SWITCH-REFUSED", "Engine", wanted, {
                    "requested": wanted, "active": current, "reason": f"no_{wanted}_venue",
                })
            return
        self._refused_mode = None
        self.venue = venue
        self.executor.venue = venue
        self.log_func("MODE-SWITCH", "Engine", wanted, {"from": current, "to": wanted})

    def _record_arbitrage_entry(self, signal, group_id: str, filled_shares: float) -> None:
        """Record one arbitrage leg fill in the Ledger, tagged with its
        basket's group_id. Entry price is the leg's limit price."""
        if self.ledger is None or filled_shares <= 0:
            return
        self.ledger.record_entry(
            token_id=self._leg_token_id(signal),
            condition_id=getattr(signal.market, "condition_id", None),
            side=signal.side,
            shares=filled_shares,
            price=float(signal.price),
            venue=self._ledger_venue(),
            strategy="arbitrage",
            asset_type=signal.market.asset_type,
            market_name=signal.market.market_name,
            group_id=group_id,
        )

    def _record_arbitrage_sale(self, signal, token_id: str, shares: float, reason: str) -> None:
        """Record a confirmed trim / unwind sale of an arbitrage leg."""
        if self.ledger is None:
            return
        self.ledger.record_exit(
            token_id=token_id, shares=float(shares), price=float(signal.price), reason=reason,
            venue=self._ledger_venue(), strategy="arbitrage", side=signal.side,
            market_name=signal.market.market_name,
        )

    def _resolve_closed_markets(self) -> None:
        """Every 15 minutes, settle closed markets on every Venue this wallet
        holds (paper positions keep resolving while trading live), recording
        each settlement in the Ledger under that Venue's name.

        NOT asyncio.to_thread: PaperVenue's pm_trader sqlite3 connection is
        bound to the thread that first used it -- this loop's thread (see
        trading/paper_venue.py). to_thread hands the call to a worker thread,
        which sqlite3 refuses, failing every resolution (28c7637). The
        blocking network calls this makes are accepted the same way other
        blocking calls are in this sequential loop.

        resolve_arbitrage=config.enable_arbitrage: the ENABLE_ARBITRAGE kill
        switch also covers this cleanup path, not just new-trade entry.
        """
        now = time.time()
        if now - self._last_resolve_check_at < 900.0:
            return
        self._last_resolve_check_at = now
        for venue in self.venues.values():
            try:
                venue.resolve_closed_markets(
                    resolve_arbitrage=self.config.enable_arbitrage,
                    log_func=self.log_func,
                    on_resolved=lambda resolved, name=venue.name: self._record_resolution(resolved, name),
                )
            except Exception as e:
                self.log_func("RESOLVE-ERROR", "Engine", venue.name, {"venue": venue.name, "error": str(e)})

    # Share counts within this of each other agree: venues floor or round
    # fills to cents, the Ledger records the requested size.
    _RECONCILE_SHARE_TOLERANCE = 0.02

    def _reconcile(self) -> None:
        """Compare the active Venue's holdings (authoritative for share
        quantities) with the Ledger's open positions on that venue, and log
        drift: tokens only one side holds, and share-count mismatches.

        Detection only; nothing is corrected. A drift is logged
        (RECONCILE-DRIFT) when it first appears or changes, and its
        clearing is logged (RECONCILE-OK), so an unchanged drift doesn't
        flood the event log every tick.
        """
        if self.ledger is None or self.venue is None:
            return
        try:
            venue_shares: dict = {}
            for p in self.venue.open_positions():
                token = str(getattr(p, "token_id", "") or "")
                venue_shares[token] = venue_shares.get(token, 0.0) + float(getattr(p, "shares", 0.0) or 0.0)
            ledger_shares = {h.token_id: float(h.shares) for h in self.ledger.open_positions(venue=self.venue.name)}
        except Exception as exc:
            self.log_func("SYNC-WARN", "Pipeline", "reconcile", {"reason": "reconcile_failed", "error": str(exc)})
            return

        self._seen_on_venue.update(t for t, n in venue_shares.items() if n > 0)
        if self.venue.name == "live" and self._close_unfilled_entries(venue_shares):
            ledger_shares = {h.token_id: float(h.shares) for h in self.ledger.open_positions(venue=self.venue.name)}

        venue_only = {t: n for t, n in venue_shares.items() if t not in ledger_shares and n > 0}
        ledger_only = {t: n for t, n in ledger_shares.items() if t not in venue_shares}
        mismatch = {
            t: {"venue": venue_shares[t], "ledger": n}
            for t, n in ledger_shares.items()
            if t in venue_shares and abs(venue_shares[t] - n) > self._RECONCILE_SHARE_TOLERANCE
        }
        drift = {"venue_only": venue_only, "ledger_only": ledger_only, "share_mismatch": mismatch}
        signature = (self.venue.name, repr(sorted((k, repr(sorted(v.items()))) for k, v in drift.items())))
        previous = self._last_drift
        if any(drift.values()):
            if signature != previous:
                self.log_func("RECONCILE-DRIFT", "Pipeline", self.venue.name, {"venue": self.venue.name, **drift})
            self._last_drift = signature
        elif previous is not None:
            self.log_func("RECONCILE-OK", "Pipeline", self.venue.name, {"venue": self.venue.name})
            self._last_drift = None

    # How long a live entry may go unseen on the CLOB before it counts as
    # never filled (the Data API lags fills by up to a minute or so).
    _UNFILLED_GRACE = timedelta(minutes=5)

    def _close_unfilled_entries(self, venue_shares: dict) -> bool:
        """Close, as unfilled, the live model entries opened since this
        engine started that the venue has never shown once the grace period
        is over. ClobVenue.buy() posts a GTC limit order, and the CLOB's
        response only says it was accepted, not that it filled; an order
        that rests unfilled would otherwise stay a Ledger holding for ever.
        Arbitrage legs are recorded from polled fills, so they are left
        alone. True if anything was closed."""
        now = self.clock()
        closed = False
        for h in self.ledger.open_positions(venue=self.venue.name):
            if h.strategy != "crypto" or h.token_id in venue_shares or h.token_id in self._seen_on_venue:
                continue
            opened = _parse_utc(h.opened_at)
            if opened is None or opened < self._started_at - timedelta(seconds=1):
                continue
            if now - opened < self._UNFILLED_GRACE:
                continue
            self.ledger.record_unfilled(token_id=h.token_id, venue=self.venue.name)
            self.log_func("RECONCILE-UNFILLED", "Pipeline", h.token_id, {
                "venue": self.venue.name, "shares": h.shares, "entry_price": h.entry_price,
                "opened_at": h.opened_at,
                "reason": "live entry order never showed up as a position on the venue",
            })
            closed = True
        return closed

    def _record_resolution(self, resolved: dict, venue: str) -> None:
        """Record one market resolution reported by *venue* (see
        Venue.resolve_closed_markets(on_resolved=...)) in the Ledger."""
        if self.ledger is None:
            return
        self.ledger.record_resolution(
            token_id=resolved["token_id"],
            shares=float(resolved["shares"]),
            payout_price=float(resolved["payout_price"]),
            venue=venue,
            entry_price=resolved.get("entry_price"),
            side=resolved.get("side"),
            market_name=resolved.get("market_name"),
        )

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
            self._record_model_entry(candidate, approved_bet)
            self.account.record_fill("crypto", float(approved_bet))
            account = self.account.refresh()
            total_equity = account.equity

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
                "spent_today": round(float(account.spent_today), 2),
                "strike_price": float(getattr(candidate.market, "strike_price", 0.0) or 0.0),
                "expiry_date": str(getattr(candidate.market, "expiry_date", "") or ""),
                # Entry-snapshot fields for forward calibration (joined to the
                # EXPIRED resolution row by token_id — see
                # scripts/forward_calibration.py). market_price_yes is the
                # market's implied YES probability at entry (the "market_prob"
                # the brain's divergence is measured against); pre_prob (in the
                # analytics fields) is the brain's raw probability; spot/tte_days
                # let the analysis bucket by moneyness and horizon. None of
                # these can be reconstructed after the fact.
                "market_price_yes": round(float(candidate.price_yes), 4),
                "spot": round(float(getattr(candidate, "spot", 0.0)), 2),
                "tte_days": round(float(getattr(candidate, "tte_days", 0.0)), 4),
            })

        self.hunter.add_to_cooldown(candidate.token_id)

    async def run_forever(self):
        while True:
            await asyncio.sleep(self.loop_delay)
            await self.tick()

    async def tick(self):
        """One engine loop iteration: mode switch, resolution, sync, exits,
        Venue/Ledger reconcile, status flush, then new entries (unless
        paused)."""
        if self.account.daily_reset(self.clock()):
            self._exhausted_arb_groups.clear()

        self._apply_mode_request()
        self._resolve_closed_markets()

        self._sync_account()
        self.portfolio_manager.manage_portfolio(self.log_func)  # exits always run, even while paused
        self._sync_account()
        account = self.account.snapshot()
        self._reconcile()

        # Flush this tick's live state to the DB for the dashboard
        # process to read — every iteration, unconditionally, before any
        # of the early returns below, so it doesn't skip ticks where
        # hunting finds nothing or a candidate gets rejected (the common
        # case). See ui/data_manager.py's engine_status/open_positions
        # tables and run_engine.py.
        data_manager.write_engine_status(
            db_path=self.db_path,
            balance=account.cash,
            starting_balance=float(self.bridge.starting_balance),
            position_value=account.open_position_value,
            unrealized_pnl=account.unrealized_pnl,
            watch_only=account.watch_only,
            venue=self.venue.name,
        )
        data_manager.write_open_positions(
            db_path=self.db_path,
            positions=list(getattr(self.bridge, "current_portfolio", None) or []),
            position_analytics=dict(getattr(self.bridge, "position_analytics", None) or {}),
        )

        if account.drawdown_paused:
            return  # circuit breaker: skip new-entry scanning/execution below

        await self._stage_strategy_scan()

        stage1 = await self._stage_hunt()
        if not stage1:
            return

        market, hunter = stage1
        candidate = self._stage_evaluate_ev(market, hunter)
        if candidate is None:
            return

        approved_bet, risk_context = self._stage_risk_and_budget(candidate)
        if approved_bet <= 0.0 or risk_context is None:
            return

        self._stage_execute(candidate, approved_bet, risk_context)


def _parse_utc(stamp: str) -> Optional[datetime]:
    """A Ledger time ("YYYY-MM-DD HH:MM:SS", UTC) as an aware datetime."""
    try:
        return datetime.strptime(str(stamp), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _default_log_func(ctx: WalletContext):
    """Bare log_func that just persists events to this wallet's own DB.

    This is what run_engine.py actually uses today (it calls
    run_market_monitor(ctx) with no log_func) — the dashboard no longer
    builds a richer one of its own; it now reads the same hunt_history rows
    back out via ui/data_manager.py's query functions instead (see
    get_terminal_feed(), get_level_counts(), etc.). A future caller that
    wants different logging behavior can still pass its own log_func here.
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

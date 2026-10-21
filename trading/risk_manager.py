import time
from typing import List, Optional

from ui import data_manager

from brains.pricing_engine import PricingEngine
from trading.correlation import CorrelationTracker
from trading.gate import MAX_EXIT_ATTEMPTS, Gate, Ok


class PortfolioManager:
    def __init__(self, bridge, executor, config, hunter=None, db_path: Optional[str] = None, ledger=None,
                 gate=None, account=None):
        self.bridge = bridge
        self.executor = executor
        self.hunter = hunter
        self.db_path = db_path  # this wallet's trades.db — no guessing, the caller (WalletContext) owns it
        # Ledger (trading/ledger.py): every confirmed sale and expiry made
        # here is recorded in it exactly once. None = not recording.
        self.ledger = ledger
        # Account (trading/account.py): the single source of cash. Sales
        # here never adjust cash themselves; free_up_capital re-reads it
        # after each sale. None (a bare test context) reads the Venue.
        self.account = account
        self.take_profit_pct = float(config.take_profit_pct)
        self.stop_loss_pct = float(config.stop_loss_pct)
        self.min_hold_ev = float(config.min_hold_ev)

        # ENABLE_ARBITRAGE kill switch: manage_portfolio() below skips every
        # arbitrage-tagged position entirely when this is False, so that
        # cleanup/exit handling — not just new-trade entry — also stops
        # touching arbitrage positions. Crypto positions are unaffected.
        self.enable_arbitrage = bool(getattr(config, "enable_arbitrage", True))

        # Wang edge decay exit (see _check_wang_edge_decay): an additional,
        # earlier exit signal alongside take-profit/stop-loss, not a
        # replacement for them.
        self.wang_min_edge = float(getattr(config, "wang_min_edge", 0.05))
        self.wang_decay_threshold = self.wang_min_edge / 2.0
        self.pricing_engine = PricingEngine(
            base_lambda=float(getattr(config, "wang_base_lambda", 0.183))
        )

        # Correlation exposure (see trading/correlation.py) — used both to
        # log correlation_exposure on new candidates and to feed the
        # dashboard's correlation matrix visualization.
        self.correlation_tracker = CorrelationTracker(config=config)
        self._last_snapshot_at: float = 0.0

        # Exit eligibility (attempt cap) and the post-exit cooldown live in
        # the Gate (trading/gate.py). Every exit path here asks
        # gate.can_exit before selling and reports the outcome; the pipeline
        # asks the same Gate before entering.
        self.gate = gate if gate is not None else Gate(ledger, venue=self._venue)

    def _refresh_portfolio(self):
        positions = self.executor.get_open_positions()
        # The positions list risk works from. Their value, unrealized P&L
        # and cash are the Account's (trading/account.py).
        self.bridge.current_portfolio = positions
        # Paper tracking mark-to-market snapshot (Phase 3)
        if getattr(self.executor, "dry_run", True):
            now = time.time()
            if now - self._last_snapshot_at >= 180.0:
                cash = self.executor.get_balance()
                positions_value = sum(float(getattr(p, "value", 0.0) or 0.0) for p in positions)
                if self.db_path:
                    data_manager.insert_paper_snapshot(self.db_path, cash, positions_value)
                self._last_snapshot_at = now

    @staticmethod
    def _position_field(position, field_name, default=None):
        if isinstance(position, dict):
            return position.get(field_name, default)
        return getattr(position, field_name, default)

    def _entry(self, position):
        """This position's entry details as the Ledger recorded them (a
        trading.ledger.Holding), or None when the Ledger has no open
        holding for its token (or there is no Ledger)."""
        if self.ledger is None:
            return None
        token_id = str(self._position_field(position, "token_id", "") or "")
        if not token_id:
            return None
        try:
            return self.ledger.holding(token_id)
        except Exception as exc:
            print(f"[PORTFOLIO] Ledger lookup failed for {token_id}: {exc}")
            return None

    def _entry_field(self, position, field_name: str):
        entry = self._entry(position)
        return getattr(entry, field_name, None) if entry is not None else None

    def _resolve_position_fair_value(self, position):
        """The model's fair value (post_prob) recorded at entry."""
        return self._entry_field(position, "post_prob")

    def _resolve_position_raw_probability(self, position):
        """The brain's raw probability estimate (p_true) recorded at entry,
        needed to re-derive the Wang edge without re-running the brain for
        a position that's already open."""
        return self._entry_field(position, "pre_prob")

    def _resolve_position_asset_type(self, position):
        """This position's asset_type (e.g. "Crypto::BTCUSDT"), needed for
        correlation exposure, since Position itself doesn't carry it."""
        return self._entry_field(position, "asset_type") or None

    def _is_arbitrage_position(self, position) -> bool:
        """True if the Ledger tags this position as an arbitrage leg. Used
        only by manage_portfolio()'s ENABLE_ARBITRAGE gate below."""
        return self._entry_field(position, "strategy") == "arbitrage"

    def _resolve_position_entry_wang_edge(self, position):
        """The Wang edge recorded at entry, for the dashboard's edge-decay display."""
        return self._entry_field(position, "wang_edge")

    def _recompute_position_wang_edge(self, position) -> Optional[dict]:
        """Re-derive the CURRENT Wang edge for an open position from its
        entry-time raw probability and the position's current market price.

        Returns the PricingEngine.compute_edge() result, or None when there's
        nothing on record to re-evaluate (e.g. a position opened before this
        existed). Used both by _check_wang_edge_decay (the exit signal) and
        by manage_portfolio's bridge.position_analytics snapshot (dashboard
        edge-decay tracking) — computed once per position per cycle, not twice.

        Note: volume/days_to_expiry aren't persisted per-position, so this
        re-derives lambda from the pooled base_lambda prior rather than the
        hierarchical (volume/duration-aware) one used at entry — the dominant
        driver of edge decay over a short hold is the market price moving
        toward or away from fair value, which this still captures directly.
        """
        pre_prob = self._resolve_position_raw_probability(position)
        if pre_prob is None:
            return None

        current_price = float(self._position_field(position, "current_price", 0.0) or 0.0)
        if current_price <= 0.0:
            return None

        # current_price is already the price of whichever side is held
        # (TradeExecutor.get_open_positions() reads it per-token), so only
        # the probability needs translating into "probability my side wins".
        side = str(self._position_field(position, "side", "YES") or "YES").upper()
        p_true_for_side = pre_prob if side != "NO" else (1.0 - pre_prob)

        return self.pricing_engine.compute_edge(p_true=p_true_for_side, market_price=current_price)

    def _check_wang_edge_decay(self, wang_result: Optional[dict]) -> Optional[dict]:
        """Given an already-recomputed Wang edge result (see
        _recompute_position_wang_edge), return it if the edge has collapsed
        below wang_min_edge/2, else None. A position with no recorded
        pre_prob is left alone — falls through to the existing
        P&L-based exits.
        """
        if wang_result is None:
            return None
        if abs(wang_result["edge"]) < self.wang_decay_threshold:
            return wang_result
        return None

    def get_book_asset_types(self) -> List[str]:
        """Resolved asset_type for every currently open position — the input
        to correlation exposure/matrix calculations."""
        asset_types = []
        for position in list(getattr(self.bridge, "current_portfolio", []) or []):
            asset_type = self._resolve_position_asset_type(position)
            if asset_type:
                asset_types.append(asset_type)
        return asset_types

    def correlation_exposure_for(self, asset_type: str) -> float:
        """Average correlation between a candidate asset_type and the current book."""
        return self.correlation_tracker.exposure_for_new_position(asset_type, self.get_book_asset_types())

    def _venue(self) -> str:
        return "paper" if getattr(self.executor, "dry_run", True) else "live"

    def _record_sale(self, position, reason: str) -> None:
        """Record one confirmed sale of this position in the Ledger. Strategy
        (crypto | arbitrage) and group_id come from the Ledger's own open
        holding for the token."""
        if self.ledger is None:
            return
        self.ledger.record_exit(
            token_id=str(self._position_field(position, "token_id", "") or ""),
            shares=float(self._position_field(position, "shares", 0.0) or 0.0),
            price=float(self._position_field(position, "current_price", 0.0) or 0.0),
            reason=reason,
            venue=self._venue(),
            entry_price=float(self._position_field(position, "initial_price", 0.0) or 0.0),
            side=str(self._position_field(position, "side", "") or "") or None,
        )

    def _liquidate_position_value(self, position, log_func) -> float:
        token_id = str(self._position_field(position, "token_id", "") or "")
        shares = float(self._position_field(position, "shares", 0.0) or 0.0)
        current_price = float(self._position_field(position, "current_price", 0.0) or 0.0)
        position_value = float(self._position_field(position, "value", shares * current_price) or 0.0)

        if not token_id or shares <= 0.0 or current_price <= 0.0:
            return 0.0

        try:
            sold, gave_up = self._sell(position, "PORTFOLIO-CULL", log_func)
        except Exception as exc:
            print(f"[PORTFOLIO-CULL] Liquidation failed for {token_id}: {exc}")
            return 0.0

        if gave_up:
            self._record_gave_up(position, log_func)
        if not sold:
            return 0.0
        self._record_sale(position, "PORTFOLIO-CULL")
        return max(0.0, float(position_value))

    def _position_live_ev(self, position) -> float:
        live_ev = getattr(position, "live_ev", None)
        if live_ev is not None:
            return float(live_ev)
        return float(getattr(position, "pnl_ratio", 0.0) or 0.0)

    def _cash(self) -> float:
        if self.account is not None:
            return self.account.refresh().cash
        return float(self.executor.get_balance())

    def _sell(self, position, reason: str, log_func):
        """Sell a whole position if the Gate allows it, and report the
        attempt back to the Gate. Returns (sold, gave_up): gave_up is True
        on the failure that reaches the exit attempt cap. A refused exit is
        not an attempt and returns (False, False)."""
        if not isinstance(self.gate.can_exit(position), Ok):
            return False, False
        token_id = str(self._position_field(position, "token_id", "") or "")
        shares = float(self._position_field(position, "shares", 0.0) or 0.0)
        current_price = float(self._position_field(position, "current_price", 0.0) or 0.0)
        sold = bool(self.executor.sell_position(token_id, shares, current_price, log_func))
        gave_up = self.gate.record_exit_attempt(position, sold=sold, reason=reason)
        return sold, gave_up

    def _exit_position(self, position, level: str, threshold: float, extra: dict, log_func) -> bool:
        """Sell a position and log the exit. Returns True if sold.

        Gives up silently once the Gate's exit attempt cap is reached for
        the token — a position that can't sell isn't going to start
        succeeding on attempt 2,001.
        """
        token_id = str(self._position_field(position, "token_id", "") or "")

        if not isinstance(self.gate.can_exit(position), Ok):
            return False

        shares = float(self._position_field(position, "shares", 0.0) or 0.0)
        current_price = float(self._position_field(position, "current_price", 0.0) or 0.0)
        # Entry price — needed (alongside the exit "price" above) so the
        # dashboard's "Avg $/share (Closed)" stat can compute a genuine
        # per-share delta (price - initial_price) rather than needing to
        # reverse-engineer it from pnl_ratio, which isn't logged by every
        # exit reason.
        initial_price = float(self._position_field(position, "initial_price", 0.0) or 0.0)

        sold, gave_up = self._sell(position, level, log_func)

        if sold:
            self._record_sale(position, level)
        log_func(level, "Portfolio", token_id, {
            "threshold": threshold,
            "shares": shares,
            "price": current_price,
            "initial_price": initial_price,
            "sold": sold,
            **({"gave_up_after_max_attempts": True} if gave_up else {}),
            **extra,
        })

        if gave_up:
            self._record_gave_up(position, log_func)

        return bool(sold)

    def _record_gave_up(self, position, log_func) -> None:
        """Every exit path calls this on the failed sell that reaches the
        Gate's attempt cap (it happens once per token).

        Live run only: giving up is the signal for "this position's market
        has no live order book left". Per the MAX_EXIT_ATTEMPTS comment in
        trading/gate.py, that's what a resolved/closed Polymarket market
        looks like from here (it doesn't disappear from the positions API,
        it just stops being sellable). sold stays False: there's no
        confirmed on-chain payout (this bot doesn't automate redemption),
        so it's excluded from get_trade_stats()/get_closed_trade_deltas()'s
        P&L aggregation, just surfaced in the Transactions ledger. Dry run's
        equivalent (PaperVenue.resolve_closed_markets) already logs a
        sold=True EXPIRED row with a real payout, so this would only double
        up on that."""
        if getattr(self.executor, "dry_run", True):
            return
        token_id = str(self._position_field(position, "token_id", "") or "")
        shares = float(self._position_field(position, "shares", 0.0) or 0.0)
        initial_price = float(self._position_field(position, "initial_price", 0.0) or 0.0)
        side = str(self._position_field(position, "side", "") or "")
        if self.ledger is not None:
            self.ledger.record_resolution(
                token_id=token_id, shares=shares, payout_price=None, venue=self._venue(),
                entry_price=initial_price, side=side or None,
            )
        log_func("EXPIRED", "Portfolio", token_id, {
            "shares": shares,
            "price": float(self._position_field(position, "current_price", 0.0) or 0.0),
            "initial_price": initial_price,
            "side": side,
            "market_name": str(self._position_field(position, "market_id", "") or ""),
            "sold": False,
            "reason": (
                f"Gave up selling after {MAX_EXIT_ATTEMPTS} failed attempts — "
                "market most likely resolved/closed on Polymarket. On-chain "
                "redemption is not automated by this bot."
            ),
        })

    def optimize_for_candidate(self, new_candidate_ev: float, min_improvement: float = 0.10, log_func=None) -> float:
        """Liquidate positions materially weaker than the new candidate to pool capital."""
        try:
            self._refresh_portfolio()
            open_positions = list(getattr(self.bridge, "current_portfolio", []) or [])
            if not open_positions:
                return 0.0

            freed_capital = 0.0
            liquidated_count = 0

            for position in open_positions:
                try:
                    post_prob = self._resolve_position_fair_value(position)
                    if post_prob is None:
                        continue

                    size = float(
                        self._position_field(position, "shares", 0.0)
                        or self._position_field(position, "size", 0.0)
                        or 0.0
                    )
                    if size <= 0.0:
                        continue

                    current_value = float(
                        self._position_field(position, "value", 0.0)
                        or self._position_field(position, "currentValue", 0.0)
                        or 0.0
                    )
                    current_price = (
                        current_value / size if current_value > 0.0
                        else float(self._position_field(position, "current_price", 0.0) or 0.0)
                    )
                    if current_price <= 0.001:
                        continue

                    live_ev = (max(0.001, min(0.999, float(post_prob))) / float(current_price)) - 1.0

                    if float(new_candidate_ev) >= float(live_ev) + float(min_improvement):
                        asset = str(self._position_field(position, "token_id", "UNKNOWN") or "UNKNOWN")
                        print(f"[PORTFOLIO-CULL] Weak position: {asset} (live_ev={live_ev:.2f})")

                        recovered = self._liquidate_position_value(
                            position, log_func or (lambda *a, **k: None)
                        )
                        if recovered <= 0.0:
                            continue

                        freed_capital += float(recovered)
                        liquidated_count += 1
                except Exception as exc:
                    print(f"[PORTFOLIO-CULL] Failed to evaluate position: {exc}")
                    continue

            print(f"[OPPORTUNITY-SWAP] Culled {liquidated_count} positions, recovered ${freed_capital:.2f}")

            if log_func is not None and liquidated_count > 0:
                log_func("PORTFOLIO-CULL", "Portfolio", "MULTI", {
                    "liquidated_count": liquidated_count,
                    "freed_capital": round(float(freed_capital), 4),
                    "new_candidate_ev": round(float(new_candidate_ev), 4),
                    "min_improvement": round(float(min_improvement), 4),
                })

            self._refresh_portfolio()
            return float(freed_capital)
        except Exception as exc:
            print(f"[PORTFOLIO-CULL] Optimization failed: {exc}")
            return 0.0

    def free_up_capital(self, required_amount: float, log_func) -> bool:
        """Sell weakest positions (by PnL) until required_amount of cash is available."""
        self._refresh_portfolio()

        if self._cash() >= float(required_amount):
            return True

        weakest_first = sorted(
            list(self.bridge.current_portfolio),
            key=lambda p: float(getattr(p, "pnl_ratio", 0.0)),
        )

        for position in weakest_first:
            token_id = str(position.token_id)
            shares = float(position.shares)
            current_price = float(position.current_price)
            position_value = float(getattr(position, "value", shares * current_price))

            sold, gave_up = self._sell(position, "OPPORTUNITY-SWAP", log_func)
            if gave_up:
                self._record_gave_up(position, log_func)
            if not sold:
                continue

            self._record_sale(position, "OPPORTUNITY-SWAP")
            cash = self._cash()
            log_func("OPPORTUNITY-SWAP", "Portfolio", token_id, {
                "message": f"Liquidated {token_id} to free capital for high-EV trade.",
                "freed_value": round(position_value, 4),
                "required_amount": round(float(required_amount), 4),
                "current_balance": round(cash, 4),
            })

            if cash >= float(required_amount):
                self._refresh_portfolio()
                return True

        self._refresh_portfolio()
        return self._cash() >= float(required_amount)

    def _update_position_analytics(self, position, wang_result: Optional[dict]) -> None:
        """Stash per-position analytics on the bridge for the dashboard —
        entry-vs-current Wang edge ("edge decay tracking") and asset_type
        (for the correlation matrix). Not part of any exit decision itself.
        """
        if not hasattr(self.bridge, "position_analytics") or not isinstance(
            getattr(self.bridge, "position_analytics", None), dict
        ):
            self.bridge.position_analytics = {}

        token_id = str(self._position_field(position, "token_id", "") or "")
        if not token_id:
            return

        entry_edge = self._resolve_position_entry_wang_edge(position)
        current_edge = wang_result["edge"] if wang_result else None

        self.bridge.position_analytics[token_id] = {
            "asset_type": self._resolve_position_asset_type(position),
            "pre_prob": self._resolve_position_raw_probability(position),
            "entry_wang_edge": entry_edge,
            "current_wang_edge": current_edge,
            "current_wang_fair_value": wang_result["fair_value"] if wang_result else None,
            "edge_delta": (
                (current_edge - entry_edge)
                if (current_edge is not None and entry_edge is not None) else None
            ),
        }

    def manage_portfolio(self, log_func):
        self._refresh_portfolio()

        for position in list(self.bridge.current_portfolio):
            # ENABLE_ARBITRAGE kill switch: while arbitrage is disabled, no
            # arbitrage-specific handling may run at all — not even exiting
            # an already-open arbitrage leg — regardless of what's in the
            # book. Non-arbitrage (crypto) positions in this same loop are
            # completely unaffected by this check.
            if not self.enable_arbitrage and self._is_arbitrage_position(position):
                continue

            pnl_ratio = float(self._position_field(position, "pnl_ratio", 0.0) or 0.0)

            # Computed once per position per cycle — feeds both the decay
            # exit check below and the dashboard's edge-decay display.
            wang_result = self._recompute_position_wang_edge(position)
            self._update_position_analytics(position, wang_result)

            if pnl_ratio >= self.take_profit_pct:
                self._exit_position(position, "TAKE-PROFIT", self.take_profit_pct,
                                    {"pnl_ratio": pnl_ratio}, log_func)
                continue

            if pnl_ratio <= self.stop_loss_pct:
                self._exit_position(position, "STOP-LOSS", self.stop_loss_pct,
                                    {"pnl_ratio": pnl_ratio}, log_func)
                continue

            # Wang edge decay: an additional, earlier exit signal alongside
            # the hard take-profit/stop-loss limits above, not a replacement
            # for them. A position that no longer has edge is dead money
            # even if it hasn't hit stop-loss yet.
            wang_decay = self._check_wang_edge_decay(wang_result)
            if wang_decay is not None:
                self._exit_position(position, "WANG-EDGE-DECAY", self.wang_decay_threshold, {
                    "pnl_ratio": pnl_ratio,
                    "wang_edge": round(wang_decay["edge"], 4),
                    "wang_fair_value": round(wang_decay["fair_value"], 4),
                    "reason": "edge collapsed below wang_min_edge/2 — closing regardless of P&L",
                }, log_func)
                continue

            estimated_hold_ev = self._position_live_ev(position)
            if estimated_hold_ev < self.min_hold_ev:
                self._exit_position(position, "EV-CONVERGENCE", self.min_hold_ev,
                                    {"estimated_ev": round(estimated_hold_ev, 4)}, log_func)

        self._refresh_portfolio()

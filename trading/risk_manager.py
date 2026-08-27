import json
import os
import sqlite3
import time
from typing import List, Optional

from ui import data_manager

from brains.pricing_engine import PricingEngine
from trading.correlation import CorrelationTracker


def _parse_payload(payload_value) -> dict:
    if isinstance(payload_value, dict):
        return payload_value
    text = str(payload_value or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        try:
            import ast
            return ast.literal_eval(text)
        except Exception:
            return {}


class PortfolioManager:
    def __init__(self, bridge, executor, config, hunter=None, db_path: Optional[str] = None):
        self.bridge = bridge
        self.executor = executor
        self.hunter = hunter
        self.db_path = db_path  # this wallet's trades.db — no guessing, the caller (WalletContext) owns it
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

    def _refresh_portfolio(self):
        positions = self.executor.get_open_positions()
        self.bridge.current_portfolio = positions
        total_open_value = sum(float(getattr(p, "value", 0.0) or 0.0) for p in positions)
        self.bridge.open_position_value = float(total_open_value)
        self.bridge.open_positions_value = float(total_open_value)
        # Unrealized only — open positions' (current - entry) x shares. The
        # top-level "Total PnL" KPI (ui/dashboard.py::_render_global_kpis)
        # uses the broader Balance - Total Deposits figure instead (realized
        # + unrealized + fees + slippage); this narrower figure stays
        # available under its own name for callers that specifically want it
        # (e.g. ui/components.py's Portfolio-view "Unrealized PnL" stat,
        # computed independently there but conceptually the same number).
        self.bridge.unrealized_pnl = sum(
            (float(getattr(p, "current_price", 0.0) or 0.0) - float(getattr(p, "initial_price", 0.0) or 0.0))
            * float(getattr(p, "shares", 0.0) or 0.0)
            for p in positions
        )

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

    def _fair_value_from_db(self, token_id: str, market_id: str):
        """Look up latest fair value for a position from the trade history DB."""
        if not self.db_path or not os.path.exists(self.db_path):
            return None
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                cursor = conn.cursor()
                for lookup_id in (token_id, market_id):
                    if not lookup_id:
                        continue
                    cursor.execute(
                        """
                        SELECT payload FROM hunt_history
                        WHERE token_id = ?
                          AND level IN ('AUTO-TRADE', 'LIVE-TRADE', 'DRY-RUN', 'PAPER-TRADE', 'TRACK')
                        ORDER BY id DESC LIMIT 10
                        """,
                        (lookup_id,),
                    )
                    for (payload_raw,) in cursor.fetchall():
                        payload = _parse_payload(payload_raw)
                        for key in ("fair_value", "post_prob"):
                            if payload.get(key) is not None:
                                try:
                                    return float(payload[key])
                                except Exception:
                                    continue
        except Exception:
            pass
        return None

    def _resolve_position_fair_value(self, position):
        # 1. Direct attribute
        direct = self._position_field(position, "fair_value")
        if direct is not None:
            try:
                return float(direct)
            except Exception:
                pass

        # 2. Bridge opportunity map
        token_id = str(self._position_field(position, "token_id", "") or "")
        market_id = str(self._position_field(position, "market_id", "") or "")
        for key in (token_id, market_id):
            snapshot = getattr(self.bridge, "opportunity_map", {}).get(key, {}) if key else {}
            for payload_key in ("fair_value", "post_prob"):
                val = snapshot.get(payload_key) if isinstance(snapshot, dict) else None
                if val is not None:
                    try:
                        return float(val)
                    except Exception:
                        pass

        # 3. DB fallback
        return self._fair_value_from_db(token_id, market_id)

    def _field_from_db(self, token_id: str, market_id: str, field_key: str, caster=float):
        """Look up a single field from this position's most recent
        trade-history row (same table/scan _fair_value_from_db uses),
        converting it with `caster` (float by default; pass str for text
        fields like asset_type)."""
        if not self.db_path or not os.path.exists(self.db_path):
            return None
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                cursor = conn.cursor()
                for lookup_id in (token_id, market_id):
                    if not lookup_id:
                        continue
                    cursor.execute(
                        """
                        SELECT payload FROM hunt_history
                        WHERE token_id = ?
                          AND level IN ('AUTO-TRADE', 'LIVE-TRADE', 'DRY-RUN', 'PAPER-TRADE', 'TRACK')
                        ORDER BY id DESC LIMIT 10
                        """,
                        (lookup_id,),
                    )
                    for (payload_raw,) in cursor.fetchall():
                        payload = _parse_payload(payload_raw)
                        if payload.get(field_key) is not None:
                            try:
                                return caster(payload[field_key])
                            except Exception:
                                continue
        except Exception:
            pass
        return None

    def _resolve_position_field(self, position, payload_key: str, direct_key: Optional[str] = None, caster=float):
        """Generic 3-tier lookup shared by the pre_prob/asset_type/
        entry-wang_edge resolvers below: direct attribute -> bridge
        opportunity_map (fast path, populated by ui/data_manager.log_event)
        -> trade-history DB (slow path, works even across process restarts).
        """
        if direct_key:
            direct = self._position_field(position, direct_key)
            if direct is not None:
                try:
                    return caster(direct)
                except Exception:
                    pass

        token_id = str(self._position_field(position, "token_id", "") or "")
        market_id = str(self._position_field(position, "market_id", "") or "")

        for key in (token_id, market_id):
            snapshot = getattr(self.bridge, "opportunity_map", {}).get(key, {}) if key else {}
            val = snapshot.get(payload_key) if isinstance(snapshot, dict) else None
            if val is not None:
                try:
                    return caster(val)
                except Exception:
                    pass

        return self._field_from_db(token_id, market_id, payload_key, caster=caster)

    def _resolve_position_raw_probability(self, position):
        """Look up the brain's raw probability estimate (p_true) recorded for
        this position at entry (see decision_pipeline.py's TRACK payload,
        Phase 3) — needed to re-derive the Wang edge without re-running the
        brain/hunter for a position that's already open.
        """
        return self._resolve_position_field(position, "pre_prob")

    def _resolve_position_asset_type(self, position):
        """Look up this position's asset_type (e.g. "Crypto::BTCUSDT") —
        needed for correlation exposure, since Position itself doesn't carry it."""
        val = self._resolve_position_field(position, "asset_type", direct_key="asset_type", caster=str)
        return val or None

    def _is_arbitrage_position(self, position) -> bool:
        """True if this position's resolved asset_type marks it as an
        arbitrage leg (e.g. "Arbitrage::EventSum", see trading/strategies/
        event_sum.py). Used only by manage_portfolio()'s ENABLE_ARBITRAGE
        gate below — not an exit decision itself, and not needed by any
        crypto-only caller (correlation exposure, dashboard analytics, ...)."""
        asset_type = self._resolve_position_asset_type(position)
        return bool(asset_type) and asset_type.startswith("Arbitrage::")

    def _resolve_position_entry_wang_edge(self, position):
        """Look up the Wang edge recorded at entry, for the dashboard's edge-decay display."""
        return self._resolve_position_field(position, "wang_edge")

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

    def _liquidate_position_value(self, position, log_func) -> float:
        token_id = str(self._position_field(position, "token_id", "") or "")
        shares = float(self._position_field(position, "shares", 0.0) or 0.0)
        current_price = float(self._position_field(position, "current_price", 0.0) or 0.0)
        position_value = float(self._position_field(position, "value", shares * current_price) or 0.0)

        if not token_id or shares <= 0.0 or current_price <= 0.0:
            return 0.0

        try:
            sold = self.executor.sell_position(token_id, shares, current_price, log_func)
        except Exception as exc:
            print(f"[PORTFOLIO-CULL] Liquidation failed for {token_id}: {exc}")
            return 0.0

        return max(0.0, float(position_value)) if sold else 0.0

    def _position_live_ev(self, position) -> float:
        live_ev = getattr(position, "live_ev", None)
        if live_ev is not None:
            return float(live_ev)
        return float(getattr(position, "pnl_ratio", 0.0) or 0.0)

    def _apply_sale_to_bridge(self, position_value: float):
        updated_cash = float(self.bridge.current_balance) + max(0.0, float(position_value))
        self.bridge.current_balance = float(updated_cash)
        self.bridge.cash = float(updated_cash)

    def _exit_position(self, position, level: str, threshold: float, extra: dict, log_func) -> bool:
        """Sell a position and log the exit. Returns True if sold."""
        token_id = str(self._position_field(position, "token_id", "") or "")
        shares = float(self._position_field(position, "shares", 0.0) or 0.0)
        current_price = float(self._position_field(position, "current_price", 0.0) or 0.0)
        # Entry price — needed (alongside the exit "price" above) so the
        # dashboard's "Avg $/share (Closed)" stat can compute a genuine
        # per-share delta (price - initial_price) rather than needing to
        # reverse-engineer it from pnl_ratio, which isn't logged by every
        # exit reason.
        initial_price = float(self._position_field(position, "initial_price", 0.0) or 0.0)
        position_value = float(self._position_field(position, "value", shares * current_price) or 0.0)

        sold = self.executor.sell_position(token_id, shares, current_price, log_func)
        if sold:
            self._apply_sale_to_bridge(position_value)
        log_func(level, "Portfolio", token_id, {
            "threshold": threshold,
            "shares": shares,
            "price": current_price,
            "initial_price": initial_price,
            "sold": sold,
            **extra,
        })
        return bool(sold)

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

        if float(self.bridge.current_balance) >= float(required_amount):
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

            sold = self.executor.sell_position(token_id, shares, current_price, log_func)
            if not sold:
                continue

            self._apply_sale_to_bridge(position_value)
            log_func("OPPORTUNITY-SWAP", "Portfolio", token_id, {
                "message": f"Liquidated {token_id} to free capital for high-EV trade.",
                "freed_value": round(position_value, 4),
                "required_amount": round(float(required_amount), 4),
                "current_balance": round(float(self.bridge.current_balance), 4),
            })

            if float(self.bridge.current_balance) >= float(required_amount):
                self._refresh_portfolio()
                return True

        self._refresh_portfolio()
        return float(self.bridge.current_balance) >= float(required_amount)

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

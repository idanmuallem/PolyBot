import json
import os
import sqlite3

from brains import get_brain_for_asset_type
from core.models import MarketData


_DB_CANDIDATES = ["/app/trades.db", "trades.db"]


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


def _db_path() -> str:
    for candidate in _DB_CANDIDATES:
        if os.path.isdir(candidate):
            candidate = os.path.join(candidate, "trades.db")
        if os.path.exists(candidate):
            return candidate
    return _DB_CANDIDATES[0]


class PortfolioManager:
    def __init__(self, bridge, executor, config, hunter=None):
        self.bridge = bridge
        self.executor = executor
        self.hunter = hunter
        self.take_profit_pct = float(config.take_profit_pct)
        self.stop_loss_pct = float(config.stop_loss_pct)
        self.min_hold_ev = float(config.min_hold_ev)

    def _refresh_portfolio(self):
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

    @staticmethod
    def _position_field(position, field_name, default=None):
        if isinstance(position, dict):
            return position.get(field_name, default)
        return getattr(position, field_name, default)

    def _fair_value_from_db(self, token_id: str, market_id: str):
        """Look up latest fair value for a position from the trade history DB."""
        db_file = _db_path()
        if not os.path.exists(db_file):
            return None
        try:
            with sqlite3.connect(db_file, timeout=5) as conn:
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
                        for key in ("fair_value", "fair"):
                            if payload.get(key) is not None:
                                try:
                                    return float(payload[key])
                                except Exception:
                                    continue
        except Exception:
            pass
        return None

    def _market_data_from_db(self, token_id: str, market_id: str, current_price: float) -> MarketData | None:
        """Reconstruct MarketData for a held position from the trade log DB."""
        db_file = _db_path()
        if not os.path.exists(db_file):
            return None
        try:
            with sqlite3.connect(db_file, timeout=5) as conn:
                cursor = conn.cursor()
                for lookup_id in (token_id, market_id):
                    if not lookup_id:
                        continue
                    cursor.execute(
                        """
                        SELECT asset_type, payload FROM hunt_history
                        WHERE token_id = ?
                          AND level IN ('TRACK', 'AUTO-TRADE', 'LIVE-TRADE', 'DRY-RUN', 'PAPER-TRADE')
                        ORDER BY id DESC LIMIT 1
                        """,
                        (lookup_id,),
                    )
                    row = cursor.fetchone()
                    if not row:
                        continue
                    asset_type, payload_raw = row
                    payload = _parse_payload(payload_raw)
                    strike_price = float(payload.get("strike_price") or 0.0)
                    expiry_date = str(payload.get("expiry_date") or "")
                    market_name = str(payload.get("market_name") or "")
                    if asset_type:
                        return MarketData(
                            market_id=market_id or token_id,
                            asset_type=asset_type,
                            strike_price=strike_price,
                            question=market_name,
                            market_name=market_name,
                            initial_price=current_price,
                            volume=0.0,
                            expiry_date=expiry_date or None,
                        )
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
            for payload_key in ("fair_value", "fair"):
                val = snapshot.get(payload_key) if isinstance(snapshot, dict) else None
                if val is not None:
                    try:
                        return float(val)
                    except Exception:
                        pass

        # 3. DB fallback
        return self._fair_value_from_db(token_id, market_id)

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
        token_id = str(self._position_field(position, "token_id", "") or "")
        market_id = str(self._position_field(position, "market_id", "") or "")
        current_price = float(self._position_field(position, "current_price", 0.0) or 0.0)

        # Reconstruct MarketData from trade log and run the brain with fresh live_truth.
        if self.hunter is not None and current_price > 0.0:
            market_data = self._market_data_from_db(token_id, market_id, current_price)
            if market_data is not None:
                try:
                    sub_hunter = self.hunter.get_hunter_for_asset_type(market_data.asset_type)
                    live_truth = sub_hunter.get_live_truth(market_data) if sub_hunter else None
                    if live_truth is not None:
                        brain = get_brain_for_asset_type(market_data.asset_type)
                        signal = brain.evaluate(market_data, float(live_truth))
                        return float(signal.expected_value)
                except Exception:
                    pass

        return float(getattr(position, "pnl_ratio", 0.0) or 0.0)

    def _apply_sale_to_bridge(self, position_value: float):
        updated_cash = float(self.bridge.current_balance) + max(0.0, float(position_value))
        self.bridge.current_balance = float(updated_cash)
        self.bridge.cash = float(updated_cash)

    def _exit_position(self, position, level: str, threshold: float, extra: dict, log_func) -> bool:
        """Sell a position and log the exit. Returns True if sold."""
        token_id = position.token_id
        shares = float(position.shares)
        current_price = float(position.current_price)
        position_value = float(getattr(position, "value", shares * current_price))

        sold = self.executor.sell_position(token_id, shares, current_price, log_func)
        if sold:
            self._apply_sale_to_bridge(position_value)
        log_func(level, "Portfolio", token_id, {
            "threshold": threshold,
            "shares": shares,
            "price": current_price,
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
                    fair_value = self._resolve_position_fair_value(position)
                    if fair_value is None:
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

                    live_ev = (max(0.001, min(0.999, float(fair_value))) / float(current_price)) - 1.0

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

    def manage_portfolio(self, log_func):
        self._refresh_portfolio()

        for position in list(self.bridge.current_portfolio):
            pnl_ratio = float(position.pnl_ratio)

            if pnl_ratio >= self.take_profit_pct:
                self._exit_position(position, "TAKE-PROFIT", self.take_profit_pct,
                                    {"pnl_ratio": pnl_ratio}, log_func)
                continue

            if pnl_ratio <= self.stop_loss_pct:
                self._exit_position(position, "STOP-LOSS", self.stop_loss_pct,
                                    {"pnl_ratio": pnl_ratio}, log_func)
                continue

            estimated_hold_ev = self._position_live_ev(position)
            if estimated_hold_ev < self.min_hold_ev:
                self._exit_position(position, "EV-CONVERGENCE", self.min_hold_ev,
                                    {"estimated_ev": round(estimated_hold_ev, 4)}, log_func)

        self._refresh_portfolio()

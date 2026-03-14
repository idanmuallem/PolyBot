import ast
import json
import os
import sqlite3


class PortfolioManager:
    def __init__(self, bridge, executor, config):
        self.bridge = bridge
        self.executor = executor
        self.take_profit_pct = float(config.take_profit_pct)
        self.stop_loss_pct = float(config.stop_loss_pct)
        self.min_hold_ev = float(config.min_hold_ev)

    def _refresh_portfolio(self):
        positions = self.executor.get_open_positions()
        self.bridge.current_portfolio = positions
        self.bridge.open_position_value = sum(p.value for p in positions)
        self.bridge.total_pnl = sum((p.current_price - p.initial_price) * p.shares for p in positions)

    @staticmethod
    def _position_field(position, field_name, default=None):
        if isinstance(position, dict):
            return position.get(field_name, default)
        return getattr(position, field_name, default)

    @staticmethod
    def _db_path() -> str:
        candidate = "/app/trades.db"
        if os.path.isdir(candidate):
            candidate = os.path.join(candidate, "trades.db")
        if os.path.exists(candidate):
            return candidate
        fallback = os.path.join("trading", "trades.db")
        return fallback

    @staticmethod
    def _parse_payload(payload_value):
        if isinstance(payload_value, dict):
            return payload_value
        if payload_value is None:
            return {}
        payload_text = str(payload_value).strip()
        if not payload_text:
            return {}
        try:
            return ast.literal_eval(payload_text)
        except Exception:
            try:
                return json.loads(payload_text)
            except Exception:
                return {}

    def _resolve_position_fair_value(self, position):
        direct_value = self._position_field(position, "fair_value")
        if direct_value is not None:
            try:
                return float(direct_value)
            except Exception:
                pass

        token_id = str(self._position_field(position, "token_id", "") or "")
        market_id = str(self._position_field(position, "market_id", "") or "")

        for key in (token_id, market_id):
            snapshot = getattr(self.bridge, "opportunity_map", {}).get(str(key), {}) if key else {}
            if isinstance(snapshot, dict):
                for payload_key in ("fair_value", "fair"):
                    if snapshot.get(payload_key) is not None:
                        try:
                            return float(snapshot.get(payload_key))
                        except Exception:
                            pass

        db_file = self._db_path()
        if not os.path.exists(db_file):
            return None

        try:
            with sqlite3.connect(db_file, timeout=5) as conn:
                cursor = conn.cursor()
                for lookup_token in (token_id, market_id):
                    if not lookup_token:
                        continue
                    cursor.execute(
                        """
                        SELECT payload
                        FROM hunt_history
                        WHERE token_id = ?
                          AND level IN ('AUTO-TRADE', 'LIVE-TRADE', 'DRY-RUN', 'PAPER-TRADE', 'TRACK')
                        ORDER BY id DESC
                        LIMIT 10
                        """,
                        (lookup_token,),
                    )
                    for (payload_raw,) in cursor.fetchall():
                        payload = self._parse_payload(payload_raw)
                        for payload_key in ("fair_value", "fair"):
                            if payload.get(payload_key) is not None:
                                try:
                                    return float(payload.get(payload_key))
                                except Exception:
                                    continue
        except Exception:
            return None

        return None

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

        if not sold:
            return 0.0

        return max(0.0, float(position_value))

    @staticmethod
    def _normalized_pnl_ratio(position) -> float:
        pnl_raw = float(getattr(position, "pnl_percent", 0.0) or 0.0)
        return pnl_raw / 100.0 if abs(pnl_raw) > 1.0 else pnl_raw

    def _position_live_ev(self, position) -> float:
        live_ev = getattr(position, "live_ev", None)
        if live_ev is None:
            return float(self._normalized_pnl_ratio(position))
        return float(live_ev)

    def _apply_sale_to_bridge(self, position_value: float):
        self.bridge.current_balance = float(self.bridge.current_balance) + max(0.0, float(position_value))

    def optimize_for_candidate(self, new_candidate_ev: float, min_improvement: float = 0.10, log_func=None) -> float:
        """Liquidate all materially weaker positions to pool capital for a stronger candidate."""
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

                    size = float(self._position_field(position, "shares", 0.0) or self._position_field(position, "size", 0.0) or 0.0)
                    current_value = float(self._position_field(position, "value", 0.0) or self._position_field(position, "currentValue", 0.0) or 0.0)

                    if size <= 0.0:
                        continue

                    current_price = current_value / size if current_value > 0.0 else float(self._position_field(position, "current_price", 0.0) or 0.0)
                    if current_price <= 0.001:
                        continue

                    bounded_fair_value = max(0.001, min(0.999, float(fair_value)))
                    live_ev = (bounded_fair_value / float(current_price)) - 1.0

                    if float(new_candidate_ev) >= float(live_ev) + float(min_improvement):
                        asset = str(self._position_field(position, "token_id", "UNKNOWN") or "UNKNOWN")
                        print(f"[PORTFOLIO-CULL] Found weak position: Asset {asset} (Live EV: {live_ev:.2f}).")

                        recovered_value = self._liquidate_position_value(position, log_func or (lambda *args, **kwargs: None))
                        if recovered_value <= 0.0:
                            continue

                        freed_capital += float(recovered_value)
                        liquidated_count += 1
                except Exception as exc:
                    print(f"[PORTFOLIO-CULL] Failed to evaluate position for cull: {exc}")
                    continue

            print(f"[OPPORTUNITY-SWAP] Total Culled: {liquidated_count} positions. Recovered Capital: ${freed_capital:.2f}")

            if log_func is not None and liquidated_count > 0:
                log_func(
                    "PORTFOLIO-CULL",
                    "Portfolio",
                    "MULTI",
                    {
                        "liquidated_count": liquidated_count,
                        "freed_capital": round(float(freed_capital), 4),
                        "new_candidate_ev": round(float(new_candidate_ev), 4),
                        "min_improvement": round(float(min_improvement), 4),
                    },
                )

            self._refresh_portfolio()
            return float(freed_capital)
        except Exception as exc:
            print(f"[PORTFOLIO-CULL] Optimization failed: {exc}")
            return 0.0

    def free_up_capital(self, required_amount: float, log_func) -> bool:
        self._refresh_portfolio()

        if float(self.bridge.current_balance) >= float(required_amount):
            return True

        weakest_first = sorted(
            list(self.bridge.current_portfolio),
            key=lambda position: float(getattr(position, "pnl_percent", 0.0)),
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
            log_func(
                "OPPORTUNITY-SWAP",
                "Portfolio",
                token_id,
                {
                    "message": f"Liquidated {token_id} to free up capital for high-EV trade.",
                    "freed_value": round(position_value, 4),
                    "required_amount": round(float(required_amount), 4),
                    "current_balance": round(float(self.bridge.current_balance), 4),
                },
            )

            if float(self.bridge.current_balance) >= float(required_amount):
                self._refresh_portfolio()
                return True

        self._refresh_portfolio()
        return float(self.bridge.current_balance) >= float(required_amount)

    def manage_portfolio(self, log_func):
        self._refresh_portfolio()

        for position in list(self.bridge.current_portfolio):
            token_id = position.token_id
            shares = float(position.shares)
            current_price = float(position.current_price)
            position_value = float(getattr(position, "value", shares * current_price))

            pnl_raw = float(position.pnl_percent)
            pnl_ratio = self._normalized_pnl_ratio(position)

            if pnl_ratio >= self.take_profit_pct:
                sold = self.executor.sell_position(token_id, shares, current_price, log_func)
                if sold:
                    self._apply_sale_to_bridge(position_value)
                log_func(
                    "TAKE-PROFIT",
                    "Portfolio",
                    token_id,
                    {
                        "pnl_percent": pnl_raw,
                        "threshold": self.take_profit_pct,
                        "shares": shares,
                        "price": current_price,
                        "sold": sold,
                    },
                )
                continue

            if pnl_ratio <= self.stop_loss_pct:
                sold = self.executor.sell_position(token_id, shares, current_price, log_func)
                if sold:
                    self._apply_sale_to_bridge(position_value)
                log_func(
                    "STOP-LOSS",
                    "Portfolio",
                    token_id,
                    {
                        "pnl_percent": pnl_raw,
                        "threshold": self.stop_loss_pct,
                        "shares": shares,
                        "price": current_price,
                        "sold": sold,
                    },
                )
                continue

            # Optional/Future EV convergence placeholder
            estimated_hold_ev = self._position_live_ev(position)
            if estimated_hold_ev < self.min_hold_ev:
                sold = self.executor.sell_position(token_id, shares, current_price, log_func)
                if sold:
                    self._apply_sale_to_bridge(position_value)
                log_func(
                    "EV-CONVERGENCE",
                    "Portfolio",
                    token_id,
                    {
                        "estimated_ev": round(estimated_hold_ev, 4),
                        "threshold": self.min_hold_ev,
                        "shares": shares,
                        "price": current_price,
                        "sold": sold,
                    },
                )

        self._refresh_portfolio()

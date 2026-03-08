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

            self.bridge.current_balance = float(self.bridge.current_balance) + max(0.0, position_value)
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

            pnl_raw = float(position.pnl_percent)
            pnl_ratio = pnl_raw / 100.0 if abs(pnl_raw) > 1.0 else pnl_raw

            if pnl_ratio >= self.take_profit_pct:
                sold = self.executor.sell_position(token_id, shares, current_price, log_func)
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
            estimated_hold_ev = pnl_ratio
            if estimated_hold_ev < self.min_hold_ev:
                sold = self.executor.sell_position(token_id, shares, current_price, log_func)
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

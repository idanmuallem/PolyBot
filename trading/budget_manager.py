import time


class BudgetManager:
    def __init__(self, bridge, config, initial_balance: float):
        self.bridge = bridge
        self.config = config

        self.daily_limit_usd = float(config.daily_limit_usd)
        self.bankroll_usd = float(config.bankroll_usd)
        self.min_trading_balance = float(config.min_trading_balance)

        self.base_balance = float(initial_balance)
        self.total_spent_today = 0.0
        self.day_start_time = time.time()

        self.watch_only = self.base_balance < self.min_trading_balance
        self._sync_bridge()

    def _sync_bridge(self):
        self.bridge.daily_spend = self.total_spent_today
        self.bridge.current_balance = max(self.base_balance - self.total_spent_today, 0.0)
        self.bridge.watch_only = self.watch_only

    def _reset_daily_if_needed(self):
        if time.time() - self.day_start_time >= 86400:
            self.total_spent_today = 0.0
            self.day_start_time = time.time()
            self._sync_bridge()

    def get_remaining_budget(self) -> float:
        self._reset_daily_if_needed()
        return self.daily_limit_usd - self.total_spent_today

    def check_and_cap_bet(self, kelly_fraction: float):
        remaining = self.get_remaining_budget()
        desired_bet = float(kelly_fraction) * self.bankroll_usd
        actual_bet = min(desired_bet, remaining)
        if actual_bet <= 0:
            return 0.0, False
        return actual_bet, True

    def record_trade(self, amount_usd: float):
        self.total_spent_today += float(amount_usd)
        self._sync_bridge()

class BudgetManager:
    def __init__(self, bridge, config, initial_balance: float):
        self.bridge = bridge
        self.config = config

        self.daily_limit_usd = float(config.daily_limit_usd)
        self.bankroll_usd = float(config.bankroll_usd)
        self.min_trading_balance = float(config.min_trading_balance)
        self.max_bet_size_usd = float(config.max_bet_size_usd)
        self.kelly_fraction = float(getattr(config, "kelly_fraction", 0.25))

        # Per-strategy daily limits (see core/trading_config.py) — arbitrage
        # and crypto (model-driven) both draw from the same wallet balance,
        # but each gets its own budget allocation so one can't starve the
        # other. Unrecognized strategy tags fall back to the global
        # daily_limit_usd ceiling.
        self.strategy_limits: dict[str, float] = {
            "arbitrage": float(getattr(config, "arbitrage_daily_limit_usd", self.daily_limit_usd)),
            "crypto": float(getattr(config, "crypto_daily_limit_usd", self.daily_limit_usd)),
        }

        self.base_balance = float(initial_balance)
        self.spent_by_strategy: dict[str, float] = {}
        self.trades_by_strategy: dict[str, int] = {}
        self.total_spent_today = 0.0

        self.watch_only = self.base_balance < self.min_trading_balance
        self._sync_bridge()

    def _strategy_limit(self, strategy_tag: str) -> float:
        return float(self.strategy_limits.get(strategy_tag, self.daily_limit_usd))

    def _sync_bridge(self):
        self.bridge.daily_spend = self.total_spent_today
        self.bridge.watch_only = self.watch_only
        current_balance = max(0.0, self.base_balance - self.total_spent_today)
        self.bridge.current_balance = current_balance
        self.bridge.cash = current_balance

    def get_remaining_budget(self, strategy_tag: str = "crypto") -> float:
        spent = self.spent_by_strategy.get(strategy_tag, 0.0)
        strategy_remaining = self._strategy_limit(strategy_tag) - spent

        # Belt-and-suspenders hard ceiling: per-strategy limits (e.g.
        # arbitrage_daily_limit_usd + crypto_daily_limit_usd) can be
        # configured to sum higher than the account-wide daily_limit_usd —
        # TradingConfig only warns about that, it never clamps it — so cap
        # every strategy's remaining budget by what's actually left of the
        # global ceiling too. No strategy can spend past the account-wide
        # daily cap through its own generous allocation.
        global_remaining = self.daily_limit_usd - self.total_spent_today
        return min(strategy_remaining, global_remaining)

    def check_and_cap_bet(self, kelly_fraction: float, strategy_tag: str = "crypto"):
        remaining = self.get_remaining_budget(strategy_tag)
        desired_bet = float(kelly_fraction) * self.bankroll_usd
        actual_bet = min(desired_bet, remaining)
        if actual_bet <= 0:
            return 0.0, False
        return actual_bet, True

    def compute_kelly_bet_size(self, edge: float, odds: float, confidence: float = 1.0) -> float:
        """Fractional-Kelly bet size in dollars for a single binary bet.

        The confidence-scale -> kelly_fraction-scale -> dollar-clamp chain
        mirrors oracle3's ModelInformedSizer.compute_size()
        (oracle3/trading/sizing.py) -
        https://github.com/YichengYang-Ethan/oracle3, licensed under the
        Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).

        kelly_optimal = edge / odds — the binary-outcome form of the Kelly
        criterion f* = (p*b - q) / b, where `edge` is (true_prob - price) and
        `odds` is the price move against you if wrong (1 - price for a YES
        bet at `price`, or `price` for the equivalent NO bet).

        Scaled down by model confidence and self.kelly_fraction (quarter-Kelly
        by default — full Kelly is optimal in expectation but has extreme
        variance), then clamped to max_bet_size_usd as a hard ceiling.
        Returns 0.0 for a non-positive edge or degenerate odds — this method
        never sizes a bet against the edge.
        """
        if odds <= 0.0 or edge <= 0.0:
            return 0.0

        kelly_raw = edge / odds
        kelly_scaled = kelly_raw * max(0.0, min(1.0, float(confidence))) * self.kelly_fraction
        bet_usd = kelly_scaled * self.bankroll_usd
        return max(0.0, min(bet_usd, self.max_bet_size_usd))

    def cap_to_remaining_budget(self, desired_bet_usd: float, strategy_tag: str = "crypto"):
        """Cap an already-sized dollar bet to what's left of this strategy's
        budget for today."""
        remaining = self.get_remaining_budget(strategy_tag)
        actual_bet = min(float(desired_bet_usd), remaining)
        if actual_bet <= 0:
            return 0.0, False
        return actual_bet, True

    def record_trade(self, amount_usd: float, strategy_tag: str = "crypto"):
        self.spent_by_strategy[strategy_tag] = self.spent_by_strategy.get(strategy_tag, 0.0) + float(amount_usd)
        self.trades_by_strategy[strategy_tag] = self.trades_by_strategy.get(strategy_tag, 0) + 1
        self.total_spent_today = sum(self.spent_by_strategy.values())
        self._sync_bridge()

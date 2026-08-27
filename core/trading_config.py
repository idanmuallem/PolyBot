import json
import logging
import os
from dataclasses import dataclass, fields

from dotenv import load_dotenv


DEFAULT_MIN_EV = 0.50
DEFAULT_MAX_BET_SIZE_USD = 3.0
DEFAULT_DAILY_LIMIT_USD = 15.0

# Entry-side pricing (see BaseBrain.evaluate() in brains/base.py). Distinct
# from wang_base_lambda below, which is the oracle3-calibrated hierarchical
# prior used only by the exit-side Wang-edge-decay check in
# trading/risk_manager.py - changing that one would silently alter open-
# position exit behavior, so entry calibration gets its own flat constant.
DEFAULT_WANG_LAMBDA = -0.75          # risk-averse distortion; 0.0 disables Wang entirely
DEFAULT_MODEL_WEIGHT = 0.40          # weight on the Wang-adjusted model vs. (1 - this) on market price


def _env_bool(name: str, default: str) -> bool:
    return str(os.getenv(name, default)).strip().lower() in ("true", "1", "t")


def _env_first(*names: str, default: str = "") -> str:
    for name in names:
        value = str(os.getenv(name, "")).strip()
        if value:
            return value
    return default


@dataclass
class TradingConfig:
    min_ev: float = DEFAULT_MIN_EV
    min_tte_minutes: int = 60
    max_tte_days: int = 180

    daily_limit_usd: float = DEFAULT_DAILY_LIMIT_USD
    max_bet_size_usd: float = DEFAULT_MAX_BET_SIZE_USD

    # Top-level kill switch for the arbitrage strategy path (EventSumStrategy
    # et al.), checked at the very top of SequentialTradingPipeline's
    # _stage_strategy_scan() before any Gamma API call or strategy.scan().
    # Deliberately NOT the same thing as arbitrage_max_daily_trades=0: that
    # would still scan/evaluate every cycle and just reject at the budget
    # gate, burning API calls and log noise for a strategy that should not
    # run at all. False means zero event-discovery calls, zero
    # STRATEGY-GROUP/STRATEGY-LEG log entries — the strategy simply never
    # runs, full stop.
    enable_arbitrage: bool = True

    # Per-strategy budget isolation (see trading/budget_manager.py): the
    # arbitrage and crypto-brain paths draw from the same wallet cash pool
    # but track spend/trade-count independently, so one can't starve the
    # other. daily_limit_usd/max_daily_trades above remain the global
    # ceiling — the sum of these should not exceed them (from_env/from_file
    # log a warning, not an error, if it does).
    arbitrage_daily_limit_usd: float = 50.0
    arbitrage_max_daily_trades: int = 50
    crypto_daily_limit_usd: float = 50.0
    crypto_max_daily_trades: int = 50

    # How long to let each arbitrage leg's limit order sit before giving up
    # (see TradeExecutor.execute_arbitrage_group). Zero fills on any leg
    # after this many seconds cancels the whole group — no arbitrage exists
    # without every leg.
    arbitrage_order_timeout_seconds: float = 60.0

    # Crypto event groups are higher-volume/more liquid than the general
    # Polymarket catalog, so EventSumStrategy scans them first and spends
    # whatever arbitrage budget remains on everything else (see
    # trading/strategies/event_sum.py). False restores the original
    # single-pass scan order, for A/B comparison during dry-run.
    arbitrage_crypto_first: bool = True

    bankroll_usd: float = 1000.0
    min_trading_balance: float = 5.0
    dry_run: bool = True
    paper_trade_mode: bool = False
    paper_balance_usd: float = 1000.0

    take_profit_pct: float = 0.20
    stop_loss_pct: float = -0.50
    min_hold_ev: float = -0.10

    loop_delay_seconds: float = 2.0
    max_daily_trades: int = 10
    private_key: str = ""
    proxy_address: str = ""
    signature_type: int = 2

    # Third-party data source keys — hunters receive these via constructor
    # injection (see hunters/weather.py, hunters/economy.py), never os.getenv().
    openweather_api_key: str = ""
    fred_api_key: str = ""

    # Wang Transform pricing (see brains/pricing_engine.py). "legacy" bypasses
    # the Wang adjustment entirely and uses the brain's raw EV, for comparing
    # the two during the transition.
    pricing_mode: str = "wang"  # "wang" or "legacy"
    wang_base_lambda: float = 0.183  # exit-side only - see trading/risk_manager.py
    wang_min_edge: float = 0.05  # minimum |wang_edge| (probability points) to consider a trade

    # Entry-side pricing knobs consumed by BaseBrain.evaluate() (see
    # DEFAULT_WANG_LAMBDA / DEFAULT_MODEL_WEIGHT above for what each does).
    wang_lambda: float = DEFAULT_WANG_LAMBDA
    model_weight: float = DEFAULT_MODEL_WEIGHT

    # Risk management (see trading/budget_manager.py, trading/risk_manager.py).
    kelly_fraction: float = 0.25  # quarter-Kelly — full Kelly is optimal in expectation but has extreme variance
    max_drawdown_pct: float = 0.20  # equity drop from peak that pauses new entries (position exits still allowed)

    def _warn_if_strategy_budgets_exceed_ceiling(self) -> None:
        """Per-strategy budgets are independent allocations, not a hard sum
        constraint — but if they add up to more than the global ceiling,
        that's very likely a misconfiguration, so warn (don't crash)."""
        strategy_usd_total = self.arbitrage_daily_limit_usd + self.crypto_daily_limit_usd
        if strategy_usd_total > self.daily_limit_usd:
            logging.warning(
                "Per-strategy daily limits (arbitrage=$%.2f + crypto=$%.2f = $%.2f) exceed "
                "the global DAILY_LIMIT_USD ceiling ($%.2f)",
                self.arbitrage_daily_limit_usd, self.crypto_daily_limit_usd,
                strategy_usd_total, self.daily_limit_usd,
            )

        strategy_trades_total = self.arbitrage_max_daily_trades + self.crypto_max_daily_trades
        if strategy_trades_total > self.max_daily_trades:
            logging.warning(
                "Per-strategy max daily trades (arbitrage=%d + crypto=%d = %d) exceed the "
                "global MAX_DAILY_TRADES ceiling (%d)",
                self.arbitrage_max_daily_trades, self.crypto_max_daily_trades,
                strategy_trades_total, self.max_daily_trades,
            )

    @classmethod
    def from_env(cls) -> "TradingConfig":
        # Load local .env values when running outside Docker/AWS. This is a
        # side effect, so it only fires when a caller actually asks for the
        # process-env config — not on import of this module.
        load_dotenv("config/.env")

        cfg = cls(
            min_ev=float(os.getenv("MIN_EV", "0.50")),
            min_tte_minutes=int(os.getenv("MIN_TTE_MINUTES", "60")),
            max_tte_days=int(os.getenv("MAX_TTE_DAYS", "180")),
            daily_limit_usd=float(os.getenv("DAILY_LIMIT_USD", "15.0")),
            max_bet_size_usd=float(os.getenv("MAX_BET_SIZE_USD", "3.0")),
            bankroll_usd=float(os.getenv("BANKROLL_USD", "1000.0")),
            min_trading_balance=float(os.getenv("MIN_TRADING_BALANCE", "5.0")),
            take_profit_pct=float(os.getenv("TAKE_PROFIT_PCT", "0.20")),
            stop_loss_pct=float(os.getenv("STOP_LOSS_PCT", "-0.50")),
            min_hold_ev=float(os.getenv("MIN_HOLD_EV", "-0.10")),
            loop_delay_seconds=float(os.getenv("ENGINE_LOOP_DELAY", "2.0")),
            max_daily_trades=int(os.getenv("MAX_DAILY_TRADES", "10")),
            dry_run=_env_bool("DRY_RUN", "True"),
            paper_trade_mode=_env_bool("PAPER_TRADE_MODE", "False"),
            paper_balance_usd=float(os.getenv("PAPER_BALANCE_USD", "1000.0")),
            private_key=_env_first("POLYMARKET_PRIVATE_KEY", "POLYGON_PRIVATE_KEY"),
            proxy_address=_env_first("POLYMARKET_PROXY_ADDRESS", "POLY_ADDRESS"),
            signature_type=int(os.getenv("SIGNATURE_TYPE", "2")),
            openweather_api_key=_env_first("OPENWEATHER_API_KEY"),
            fred_api_key=_env_first("FRED_API_KEY"),
            pricing_mode=_env_first("PRICING_MODE", default="wang"),
            wang_base_lambda=float(os.getenv("WANG_BASE_LAMBDA", "0.183")),
            wang_min_edge=float(os.getenv("WANG_MIN_EDGE", "0.05")),
            wang_lambda=float(os.getenv("WANG_LAMBDA", str(DEFAULT_WANG_LAMBDA))),
            model_weight=float(os.getenv("MODEL_WEIGHT", str(DEFAULT_MODEL_WEIGHT))),
            kelly_fraction=float(os.getenv("KELLY_FRACTION", "0.25")),
            max_drawdown_pct=float(os.getenv("MAX_DRAWDOWN_PCT", "0.20")),
            enable_arbitrage=_env_bool("ENABLE_ARBITRAGE", "True"),
            arbitrage_daily_limit_usd=float(os.getenv("ARBITRAGE_DAILY_LIMIT_USD", "50.0")),
            arbitrage_max_daily_trades=int(os.getenv("ARBITRAGE_MAX_DAILY_TRADES", "50")),
            crypto_daily_limit_usd=float(os.getenv("CRYPTO_DAILY_LIMIT_USD", "50.0")),
            crypto_max_daily_trades=int(os.getenv("CRYPTO_MAX_DAILY_TRADES", "50")),
            arbitrage_order_timeout_seconds=float(os.getenv("ARBITRAGE_ORDER_TIMEOUT_SECONDS", "60")),
            arbitrage_crypto_first=_env_bool("ARBITRAGE_CRYPTO_FIRST", "True"),
        )
        cfg._warn_if_strategy_budgets_exceed_ceiling()
        return cfg

    @classmethod
    def from_file(cls, path: str) -> "TradingConfig":
        """Load a wallet's trading parameters from its config.json."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        defaults = cls()
        field_types = {f.name: f.type for f in fields(cls)}

        def _cast(name: str, raw_value):
            target_type = field_types.get(name)
            try:
                if target_type is bool:
                    return bool(raw_value)
                if target_type is int:
                    return int(raw_value)
                if target_type is float:
                    return float(raw_value)
            except (TypeError, ValueError):
                return getattr(defaults, name)
            return raw_value

        kwargs = {
            name: _cast(name, data[name])
            for name in field_types
            if name in data and data[name] is not None
        }
        cfg = cls(**kwargs)
        cfg._warn_if_strategy_budgets_exceed_ceiling()
        return cfg

import json
import os
from dataclasses import dataclass, fields

from dotenv import load_dotenv


DEFAULT_MIN_EV = 0.30
DEFAULT_MAX_BET_SIZE_USD = 3.0
DEFAULT_DAILY_LIMIT_USD = 15.0


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
    wang_base_lambda: float = 0.183
    wang_min_edge: float = 0.05  # minimum |wang_edge| (probability points) to consider a trade

    # Risk management (see trading/budget_manager.py, trading/risk_manager.py).
    kelly_fraction: float = 0.25  # quarter-Kelly — full Kelly is optimal in expectation but has extreme variance
    max_drawdown_pct: float = 0.20  # equity drop from peak that pauses new entries (position exits still allowed)

    @classmethod
    def from_env(cls) -> "TradingConfig":
        # Load local .env values when running outside Docker/AWS. This is a
        # side effect, so it only fires when a caller actually asks for the
        # process-env config — not on import of this module.
        load_dotenv("config/.env")

        return cls(
            min_ev=float(os.getenv("MIN_EV", "0.30")),
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
            kelly_fraction=float(os.getenv("KELLY_FRACTION", "0.25")),
            max_drawdown_pct=float(os.getenv("MAX_DRAWDOWN_PCT", "0.20")),
        )

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
        return cls(**kwargs)

import os
from dataclasses import dataclass

from dotenv import load_dotenv


# Load local .env values when running outside Docker/AWS.
load_dotenv()


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

    take_profit_pct: float = 0.20
    stop_loss_pct: float = -0.50
    min_hold_ev: float = -0.10

    loop_delay_seconds: float = 2.0
    private_key: str = ""
    proxy_address: str = ""
    signature_type: int = 2


    @classmethod
    def from_env(cls) -> "TradingConfig":
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
            private_key=_env_first("POLYMARKET_PRIVATE_KEY", "POLYGON_PRIVATE_KEY"),
            proxy_address=_env_first("POLYMARKET_PROXY_ADDRESS", "POLY_ADDRESS"),
            signature_type=int(os.getenv("SIGNATURE_TYPE", "2")),
        )

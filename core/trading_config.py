from dataclasses import dataclass
import os


DEFAULT_MIN_EV = 0.05


@dataclass
class TradingConfig:
    min_ev: float = DEFAULT_MIN_EV
    min_tte_minutes: int = 60
    max_tte_days: int = 180

    daily_limit_usd: float = 100.0
    bankroll_usd: float = 1000.0
    min_trading_balance: float = 5.0

    take_profit_pct: float = 0.20
    stop_loss_pct: float = -0.15
    min_hold_ev: float = 0.05

    loop_delay_seconds: float = 2.0


    @classmethod
    def from_env(cls) -> "TradingConfig":
        return cls(
            min_ev=float(os.getenv("MIN_EV", str(DEFAULT_MIN_EV))),
            min_tte_minutes=int(os.getenv("MIN_TTE_MINUTES", "60")),
            max_tte_days=int(os.getenv("MAX_TTE_DAYS", "180")),
            daily_limit_usd=float(os.getenv("DAILY_LIMIT_USD", "100.0")),
            bankroll_usd=float(os.getenv("BANKROLL_USD", "1000.0")),
            min_trading_balance=float(os.getenv("MIN_TRADING_BALANCE", "5.0")),
            take_profit_pct=float(os.getenv("TAKE_PROFIT_PCT", "0.20")),
            stop_loss_pct=float(os.getenv("STOP_LOSS_PCT", "-0.15")),
            min_hold_ev=float(os.getenv("MIN_HOLD_EV", "0.05")),
            loop_delay_seconds=float(os.getenv("ENGINE_LOOP_DELAY", "2.0")),
        )

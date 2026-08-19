"""WalletContext: bundles everything a single wallet needs to run in isolation.

Each wallet is a fully self-contained unit — its own config, its own live
state (DataBridge), and its own SQLite trade history. No component reads
from globals or module-level state; everything a wallet-scoped piece of code
needs is reached through its WalletContext.

Each wallet's data directory lives at data/{wallet_id}/ containing:
    config.json — that wallet's trading parameters
    trades.db   — that wallet's trade history database
"""

from dataclasses import dataclass

from core.bridge import DataBridge
from core.trading_config import TradingConfig

DATA_ROOT = "data"


@dataclass
class WalletContext:
    wallet_id: str                    # unique identifier (e.g. "wallet_alpha")
    config: TradingConfig             # this wallet's risk/trading parameters
    bridge: DataBridge                # this wallet's live state
    db_path: str                      # path to this wallet's SQLite DB (e.g. "data/wallet_alpha/trades.db")
    status: str = "idle"              # "running" | "stopped" | "error"
    drawdown_paused: bool = False     # set by the drawdown circuit breaker: True pauses new trade entry,
                                       # but position management/exits keep running (see decision_pipeline.py)

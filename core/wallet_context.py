"""WalletContext: bundles everything a single wallet needs to run in isolation.

Each wallet is a fully self-contained unit — its own config, its own live
state (DataBridge), and its own SQLite trade history. No component reads
from globals or module-level state; everything a wallet-scoped piece of code
needs is reached through its WalletContext.

Each wallet's data directory lives at data/{wallet_id}/ containing:
    config.json — that wallet's trading parameters
    trades.db   — that wallet's trade history database

WalletContext is a container, not a factory: it carries a wallet's runtime
components (executor, scanner, portfolio_manager, budget_manager) but never
constructs, initializes, or calls methods on them — the caller builds those
and hands them in. Construction order is executor -> scanner ->
portfolio_manager -> budget_manager (scanner needs executor; budget_manager
and portfolio_manager both need the bridge/config the earlier components
were built from) wherever all four are assembled together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from core.bridge import DataBridge
from core.trading_config import TradingConfig

if TYPE_CHECKING:
    from trading.executor import TradeExecutor
    from polymarket import PolymarketScannerHunter
    from trading.risk_manager import PortfolioManager
    from trading.budget_manager import BudgetManager

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

    # Runtime components — built by the caller and carried here so they
    # travel with the wallet. WalletContext never constructs these itself.
    executor: Optional["TradeExecutor"] = None
    scanner: Optional["PolymarketScannerHunter"] = None
    portfolio_manager: Optional["PortfolioManager"] = None
    budget_manager: Optional["BudgetManager"] = None

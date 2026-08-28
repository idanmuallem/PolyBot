"""build_wallet_runtime: builds and wires one wallet's runtime components
onto an already-constructed WalletContext.

Used by run_engine.py's standalone entrypoint. WalletContext stays a plain
container — this is the one place that builds and wires what it holds.
"""

from core.wallet_context import WalletContext
from polymarket import PolymarketScannerHunter
from trading.budget_manager import BudgetManager
from trading.decision_pipeline import _default_log_func, sync_live_account_state
from trading.executor import TradeExecutor
from trading.risk_manager import PortfolioManager


def build_wallet_runtime(ctx: WalletContext) -> None:
    """Build this wallet's runtime components in dependency order and attach
    them to ctx: executor -> scanner -> portfolio_manager -> (live balance
    sync) -> budget_manager.
    """
    ctx.executor = TradeExecutor(config=ctx.config)
    ctx.scanner = PolymarketScannerHunter(
        bridge=ctx.bridge,
        executor=ctx.executor,
        config=ctx.config,
    )
    ctx.portfolio_manager = PortfolioManager(
        bridge=ctx.bridge,
        executor=ctx.executor,
        config=ctx.config,
        hunter=ctx.scanner,
        db_path=ctx.db_path,
    )

    # Sync live balance before building BudgetManager so its initial_balance
    # reflects the account's real collateral, not the DataBridge default of
    # 0.0 (or, for run_engine.py's caller, whatever restore_wallet_state()
    # already restored from trade history).
    sync_live_account_state(ctx.bridge, ctx.executor, ctx.portfolio_manager, _default_log_func(ctx))

    ctx.budget_manager = BudgetManager(
        bridge=ctx.bridge,
        config=ctx.config,
        initial_balance=float(ctx.bridge.current_balance),
    )

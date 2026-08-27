"""WalletManager: top-level orchestrator for running multiple wallets concurrently.

Single process, async-based concurrency — no threads. Each wallet is a fully
isolated WalletContext; WalletManager just owns the registry and drives each
wallet's run_market_monitor loop as its own asyncio task.
"""

import asyncio
import os

import ui.data_manager as data_manager
from core.bridge import DataBridge
from core.trading_config import TradingConfig
from core.wallet_context import DATA_ROOT, WalletContext
from polymarket import PolymarketScannerHunter
from trading.budget_manager import BudgetManager
from trading.decision_pipeline import _default_log_func, run_market_monitor, sync_live_account_state
from trading.executor import TradeExecutor
from trading.risk_manager import PortfolioManager


def build_wallet_runtime(ctx: WalletContext) -> None:
    """Build this wallet's runtime components in dependency order and attach
    them to ctx: executor -> scanner -> portfolio_manager -> (live balance
    sync) -> budget_manager. WalletContext stays a plain container — this is
    the one place (alongside register_wallet below, which calls this) that
    builds and wires what it holds.

    Factored out of register_wallet() so run_engine.py's standalone
    single-wallet entrypoint can reuse the exact same wiring sequence
    without going through the file-based, multi-wallet-oriented
    register_wallet() API — both entrypoints end up building components the
    same way by construction, not by two independently-maintained copies of
    the same sequence.
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


class WalletManager:
    def __init__(self):
        self.wallets: dict[str, WalletContext] = {}

    def register_wallet(self, wallet_id: str, config_path: str) -> WalletContext:
        """Load config from file, create bridge and context, init DB."""
        config = TradingConfig.from_file(config_path)
        wallet_dir = os.path.join(DATA_ROOT, wallet_id)
        db_path = os.path.join(wallet_dir, "trades.db")
        os.makedirs(wallet_dir, exist_ok=True)
        bridge = DataBridge(wallet_id=wallet_id)
        ctx = WalletContext(
            wallet_id=wallet_id,
            config=config,
            bridge=bridge,
            db_path=db_path,
        )
        data_manager.init_db(db_path)

        build_wallet_runtime(ctx)

        self.wallets[wallet_id] = ctx
        return ctx

    async def start_wallet(self, wallet_id: str):
        """Start the trading pipeline for a specific wallet as an async task."""
        ctx = self.wallets[wallet_id]
        ctx.status = "running"
        try:
            await run_market_monitor(ctx)
        except Exception:
            ctx.status = "error"
            raise

    async def start_all(self):
        """Run all registered wallets concurrently via asyncio.gather."""
        tasks = [self.start_wallet(wid) for wid in self.wallets]
        await asyncio.gather(*tasks, return_exceptions=True)

    def get_all_statuses(self) -> dict:
        """Summary for the manager dashboard."""
        return {
            wid: {
                "status": ctx.status,
                "balance": ctx.bridge.current_balance,
                "daily_spend": ctx.bridge.daily_spend,
                "positions": len(ctx.bridge.current_portfolio),
            }
            for wid, ctx in self.wallets.items()
        }

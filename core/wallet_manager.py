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
from trading.decision_pipeline import run_market_monitor


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

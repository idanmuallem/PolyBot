"""Integration: the periodic paper-engine resolve check must run on the same
thread that (would have) constructed PaperAdapter's pm_trader.Engine.

pm_trader.Engine opens its sqlite3 connection with the library default
(check_same_thread=True, and the installed package exposes no way to
override it) in whatever thread first constructs PaperAdapter -- in
production that's the polybot-engine background thread, which also drives
this same asyncio loop. Routing resolve_closed_markets() through
asyncio.to_thread() hands the call to a *different* thread (a
ThreadPoolExecutor worker), which sqlite3 then refuses outright:
"SQLite objects created in a thread can only be used in that same thread."
"""
import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.bridge import DataBridge
from core.trading_config import TradingConfig
from core.wallet_context import WalletContext
from polymarket import PolymarketScannerHunter
from trading.budget_manager import BudgetManager
from trading.executor import TradeExecutor
from trading.risk_manager import PortfolioManager


def _make_pipeline(balance=100.0):
    from trading.decision_pipeline import SequentialTradingPipeline, sync_live_account_state

    bridge = DataBridge()
    bridge.current_balance = balance
    bridge.current_portfolio = []
    bridge.open_position_value = 0.0
    bridge.open_positions_value = 0.0

    config = TradingConfig(
        trading_mode="dry_run", min_ev=0.30, bankroll_usd=1000.0,
        daily_limit_usd=15.0, max_bet_size_usd=3.0,
        max_daily_trades=10, min_trading_balance=1.0,
        loop_delay_seconds=0.01,
    )

    ctx = WalletContext(wallet_id="test_wallet", config=config, bridge=bridge, db_path="test_wallet_trades.db")
    log_calls = []
    log_func = lambda level, *a, **kw: log_calls.append(level)

    with patch("trading.executor.TradeExecutor.get_open_positions", return_value=[]), \
         patch("trading.executor.TradeExecutor.get_balance", return_value=balance):
        ctx.executor = TradeExecutor(config=ctx.config)
        ctx.scanner = PolymarketScannerHunter(bridge=ctx.bridge, executor=ctx.executor, config=ctx.config)
        ctx.portfolio_manager = PortfolioManager(
            bridge=ctx.bridge, executor=ctx.executor, config=ctx.config,
            hunter=ctx.scanner, db_path=ctx.db_path,
        )
        sync_live_account_state(ctx.bridge, ctx.executor, ctx.portfolio_manager, log_func)
        ctx.budget_manager = BudgetManager(
            bridge=ctx.bridge, config=ctx.config, initial_balance=float(ctx.bridge.current_balance),
        )

        pipeline = SequentialTradingPipeline(
            ctx=ctx,
            log_func=log_func,
            delay=0.01,
        )

    return pipeline, ctx, bridge, log_calls


def _silence_other_stages(pipeline):
    """Stub out the hunt/strategy/portfolio stages so only the resolve-check
    block under test actually does anything on this loop iteration."""
    pipeline._stage_hunt = AsyncMock(return_value=None)
    pipeline._stage_strategy_scan = MagicMock(side_effect=lambda: asyncio.sleep(0))
    pipeline.portfolio_manager.manage_portfolio = MagicMock()


@pytest.mark.asyncio
async def test_resolve_closed_markets_runs_on_the_calling_thread():
    """Regression for the cross-thread sqlite3 crash: assert the call lands
    on the same thread that's driving the loop (i.e. NOT shuttled through
    asyncio.to_thread to some other worker thread)."""
    pipeline, ctx, bridge, log_calls = _make_pipeline(balance=100.0)
    _silence_other_stages(pipeline)

    calling_thread_id = threading.get_ident()
    seen_thread_ids = []

    mock_paper = MagicMock()
    mock_paper.resolve_closed_markets.side_effect = lambda **kw: seen_thread_ids.append(threading.get_ident())
    pipeline.executor.paper = mock_paper

    # Force the periodic (15-min) check to fire on the very first iteration.
    pipeline._last_resolve_check_at = 0.0

    try:
        await asyncio.wait_for(pipeline.run_forever(), timeout=0.3)
    except asyncio.TimeoutError:
        pass

    mock_paper.resolve_closed_markets.assert_called()
    assert seen_thread_ids == [calling_thread_id], (
        "resolve_closed_markets() ran on a different thread than the caller "
        "-- this is exactly the cross-thread sqlite3 bug (regression to "
        "asyncio.to_thread would reproduce it)"
    )


@pytest.mark.asyncio
async def test_resolve_closed_markets_failure_is_caught_and_logged():
    pipeline, ctx, bridge, log_calls = _make_pipeline(balance=100.0)
    _silence_other_stages(pipeline)

    mock_paper = MagicMock()
    mock_paper.resolve_closed_markets.side_effect = RuntimeError("boom")
    pipeline.executor.paper = mock_paper
    pipeline._last_resolve_check_at = 0.0

    try:
        await asyncio.wait_for(pipeline.run_forever(), timeout=0.3)
    except asyncio.TimeoutError:
        pass

    assert "PAPER-ERROR" in log_calls


@pytest.mark.asyncio
async def test_resolve_closed_markets_skipped_when_no_paper_adapter():
    pipeline, ctx, bridge, log_calls = _make_pipeline(balance=100.0)
    _silence_other_stages(pipeline)

    pipeline.executor.paper = None
    pipeline._last_resolve_check_at = 0.0

    try:
        await asyncio.wait_for(pipeline.run_forever(), timeout=0.3)
    except asyncio.TimeoutError:
        pass

    assert "PAPER-ERROR" not in log_calls

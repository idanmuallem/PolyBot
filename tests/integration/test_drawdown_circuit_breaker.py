"""Integration: drawdown circuit breaker pauses new trade entry but not
position management/exits."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.bridge import DataBridge
from core.trading_config import TradingConfig
from core.wallet_context import WalletContext
from polymarket import PolymarketScannerHunter
from trading.budget_manager import BudgetManager
from trading.executor import TradeExecutor
from trading.risk_manager import PortfolioManager


def _make_pipeline(balance=100.0, max_drawdown_pct=0.20):
    from trading.decision_pipeline import SequentialTradingPipeline, sync_live_account_state

    bridge = DataBridge()
    bridge.current_balance = balance
    bridge.current_portfolio = []
    bridge.open_position_value = 0.0
    bridge.open_positions_value = 0.0

    config = TradingConfig(
        dry_run=True, min_ev=0.30, bankroll_usd=1000.0,
        daily_limit_usd=15.0, max_bet_size_usd=3.0,
        max_daily_trades=10, min_trading_balance=1.0,
        max_drawdown_pct=max_drawdown_pct,
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


# ── _update_drawdown_guard: peak tracking, threshold, logging ────────────────

def test_no_pause_within_drawdown_limit():
    pipeline, ctx, bridge, log_calls = _make_pipeline(balance=100.0, max_drawdown_pct=0.20)
    bridge.current_balance = 90.0  # 10% down from the $100 peak

    pipeline._update_drawdown_guard()

    assert ctx.drawdown_paused is False
    assert bridge.drawdown_paused is False
    assert "CIRCUIT-BREAKER" not in log_calls


def test_pause_triggers_past_drawdown_limit():
    pipeline, ctx, bridge, log_calls = _make_pipeline(balance=100.0, max_drawdown_pct=0.20)
    bridge.current_balance = 75.0  # 25% down from the $100 peak

    pipeline._update_drawdown_guard()

    assert ctx.drawdown_paused is True
    assert bridge.drawdown_paused is True
    assert "CIRCUIT-BREAKER" in log_calls


def test_pause_clears_once_equity_recovers():
    pipeline, ctx, bridge, log_calls = _make_pipeline(balance=100.0, max_drawdown_pct=0.20)
    bridge.current_balance = 75.0
    pipeline._update_drawdown_guard()
    assert ctx.drawdown_paused is True

    bridge.current_balance = 100.0
    pipeline._update_drawdown_guard()

    assert ctx.drawdown_paused is False
    assert bridge.drawdown_paused is False
    assert "CIRCUIT-BREAKER-CLEAR" in log_calls


def test_peak_equity_ratchets_up_with_new_highs():
    pipeline, ctx, bridge, _ = _make_pipeline(balance=100.0, max_drawdown_pct=0.20)
    bridge.current_balance = 150.0
    pipeline._update_drawdown_guard()
    assert pipeline.peak_equity == pytest.approx(150.0)

    # A drop to $125 is only ~16.7% down from the new $150 peak -> no pause.
    bridge.current_balance = 125.0
    pipeline._update_drawdown_guard()
    assert ctx.drawdown_paused is False


def test_drawdown_pause_logs_only_once_on_transition():
    pipeline, ctx, bridge, log_calls = _make_pipeline(balance=100.0, max_drawdown_pct=0.20)
    bridge.current_balance = 75.0

    pipeline._update_drawdown_guard()
    pipeline._update_drawdown_guard()  # still paused -- should not re-log

    assert log_calls.count("CIRCUIT-BREAKER") == 1


def test_bridge_tracks_peak_equity_and_drawdown_pct():
    pipeline, ctx, bridge, _ = _make_pipeline(balance=100.0, max_drawdown_pct=0.20)
    bridge.current_balance = 80.0

    pipeline._update_drawdown_guard()

    assert bridge.peak_equity == pytest.approx(100.0)
    assert bridge.current_drawdown_pct == pytest.approx(0.20)


# ── run_forever: paused state skips new entries but not exits ────────────────

@pytest.mark.asyncio
async def test_paused_pipeline_skips_new_entry_but_runs_exits():
    pipeline, ctx, bridge, log_calls = _make_pipeline(balance=100.0, max_drawdown_pct=0.20)
    pipeline.peak_equity = 100.0

    pipeline._stage_hunt = AsyncMock(return_value=None)  # _stage_hunt is now async (Phase 6a)
    pipeline._stage_strategy_scan = MagicMock(side_effect=lambda: asyncio.sleep(0))
    pipeline.portfolio_manager.manage_portfolio = MagicMock()

    # _sync_live_account_state() re-pulls the balance from the executor every
    # tick, so the drop has to come from there, not a one-off bridge mutation.
    with patch("trading.executor.TradeExecutor.get_balance", return_value=75.0):
        try:
            await asyncio.wait_for(pipeline.run_forever(), timeout=0.3)
        except asyncio.TimeoutError:
            pass

    assert ctx.drawdown_paused is True
    pipeline.portfolio_manager.manage_portfolio.assert_called()
    pipeline._stage_hunt.assert_not_called()
    pipeline._stage_strategy_scan.assert_not_called()


@pytest.mark.asyncio
async def test_unpaused_pipeline_still_scans_for_new_entries():
    pipeline, ctx, bridge, log_calls = _make_pipeline(balance=100.0, max_drawdown_pct=0.20)
    # No drawdown -- stays well within limits.

    pipeline._stage_hunt = AsyncMock(return_value=None)  # _stage_hunt is now async (Phase 6a)
    pipeline._stage_strategy_scan = MagicMock(side_effect=lambda: asyncio.sleep(0))
    pipeline.portfolio_manager.manage_portfolio = MagicMock()

    try:
        await asyncio.wait_for(pipeline.run_forever(), timeout=0.3)
    except asyncio.TimeoutError:
        pass

    assert ctx.drawdown_paused is False
    pipeline.portfolio_manager.manage_portfolio.assert_called()
    pipeline._stage_hunt.assert_called()
    pipeline._stage_strategy_scan.assert_called()

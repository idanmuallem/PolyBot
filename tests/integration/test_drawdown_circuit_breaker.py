"""Integration: the Account's drawdown circuit breaker pauses new trade
entry for the pipeline tick, but position management/exits keep running.

The breaker's own arithmetic (peak tracking, threshold, transition logging)
is covered at the Account interface in tests/unit/test_account.py."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import ui.data_manager as dm
from core.bridge import DataBridge
from core.trading_config import TradingConfig
from core.wallet_context import WalletContext
from polymarket import PolymarketScannerHunter
from trading.executor import TradeExecutor
from trading.fake_venue import FakeVenue
from trading.ledger import Ledger
from trading.risk_manager import PortfolioManager
from trading.venue import MarketRef

BTC = MarketRef(yes_token_id="btc-yes", no_token_id="btc-no", condition_id="0xbtc", slug="btc-100k")


def _make_pipeline(tmp_path, max_drawdown_pct=0.20):
    """A $100 account: $50 cash plus 100 BTC YES shares bought at 0.50."""
    from trading.decision_pipeline import SequentialTradingPipeline

    venue = FakeVenue(cash=100.0)
    venue.list_market(BTC, yes_price=0.50)
    venue.buy(BTC, "YES", 50.0)

    config = TradingConfig(
        trading_mode="dry_run", min_ev=0.30, bankroll_usd=1000.0,
        daily_limit_usd=15.0, max_bet_size_usd=3.0,
        max_daily_trades=10, min_trading_balance=1.0,
        max_drawdown_pct=max_drawdown_pct,
        loop_delay_seconds=0.01,
    )
    db_path = str(tmp_path / "trades.db")
    dm.init_db(db_path)
    ctx = WalletContext(wallet_id="test_wallet", config=config, bridge=DataBridge(), db_path=db_path)
    ctx.ledger = Ledger(db_path)
    ctx.executor = TradeExecutor(config=config, venue=venue)
    ctx.scanner = PolymarketScannerHunter(bridge=ctx.bridge, executor=ctx.executor, config=config)
    ctx.portfolio_manager = PortfolioManager(
        bridge=ctx.bridge, executor=ctx.executor, config=config,
        hunter=ctx.scanner, db_path=db_path, ledger=ctx.ledger,
    )
    pipeline = SequentialTradingPipeline(ctx=ctx, log_func=lambda *a, **kw: None, delay=0.01)

    pipeline._stage_hunt = AsyncMock(return_value=None)
    pipeline._stage_strategy_scan = MagicMock(side_effect=lambda: asyncio.sleep(0))
    pipeline.portfolio_manager.manage_portfolio = MagicMock()
    return pipeline, venue, ctx


async def _run_briefly(pipeline):
    try:
        await asyncio.wait_for(pipeline.run_forever(), timeout=0.3)
    except asyncio.TimeoutError:
        pass


@pytest.mark.asyncio
async def test_paused_pipeline_skips_new_entry_but_runs_exits(tmp_path):
    pipeline, venue, ctx = _make_pipeline(tmp_path)
    venue.set_price(BTC, "YES", 0.25)         # equity 75: 25% off the $100 peak

    await _run_briefly(pipeline)

    assert ctx.account.drawdown_paused is True
    assert ctx.bridge.drawdown_paused is True
    pipeline.portfolio_manager.manage_portfolio.assert_called()
    pipeline._stage_hunt.assert_not_called()
    pipeline._stage_strategy_scan.assert_not_called()


@pytest.mark.asyncio
async def test_unpaused_pipeline_still_scans_for_new_entries(tmp_path):
    pipeline, venue, ctx = _make_pipeline(tmp_path)
    venue.set_price(BTC, "YES", 0.40)         # equity 90: within the 20% limit

    await _run_briefly(pipeline)

    assert ctx.account.drawdown_paused is False
    pipeline.portfolio_manager.manage_portfolio.assert_called()
    pipeline._stage_hunt.assert_called()
    pipeline._stage_strategy_scan.assert_called()


@pytest.mark.asyncio
async def test_engine_status_shows_the_account_snapshot(tmp_path):
    """The dashboard reads engine_status: it must carry the Account's cash
    and position value, the same numbers the breaker used."""
    pipeline, venue, ctx = _make_pipeline(tmp_path)
    venue.set_price(BTC, "YES", 0.40)

    await _run_briefly(pipeline)

    status = dm.read_engine_status(ctx.db_path)
    assert status["current_balance"] == pytest.approx(50.0)
    assert status["open_position_value"] == pytest.approx(40.0)

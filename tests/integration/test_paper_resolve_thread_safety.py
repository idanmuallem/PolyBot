"""Integration: the pipeline's periodic paper-resolution check must call the
Venue on the loop's own thread (regression for 28c7637).

PaperVenue's pm_trader.Engine opens its sqlite3 connection with the library
default check_same_thread=True (no override exposed), in whichever thread
first uses it -- in production, the engine loop's thread. Routing
resolve_closed_markets() through asyncio.to_thread() hands it to a worker
thread, which sqlite3 refuses ("SQLite objects created in a thread can only
be used in that same thread"), so nothing ever resolved.

These tests drive the real loop against a real PaperVenue (pm_trader + sqlite
in tmp_path, stubbed market data) and assert on the outcome: the closed
market's position is settled and reported.
"""
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
from trading.venue import MarketRef

BTC = MarketRef(yes_token_id="btc-yes", no_token_id="btc-no", condition_id="0xbtc", slug="btc-100k")


def _make_pipeline(tmp_path, venue):
    from trading.decision_pipeline import SequentialTradingPipeline

    bridge = DataBridge()
    bridge.current_portfolio = []
    config = TradingConfig(
        trading_mode="dry_run", min_ev=0.30, bankroll_usd=1000.0,
        daily_limit_usd=15.0, max_bet_size_usd=3.0,
        max_daily_trades=10, min_trading_balance=1.0,
        loop_delay_seconds=0.01,
    )
    ctx = WalletContext(wallet_id="test_wallet", config=config, bridge=bridge,
                        db_path=str(tmp_path / "trades.db"))
    rows = []
    log_func = lambda level, asset_type, token_id, payload: rows.append((level, token_id, payload))

    ctx.executor = TradeExecutor(config=ctx.config, venue=venue)
    ctx.scanner = PolymarketScannerHunter(bridge=ctx.bridge, executor=ctx.executor, config=ctx.config)
    ctx.portfolio_manager = PortfolioManager(
        bridge=ctx.bridge, executor=ctx.executor, config=ctx.config,
        hunter=ctx.scanner, db_path=ctx.db_path,
    )
    ctx.budget_manager = BudgetManager(
        bridge=ctx.bridge, config=ctx.config, initial_balance=float(ctx.bridge.current_balance),
    )
    pipeline = SequentialTradingPipeline(ctx=ctx, log_func=log_func, delay=0.01)

    # Only the resolve-check block under test does anything this iteration.
    pipeline._stage_hunt = AsyncMock(return_value=None)
    pipeline._stage_strategy_scan = AsyncMock(return_value=None)
    pipeline.portfolio_manager.manage_portfolio = MagicMock()
    pipeline._last_resolve_check_at = 0.0  # fire the 15-min check on the first tick
    return pipeline, rows


async def _run_one_tick(pipeline):
    with patch("trading.decision_pipeline.data_manager.read_live_trading_requested", return_value=False), \
         patch("trading.decision_pipeline.data_manager.write_engine_status"), \
         patch("trading.decision_pipeline.data_manager.write_open_positions"):
        try:
            await asyncio.wait_for(pipeline.run_forever(), timeout=0.3)
        except asyncio.TimeoutError:
            pass


@pytest.mark.asyncio
async def test_paper_resolution_settles_closed_market_on_the_loop_thread(tmp_path):
    pytest.importorskip("pm_trader")
    from tests.venue_harness import make_paper_venue

    # Built (and its sqlite connection opened) on this, the loop's, thread.
    venue, exchange = make_paper_venue(tmp_path / "paper_trading", 1000.0)
    exchange.list_market(BTC, yes_price=0.40)
    assert venue.buy(BTC, "YES", 20.0) is True
    exchange.close_market(BTC, "YES")
    pipeline, rows = _make_pipeline(tmp_path, venue)

    await _run_one_tick(pipeline)

    expired = [(token, payload) for level, token, payload in rows if level == "EXPIRED"]
    assert expired and expired[0][0] == "btc-yes", (
        "the closed market was never settled -- a cross-thread call into "
        "the paper venue's sqlite connection fails exactly like this"
    )
    assert venue.open_positions() == []
    assert venue.cash() == pytest.approx(1000.0 - 20.0 + 50.0)
    assert "PAPER-ERROR" not in [level for level, _, _ in rows]


@pytest.mark.asyncio
async def test_paper_resolution_failure_is_caught_and_logged(tmp_path):
    from trading.fake_venue import FakeVenue

    class _ResolveExplodes(FakeVenue):
        def resolve_closed_markets(self, **kw):
            raise RuntimeError("boom")

    pipeline, rows = _make_pipeline(tmp_path, _ResolveExplodes(cash=100.0))

    await _run_one_tick(pipeline)

    assert "PAPER-ERROR" in [level for level, _, _ in rows]


@pytest.mark.asyncio
async def test_paper_resolution_skipped_without_venue(tmp_path):
    pipeline, rows = _make_pipeline(tmp_path, venue=None)

    await _run_one_tick(pipeline)

    assert "PAPER-ERROR" not in [level for level, _, _ in rows]

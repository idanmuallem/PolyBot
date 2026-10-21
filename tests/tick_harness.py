"""Test harness: one engine pipeline on FakeVenues and a real Ledger, driven
one tick at a time through SequentialTradingPipeline.tick().

    pipeline, ctx, rows = make_tick_pipeline(tmp_path, paper=FakeVenue(), live=live_fake())
    data_manager.write_live_trading_requested(ctx.db_path, True)
    await pipeline.tick()

Hunting finds nothing (QuietScanner) and arbitrage is off unless a test
turns it on, so a tick only does what the test sets up.
"""

from __future__ import annotations

from typing import List, Optional

import ui.data_manager as dm
from core.bridge import DataBridge
from core.trading_config import TradingConfig
from core.wallet_context import WalletContext
from trading.executor import TradeExecutor
from trading.fake_venue import FakeVenue
from trading.ledger import Ledger
from trading.risk_manager import PortfolioManager


def live_fake(cash: float = 1000.0) -> FakeVenue:
    """A FakeVenue standing in for the live (CLOB) venue."""
    venue = FakeVenue(cash=cash)
    venue.name = "live"
    return venue


class QuietScanner:
    """Hunter that never finds a market."""

    def get_active_markets(self, log_func):
        return None, None

    def add_to_cooldown(self, token_id):
        pass


class FixedMarketScanner:
    """Hunter that finds the same market on every scan, with a fixed live
    truth (the underlying's spot) for the brain."""

    def __init__(self, market, live_truth: float):
        self.market = market
        self.live_truth = live_truth

    def get_active_markets(self, log_func):
        return self.market, self

    def get_live_truth(self, market):
        return self.live_truth

    def add_to_cooldown(self, token_id):
        pass


class StaticEvents:
    """PolymarketClient stand-in for the strategy scan."""

    def get_multi_outcome_events(self, limit=100):
        return []


class FixedSignals:
    """Strategy that emits the same signals on every scan."""

    strategy_type = "arbitrage"

    def __init__(self, signals: list):
        self.signals = list(signals)

    async def scan(self, events):
        return list(self.signals)


def make_tick_pipeline(
    tmp_path,
    paper: Optional[FakeVenue] = None,
    live: Optional[FakeVenue] = None,
    active: str = "paper",
    config: Optional[TradingConfig] = None,
    balance: float = 100.0,
    scanner=None,
    clock=None,
):
    """(pipeline, ctx, rows): rows collects every (level, token_id, payload)
    the engine logs. The venues present become ctx.venues; *active* is the
    one the executor starts on. *scanner* replaces the QuietScanner (e.g. a
    FixedMarketScanner); *clock* is the pipeline's UTC clock (callable)."""
    from trading.decision_pipeline import SequentialTradingPipeline

    db_path = str(tmp_path / "trades.db")
    dm.init_db(db_path)
    config = config or TradingConfig(
        trading_mode="dry_run" if active == "paper" else "live_run",
        min_ev=0.30, bankroll_usd=1000.0, daily_limit_usd=15.0, max_bet_size_usd=3.0,
        max_daily_trades=10, min_trading_balance=1.0, loop_delay_seconds=0.01,
        enable_arbitrage=False,
    )
    bridge = DataBridge()
    bridge.current_balance = balance
    bridge.current_portfolio = []
    ctx = WalletContext(wallet_id="test_wallet", config=config, bridge=bridge, db_path=db_path)
    ctx.ledger = Ledger(db_path)
    ctx.venues = {v.name: v for v in (paper, live) if v is not None}

    rows: List[tuple] = []

    def log_func(level, asset_type, token_id, payload):
        rows.append((level, token_id, payload))

    ctx.executor = TradeExecutor(config=config, venue=ctx.venues[active])
    ctx.scanner = scanner or QuietScanner()
    ctx.portfolio_manager = PortfolioManager(
        bridge=bridge, executor=ctx.executor, config=config,
        hunter=ctx.scanner, db_path=db_path, ledger=ctx.ledger,
    )
    pipeline = SequentialTradingPipeline(ctx=ctx, log_func=log_func, delay=0.01, clock=clock)
    pipeline.polymarket_client = StaticEvents()
    return pipeline, ctx, rows


def levels(rows) -> List[str]:
    return [level for level, _, _ in rows]

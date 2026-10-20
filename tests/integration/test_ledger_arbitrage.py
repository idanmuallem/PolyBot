"""Integration: arbitrage baskets record every leg fill and unwind sale in
the Ledger, tagged strategy `arbitrage` with the basket's group_id, and
paper arbitrage fills show up in dashboard stats and transactions (#14).

Driven through the pipeline's arbitrage group execution path with the
executor trading on a FakeVenue and a real Ledger on tmp_path. A leg's fill
is set by its listed price: $1 at a listed price p buys 1/p shares, and an
unlisted leg never fills. #19 moves this to a tick-level test. Resolutions are
covered at the Ledger seam (tests/unit/test_ledger.py): a resolution of an
arbitrage leg inherits its strategy and group_id from the open position.
"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import ui.data_manager as dm
from core.bridge import DataBridge
from core.models import MarketData, Position
from core.trading_config import TradingConfig
from core.wallet_context import WalletContext
from polymarket import PolymarketScannerHunter
from trading.budget_manager import BudgetManager
from trading.executor import TradeExecutor
from trading.fake_venue import FakeVenue
from trading.ledger import Ledger
from trading.risk_manager import PortfolioManager
from trading.strategies.base import StrategySignal
from trading.venue import MarketRef

GROUP = "event_sum:evt1"
_ARBITRAGE = SimpleNamespace(strategy_type="arbitrage")


def _make_pipeline(db_path, balance=100.0):
    from trading.decision_pipeline import SequentialTradingPipeline

    dm.init_db(db_path)
    bridge = DataBridge()
    bridge.current_balance = balance
    bridge.current_portfolio = []
    config = TradingConfig(
        trading_mode="dry_run", min_ev=0.30, bankroll_usd=1000.0,
        daily_limit_usd=15.0, max_bet_size_usd=3.0,
        max_daily_trades=10, min_trading_balance=1.0,
    )
    config.arbitrage_order_timeout_seconds = 0.05
    ctx = WalletContext(wallet_id="test_wallet", config=config, bridge=bridge, db_path=db_path)
    ctx.ledger = Ledger(db_path)
    log_func = lambda *args: None

    with patch("trading.executor.TradeExecutor.get_open_positions", return_value=[]), \
         patch("trading.executor.TradeExecutor.get_balance", return_value=balance):
        ctx.executor = TradeExecutor(config=ctx.config, venue=FakeVenue(cash=1000.0))
        ctx.scanner = PolymarketScannerHunter(bridge=bridge, executor=ctx.executor, config=config)
        ctx.portfolio_manager = PortfolioManager(
            bridge=bridge, executor=ctx.executor, config=config,
            hunter=ctx.scanner, db_path=db_path, ledger=ctx.ledger,
        )
        ctx.budget_manager = BudgetManager(bridge=bridge, config=config, initial_balance=balance)
        pipeline = SequentialTradingPipeline(ctx=ctx, log_func=log_func, delay=0.01)
    return pipeline, ctx


def _signal(token_id, price, bet_amount_usd=1.0):
    market = MarketData(
        market_id=token_id, asset_type="Arbitrage::EventSum", strike_price=0.0,
        question=f"Outcome {token_id}?", market_name=f"Test Event - {token_id}",
        initial_price=price, volume=100_000.0, no_market_id=f"{token_id}_no",
        condition_id=f"cond_{token_id}", slug=f"slug-{token_id}",
    )
    return StrategySignal(
        market=market, side="YES", price=price, bet_amount_usd=bet_amount_usd,
        strategy_type="arbitrage", reasoning="test", group_id=GROUP, edge=0.10,
    )


async def _run(pipeline, signals, listed_prices, sell_ok=True):
    """List each leg's market on the FakeVenue at *listed_prices* {token: price}
    (unlisted legs never fill), then execute the basket."""
    venue = pipeline.executor.venue
    for signal in signals:
        price = listed_prices.get(signal.market.market_id)
        if price is not None:
            venue.list_market(MarketRef.from_market(signal.market), price)
    if not sell_ok:
        venue.sell = lambda token_id, shares: False
    await pipeline._execute_strategy_group(_ARBITRAGE, GROUP, signals)


def _kinds(ledger):
    return [(t["kind"], t["token_id"], t["shares"]) for t in reversed(ledger.transactions())]


@pytest.mark.asyncio
async def test_filled_basket_records_every_leg_as_an_arbitrage_entry(tmp_path):
    pipeline, ctx = _make_pipeline(str(tmp_path / "trades.db"))

    # $1 per leg at $0.40 fills 2.5 shares each.
    await _run(pipeline, [_signal("tokA", 0.40), _signal("tokB", 0.40)], {"tokA": 0.40, "tokB": 0.40})

    rows = ctx.ledger.transactions()
    assert sorted((r["kind"], r["token_id"]) for r in rows) == [("BUY", "tokA"), ("BUY", "tokB")]
    assert {(r["strategy"], r["group_id"], r["venue"]) for r in rows} == {("arbitrage", GROUP, "paper")}
    held = {h.token_id: h for h in ctx.ledger.open_positions()}
    assert held["tokA"].shares == pytest.approx(2.5)
    assert held["tokA"].entry_price == pytest.approx(0.40)
    assert held["tokA"].condition_id == "cond_tokA"
    assert held["tokB"].shares == pytest.approx(2.5)


@pytest.mark.asyncio
async def test_paper_arbitrage_fills_appear_in_dashboard_stats_and_transactions(tmp_path):
    db = str(tmp_path / "trades.db")
    pipeline, _ = _make_pipeline(db)

    await _run(pipeline, [_signal("tokA", 0.40), _signal("tokB", 0.40)], {"tokA": 0.40, "tokB": 0.40})

    assert dm.get_trade_stats(db)["total_trades"] == 2
    tx = dm.fetch_transactions(db)
    assert sorted(tx["Market Name"]) == ["Test Event - tokA", "Test Event - tokB"]
    assert set(tx["Type"]) == {"BUY"}


@pytest.mark.asyncio
async def test_surplus_trim_is_recorded_as_a_sale_leaving_the_matched_set_held(tmp_path):
    pipeline, ctx = _make_pipeline(str(tmp_path / "trades.db"))

    # tokA fills 5 shares at $0.20, tokB 2.5 at $0.40: 2.5 tokA are surplus.
    await _run(pipeline, [_signal("tokA", 0.30), _signal("tokB", 0.40)], {"tokA": 0.20, "tokB": 0.40})

    sale = [t for t in ctx.ledger.transactions() if t["kind"] == "SELL"]
    assert len(sale) == 1
    assert (sale[0]["token_id"], sale[0]["shares"], sale[0]["reason"]) == ("tokA", 2.5, "ARBITRAGE-TRIM")
    assert (sale[0]["strategy"], sale[0]["group_id"]) == ("arbitrage", GROUP)
    held = {h.token_id: h.shares for h in ctx.ledger.open_positions()}
    assert held == {"tokA": pytest.approx(2.5), "tokB": pytest.approx(2.5)}


@pytest.mark.asyncio
async def test_dashboard_labels_arbitrage_trim_and_unwind_sales(tmp_path):
    db = str(tmp_path / "trades.db")
    trimmed, _ = _make_pipeline(db)
    await _run(trimmed, [_signal("tokA", 0.30), _signal("tokB", 0.40)], {"tokA": 0.20, "tokB": 0.40})
    unwound, _ = _make_pipeline(db)
    await _run(unwound, [_signal("tokC", 0.30), _signal("tokD", 0.35)], {"tokC": 0.25})

    labels = set(dm.fetch_transactions(db)["Type"])
    assert {"SELL (Arb Trim)", "SELL (Arb Unwind)"} <= labels


@pytest.mark.asyncio
async def test_incomplete_basket_unwind_is_recorded_and_leaves_nothing_held(tmp_path):
    pipeline, ctx = _make_pipeline(str(tmp_path / "trades.db"))

    # tokA fills 4 shares at $0.25; tokB is never listed, so never fills.
    await _run(pipeline, [_signal("tokA", 0.30), _signal("tokB", 0.35)], {"tokA": 0.25})

    assert _kinds(ctx.ledger) == [("BUY", "tokA", 4.0), ("SELL", "tokA", 4.0)]
    assert ctx.ledger.transactions()[0]["reason"] == "ARBITRAGE-UNWIND"
    assert {t["group_id"] for t in ctx.ledger.transactions()} == {GROUP}
    assert ctx.ledger.open_positions() == []


@pytest.mark.asyncio
async def test_unprofitable_basket_unwind_is_recorded_and_leaves_nothing_held(tmp_path):
    pipeline, ctx = _make_pipeline(str(tmp_path / "trades.db"))

    # $1 per leg at $0.50 buys 2 shares each: the basket costs its full $1 payout.
    await _run(pipeline, [_signal("tokA", 0.50), _signal("tokB", 0.50)], {"tokA": 0.50, "tokB": 0.50})

    sales = [t for t in ctx.ledger.transactions() if t["kind"] == "SELL"]
    assert sorted(t["token_id"] for t in sales) == ["tokA", "tokB"]
    assert {t["reason"] for t in sales} == {"ARBITRAGE-UNWIND"}
    assert ctx.ledger.open_positions() == []


@pytest.mark.asyncio
async def test_failed_unwind_sale_is_not_recorded_so_the_leg_stays_held(tmp_path):
    pipeline, ctx = _make_pipeline(str(tmp_path / "trades.db"))

    await _run(pipeline, [_signal("tokA", 0.30), _signal("tokB", 0.35)], {"tokA": 0.25}, sell_ok=False)

    assert _kinds(ctx.ledger) == [("BUY", "tokA", 4.0)]
    assert [h.token_id for h in ctx.ledger.open_positions()] == ["tokA"]


@pytest.mark.asyncio
async def test_partial_basket_completion_adds_only_the_missing_leg(tmp_path):
    pipeline, ctx = _make_pipeline(str(tmp_path / "trades.db"))
    # tokA was bought by an earlier attempt and is still held.
    ctx.ledger.record_entry(
        token_id="tokA", condition_id="cond_tokA", side="YES", shares=3.0, price=0.30,
        venue="paper", strategy="arbitrage", group_id=GROUP,
    )
    ctx.bridge.current_portfolio = [
        Position(market_id="tokA", token_id="tokA", initial_price=0.30, current_price=0.30,
                 shares=3.0, value=0.90, pnl_ratio=0.0, side="YES"),
    ]

    # Only tokB is bought: $1 at $0.40 fills 2.5 shares.
    await _run(pipeline, [_signal("tokA", 0.30), _signal("tokB", 0.40)], {"tokB": 0.40})

    buys = [t for t in ctx.ledger.transactions() if t["kind"] == "BUY"]
    assert sorted(t["token_id"] for t in buys) == ["tokA", "tokB"]
    held = {h.token_id: (h.shares, h.group_id) for h in ctx.ledger.open_positions()}
    assert held == {"tokA": (pytest.approx(3.0), GROUP), "tokB": (pytest.approx(2.5), GROUP)}

"""Integration: arbitrage baskets record every leg fill and unwind sale in
the Ledger, tagged strategy `arbitrage` with the basket's group_id and the
active venue, and paper arbitrage fills show up in dashboard stats and
transactions (#14, #19).

Driven one engine tick at a time (tests/tick_harness.py): a strategy emits
the basket's legs, the executor trades on a FakeVenue, and a real Ledger
lives on tmp_path. A leg's fill is set by its listed price: $1 at a listed
price p buys 1/p shares, and an unlisted leg never fills. Resolutions are
covered at the Ledger seam (tests/unit/test_ledger.py): a resolution of an
arbitrage leg inherits its strategy and group_id from the open position.
"""
import pytest

import ui.data_manager as dm
from core.models import MarketData
from core.trading_config import TradingConfig
from tests.tick_harness import FixedSignals, live_fake, make_tick_pipeline
from trading.fake_venue import FakeVenue
from trading.strategies.base import StrategySignal
from trading.venue import MarketRef

GROUP = "event_sum:evt1"


def _make_pipeline(tmp_path, live=None):
    config = TradingConfig(
        trading_mode="dry_run", min_ev=0.30, bankroll_usd=1000.0,
        daily_limit_usd=15.0, max_bet_size_usd=3.0,
        max_daily_trades=10, min_trading_balance=1.0, loop_delay_seconds=0.01,
    )
    config.arbitrage_order_timeout_seconds = 0.05
    pipeline, ctx, _ = make_tick_pipeline(
        tmp_path, paper=FakeVenue(cash=1000.0), live=live, config=config,
    )
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
    """List each leg's market on every FakeVenue at *listed_prices*
    {token: price} (unlisted legs never fill), then run one tick in which the
    strategy emits the basket."""
    for venue in pipeline.venues.values():
        for signal in signals:
            price = listed_prices.get(signal.market.market_id)
            if price is not None:
                venue.list_market(MarketRef.from_market(signal.market), price)
        if not sell_ok:
            venue.sell = lambda *args, **kwargs: False
    pipeline.strategies = [FixedSignals(signals)]
    await pipeline.tick()


def _kinds(ledger):
    return [(t["kind"], t["token_id"], t["shares"]) for t in reversed(ledger.transactions())]


@pytest.mark.asyncio
async def test_filled_basket_records_every_leg_as_an_arbitrage_entry(tmp_path):
    pipeline, ctx = _make_pipeline(tmp_path)

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
    pipeline, ctx = _make_pipeline(tmp_path)
    db = ctx.db_path

    await _run(pipeline, [_signal("tokA", 0.40), _signal("tokB", 0.40)], {"tokA": 0.40, "tokB": 0.40})

    assert dm.get_trade_stats(db)["total_trades"] == 2
    tx = dm.fetch_transactions(db)
    assert sorted(tx["Market Name"]) == ["Test Event - tokA", "Test Event - tokB"]
    assert set(tx["Type"]) == {"BUY"}


@pytest.mark.asyncio
async def test_surplus_trim_is_recorded_as_a_sale_leaving_the_matched_set_held(tmp_path):
    pipeline, ctx = _make_pipeline(tmp_path)

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
    trimmed, ctx = _make_pipeline(tmp_path)
    db = ctx.db_path
    await _run(trimmed, [_signal("tokA", 0.30), _signal("tokB", 0.40)], {"tokA": 0.20, "tokB": 0.40})
    unwound, _ = _make_pipeline(tmp_path)
    await _run(unwound, [_signal("tokC", 0.30), _signal("tokD", 0.35)], {"tokC": 0.25})

    labels = set(dm.fetch_transactions(db)["Type"])
    assert {"SELL (Arb Trim)", "SELL (Arb Unwind)"} <= labels


@pytest.mark.asyncio
async def test_incomplete_basket_unwind_is_recorded_and_leaves_nothing_held(tmp_path):
    pipeline, ctx = _make_pipeline(tmp_path)

    # tokA fills 4 shares at $0.25; tokB is never listed, so never fills.
    await _run(pipeline, [_signal("tokA", 0.30), _signal("tokB", 0.35)], {"tokA": 0.25})

    assert _kinds(ctx.ledger) == [("BUY", "tokA", 4.0), ("SELL", "tokA", 4.0)]
    assert ctx.ledger.transactions()[0]["reason"] == "ARBITRAGE-UNWIND"
    assert {t["group_id"] for t in ctx.ledger.transactions()} == {GROUP}
    assert ctx.ledger.open_positions() == []


@pytest.mark.asyncio
async def test_unprofitable_basket_unwind_is_recorded_and_leaves_nothing_held(tmp_path):
    pipeline, ctx = _make_pipeline(tmp_path)

    # $1 per leg at $0.50 buys 2 shares each: the basket costs its full $1 payout.
    await _run(pipeline, [_signal("tokA", 0.50), _signal("tokB", 0.50)], {"tokA": 0.50, "tokB": 0.50})

    sales = [t for t in ctx.ledger.transactions() if t["kind"] == "SELL"]
    assert sorted(t["token_id"] for t in sales) == ["tokA", "tokB"]
    assert {t["reason"] for t in sales} == {"ARBITRAGE-UNWIND"}
    assert ctx.ledger.open_positions() == []


@pytest.mark.asyncio
async def test_failed_unwind_sale_is_not_recorded_so_the_leg_stays_held(tmp_path):
    pipeline, ctx = _make_pipeline(tmp_path)

    await _run(pipeline, [_signal("tokA", 0.30), _signal("tokB", 0.35)], {"tokA": 0.25}, sell_ok=False)

    assert _kinds(ctx.ledger) == [("BUY", "tokA", 4.0)]
    assert [h.token_id for h in ctx.ledger.open_positions()] == ["tokA"]


@pytest.mark.asyncio
async def test_partial_basket_completion_adds_only_the_missing_leg(tmp_path):
    pipeline, ctx = _make_pipeline(tmp_path)
    # tokA was bought by an earlier attempt and is still held.
    ctx.ledger.record_entry(
        token_id="tokA", condition_id="cond_tokA", side="YES", shares=3.0, price=0.30,
        venue="paper", strategy="arbitrage", group_id=GROUP,
    )

    # Only tokB is bought: $1 at $0.40 fills 2.5 shares.
    await _run(pipeline, [_signal("tokA", 0.30), _signal("tokB", 0.40)], {"tokB": 0.40})

    buys = [t for t in ctx.ledger.transactions() if t["kind"] == "BUY"]
    assert sorted(t["token_id"] for t in buys) == ["tokA", "tokB"]
    held = {h.token_id: (h.shares, h.group_id) for h in ctx.ledger.open_positions()}
    assert held == {"tokA": (pytest.approx(3.0), GROUP), "tokB": (pytest.approx(2.5), GROUP)}


@pytest.mark.asyncio
async def test_basket_filled_after_switching_live_is_tagged_live(tmp_path):
    pipeline, ctx = _make_pipeline(tmp_path, live=live_fake(cash=1000.0))
    dm.write_live_trading_requested(ctx.db_path, True)

    await _run(pipeline, [_signal("tokA", 0.40), _signal("tokB", 0.40)], {"tokA": 0.40, "tokB": 0.40})

    assert {t["venue"] for t in ctx.ledger.transactions()} == {"live"}
    assert ctx.ledger.open_positions(venue="paper") == []

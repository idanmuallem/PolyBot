"""Integration: the pipeline owns the active Venue (#19).

One engine tick at a time (SequentialTradingPipeline.tick()) on FakeVenues
and a real Ledger: the dashboard's live-trading request swaps the Venue at
the next tick boundary, resolution goes through the Venue port, Ledger rows
are tagged with the venue that traded, and a reconcile step logs drift
between the Venue's holdings and the Ledger's.
"""
import pytest

import ui.data_manager as dm
from tests.tick_harness import levels, live_fake, make_tick_pipeline
from trading.fake_venue import FakeVenue
from trading.venue import MarketRef

BTC = MarketRef(yes_token_id="btc-yes", no_token_id="btc-no", condition_id="0xbtc", slug="btc-100k")


# ── mode switch ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_live_request_takes_effect_at_the_next_tick(tmp_path):
    paper, live = FakeVenue(cash=300.0), live_fake(cash=55.0)
    pipeline, ctx, rows = make_tick_pipeline(tmp_path, paper=paper, live=live)

    dm.write_live_trading_requested(ctx.db_path, True)
    assert ctx.executor.venue is paper  # nothing changes mid-tick

    await pipeline.tick()

    assert pipeline.venue is live
    assert ctx.executor.venue is live
    assert ctx.bridge.current_balance == 55.0
    assert "MODE-SWITCH" in levels(rows)


@pytest.mark.asyncio
async def test_paper_request_switches_back(tmp_path):
    paper, live = FakeVenue(cash=300.0), live_fake(cash=55.0)
    pipeline, ctx, _ = make_tick_pipeline(tmp_path, paper=paper, live=live, active="live")
    dm.write_live_trading_requested(ctx.db_path, True)
    await pipeline.tick()
    assert pipeline.venue is live

    dm.write_live_trading_requested(ctx.db_path, False)
    await pipeline.tick()

    assert pipeline.venue is paper
    assert ctx.executor.venue is paper
    assert ctx.bridge.current_balance == 300.0


@pytest.mark.asyncio
async def test_live_request_without_a_live_venue_stays_on_paper_and_says_so_once(tmp_path):
    """Live without a usable client never trades: the switch is refused."""
    paper = FakeVenue(cash=300.0)
    pipeline, ctx, rows = make_tick_pipeline(tmp_path, paper=paper)
    dm.write_live_trading_requested(ctx.db_path, True)

    await pipeline.tick()
    await pipeline.tick()

    assert pipeline.venue is paper
    assert ctx.executor.venue is paper
    assert levels(rows).count("MODE-SWITCH-REFUSED") == 1


# ── resolution through the Venue port ────────────────────────────────────────

def _hold(venue, ledger, side="YES", usd=4.0, yes_price=0.40):
    """Buy *usd* of BTC *side* on *venue* and record it in the Ledger."""
    venue.list_market(BTC, yes_price=yes_price)
    price = yes_price if side == "YES" else round(1 - yes_price, 6)
    venue.buy(BTC, side, usd)
    ledger.record_entry(token_id=BTC.token_for(side), condition_id=BTC.condition_id, side=side,
                        shares=round(usd / price, 2), price=price, venue=venue.name, strategy="crypto")


@pytest.mark.asyncio
async def test_resolution_is_recorded_under_the_venue_that_held_it(tmp_path):
    live = live_fake(cash=100.0)
    pipeline, ctx, _ = make_tick_pipeline(tmp_path, paper=FakeVenue(), live=live, active="live")
    dm.write_live_trading_requested(ctx.db_path, True)
    _hold(live, ctx.ledger)  # 10 shares at 0.40
    live.close_market(BTC, "YES")

    await pipeline.tick()

    assert ctx.ledger.open_positions() == []
    resolved = [t for t in ctx.ledger.transactions() if t["kind"] == "RESOLVED"]
    assert [(t["token_id"], t["venue"]) for t in resolved] == [("btc-yes", "live")]
    assert ctx.ledger.realized_pnl() == pytest.approx(6.0)


@pytest.mark.asyncio
async def test_paper_positions_keep_resolving_while_trading_live(tmp_path):
    paper = FakeVenue(cash=100.0)
    pipeline, ctx, _ = make_tick_pipeline(tmp_path, paper=paper, live=live_fake(), active="live")
    dm.write_live_trading_requested(ctx.db_path, True)
    _hold(paper, ctx.ledger)
    paper.close_market(BTC, "NO")

    await pipeline.tick()

    assert paper.open_positions() == []
    assert ctx.ledger.open_positions(venue="paper") == []
    assert ctx.ledger.realized_pnl() == pytest.approx(-4.0)


@pytest.mark.asyncio
async def test_resolution_runs_at_most_every_15_minutes(tmp_path):
    paper = FakeVenue(cash=100.0)
    pipeline, ctx, _ = make_tick_pipeline(tmp_path, paper=paper)
    await pipeline.tick()  # first check: nothing to resolve
    _hold(paper, ctx.ledger)
    paper.close_market(BTC, "YES")

    await pipeline.tick()

    assert ctx.ledger.holding("btc-yes") is not None


# ── reconcile: Venue holdings vs Ledger ──────────────────────────────────────

def _drift(rows):
    return [payload for level, _, payload in rows if level == "RECONCILE-DRIFT"]


@pytest.mark.asyncio
async def test_matching_venue_and_ledger_log_no_drift(tmp_path):
    paper = FakeVenue(cash=100.0)
    pipeline, ctx, rows = make_tick_pipeline(tmp_path, paper=paper)
    _hold(paper, ctx.ledger)

    await pipeline.tick()

    assert _drift(rows) == []


@pytest.mark.asyncio
async def test_drift_between_venue_and_ledger_is_logged(tmp_path):
    paper = FakeVenue(cash=100.0)
    pipeline, ctx, rows = make_tick_pipeline(tmp_path, paper=paper)
    paper.list_market(BTC, yes_price=0.40)
    paper.buy(BTC, "YES", 4.0)                      # held on the venue, unknown to the Ledger
    ctx.ledger.record_entry(token_id="eth-yes", condition_id="0xeth", side="YES", shares=5.0,
                            price=0.5, venue="paper", strategy="crypto")  # Ledger only
    ctx.ledger.record_entry(token_id="sol-yes", condition_id="0xsol", side="YES", shares=5.0,
                            price=0.5, venue="live", strategy="crypto")   # other venue: ignored

    await pipeline.tick()

    [drift] = _drift(rows)
    assert drift["venue"] == "paper"
    assert drift["venue_only"] == {"btc-yes": pytest.approx(10.0)}
    assert drift["ledger_only"] == {"eth-yes": pytest.approx(5.0)}
    assert drift["share_mismatch"] == {}


@pytest.mark.asyncio
async def test_share_mismatch_is_drift(tmp_path):
    paper = FakeVenue(cash=100.0)
    pipeline, ctx, rows = make_tick_pipeline(tmp_path, paper=paper)
    _hold(paper, ctx.ledger)        # 10 shares on both
    paper.sell("btc-yes", 4.0)      # sold on the venue, never recorded

    await pipeline.tick()

    [drift] = _drift(rows)
    assert drift["share_mismatch"] == {"btc-yes": {"venue": pytest.approx(6.0), "ledger": pytest.approx(10.0)}}


@pytest.mark.asyncio
async def test_unchanged_drift_is_logged_once_and_its_clearing_is_logged(tmp_path):
    paper = FakeVenue(cash=100.0)
    pipeline, ctx, rows = make_tick_pipeline(tmp_path, paper=paper)
    ctx.ledger.record_entry(token_id="eth-yes", condition_id="0xeth", side="YES", shares=5.0,
                            price=0.5, venue="paper", strategy="crypto")

    await pipeline.tick()
    await pipeline.tick()
    assert len(_drift(rows)) == 1

    ctx.ledger.record_resolution(token_id="eth-yes", shares=5.0, payout_price=1.0, venue="paper")
    await pipeline.tick()

    assert "RECONCILE-OK" in levels(rows)


# ── Live run: a resting entry order is cancelled, then kept for what filled ──
# A live entry is a GTC limit order: accepted is not filled. Once a grace
# period has passed the engine cancels the entry's resting order and asks
# the venue what filled: nothing -> closed as unfilled (no P&L); part ->
# kept at the filled shares; all -> kept. A cancel that fails is retried
# next tick. The order id is in the Ledger, so this survives a restart.

def _clocked_live_pipeline(tmp_path):
    from datetime import datetime, timezone
    now = {"t": datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)}
    live = live_fake(cash=100.0)
    live.rest_buys()
    pipeline, ctx, rows = make_tick_pipeline(tmp_path, live=live, active="live", clock=lambda: now["t"])
    dm.write_live_trading_requested(ctx.db_path, True)
    return pipeline, ctx, rows, live, now


def _minutes(now, n):
    from datetime import timedelta
    now["t"] = now["t"] + timedelta(minutes=n)


def _resting_entry(live, ledger, at="2026-10-21 12:00:00"):
    """A live BTC YES entry of 10 shares at 0.40, resting on the book."""
    live.list_market(BTC, yes_price=0.40)
    order = live.buy(BTC, "YES", 4.0, price=0.40)
    ledger.record_entry(token_id="btc-yes", condition_id="0xbtc", side="YES", shares=10.0, price=0.40,
                        venue="live", strategy="crypto", order_id=order.id, at=at)
    return order


def _closes(ledger):
    return [t for t in ledger.transactions() if t["kind"] != "BUY"]


@pytest.mark.asyncio
async def test_a_live_entry_that_never_fills_is_cancelled_then_closed_as_unfilled(tmp_path):
    pipeline, ctx, rows, live, now = _clocked_live_pipeline(tmp_path)
    order = _resting_entry(live, ctx.ledger)

    _minutes(now, 1)
    await pipeline.tick()
    assert ctx.ledger.holding("btc-yes") is not None   # still inside the grace period
    assert live.cancelled == []

    _minutes(now, 10)
    await pipeline.tick()

    assert live.cancelled == [order.id]
    assert ctx.ledger.holding("btc-yes") is None
    [closed] = _closes(ctx.ledger)
    assert closed["kind"] == "UNFILLED"
    assert closed["confirmed"] is False
    assert closed["pnl_per_share"] is None
    assert ctx.ledger.realized_pnl() == 0.0
    assert "RECONCILE-UNFILLED" in levels(rows)


@pytest.mark.asyncio
async def test_a_partly_filled_live_entry_is_kept_with_the_filled_shares(tmp_path):
    pipeline, ctx, rows, live, now = _clocked_live_pipeline(tmp_path)
    order = _resting_entry(live, ctx.ledger)
    live.fill_resting(order, 4.0)

    _minutes(now, 10)
    await pipeline.tick()

    assert live.cancelled == [order.id]                 # the rest can't fill later
    assert ctx.ledger.holding("btc-yes").shares == pytest.approx(4.0)
    assert _closes(ctx.ledger) == []

    live.fill_resting(order, 6.0)                       # too late: it was cancelled
    _minutes(now, 10)
    await pipeline.tick()
    assert ctx.ledger.holding("btc-yes").shares == pytest.approx(4.0)
    assert [p.shares for p in live.open_positions()] == [pytest.approx(4.0)]


@pytest.mark.asyncio
async def test_a_fully_filled_live_entry_is_kept(tmp_path):
    pipeline, ctx, rows, live, now = _clocked_live_pipeline(tmp_path)
    order = _resting_entry(live, ctx.ledger)
    live.fill_resting(order, 10.0)

    _minutes(now, 10)
    await pipeline.tick()

    holding = ctx.ledger.holding("btc-yes")
    assert holding.shares == pytest.approx(10.0)
    assert holding.order_id is None
    assert _closes(ctx.ledger) == []


@pytest.mark.asyncio
async def test_a_failed_cancel_keeps_the_entry_and_is_retried_next_tick(tmp_path):
    pipeline, ctx, rows, live, now = _clocked_live_pipeline(tmp_path)
    order = _resting_entry(live, ctx.ledger)
    live.fail_cancels()

    _minutes(now, 10)
    await pipeline.tick()

    assert ctx.ledger.holding("btc-yes").order_id == order.id
    assert _closes(ctx.ledger) == []

    live.restore_cancels()
    _minutes(now, 1)
    await pipeline.tick()

    assert live.cancelled == [order.id]
    assert ctx.ledger.holding("btc-yes") is None
    assert [c["kind"] for c in _closes(ctx.ledger)] == ["UNFILLED"]


@pytest.mark.asyncio
async def test_a_dry_run_holding_of_the_same_token_is_never_closed_as_unfilled(tmp_path):
    pipeline, ctx, rows, live, now = _clocked_live_pipeline(tmp_path)
    _resting_entry(live, ctx.ledger)
    ctx.ledger.record_entry(token_id="btc-yes", condition_id="0xbtc", side="YES", shares=7.0, price=0.40,
                            venue="paper", strategy="crypto", at="2026-10-21 12:01:00")

    _minutes(now, 10)
    await pipeline.tick()

    [paper] = ctx.ledger.open_positions()
    assert paper.venue == "paper"
    assert paper.shares == pytest.approx(7.0)
    assert [c["venue"] for c in _closes(ctx.ledger)] == ["live"]


@pytest.mark.asyncio
async def test_a_live_entry_opened_before_a_restart_is_still_cancelled_and_closed(tmp_path):
    pipeline, ctx, rows, live, now = _clocked_live_pipeline(tmp_path)
    order = _resting_entry(live, ctx.ledger, at="2026-10-20 08:00:00")   # yesterday, before this start

    await pipeline.tick()

    assert live.cancelled == [order.id]
    assert ctx.ledger.holding("btc-yes") is None


@pytest.mark.asyncio
async def test_a_live_entry_with_no_order_to_follow_is_never_closed_as_unfilled(tmp_path):
    """Entries recorded before order ids were kept: nothing to cancel or
    check, so nothing is assumed about them."""
    pipeline, ctx, rows, live, now = _clocked_live_pipeline(tmp_path)
    ctx.ledger.record_entry(token_id="eth-yes", condition_id="0xeth", side="YES", shares=5.0, price=0.5,
                            venue="live", strategy="crypto", at="2026-10-21 12:00:00")

    _minutes(now, 30)
    await pipeline.tick()

    assert ctx.ledger.holding("eth-yes") is not None
    assert "RECONCILE-UNFILLED" not in levels(rows)


@pytest.mark.asyncio
async def test_a_live_model_entry_records_its_resting_order_id(tmp_path):
    from unittest.mock import patch
    from tests.integration.test_account_budget_tick import _btc_market, _config
    from tests.tick_harness import FixedMarketScanner

    market = _btc_market()
    live = live_fake(cash=1000.0)
    live.rest_buys()
    live.list_market(MarketRef.from_market(market), market.initial_price)
    pipeline, ctx, rows = make_tick_pipeline(
        tmp_path, live=live, active="live", config=_config(trading_mode="live_run"),
        scanner=FixedMarketScanner(market, live_truth=97_000.0),
    )
    dm.write_live_trading_requested(ctx.db_path, True)

    with patch("brains.crypto.HybridCryptoBrain._calculate_probability", return_value=0.70):
        await pipeline.tick()

    [entry] = ctx.ledger.open_positions(venue="live")
    assert entry.order_id is not None
    assert live.resting_order_ids() == [entry.order_id]


# ── positions unavailable (a Data API outage) ────────────────────────────────

@pytest.mark.asyncio
async def test_an_outage_longer_than_the_grace_period_closes_nothing(tmp_path):
    pipeline, ctx, rows, live, now = _clocked_live_pipeline(tmp_path)
    _resting_entry(live, ctx.ledger)
    live.fail_position_reads()

    for _ in range(3):
        _minutes(now, 10)
        await pipeline.tick()

    assert ctx.ledger.holding("btc-yes") is not None
    assert live.cancelled == []
    assert "RECONCILE-UNFILLED" not in levels(rows)
    assert _drift(rows) == []                          # an outage is not absence
    assert levels(rows).count("RECONCILE-UNAVAILABLE") == 1   # logged once, not every tick


@pytest.mark.asyncio
async def test_reconcile_resumes_when_positions_are_readable_again(tmp_path):
    pipeline, ctx, rows, live, now = _clocked_live_pipeline(tmp_path)
    ctx.ledger.record_entry(token_id="eth-yes", condition_id="0xeth", side="YES", shares=5.0, price=0.5,
                            venue="live", strategy="crypto", at="2026-10-21 12:00:00")
    live.fail_position_reads()
    await pipeline.tick()
    live.restore_position_reads()

    await pipeline.tick()

    [drift] = _drift(rows)
    assert drift["ledger_only"] == {"eth-yes": pytest.approx(5.0)}

    live.fail_position_reads()
    await pipeline.tick()
    assert levels(rows).count("RECONCILE-UNAVAILABLE") == 2   # a new outage is logged again


@pytest.mark.asyncio
async def test_engine_status_names_the_active_venue(tmp_path):
    paper, live = FakeVenue(cash=300.0), live_fake(cash=55.0)
    pipeline, ctx, _ = make_tick_pipeline(tmp_path, paper=paper, live=live)

    await pipeline.tick()
    assert dm.trading_venue(ctx.db_path) == "paper"

    dm.write_live_trading_requested(ctx.db_path, True)
    await pipeline.tick()
    assert dm.trading_venue(ctx.db_path) == "live"

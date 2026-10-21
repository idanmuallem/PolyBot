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


# ── Live run: entries the venue never shows are closed as unfilled ───────────
# A live entry is a resting limit order: posting it is not a fill. A model
# entry the CLOB never shows as a position is closed in the Ledger once a
# grace period has passed, as unfilled (no P&L), so it isn't a phantom
# holding.

def _clocked_live_pipeline(tmp_path):
    from datetime import datetime, timezone
    now = {"t": datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)}
    live = live_fake(cash=100.0)
    pipeline, ctx, rows = make_tick_pipeline(tmp_path, live=live, active="live", clock=lambda: now["t"])
    dm.write_live_trading_requested(ctx.db_path, True)
    return pipeline, ctx, rows, live, now


def _minutes(now, n):
    from datetime import timedelta
    now["t"] = now["t"] + timedelta(minutes=n)


@pytest.mark.asyncio
async def test_a_live_entry_the_venue_never_shows_is_closed_as_unfilled(tmp_path):
    pipeline, ctx, rows, live, now = _clocked_live_pipeline(tmp_path)
    ctx.ledger.record_entry(token_id="eth-yes", condition_id="0xeth", side="YES", shares=5.0, price=0.5,
                            venue="live", strategy="crypto", at="2026-10-21 12:00:00")

    _minutes(now, 1)
    await pipeline.tick()
    assert ctx.ledger.holding("eth-yes") is not None   # still inside the grace period

    _minutes(now, 10)
    await pipeline.tick()

    assert ctx.ledger.holding("eth-yes") is None
    [closed] = [t for t in ctx.ledger.transactions() if t["kind"] != "BUY"]
    assert closed["kind"] == "UNFILLED"
    assert closed["confirmed"] is False
    assert closed["pnl_per_share"] is None
    assert ctx.ledger.realized_pnl() == 0.0
    assert "RECONCILE-UNFILLED" in levels(rows)


@pytest.mark.asyncio
async def test_a_live_entry_the_venue_has_shown_is_not_closed_as_unfilled(tmp_path):
    pipeline, ctx, rows, live, now = _clocked_live_pipeline(tmp_path)
    live.list_market(BTC, yes_price=0.40)
    live.buy(BTC, "YES", 4.0)
    ctx.ledger.record_entry(token_id="btc-yes", condition_id="0xbtc", side="YES", shares=10.0, price=0.4,
                            venue="live", strategy="crypto", at="2026-10-21 12:00:00")
    await pipeline.tick()                       # the venue shows it: it filled

    live.sell("btc-yes", 10.0)                  # gone from the venue later (sold elsewhere)
    _minutes(now, 30)
    await pipeline.tick()

    assert ctx.ledger.holding("btc-yes") is not None
    assert "RECONCILE-UNFILLED" not in levels(rows)


@pytest.mark.asyncio
async def test_holdings_from_before_this_run_are_not_closed_as_unfilled(tmp_path):
    pipeline, ctx, rows, live, now = _clocked_live_pipeline(tmp_path)
    ctx.ledger.record_entry(token_id="eth-yes", condition_id="0xeth", side="YES", shares=5.0, price=0.5,
                            venue="live", strategy="crypto", at="2026-10-20 08:00:00")

    _minutes(now, 30)
    await pipeline.tick()

    assert ctx.ledger.holding("eth-yes") is not None


@pytest.mark.asyncio
async def test_engine_status_names_the_active_venue(tmp_path):
    paper, live = FakeVenue(cash=300.0), live_fake(cash=55.0)
    pipeline, ctx, _ = make_tick_pipeline(tmp_path, paper=paper, live=live)

    await pipeline.tick()
    assert dm.trading_venue(ctx.db_path) == "paper"

    dm.write_live_trading_requested(ctx.db_path, True)
    await pipeline.tick()
    assert dm.trading_venue(ctx.db_path) == "live"

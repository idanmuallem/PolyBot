"""Integration: the pipeline's periodic paper-resolution check must call the
Venue on the loop's own thread (regression for 28c7637).

PaperVenue's pm_trader.Engine opens its sqlite3 connection with the library
default check_same_thread=True (no override exposed), in whichever thread
first uses it -- in production, the engine loop's thread. Routing
resolve_closed_markets() through asyncio.to_thread() hands it to a worker
thread, which sqlite3 refuses ("SQLite objects created in a thread can only
be used in that same thread"), so nothing ever resolved.

These tests drive one engine tick against a real PaperVenue (pm_trader + sqlite
in tmp_path, stubbed market data) and assert on the outcome: the closed
market's position is settled and reported.
"""
import pytest

from tests.tick_harness import levels, make_tick_pipeline
from trading.venue import MarketRef

BTC = MarketRef(yes_token_id="btc-yes", no_token_id="btc-no", condition_id="0xbtc", slug="btc-100k")


@pytest.mark.asyncio
async def test_paper_resolution_settles_closed_market_on_the_loop_thread(tmp_path):
    pytest.importorskip("pm_trader")
    from tests.venue_harness import make_paper_venue

    # Built (and its sqlite connection opened) on this, the loop's, thread.
    venue, exchange = make_paper_venue(tmp_path / "paper_trading", 1000.0)
    exchange.list_market(BTC, yes_price=0.40)
    assert venue.buy(BTC, "YES", 20.0) is True
    exchange.close_market(BTC, "YES")
    pipeline, _, rows = make_tick_pipeline(tmp_path, paper=venue)

    await pipeline.tick()

    expired = [(token, payload) for level, token, payload in rows if level == "EXPIRED"]
    assert expired and expired[0][0] == "btc-yes", (
        "the closed market was never settled -- a cross-thread call into "
        "the paper venue's sqlite connection fails exactly like this"
    )
    assert venue.open_positions() == []
    assert venue.cash() == pytest.approx(1000.0 - 20.0 + 50.0)
    assert "RESOLVE-ERROR" not in levels(rows)


@pytest.mark.asyncio
async def test_paper_resolution_failure_is_caught_and_logged(tmp_path):
    from trading.fake_venue import FakeVenue

    class _ResolveExplodes(FakeVenue):
        def resolve_closed_markets(self, **kw):
            raise RuntimeError("boom")

    pipeline, _, rows = make_tick_pipeline(tmp_path, paper=_ResolveExplodes(cash=100.0))

    await pipeline.tick()

    [(token, payload)] = [(t, p) for level, t, p in rows if level == "RESOLVE-ERROR"]
    assert token == "paper"
    assert payload["venue"] == "paper" and "boom" in payload["error"]

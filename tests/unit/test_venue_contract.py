"""Shared Venue contract: FakeVenue and PaperVenue (real pm_trader in a temp
dir, stubbed market data) must behave the same through the Venue port.

If FakeVenue passes something PaperVenue fails, the fake is lying -- fix
the fake, not the test.
"""

import pytest

from trading.venue import MarketRef

START_CASH = 1000.0

BTC = MarketRef(yes_token_id="btc-yes", no_token_id="btc-no", condition_id="0xbtc", slug="btc-100k")
ETH = MarketRef(yes_token_id="eth-yes", no_token_id="eth-no", condition_id="0xeth", slug="eth-5k")


@pytest.fixture(params=["fake", "paper"])
def venue_and_exchange(request, tmp_path):
    if request.param == "fake":
        from trading.fake_venue import FakeVenue

        venue = FakeVenue(cash=START_CASH)
        return venue, venue
    pytest.importorskip("pm_trader")
    from tests.venue_harness import make_paper_venue

    return make_paper_venue(tmp_path / "paper_trading", START_CASH)


def _position(venue, token_id):
    matches = [p for p in venue.open_positions() if p.token_id == token_id]
    return matches[0] if matches else None


def test_starts_with_configured_cash_and_no_positions(venue_and_exchange):
    venue, _ = venue_and_exchange
    assert venue.cash() == pytest.approx(START_CASH)
    assert venue.open_positions() == []


def test_buy_yes_spends_cash_and_opens_yes_position(venue_and_exchange):
    venue, exchange = venue_and_exchange
    exchange.list_market(BTC, yes_price=0.40)
    assert venue.open_positions() == []  # read before the buy, too

    assert venue.buy(BTC, "YES", 20.0) is True

    assert venue.cash() == pytest.approx(980.0)
    position = _position(venue, "btc-yes")
    assert position is not None
    assert position.side == "YES"
    assert position.shares == pytest.approx(50.0)
    assert position.initial_price == pytest.approx(0.40)


def test_buy_no_trades_the_no_token_at_the_no_price(venue_and_exchange):
    venue, exchange = venue_and_exchange
    exchange.list_market(BTC, yes_price=0.40)

    assert venue.buy(BTC, "NO", 30.0) is True

    assert _position(venue, "btc-yes") is None
    position = _position(venue, "btc-no")
    assert position is not None
    assert position.side == "NO"
    assert position.shares == pytest.approx(50.0)  # $30 at $0.60
    assert position.initial_price == pytest.approx(0.60)


def test_buy_beyond_cash_is_refused_and_spends_nothing(venue_and_exchange):
    venue, exchange = venue_and_exchange
    exchange.list_market(BTC, yes_price=0.40)

    assert venue.buy(BTC, "YES", START_CASH + 1.0) is False

    assert venue.cash() == pytest.approx(START_CASH)
    assert venue.open_positions() == []


def test_buy_in_unknown_market_is_refused(venue_and_exchange):
    venue, _ = venue_and_exchange
    assert venue.buy(ETH, "YES", 10.0) is False
    assert venue.cash() == pytest.approx(START_CASH)


def test_sell_closes_position_immediately_and_credits_cash(venue_and_exchange):
    """Phantom-sell regression (6a06f35): after a sale lands, the very next
    open_positions() read must not still show the position, or exit logic
    sells it again on the next tick."""
    venue, exchange = venue_and_exchange
    exchange.list_market(BTC, yes_price=0.40)
    venue.buy(BTC, "YES", 20.0)
    assert _position(venue, "btc-yes") is not None  # prime any position cache
    exchange.set_price(BTC, "YES", 0.50)

    assert venue.sell("btc-yes", 50.0) is True

    assert _position(venue, "btc-yes") is None
    assert venue.cash() == pytest.approx(980.0 + 25.0)


def test_sell_of_unheld_token_fails_without_crediting_cash(venue_and_exchange):
    """Phantom-sell regression (6a06f35): a sale that never lands must say so."""
    venue, exchange = venue_and_exchange
    exchange.list_market(BTC, yes_price=0.40)

    assert venue.sell("btc-yes", 10.0) is False
    assert venue.sell("never-seen-token", 10.0) is False
    assert venue.cash() == pytest.approx(START_CASH)


def test_repeated_sell_of_closed_position_fails(venue_and_exchange):
    """Phantom-sell regression (6a06f35): selling the same position twice
    must not report two sales."""
    venue, exchange = venue_and_exchange
    exchange.list_market(BTC, yes_price=0.40)
    venue.buy(BTC, "YES", 20.0)

    assert venue.sell("btc-yes", 50.0) is True
    cash_after_first = venue.cash()
    assert venue.sell("btc-yes", 50.0) is False
    assert venue.cash() == pytest.approx(cash_after_first)


def test_partial_sell_keeps_the_rest(venue_and_exchange):
    venue, exchange = venue_and_exchange
    exchange.list_market(BTC, yes_price=0.40)
    venue.buy(BTC, "NO", 30.0)

    assert venue.sell("btc-no", 20.0) is True

    assert _position(venue, "btc-no").shares == pytest.approx(30.0)


def test_limit_buy_at_or_above_market_fills_on_poll(venue_and_exchange):
    venue, exchange = venue_and_exchange
    exchange.list_market(BTC, yes_price=0.40)

    order = venue.place_limit_buy(BTC, "YES", 20.0, limit_price=0.40)
    assert order is not None
    assert order.token_id == "btc-yes"

    fills = venue.poll_limit_orders([order])

    assert fills[order.id] == pytest.approx(50.0)
    assert _position(venue, "btc-yes").shares == pytest.approx(50.0)
    assert venue.cash() == pytest.approx(980.0)


def test_limit_buy_below_market_rests_and_can_be_cancelled(venue_and_exchange):
    venue, exchange = venue_and_exchange
    exchange.list_market(BTC, yes_price=0.40)

    order = venue.place_limit_buy(BTC, "NO", 20.0, limit_price=0.50)  # NO trades at 0.60
    assert order.token_id == "btc-no"

    assert venue.poll_limit_orders([order])[order.id] == 0.0
    assert venue.cancel_limit_order(order) is True
    assert venue.cancel_limit_order(order) is False  # no longer pending

    exchange.set_price(BTC, "NO", 0.45)
    assert venue.poll_limit_orders([order])[order.id] == 0.0  # cancelled orders never fill
    assert venue.cash() == pytest.approx(START_CASH)


def test_limit_buy_fills_once_the_price_comes_in(venue_and_exchange):
    venue, exchange = venue_and_exchange
    exchange.list_market(BTC, yes_price=0.40)
    order = venue.place_limit_buy(BTC, "YES", 10.0, limit_price=0.25)
    assert venue.poll_limit_orders([order])[order.id] == 0.0

    exchange.set_price(BTC, "YES", 0.25)

    assert venue.poll_limit_orders([order])[order.id] == pytest.approx(40.0)


def test_resolution_pays_winners_and_closes_positions(venue_and_exchange):
    venue, exchange = venue_and_exchange
    exchange.list_market(BTC, yes_price=0.40)
    exchange.list_market(ETH, yes_price=0.50)
    venue.buy(BTC, "YES", 20.0)  # 50 shares, wins
    venue.buy(ETH, "YES", 10.0)  # 20 shares, loses
    assert len(venue.open_positions()) == 2
    exchange.close_market(BTC, "YES")
    exchange.close_market(ETH, "NO")
    rows = []

    resolved = venue.resolve_closed_markets(log_func=lambda *row: rows.append(row))

    assert resolved == 2
    assert venue.open_positions() == []
    assert venue.cash() == pytest.approx(START_CASH - 30.0 + 50.0)
    expired = {token: payload for level, _asset, token, payload in rows if level == "EXPIRED"}
    assert expired["btc-yes"]["price"] == pytest.approx(1.0)
    assert expired["btc-yes"]["shares"] == pytest.approx(50.0)
    assert expired["btc-yes"]["initial_price"] == pytest.approx(0.40)
    assert expired["btc-yes"]["side"] == "YES"
    assert expired["btc-yes"]["sold"] is True
    assert expired["eth-yes"]["price"] == pytest.approx(0.0)


def test_resolution_leaves_open_markets_alone(venue_and_exchange):
    venue, exchange = venue_and_exchange
    exchange.list_market(BTC, yes_price=0.40)
    venue.buy(BTC, "YES", 20.0)

    assert venue.resolve_closed_markets() == 0
    assert _position(venue, "btc-yes") is not None


def test_resolution_can_skip_arbitrage_positions(venue_and_exchange):
    """ENABLE_ARBITRAGE kill switch: resolve_arbitrage=False must leave
    positions opened through limit orders (arbitrage legs) untouched."""
    venue, exchange = venue_and_exchange
    exchange.list_market(BTC, yes_price=0.40)
    exchange.list_market(ETH, yes_price=0.50)
    venue.buy(BTC, "YES", 20.0)
    order = venue.place_limit_buy(ETH, "YES", 10.0, limit_price=0.50)
    venue.poll_limit_orders([order])
    exchange.close_market(BTC, "YES")
    exchange.close_market(ETH, "YES")

    assert venue.resolve_closed_markets(resolve_arbitrage=False) == 1
    assert _position(venue, "eth-yes") is not None

    assert venue.resolve_closed_markets(resolve_arbitrage=True) == 1
    assert venue.open_positions() == []

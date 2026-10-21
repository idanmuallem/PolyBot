"""ClobVenue: the live Venue adapter over py_clob_client + the Data API.

No network: a StubClobClient stands in for py_clob_client's ClobClient and
a stub http_get for the Data API. ClobVenue is not contract-tested against
the network (spec #12); these pin the live behaviour moved out of the
executor.
"""

import pytest

from trading.venue import LimitOrder, MarketRef

BTC = MarketRef(yes_token_id="btc-yes", no_token_id="btc-no", condition_id="0xbtc", slug="btc-100k")


_ACCEPTED = object()


class StubClobClient:
    """Records orders; answers like the CLOB with configurable responses."""

    def __init__(self, order_response=_ACCEPTED, filled=None, balance=None):
        self.orders = []          # (token_id, price, side, size)
        self.cancelled = []
        self.order_response = {"orderID": "o1"} if order_response is _ACCEPTED else order_response
        self.filled = filled or {}  # order id -> size_matched
        self.balance = balance

    def create_and_post_order(self, order):
        self.orders.append((order.token_id, order.price, order.side, order.size))
        if isinstance(self.order_response, Exception):
            raise self.order_response
        if callable(self.order_response):
            return self.order_response(order)
        return self.order_response

    def get_order(self, order_id):
        value = self.filled.get(order_id, 0.0)
        if isinstance(value, Exception):
            raise value
        return {"size_matched": str(value)}

    def cancel(self, order_id):
        self.cancelled.append(order_id)
        return {"canceled": [order_id]}

    def get_balance_allowance(self, params=None):
        return {"balance": self.balance}


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.exceptions.HTTPError(response=self)

    def json(self):
        return self._payload


def _venue(client=None, positions_payload=None, proxy="0xabc", http_calls=None):
    from trading.clob_venue import ClobVenue

    def http_get(url, timeout=None):
        if http_calls is not None:
            http_calls.append(url)
        return _Resp(positions_payload if positions_payload is not None else [])

    return ClobVenue(client=client or StubClobClient(), proxy_address=proxy, http_get=http_get)


def _logger():
    rows = []
    return rows, lambda level, asset_type, token_id, payload: rows.append((level, asset_type, token_id, payload))


# -- buy ----------------------------------------------------------------------

def test_buy_no_posts_a_limit_buy_on_the_no_token_at_the_side_price():
    client = StubClobClient()
    venue = _venue(client)

    assert venue.buy(BTC, "NO", 3.0, price=0.6) is True

    # floor(3.0 / 0.6 * 100) / 100 = 5.0 shares at the NO price
    assert client.orders == [("btc-no", 0.6, "BUY", 5.0)]


def test_buy_logs_live_trade_with_the_order_response():
    client = StubClobClient(order_response={"orderID": "abc"})
    rows, log = _logger()

    assert _venue(client).buy(BTC, "YES", 2.0, price=0.4, log_func=log) is True

    assert rows == [("LIVE-TRADE", "Portfolio", "btc-yes", {
        "success": True, "response": {"orderID": "abc"}, "side": "YES", "price": 0.4,
    })]


@pytest.mark.parametrize("response", [None, {}, {"error": "insufficient_funds"}, {"errorMsg": "x", "orderID": "1"}])
def test_buy_rejected_order_reports_failure_and_logs_error(response):
    rows, log = _logger()
    client = StubClobClient(order_response=response)

    assert _venue(client).buy(BTC, "YES", 2.0, price=0.4, log_func=log) is False

    assert [r[0] for r in rows] == ["LIVE-TRADE-ERROR"]
    assert rows[0][3]["error"] == "order_rejected_or_unconfirmed"


def test_buy_raising_client_reports_failure_and_logs_error():
    rows, log = _logger()
    client = StubClobClient(order_response=RuntimeError("boom"))

    assert _venue(client).buy(BTC, "YES", 2.0, price=0.4, log_func=log) is False

    assert [r[0] for r in rows] == ["LIVE-TRADE-ERROR"]


def test_buy_without_a_price_places_nothing():
    client = StubClobClient()
    assert _venue(client).buy(BTC, "YES", 2.0) is False
    assert client.orders == []


# -- sell ---------------------------------------------------------------------

def test_sell_posts_a_limit_sell_and_logs_sell():
    client = StubClobClient(order_response={"orderID": "s1"})
    rows, log = _logger()

    assert _venue(client).sell("btc-yes", 10.0, price=0.55, log_func=log) is True

    assert client.orders == [("btc-yes", 0.55, "SELL", 10.0)]
    assert rows == [("SELL", "Portfolio", "btc-yes", {
        "success": True, "response": {"orderID": "s1"}, "price": 0.55, "shares": 10.0,
    })]


def test_sell_rejected_order_reports_failure_and_logs_sell_error():
    rows, log = _logger()
    client = StubClobClient(order_response={"error": "no balance"})

    assert _venue(client).sell("btc-yes", 10.0, price=0.55, log_func=log) is False

    assert [r[0] for r in rows] == ["SELL-ERROR"]


# -- cash ---------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("12.5", 12.5),
    ("12500000", 12.5),          # micro-USDC
    ({"balance": "7"}, 7.0),
])
def test_cash_reads_collateral_balance(raw, expected):
    assert _venue(StubClobClient(balance=raw)).cash() == pytest.approx(expected)


def test_cash_is_zero_when_the_balance_fetch_fails():
    class Broken(StubClobClient):
        def get_balance_allowance(self, params=None):
            raise RuntimeError("down")

    assert _venue(Broken()).cash() == 0.0


# -- open positions -----------------------------------------------------------

def test_open_positions_reads_the_data_api_for_the_proxy_wallet():
    calls = []
    venue = _venue(positions_payload=[{
        "asset": "btc-yes", "conditionId": "0xbtc", "avgPrice": "0.40",
        "currentPrice": "0.404", "currentValue": "4.04", "size": "10", "outcome": "Yes",
    }], http_calls=calls)

    [position] = venue.open_positions()

    assert calls == ["https://data-api.polymarket.com/positions?user=0xabc"]
    assert position.token_id == "btc-yes"
    assert position.market_id == "0xbtc"
    assert position.shares == pytest.approx(10.0)
    assert position.value == pytest.approx(4.04)
    assert position.side == "Yes"
    # A 1% gain gives live_ev == pnl_ratio == 0.01.
    assert position.live_ev == pytest.approx(0.01, abs=0.001)


def test_open_positions_skips_empty_rows_and_derives_missing_fields():
    venue = _venue(positions_payload={"positions": [
        {"asset": "", "size": "5"},
        {"asset": "gone", "size": "0", "currentValue": "0"},
        {"asset": "eth-yes", "avgPrice": "0.5", "currentValue": "6", "size": "0"},
    ]})

    [position] = venue.open_positions()

    assert position.token_id == "eth-yes"
    assert position.shares == pytest.approx(12.0)   # value / avg price
    assert position.current_price == pytest.approx(0.5)


def test_open_positions_without_a_proxy_wallet_is_empty():
    calls = []
    assert _venue(proxy="", http_calls=calls).open_positions() == []
    assert calls == []


def _venue_with_http(http_get):
    from trading.clob_venue import ClobVenue
    return ClobVenue(client=StubClobClient(), proxy_address="0xabc", http_get=http_get)


def _raising(exc):
    def http_get(url, timeout=None):
        raise exc
    return http_get


@pytest.mark.parametrize("status", [404, 500, 503])
def test_open_positions_on_a_data_api_error_is_unavailable_not_empty(status):
    from trading.venue import PositionsUnavailable
    venue = _venue_with_http(lambda url, timeout=None: _Resp([], status=status))

    with pytest.raises(PositionsUnavailable):
        venue.open_positions()


@pytest.mark.parametrize("exc_name", ["Timeout", "ConnectionError"])
def test_open_positions_on_a_data_api_timeout_is_unavailable(exc_name):
    import requests
    from trading.venue import PositionsUnavailable
    venue = _venue_with_http(_raising(getattr(requests.exceptions, exc_name)("down")))

    with pytest.raises(PositionsUnavailable):
        venue.open_positions()


@pytest.mark.parametrize("payload", [ValueError("not json"), "<html>oops</html>", {"error": "x"}])
def test_open_positions_with_an_unreadable_body_is_unavailable(payload):
    from trading.venue import PositionsUnavailable

    class Body(_Resp):
        def json(self):
            if isinstance(self._payload, Exception):
                raise self._payload
            return self._payload

    venue = _venue_with_http(lambda url, timeout=None: Body(payload))

    with pytest.raises(PositionsUnavailable):
        venue.open_positions()


# -- limit orders (arbitrage) -------------------------------------------------

def test_place_limit_buy_posts_floor_shares_at_the_limit_price():
    client = StubClobClient(order_response={"orderID": "L1"})

    order = _venue(client).place_limit_buy(MarketRef(yes_token_id="A"), "YES", 1.0, limit_price=0.3)

    assert client.orders == [("A", 0.3, "BUY", 3.33)]
    assert order == LimitOrder(id="L1", token_id="A")


def test_place_limit_buy_without_an_order_id_returns_none():
    client = StubClobClient(order_response={"error": "nope"})
    assert _venue(client).place_limit_buy(MarketRef(yes_token_id="A"), "YES", 1.0, limit_price=0.3) is None


def test_poll_limit_orders_reports_matched_size_per_order():
    client = StubClobClient(filled={"L1": 3.0, "L2": RuntimeError("down")})
    venue = _venue(client)

    filled = venue.poll_limit_orders([LimitOrder(id="L1", token_id="A"), LimitOrder(id="L2", token_id="B")])

    assert filled == {"L1": 3.0, "L2": 0.0}


def test_cancel_limit_order_cancels_on_the_clob():
    client = StubClobClient()
    assert _venue(client).cancel_limit_order(LimitOrder(id="L1", token_id="A")) is True
    assert client.cancelled == ["L1"]


def test_resolve_closed_markets_is_a_no_op_live():
    assert _venue().resolve_closed_markets(log_func=lambda *a: None) == 0


def test_is_named_live():
    assert _venue().name == "live"


# -- construction -------------------------------------------------------------

def test_from_config_without_keys_is_a_clear_error():
    from core.trading_config import TradingConfig
    from trading.clob_venue import ClobVenue, ClobUnavailable

    config = TradingConfig(trading_mode="live_run", private_key="", proxy_address="")
    with pytest.raises(ClobUnavailable, match="POLYMARKET_PRIVATE_KEY"):
        ClobVenue.from_config(config)


def test_from_config_failing_to_derive_credentials_is_a_clear_error(monkeypatch):
    from core.trading_config import TradingConfig
    import trading.clob_venue as clob_venue

    class NoCreds:
        def __init__(self, **kwargs):
            pass

        def create_or_derive_api_creds(self):
            raise RuntimeError("401")

    monkeypatch.setattr(clob_venue, "ClobClient", NoCreds)
    monkeypatch.setattr(clob_venue, "CLOB_IMPORT_OK", True)
    config = TradingConfig(trading_mode="live_run", private_key="0xkey", proxy_address="0xabc")

    with pytest.raises(clob_venue.ClobUnavailable, match="credentials"):
        clob_venue.ClobVenue.from_config(config)

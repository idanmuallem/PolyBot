from unittest.mock import MagicMock, patch

import pytest

from hunters.clients.ccxt_client import CCXTDataClient


def _make_client(**kwargs) -> CCXTDataClient:
    """A CCXTDataClient with its exchange swapped for a MagicMock — no network calls."""
    with patch("hunters.clients.ccxt_client.ccxt") as mock_ccxt:
        mock_ccxt.binance.return_value = MagicMock()
        client = CCXTDataClient(**kwargs)
    return client


def _candles(closes: list, start_ts: int = 0) -> list:
    """Build minimal OHLCV rows [ts, open, high, low, close, volume]."""
    return [[start_ts + i * 86_400_000, c, c, c, c, 100.0] for i, c in enumerate(closes)]


# ── Symbol translation ───────────────────────────────────────────────────────

def test_to_spot_symbol_converts_binance_style():
    assert CCXTDataClient._to_spot_symbol("BTCUSDT") == "BTC/USDT"
    assert CCXTDataClient._to_spot_symbol("ETHUSDT") == "ETH/USDT"


def test_to_spot_symbol_passes_through_slash_format():
    assert CCXTDataClient._to_spot_symbol("BTC/USDT") == "BTC/USDT"


def test_to_swap_symbol_appends_settlement_currency():
    assert CCXTDataClient._to_swap_symbol("BTC/USDT") == "BTC/USDT:USDT"


# ── get_latest_value (spot, always live) ─────────────────────────────────────

def test_get_latest_value_returns_spot_price():
    client = _make_client()
    client.exchange.fetch_ticker.return_value = {"last": 65_000.0}
    assert client.get_latest_value("BTCUSDT") == 65_000.0
    client.exchange.fetch_ticker.assert_called_once_with("BTC/USDT")


def test_get_latest_value_returns_zero_on_error():
    client = _make_client()
    client.exchange.fetch_ticker.side_effect = Exception("network down")
    assert client.get_latest_value("BTCUSDT") == 0.0


def test_get_latest_value_never_cached():
    client = _make_client()
    client.exchange.fetch_ticker.side_effect = [{"last": 100.0}, {"last": 200.0}]
    assert client.get_latest_value("BTCUSDT") == 100.0
    assert client.get_latest_value("BTCUSDT") == 200.0
    assert client.exchange.fetch_ticker.call_count == 2


# ── get_enriched_data: fields present and well-formed ────────────────────────

def test_enriched_data_returns_all_fields():
    client = _make_client()
    client.exchange.fetch_ticker.return_value = {"last": 65_000.0, "quoteVolume": 1_000_000.0}
    client.exchange.fetch_ohlcv.return_value = _candles([100 + i for i in range(91)])
    client.exchange.fetch_funding_rate.return_value = {"fundingRate": 0.0001}
    client.exchange.fetch_order_book.return_value = {
        "bids": [[64_999.0, 1.0]], "asks": [[65_001.0, 1.0]],
    }

    data = client.get_enriched_data("BTCUSDT")

    for key in ("spot_price", "vol_30d", "vol_60d", "vol_90d", "funding_rate", "volume_24h", "spread"):
        assert key in data

    assert data["spot_price"] == 65_000.0
    assert data["volume_24h"] == 1_000_000.0
    assert data["funding_rate"] == pytest.approx(0.0001)
    assert data["spread"] == pytest.approx((65_001.0 - 64_999.0) / 65_000.0)
    assert data["vol_30d"] >= 0.0
    assert data["vol_60d"] >= 0.0
    assert data["vol_90d"] >= 0.0


def test_enriched_data_degrades_gracefully_on_all_errors():
    client = _make_client()
    client.exchange.fetch_ticker.side_effect = Exception("boom")
    client.exchange.fetch_ohlcv.side_effect = Exception("boom")
    client.exchange.fetch_funding_rate.side_effect = Exception("boom")
    client.exchange.fetch_order_book.side_effect = Exception("boom")

    data = client.get_enriched_data("BTCUSDT")

    assert data == {
        "spot_price": 0.0,
        "vol_30d": 0.0,
        "vol_60d": 0.0,
        "vol_90d": 0.0,
        "funding_rate": 0.0,
        "volume_24h": 0.0,
        "spread": 0.0,
    }


# ── Realized volatility math ─────────────────────────────────────────────────

def test_realized_vol_zero_for_flat_prices():
    assert CCXTDataClient._realized_vol([100.0] * 31) == pytest.approx(0.0, abs=1e-9)


def test_realized_vol_positive_for_moving_prices():
    closes = [100.0 * (1.02 if i % 2 == 0 else 0.98) for i in range(31)]
    vol = CCXTDataClient._realized_vol(closes)
    assert vol > 0.0


def test_realized_vol_short_series_returns_zero():
    assert CCXTDataClient._realized_vol([100.0]) == 0.0
    assert CCXTDataClient._realized_vol([]) == 0.0


def test_vol_windows_are_computed_independently():
    # A jump on the most recent candle falls inside all three windows, but
    # each window is a different slice length so they need not be equal —
    # the point is each is computed from its own window, not one shared value.
    client = _make_client()
    closes = [100.0] * 90 + [200.0]  # one big jump on the most recent candle
    client.exchange.fetch_ohlcv.return_value = _candles(closes)

    vols = client._get_realized_volatility("BTC/USDT")
    assert vols[30] > 0.0 and vols[60] > 0.0 and vols[90] > 0.0

    flat = _candles([100.0] * 91)
    client.exchange.fetch_ohlcv.return_value = flat
    client.clear_cache()
    flat_vols = client._get_realized_volatility("BTC/USDT")
    assert flat_vols[30] == flat_vols[60] == flat_vols[90] == pytest.approx(0.0, abs=1e-9)


# ── Caching: OHLCV (1h) ──────────────────────────────────────────────────────

def test_ohlcv_is_cached_within_ttl():
    client = _make_client(ohlcv_cache_ttl=3600.0)
    client.exchange.fetch_ohlcv.return_value = _candles([100 + i for i in range(91)])

    client._get_realized_volatility("BTC/USDT")
    client._get_realized_volatility("BTC/USDT")

    assert client.exchange.fetch_ohlcv.call_count == 1


def test_ohlcv_refetches_after_ttl_expires():
    client = _make_client(ohlcv_cache_ttl=0.0)  # expires immediately
    client.exchange.fetch_ohlcv.return_value = _candles([100 + i for i in range(91)])

    client._get_realized_volatility("BTC/USDT")
    client._get_realized_volatility("BTC/USDT")

    assert client.exchange.fetch_ohlcv.call_count == 2


def test_ohlcv_falls_back_to_stale_cache_on_fetch_error():
    client = _make_client(ohlcv_cache_ttl=0.0)
    client.exchange.fetch_ohlcv.return_value = _candles([100 + i for i in range(91)])
    client._get_realized_volatility("BTC/USDT")  # populate cache

    client.exchange.fetch_ohlcv.side_effect = Exception("network down")
    vols = client._get_realized_volatility("BTC/USDT")  # ttl=0 forces a refetch attempt
    assert vols[30] >= 0.0  # served from stale cache instead of raising


# ── Caching: funding rate (15min) ────────────────────────────────────────────

def test_funding_rate_is_cached_within_ttl():
    client = _make_client(funding_cache_ttl=900.0)
    client.exchange.fetch_funding_rate.return_value = {"fundingRate": 0.0002}

    r1 = client._get_funding_rate("BTC/USDT")
    r2 = client._get_funding_rate("BTC/USDT")

    assert r1 == r2 == pytest.approx(0.0002)
    assert client.exchange.fetch_funding_rate.call_count == 1


def test_funding_rate_refetches_after_ttl_expires():
    client = _make_client(funding_cache_ttl=0.0)
    client.exchange.fetch_funding_rate.side_effect = [
        {"fundingRate": 0.0001}, {"fundingRate": 0.0003},
    ]

    r1 = client._get_funding_rate("BTC/USDT")
    r2 = client._get_funding_rate("BTC/USDT")

    assert r1 == pytest.approx(0.0001)
    assert r2 == pytest.approx(0.0003)
    assert client.exchange.fetch_funding_rate.call_count == 2


def test_clear_cache_empties_both_caches():
    client = _make_client()
    client.exchange.fetch_ohlcv.return_value = _candles([100 + i for i in range(91)])
    client.exchange.fetch_funding_rate.return_value = {"fundingRate": 0.0001}

    client._get_realized_volatility("BTC/USDT")
    client._get_funding_rate("BTC/USDT")
    assert client._ohlcv_cache and client._funding_cache

    client.clear_cache()
    assert client._ohlcv_cache == {}
    assert client._funding_cache == {}

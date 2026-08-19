"""CCXTDataClient: enriched crypto market data via the CCXT unified exchange API.

Replaces BinanceClient with a richer data package for the pricing engine:
spot price, realized volatility (30/60/90d), perpetual funding rate, 24h
volume, and order book spread. CCXT's public endpoints are free and don't
require API keys.

Caching: volatility and funding rates don't change second-to-second, so
OHLCV candles (used to compute realized volatility) are cached for 1 hour
and funding rates for 15 minutes. Spot price, 24h volume, and order book
spread are fetched live on every call — they're cheap single-ticker/orderbook
requests and are exactly the numbers that DO move second-to-second, so caching
them would stale the pricing engine's most time-sensitive inputs.

This client is synchronous (uses the plain `ccxt` package, not
`ccxt.async_support`), matching the rest of PolyBot's hunter/client stack
(BinanceClient, FredClient — all sync, curl_cffi/requests based).
"""
import math
import time

import ccxt

from . import BaseApiClient

OHLCV_CACHE_TTL = 3600.0    # 1 hour
FUNDING_CACHE_TTL = 900.0   # 15 minutes

_QUOTE_SUFFIXES = ("USDT", "USDC", "BUSD", "USD")
_VOL_WINDOWS = (30, 60, 90)


class CCXTDataClient(BaseApiClient):
    """Wraps a CCXT exchange to produce the enriched data package brains/pricing need."""

    def __init__(
        self,
        exchange_id: str = "binance",
        ohlcv_cache_ttl: float = OHLCV_CACHE_TTL,
        funding_cache_ttl: float = FUNDING_CACHE_TTL,
    ):
        self.exchange_id = exchange_id
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class()
        self.ohlcv_cache_ttl = ohlcv_cache_ttl
        self.funding_cache_ttl = funding_cache_ttl
        self._ohlcv_cache: dict = {}    # ccxt_symbol -> (fetched_at, candles)
        self._funding_cache: dict = {}  # swap_symbol -> (fetched_at, funding_rate)

    # ── Symbol translation ───────────────────────────────────────────────

    @staticmethod
    def _to_spot_symbol(symbol: str) -> str:
        """"BTCUSDT" -> "BTC/USDT". Passes through if already CCXT-formatted."""
        if "/" in symbol:
            return symbol.upper()
        upper = symbol.upper()
        for quote in _QUOTE_SUFFIXES:
            if upper.endswith(quote) and len(upper) > len(quote):
                return f"{upper[:-len(quote)]}/{quote}"
        return upper

    @staticmethod
    def _to_swap_symbol(ccxt_symbol: str) -> str:
        """"BTC/USDT" -> "BTC/USDT:USDT" (linear perpetual swap, for funding rate)."""
        base, _, quote = ccxt_symbol.partition("/")
        return f"{base}/{quote}:{quote}"

    # ── BaseApiClient interface (drop-in replacement for BinanceClient) ──

    def get_latest_value(self, symbol: str = "BTCUSDT") -> float:
        """Fetch the latest spot price. Always live — never cached.

        Kept float-in/float-out so CryptoHunter's market-discovery path
        (fast, frequent anchor lookups) is unaffected by this swap.
        """
        try:
            ticker = self.exchange.fetch_ticker(self._to_spot_symbol(symbol))
            return float(ticker.get("last") or 0.0)
        except Exception as e:
            print(f"[CCXTDataClient] error fetching spot price for {symbol}: {e}")
            return 0.0

    # ── Enriched data package ────────────────────────────────────────────

    def get_enriched_data(self, symbol: str) -> dict:
        """Return the full enriched data package the pricing engine consumes.

        Args:
            symbol: e.g. "BTCUSDT" or "BTC/USDT".

        Returns:
            dict with spot_price, vol_30d, vol_60d, vol_90d, funding_rate,
            volume_24h, spread. Any field that fails to fetch falls back to
            0.0 rather than raising, matching BinanceClient's error handling.
        """
        ccxt_symbol = self._to_spot_symbol(symbol)

        ticker = self._fetch_ticker_safe(ccxt_symbol)
        spot_price = float(ticker.get("last") or 0.0)
        volume_24h = float(ticker.get("quoteVolume") or ticker.get("baseVolume") or 0.0)

        vols = self._get_realized_volatility(ccxt_symbol)
        funding_rate = self._get_funding_rate(ccxt_symbol)
        spread = self._get_spread(ccxt_symbol)

        return {
            "spot_price": spot_price,
            "vol_30d": vols.get(30, 0.0),
            "vol_60d": vols.get(60, 0.0),
            "vol_90d": vols.get(90, 0.0),
            "funding_rate": funding_rate,
            "volume_24h": volume_24h,
            "spread": spread,
        }

    # ── Internal fetchers ────────────────────────────────────────────────

    def _fetch_ticker_safe(self, ccxt_symbol: str) -> dict:
        try:
            return self.exchange.fetch_ticker(ccxt_symbol) or {}
        except Exception as e:
            print(f"[CCXTDataClient] error fetching ticker for {ccxt_symbol}: {e}")
            return {}

    def _get_spread(self, ccxt_symbol: str) -> float:
        """Relative bid-ask spread: (best_ask - best_bid) / mid_price."""
        try:
            book = self.exchange.fetch_order_book(ccxt_symbol, limit=5)
            bids, asks = book.get("bids") or [], book.get("asks") or []
            if not bids or not asks:
                return 0.0
            best_bid, best_ask = float(bids[0][0]), float(asks[0][0])
            mid = (best_bid + best_ask) / 2.0
            return (best_ask - best_bid) / mid if mid > 0 else 0.0
        except Exception as e:
            print(f"[CCXTDataClient] error fetching order book for {ccxt_symbol}: {e}")
            return 0.0

    def _get_funding_rate(self, ccxt_symbol: str) -> float:
        """Perpetual swap funding rate, cached for `funding_cache_ttl` seconds."""
        swap_symbol = self._to_swap_symbol(ccxt_symbol)
        now = time.monotonic()
        cached = self._funding_cache.get(swap_symbol)
        if cached and now - cached[0] < self.funding_cache_ttl:
            return cached[1]

        try:
            data = self.exchange.fetch_funding_rate(swap_symbol)
            rate = float(data.get("fundingRate") or 0.0)
            self._funding_cache[swap_symbol] = (now, rate)
            return rate
        except Exception as e:
            print(f"[CCXTDataClient] error fetching funding rate for {swap_symbol}: {e}")
            return cached[1] if cached else 0.0

    def _get_realized_volatility(self, ccxt_symbol: str) -> dict:
        """Annualized realized volatility over 30/60/90-day windows from daily
        OHLCV candles. Candles are cached for `ohlcv_cache_ttl` seconds."""
        now = time.monotonic()
        cached = self._ohlcv_cache.get(ccxt_symbol)
        if cached and now - cached[0] < self.ohlcv_cache_ttl:
            candles = cached[1]
        else:
            try:
                candles = self.exchange.fetch_ohlcv(ccxt_symbol, timeframe="1d", limit=max(_VOL_WINDOWS) + 1)
                self._ohlcv_cache[ccxt_symbol] = (now, candles)
            except Exception as e:
                print(f"[CCXTDataClient] error fetching OHLCV for {ccxt_symbol}: {e}")
                candles = cached[1] if cached else []

        closes = [float(c[4]) for c in candles]
        return {window: self._realized_vol(closes[-(window + 1):]) for window in _VOL_WINDOWS}

    @staticmethod
    def _realized_vol(closes: list) -> float:
        """Annualized stdev of daily log returns: std(ln(c_t / c_t-1)) * sqrt(365)."""
        if len(closes) < 2:
            return 0.0
        returns = [
            math.log(closes[i] / closes[i - 1])
            for i in range(1, len(closes))
            if closes[i - 1] > 0 and closes[i] > 0
        ]
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        return math.sqrt(variance) * math.sqrt(365.0)

    def clear_cache(self):
        """Drop all cached OHLCV/funding data — mostly useful for tests."""
        self._ohlcv_cache.clear()
        self._funding_cache.clear()

from curl_cffi import requests as crequests

from .base import BaseApiClient


class BinanceClient(BaseApiClient):
    """Simple wrapper around Binance ticker API."""

    BASE_URL = "https://api.binance.com/api/v3/ticker/price"

    def get_latest_value(self, symbol: str = "BTCUSDT") -> float:
        """Fetch the latest price for a given trading pair.

        Args:
            symbol: e.g. "BTCUSDT" or "ETHUSDT".

        Returns:
            Float price, or 0.0 on error.
        """
        try:
            res = crequests.get(
                f"{self.BASE_URL}?symbol={symbol}",
                timeout=5,
                impersonate="chrome120",
            )
            return float(res.json().get("price", 0.0))
        except Exception as e:
            print(f"[BinanceClient] error fetching {symbol}: {e}")
            return 0.0

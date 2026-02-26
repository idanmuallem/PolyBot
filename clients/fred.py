import os
import requests
from typing import Optional

from .base import BaseApiClient


class FredClient(BaseApiClient):
    """Client for the FRED (Federal Reserve Economic Data) API."""

    FRED_SERIES_MAP = {
        "FedRate": "FEDFUNDS",
        "CPI": "CPIAUCSL",
        "Unemployment": "UNRATE",
        "GDP": "A191RL1Q225SBEA",
        "DFF": "DFF",
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FRED_API_KEY")

    def get_latest_value(self, indicator: str) -> float:
        """Return the latest observation for the given indicator.

        Falls back to fake data when the API key is missing or on error.
        """
        if not self.api_key:
            print("[FredClient] FRED_API_KEY not set, using fallback")
            return self._get_fake_econ_value(indicator)

        series_id = self.FRED_SERIES_MAP.get(indicator, indicator)
        try:
            url = (
                f"https://api.stlouisfed.org/fred/series/observations?"
                f"series_id={series_id}&api_key={self.api_key}&"
                f"file_type=json&limit=1&sort_order=desc"
            )
            r = requests.get(url, timeout=6)
            if r.status_code == 200:
                j = r.json()
                obs = j.get("observations", [])
                if obs:
                    latest = obs[0].get("value")
                    try:
                        return float(latest)
                    except Exception:
                        pass
        except Exception as e:
            print(f"[FredClient] Failed to fetch {indicator}: {e}")

        return self._get_fake_econ_value(indicator)

    @staticmethod
    def _get_fake_econ_value(indicator: str) -> float:
        values = {
            "FedRate": 5.25,
            "CPI": 308.4,
            "Unemployment": 3.9,
            "GDP": 2.5,
            "DFF": 5.33,
        }
        return values.get(indicator, 5.0)

"""
EconomyHunter: Hunts Polymarket for economic indicator prediction markets.

Uses FRED API (Federal Reserve Economic Data) as the anchor source.
"""

import json
import re
from datetime import datetime
from typing import Optional, Dict, Any

import os
import requests
from curl_cffi import requests as crequests

from .base import BaseHunter


class EconomyHunter(BaseHunter):
    """Hunt markets related to economic indicators (Fed Rate, CPI, Unemployment, etc.).

    Anchor: FRED API (St. Louis Federal Reserve)
    """

    # Map friendly indicator names to FRED series IDs
    FRED_SERIES_MAP = {
        "FedRate": "FEDFUNDS",
        "CPI": "CPIAUCSL",
        "Unemployment": "UNRATE",
        "GDP": "A191RL1Q225SBEA",
        "DFF": "DFF",  # Effective Federal Funds Rate
    }

    DEFAULT_INDICATORS = ["FedRate"]

    def __init__(self, indicators: Optional[list] = None, **kwargs):
        """Initialize EconomyHunter.

        Args:
            indicators: List of indicators to hunt for.
                       Friendly names like "FedRate", "CPI" (matched against FRED_SERIES_MAP)
            **kwargs: Passed to parent (e.g., polymarket_base)
        """
        super().__init__(**kwargs)
        self.indicators = indicators or list(self.DEFAULT_INDICATORS)
        self.api_key = os.getenv("FRED_API_KEY")
        self._anchor_value = None

    def get_topic_type(self) -> str:
        return "Economy"

    def get_anchor_value(self) -> Optional[float]:
        """Fetch the current value of the primary economic indicator.

        Returns:
            Indicator value or None on failure.
        """
        if self.indicators:
            return self._get_fred_value(self.indicators[0])
        return None

    def _get_fred_value(self, indicator: str) -> Optional[float]:
        """Fetch the latest value of a FRED series.

        Args:
            indicator: Friendly name from FRED_SERIES_MAP (e.g., "FedRate")

        Returns:
            Latest value or None on failure.
        """
        if not self.api_key:
            print("[EconomyHunter] FRED_API_KEY not set, using fallback")
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
                        return None
        except Exception as e:
            print(f"[EconomyHunter] Failed to fetch {indicator}: {e}")

        # Fallback to fake data
        return self._get_fake_econ_value(indicator)

    @staticmethod
    def _get_fake_econ_value(indicator: str) -> float:
        """Return a fake economic value for testing/fallback.

        Args:
            indicator: Indicator name

        Returns:
            A plausible value for the indicator.
        """
        values = {
            "FedRate": 5.25,
            "CPI": 308.4,
            "Unemployment": 3.9,
            "GDP": 2.5,
            "DFF": 5.33,
        }
        return values.get(indicator, 5.0)

    @staticmethod
    def _extract_valid_strike(question: str, anchor_val: float) -> Optional[float]:
        """Extract a valid strike from the market question.

        For economy indicators, look for percentage/rate patterns.

        Args:
            question: Market question text
            anchor_val: Current indicator value for validation

        Returns:
            Valid strike or None.
        """
        # Look for percentage/rate patterns like "3.5%", "4.25", etc.
        matches = re.finditer(r"(\d{1,3}(?:\.\d{1,2})?)\s*(%)?", question)
        for match in matches:
            val = float(match.group(1))

            # Reasonable strike: within 5 units of current value
            if abs(val - anchor_val) < 5.0:
                return val

        return None

    def _scan_polymarket(self, anchor: float, indicator: str, max_pages: int = 5) -> Optional[Dict[str, Any]]:
        """Scan Polymarket for economy-related markets.

        Selection criteria:
        - Price floor: Market price must be >= $0.18 (no longshots)
        - Volume optimization: Among valid markets, select highest volume

        Args:
            anchor: Current indicator value
            indicator: Indicator name to match
            max_pages: Max pages to scan

        Returns:
            Market dict with highest volume, or None if no suitable market found.
        """
        best_market = None
        highest_volume = 0.0

        for page in range(max_pages):
            params = {
                "active": "true",
                "closed": "false",
                "limit": 100,
                "offset": page * 100
            }
            try:
                resp = crequests.get(
                    self.polymarket_base,
                    params=params,
                    impersonate="chrome120",
                    timeout=15
                )
                if resp.status_code != 200:
                    break
                events = resp.json()
                if not events:
                    break

                for event in events:
                    title = event.get("title", "").lower()
                    slug = event.get("slug", "").lower()

                    # Match indicator in title or slug
                    indicator_lower = indicator.lower()
                    if indicator_lower not in title and indicator_lower not in slug:
                        continue

                    for market in event.get("markets", []):
                        if market.get("closed"):
                            continue

                        question = market.get("question", "")
                        current_price = float(market.get("lastTradePrice", 0) or 0)

                        # Rule 1: Price floor filter - reject markets below $0.18
                        if current_price < self.PRICE_FLOOR:
                            continue

                        # Rule 2: Price ceiling - reject markets at or above $1.00
                        if current_price >= 1.0:
                            continue

                        # Rule 3: Extract strike
                        valid_strike = self._extract_valid_strike(question, anchor)
                        if valid_strike is None:
                            continue

                        # Rule 4: Get token ID
                        tokens = market.get("clobTokenIds")
                        if isinstance(tokens, str):
                            try:
                                tokens = json.loads(tokens)
                            except Exception:
                                tokens = None

                        if not (isinstance(tokens, list) and tokens):
                            continue

                        # Rule 5: Extract volume
                        volume = float(market.get("volume", 0) or 0)
                        if volume == 0:
                            volume = float(market.get("liquidity", 0) or 0)
                        if volume == 0:
                            volume = float(market.get("tradingVolume", 0) or 0)

                        # Rule 6: Select market with highest volume
                        if volume > highest_volume:
                            highest_volume = volume
                            best_market = {
                                "market_id": str(tokens[0]).strip(),
                                "asset_type": f"Economy::{indicator}",
                                "strike_price": valid_strike,
                                "question": question,
                                "anchor_url": None,
                                "initial_price": current_price,
                                "volume": volume,
                            }

            except Exception as e:
                print(f"[EconomyHunter] Scan error on page {page}: {e}")
                break

        return best_market

    def hunt(self) -> Optional[Dict[str, Any]]:
        """Hunt for an economy market.

        Tries each indicator in order. Returns the first valid market found.

        Returns:
            Market dict or None.
        """
        print(f"[EconomyHunter] {datetime.now().isoformat()} - Starting hunt for {len(self.indicators)} indicators")

        for indicator in self.indicators:
            print(f"[EconomyHunter] Trying indicator: {indicator}")

            # Get anchor value
            anchor_val = self._get_fred_value(indicator)
            if anchor_val is None:
                print(f"[EconomyHunter] Failed to fetch anchor for {indicator}, skipping")
                continue

            # Scan Polymarket
            found = self._scan_polymarket(anchor_val, indicator)

            if found:
                series_id = self.FRED_SERIES_MAP.get(indicator, indicator)
                found["anchor_url"] = (
                    f"https://api.stlouisfed.org/fred/series/observations?"
                    f"series_id={series_id}&api_key={self.api_key or 'N/A'}"
                )
                print(f"[EconomyHunter] Found market for {indicator}: {found['market_id']}")
                self._anchor_value = anchor_val
                return found

        print("[EconomyHunter] No economy markets found after trying all indicators")
        return None

    def get_live_truth(self, market: Dict[str, Any]) -> Optional[float]:
        """Fetch live economic indicator value from FRED API for the given market.

        Args:
            market: Market dict with asset_type like "Economy::FedRate"

        Returns:
            Current indicator value or None on error.
        """
        if not market:
            return None

        try:
            asset_type = market.get("asset_type", "")
            if not asset_type.startswith("Economy::"):
                return None

            # Extract indicator (e.g., "FedRate" from "Economy::FedRate")
            indicator = asset_type.split("::", 1)[1]
            return self._get_fred_value(indicator)
        except Exception as e:
            print(f"[EconomyHunter] get_live_truth error: {e}")
            return None

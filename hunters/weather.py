"""
WeatherHunter: Hunts Polymarket for weather-related prediction markets.

Uses OpenWeather API as the anchor source for current conditions.
"""

import json
import re
from datetime import datetime
from typing import Optional, Dict, Any

import os
import requests
from curl_cffi import requests as crequests

from .base import BaseHunter


class WeatherHunter(BaseHunter):
    """Hunt markets related to weather conditions (temperature, precipitation, etc.).

    Anchor: OpenWeather API
    """

    DEFAULT_LOCATIONS = ["Miami", "New York", "London"]

    def __init__(self, locations: Optional[list] = None, **kwargs):
        """Initialize WeatherHunter.

        Args:
            locations: List of locations to hunt for (default: Miami, New York, London).
            **kwargs: Passed to parent (e.g., polymarket_base)
        """
        super().__init__(**kwargs)
        self.locations = locations or list(self.DEFAULT_LOCATIONS)
        self.api_key = os.getenv("OPENWEATHER_API_KEY")
        self._anchor_value = None

    def get_topic_type(self) -> str:
        return "Weather"

    def get_anchor_value(self) -> Optional[float]:
        """Fetch the current temperature for the primary location.

        Returns:
            Temperature in Celsius or None on failure.
        """
        if self.locations:
            return self._get_openweather_temperature(self.locations[0])
        return None

    def _get_openweather_temperature(self, location: str) -> Optional[float]:
        """Fetch current temperature from OpenWeather API.

        If OPENWEATHER_API_KEY is not set, return None (do not use fake data).

        Args:
            location: City name (e.g., "Miami")

        Returns:
            Temperature in Celsius or None on failure or missing API key.
        """
        if not self.api_key:
            # Print warning once instead of using fallback
            if not hasattr(self, "_api_key_warning_printed"):
                print("[WeatherHunter] WARNING: OPENWEATHER_API_KEY not set. Weather hunting disabled.")
                self._api_key_warning_printed = True
            return None

        try:
            url = (
                f"https://api.openweathermap.org/data/2.5/weather?"
                f"q={location}&units=metric&appid={self.api_key}"
            )
            r = requests.get(url, timeout=6)
            if r.status_code == 200:
                j = r.json()
                temp = float(j.get("main", {}).get("temp"))
                return temp
        except Exception as e:
            print(f"[WeatherHunter] Failed to fetch temp for {location}: {e}")

        # No fallback: return None on error
        return None

    @staticmethod
    def _get_fake_temperature(location: str) -> float:
        """Return a fake temperature for testing/fallback.

        Args:
            location: City name

        Returns:
            A plausible temperature based on location.
        """
        temps = {
            "miami": 28.0,
            "newyork": 10.0,
            "london": 8.0,
            "tokyo": 15.0,
            "sydney": 22.0,
        }
        return temps.get(location.lower(), 20.0)

    @staticmethod
    def _extract_valid_strike(question: str, anchor_temp: float) -> Optional[float]:
        """Extract a valid strike temperature from the market question.

        Looks for patterns like "above 75F", "below 30", etc.

        Args:
            question: Market question text
            anchor_temp: Current temperature for validation

        Returns:
            Valid strike temperature or None.
        """
        # Look for number patterns (degrees)
        matches = re.finditer(r"(\d{1,3})(?:\s*°\s*|)([FCf])?", question)
        for match in matches:
            temp_val = float(match.group(1))
            unit = match.group(2)

            # If Fahrenheit, convert to Celsius for consistency
            if unit and unit.upper() == "F":
                temp_val = (temp_val - 32) * 5 / 9

            # Reasonable strike: within ~20 degrees of current
            if abs(temp_val - anchor_temp) < 20:
                return temp_val

        return None

    def _scan_polymarket(self, anchor: float, location: str, max_pages: int = 5) -> Optional[Dict[str, Any]]:
        """Scan Polymarket for weather-related markets.

        Selection criteria:
        - Price floor: Market price must be >= $0.18 (no longshots)
        - Volume optimization: Among valid markets, select highest volume

        Args:
            anchor: Current temperature
            location: Location name to match
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

                    # Match location in title or slug
                    if location.lower() not in title and location.lower() not in slug:
                        continue
                    
                    # Expanded keywords to catch "Miami High", "NYC Temp", etc.
                    valid_keywords = ["weather", "temperature", "temp", "high", "low", "precipitation", "rain", "snow", "degree"]
                    if not any(k in title or k in slug for k in valid_keywords):
                        continue

                    for market in event.get("markets", []):
                        if market.get("closed"):
                            continue

                        question = market.get("question", "")
                        current_price = float(market.get("lastTradePrice", 0) or 0)

                        # Rule 0: Get token ID early for skip_ids check
                        tokens = market.get("clobTokenIds")
                        if isinstance(tokens, str):
                            try:
                                tokens = json.loads(tokens)
                            except Exception:
                                tokens = None

                        if not (isinstance(tokens, list) and tokens):
                            continue
                        
                        market_id = str(tokens[0]).strip()
                        
                        # Skip markets in cooldown cache
                        if market_id in skip_ids:
                            print(f"[WeatherHunter] Skipping {market_id} (in 10m cooldown)")
                            continue

                        # Rule 1: Price floor filter - reject markets below $0.18
                        if current_price < self.PRICE_FLOOR:
                            continue

                        # Rule 2: Price ceiling - reject markets at or above $1.00
                        if current_price >= 1.0:
                            continue

                        # Rule 3: Extract strike or strike-range from question or outcomes
                        valid_strike = None
                        strike_low = None
                        strike_high = None

                        # First try to detect explicit ranges in outcomes or question
                        def parse_range(text: str):
                            if not text:
                                return None
                            # Range like '74-75F' or '74 - 75 °F'
                            m = re.search(r"(\d{1,3})\s*[-–]\s*(\d{1,3})(?:\s*°\s*|)\s*([FCf])?", text)
                            if m:
                                a = float(m.group(1))
                                b = float(m.group(2))
                                unit = m.group(3)
                                if unit and unit.upper() == "F":
                                    a = (a - 32) * 5 / 9
                                    b = (b - 32) * 5 / 9
                                return (min(a, b), max(a, b))

                            # '80F or higher' or 'above 80F' patterns
                            m2 = re.search(r"(?:above|greater than|at least|or higher)\s*(\d{1,3})(?:\s*°\s*|)\s*([FCf])?", text, re.IGNORECASE)
                            if m2:
                                a = float(m2.group(1))
                                unit = m2.group(2)
                                if unit and unit.upper() == "F":
                                    a = (a - 32) * 5 / 9
                                return (a, None)

                            return None

                        # Check question first
                        pr = parse_range(question)
                        if pr:
                            strike_low, strike_high = pr
                            if strike_high is None:
                                valid_strike = strike_low
                            else:
                                valid_strike = (strike_low + strike_high) / 2

                        # If not in question, scan outcomes for ranges or explicit numbers
                        if valid_strike is None:
                            outcomes = market.get("outcomes") or []
                            for o in outcomes:
                                text = None
                                if isinstance(o, dict):
                                    text = o.get("name") or o.get("label") or o.get("title")
                                else:
                                    text = str(o)

                                pr = parse_range(text)
                                if pr:
                                    strike_low, strike_high = pr
                                    if strike_high is None:
                                        valid_strike = strike_low
                                    else:
                                        valid_strike = (strike_low + strike_high) / 2
                                    break

                        # Fallback: try extracting a single-value strike from the question
                        if valid_strike is None:
                            valid_strike = self._extract_valid_strike(question, anchor)
                        if valid_strike is None:
                            continue

                        # Rule 4: Token ID already extracted above for skip_ids check

                        # Rule 5: Extract volume and enforce MIN_VOLUME
                        volume = float(market.get("volume", 0) or 0)
                        if volume == 0:
                            volume = float(market.get("liquidity", 0) or 0)
                        if volume == 0:
                            volume = float(market.get("tradingVolume", 0) or 0)

                        if volume < getattr(self, "MIN_VOLUME", 0):
                            continue

                        # Rule 6: Select market with highest volume
                        if volume > highest_volume:
                            highest_volume = volume
                            bm = {
                                "market_id": str(tokens[0]).strip(),
                                "asset_type": f"Weather::{location}",
                                "strike_price": valid_strike,
                                "question": question,
                                "anchor_url": None,
                                "initial_price": current_price,
                                "volume": volume,
                            }
                            # attach range metadata if present
                            if strike_low is not None:
                                bm["strike_low"] = strike_low
                            if strike_high is not None:
                                bm["strike_high"] = strike_high

                            best_market = bm

            except Exception as e:
                print(f"[WeatherHunter] Scan error on page {page}: {e}")
                break

        return best_market

    def hunt(self, skip_ids: list = None) -> Optional[Dict[str, Any]]:
        """Hunt for a weather market.

        If API key is not set, return None immediately without attempting to hunt.
        Otherwise, tries each location in order. Returns the first valid market found.
        Respects skip_ids list to avoid markets in 10-minute cooldown.

        Args:
            skip_ids: List of market_ids to skip (in cooldown). Defaults to [].

        Returns:
            Market dict or None.
        """
        if skip_ids is None:
            skip_ids = []
        
        # Early exit if API key not configured
        if not self.api_key:
            if not hasattr(self, "_hunt_warning_printed"):
                print("[WeatherHunter] Skipping hunt: OPENWEATHER_API_KEY not configured.")
                self._hunt_warning_printed = True
            return None

        print(f"[WeatherHunter] {datetime.now().isoformat()} - Starting hunt for {len(self.locations)} locations (skipping {len(skip_ids)} cooldown markets)")

        for location in self.locations:
            print(f"[WeatherHunter] Trying location: {location}")

            # Get anchor temperature
            anchor_temp = self._get_openweather_temperature(location)
            if anchor_temp is None:
                print(f"[WeatherHunter] Failed to fetch anchor for {location}, skipping")
                continue

            # Scan Polymarket
            found = self._scan_polymarket(anchor_temp, location)

            if found:
                found["anchor_url"] = (
                    f"https://api.openweathermap.org/data/2.5/weather?"
                    f"q={location}&units=metric&appid={self.api_key}"
                )
                print(f"[WeatherHunter] Found market for {location}: {found['market_id']}")
                self._anchor_value = anchor_temp
                return found

        print("[WeatherHunter] No weather markets found after trying all locations")
        return None

    def get_live_truth(self, market: Dict[str, Any]) -> Optional[float]:
        """Fetch live temperature from OpenWeather API for the given market.

        Args:
            market: Market dict with asset_type like "Weather::Miami"

        Returns:
            Current temperature in Celsius or None on error.
        """
        if not market:
            return None

        try:
            asset_type = market.get("asset_type", "")
            if not asset_type.startswith("Weather::"):
                return None

            # Extract location (e.g., "Miami" from "Weather::Miami")
            location = asset_type.split("::", 1)[1]
            return self._get_openweather_temperature(location)
        except Exception as e:
            print(f"[WeatherHunter] get_live_truth error: {e}")
            return None

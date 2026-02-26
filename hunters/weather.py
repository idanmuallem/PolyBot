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

from .base import BasePolymarketHunter


class WeatherHunter(BasePolymarketHunter):
    """Hunt markets related to weather conditions (temperature, precipitation, etc.).

    Anchor: OpenWeather API

    This implementation leverages :class:`BasePolymarketHunter` for the
    underlying scan logic.  A small number of hooks are supplied to handle
    temperature-specific strike extraction and keyword aliases; additionally
    we require that the event title/slug contain the location string so that
    anchors are not mismatched (e.g. we don't want the "weather" event for
    London when hunting Miami temperatures).
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

    def extract_strike(self, text: str, anchor_temp: float) -> Optional[float]:
        """Concrete hook for :meth:`BasePolymarketHunter.extract_strike`.

        The original _extract_valid_strike logic lives here; we examine any
        text blob (title/question/outcomes) and look for a plausible
        temperature that is within ~20°C of the anchor.  Fahrenheit values
        are converted to Celsius.
        """
        # Look for number patterns (degrees)
        matches = re.finditer(r"(\d{1,3})(?:\s*°\s*|)([FCf])?", text)
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

    # the old scanning logic has been replaced by the generic base implementation
    # which is much shorter and easier to maintain.  We still keep a copy of the
    # original code in git history in case the location-handling requirements
    # change in the future.

    def get_search_aliases(self) -> list:
        """Return keywords used to identify weather markets in Polymarket events.

        These are generic terms such as "weather", "temperature", etc.  The
        _scan_polymarket helper will always include the location string passed as
        the keyword argument, so the combination guarantees that the returned
        market relates to *both* weather and the requested city.
        """
        return ["weather", "temperature", "temp", "high", "low", "precipitation", "rain", "snow", "degree"]

    def hunt(self, skip_ids: list = None) -> Optional[Dict[str, Any]]:
        """Hunt for a weather market.

        Tries each configured location in turn.  Uses the shared
        :meth:`_scan_polymarket` routine with ``required_keywords`` set to the
        current location so that results are location-specific.
        """
        if skip_ids is None:
            skip_ids = []

        # Early exit if API key not configured
        if not self.api_key:
            if not hasattr(self, "_hunt_warning_printed"):
                print("[WeatherHunter] Skipping hunt: OPENWEATHER_API_KEY not configured.")
                self._hunt_warning_printed = True
            return None

        print(
            f"[WeatherHunter] {datetime.now().isoformat()} - Starting hunt for {len(self.locations)} locations (skipping {len(skip_ids)} cooldown markets)"
        )

        for location in self.locations:
            print(f"[WeatherHunter] Trying location: {location}")

            # Get anchor temperature
            anchor_temp = self._get_openweather_temperature(location)
            if anchor_temp is None:
                print(f"[WeatherHunter] Failed to fetch anchor for {location}, skipping")
                continue

            # Scan Polymarket (require location to appear in title/slug)
            found = self._scan_polymarket(
                anchor_temp,
                location,
                skip_ids=skip_ids,
                required_keywords=[location],
            )

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

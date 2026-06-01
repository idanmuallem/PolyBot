"""
WeatherHunter: Hunts Polymarket for weather prediction markets.

Requires OPENWEATHER_API_KEY; returns None from hunt() if the key is absent.
"""

import os
import re
from typing import Optional

import requests

from core.models import MarketData
from .base import BasePolymarketHunter


class WeatherHunter(BasePolymarketHunter):
    """Hunt markets related to weather conditions (temperature, precipitation, etc.).

    Anchor: OpenWeather API current temperature.
    Location string is enforced as a required keyword so results are city-specific.
    """

    DEFAULT_LOCATIONS = ["Miami", "New York", "London"]

    def __init__(self, locations: Optional[list] = None, **kwargs):
        super().__init__(**kwargs)
        self.locations = locations or list(self.DEFAULT_LOCATIONS)
        self.api_key = os.getenv("OPENWEATHER_API_KEY")

    def get_topic_type(self) -> str:
        return "Weather"

    def get_anchor_value(self) -> Optional[float]:
        if self.locations:
            return self._fetch_temperature(self.locations[0])
        return None

    def _fetch_temperature(self, location: str) -> Optional[float]:
        if not self.api_key:
            print("[WeatherHunter] OPENWEATHER_API_KEY not set — weather disabled.")
            return None
        try:
            url = (
                f"https://api.openweathermap.org/data/2.5/weather"
                f"?q={location}&units=metric&appid={self.api_key}"
            )
            r = requests.get(url, timeout=6)
            if r.status_code == 200:
                return float(r.json().get("main", {}).get("temp"))
        except Exception as e:
            print(f"[WeatherHunter] Failed to fetch temp for {location}: {e}")
        return None

    def extract_strike(self, text: str, anchor_temp: float) -> Optional[float]:
        for match in re.finditer(r"(\d{1,3})(?:\s*°\s*|)([FCf])?", text):
            temp_val = float(match.group(1))
            unit = match.group(2)
            if unit and unit.upper() == "F":
                temp_val = (temp_val - 32) * 5 / 9
            if abs(temp_val - anchor_temp) < 20:
                return temp_val
        return None

    def get_search_aliases(self) -> list:
        return ["weather", "temperature", "temp", "high", "low", "precipitation", "rain", "snow", "degree"]

    def hunt(self, skip_ids: list = None, add_cooldown_func=None) -> Optional[MarketData]:
        if skip_ids is None:
            skip_ids = []

        if not self.api_key:
            print("[WeatherHunter] Skipping hunt: OPENWEATHER_API_KEY not configured.")
            return None

        print(f"[WeatherHunter] Starting hunt: {len(self.locations)} locations, {len(skip_ids)} skipped")

        for location in self.locations:
            anchor_temp = self._fetch_temperature(location)
            if anchor_temp is None:
                print(f"[WeatherHunter] No anchor for {location}, skipping")
                continue

            found = self._scan_polymarket(
                anchor_temp,
                location,
                skip_ids=skip_ids,
                required_keywords=[location],
                add_cooldown_func=add_cooldown_func,
            )
            if found:
                print(f"[WeatherHunter] Found market for {location}: {found.market_name}")
                return found

        print("[WeatherHunter] No markets found")
        return None

    def get_live_truth(self, market: MarketData) -> Optional[float]:
        if not market or not market.asset_type.startswith("Weather::"):
            return None
        try:
            location = market.asset_type.split("::", 1)[1]
            return self._fetch_temperature(location)
        except Exception as e:
            print(f"[WeatherHunter] get_live_truth error: {e}")
            return None

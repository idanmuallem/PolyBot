# hunter.py
import re
import json
import time
from datetime import datetime
import requests
from curl_cffi import requests as crequests


class MarketHunter:
    """Hunts Polymarket for markets using a Dynamic Anchor cascade.

    Targets are attempted in order; the first valid market found is returned.
    """

    DEFAULT_TARGETS = [
        {"type": "Crypto", "symbol": "BTCUSDT"},
        {"type": "Crypto", "symbol": "ETHUSDT"},
        {"type": "Weather", "location": "Miami"},
        {"type": "Economy", "indicator": "FedRate"},
    ]

    EQUILIBRIUM_TARGET = 0.50
    PRICE_TOLERANCE = 0.35

    def __init__(self, targets=None, polymarket_base="https://gamma-api.polymarket.com/events"):
        self.targets = targets or list(self.DEFAULT_TARGETS)
        self.polymarket_base = polymarket_base

    # ---- Helpers ----
    @staticmethod
    def extract_valid_strike(question, live_price):
        matches = re.finditer(r"\$(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*([kKmMbB])?", question)
        for match in matches:
            base_num = float(match.group(1).replace(",", ""))
            suffix = match.group(2).lower() if match.group(2) else ""
            if suffix == "k":
                base_num *= 1_000
            elif suffix == "m":
                base_num *= 1_000_000
            elif suffix == "b":
                base_num *= 1_000_000_000

            ratio = base_num / live_price if live_price else 0
            if 0.2 < ratio < 5.0:
                return base_num
        return None

    @staticmethod
    def get_binance_price(symbol="BTCUSDT"):
        try:
            res = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}", timeout=5)
            return float(res.json().get("price", 0.0))
        except Exception:
            return None

    # ---- Polymarket scanning ----
    def scan_polymarket(self, anchor, topic_type, max_pages=5, page_size=100):
        """Scan Polymarket for markets that are mathematically near the anchor.

        anchor: numeric anchor value (e.g., price or temperature)
        topic_type: a short string used to match event titles/questions
        """
        closest = None
        closest_dist = 1.0

        for page in range(max_pages):
            params = {"active": "true", "closed": "false", "limit": page_size, "offset": page * page_size}
            try:
                resp = crequests.get(self.polymarket_base, params=params, impersonate="chrome120", timeout=15)
                if resp.status_code != 200:
                    break
                events = resp.json()
                if not events:
                    break

                for event in events:
                    title = event.get("title", "").lower()
                    # simple topic matching
                    if topic_type.lower() not in title and topic_type.lower() not in event.get("slug", "").lower():
                        continue

                    for m in event.get("markets", []):
                        if m.get("closed"):
                            continue

                        question = m.get("question", "")
                        current_price = float(m.get("lastTradePrice", 0) or 0)
                        dist_to_center = abs(current_price - self.EQUILIBRIUM_TARGET)
                        if dist_to_center > self.PRICE_TOLERANCE:
                            continue

                        valid_strike = self.extract_valid_strike(question, anchor)
                        if not valid_strike:
                            continue

                        tokens = m.get("clobTokenIds")
                        if isinstance(tokens, str):
                            try:
                                tokens = json.loads(tokens)
                            except Exception:
                                tokens = None

                        if isinstance(tokens, list) and tokens:
                            if dist_to_center < closest_dist:
                                closest_dist = dist_to_center
                                closest = {
                                    "market_id": str(tokens[0]).strip(),
                                    "asset_type": topic_type,
                                    "strike_price": valid_strike,
                                    "question": question,
                                    "anchor_url": None,
                                    "initial_price": current_price,
                                }
            except Exception:
                break

            if closest_dist <= 0.10:
                break

        return closest

    # ---- Main public API ----
    def hunt(self):
        """Iterates through targets and returns the first valid market found.

        Return: dict with keys market_id, asset_type, strike_price, question, anchor_url
        """
        print(f"[MarketHunter] {datetime.now().isoformat()} - Starting hunt with {len(self.targets)} targets")

        for t in self.targets:
            ttype = t.get("type")
            print(f"[MarketHunter] Trying target: {t}")

            if ttype == "Crypto":
                symbol = t.get("symbol")
                price = self.get_binance_price(symbol)
                if not price:
                    print(f"[MarketHunter] Failed to fetch anchor for {symbol}")
                    continue

                found = self.scan_polymarket(price, symbol)
                if found:
                    found["anchor_url"] = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
                    found["asset_type"] = f"Crypto::{symbol}"
                    print(f"[MarketHunter] Found market for {symbol}: {found['market_id']}")
                    return found

            elif ttype == "Weather":
                loc = t.get("location")
                # For Weather, use a simple placeholder anchor: current temperature (C)
                temp = self.get_fake_weather_temperature(loc)
                if temp is None:
                    continue
                found = self.scan_polymarket(temp, loc)
                if found:
                    found["anchor_url"] = f"weather://{loc}"
                    found["asset_type"] = f"Weather::{loc}"
                    print(f"[MarketHunter] Found weather market for {loc}: {found['market_id']}")
                    return found

            elif ttype == "Economy":
                indicator = t.get("indicator")
                value = self.get_fake_econ_indicator(indicator)
                if value is None:
                    continue
                found = self.scan_polymarket(value, indicator)
                if found:
                    found["anchor_url"] = f"economy://{indicator}"
                    found["asset_type"] = f"Economy::{indicator}"
                    print(f"[MarketHunter] Found econ market for {indicator}: {found['market_id']}")
                    return found

            time.sleep(0.5)

        print("[MarketHunter] No markets found in cascade.")
        return None

    # ---- Simple fake anchors for Weather/Economy when no API key is available ----
    @staticmethod
    def get_fake_weather_temperature(location):
        # Placeholder: in a production system replace with OpenWeather fetch
        sample_temps = {"Miami": 32.0, "NYC": 10.0}
        return sample_temps.get(location)

    @staticmethod
    def get_fake_econ_indicator(indicator):
        sample = {"FedRate": 4.25}
        return sample.get(indicator)
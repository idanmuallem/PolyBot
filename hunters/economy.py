"""
EconomyHunter: Hunts Polymarket for economic indicator prediction markets.
"""

from typing import Optional

from core.models import MarketData
from .parsers import extract_economy_strike
from .clients.fred import FredClient

from .base import BasePolymarketHunter


class EconomyHunter(BasePolymarketHunter):
    """Hunt markets related to economic indicators (Fed Rate, CPI, etc.).

    Anchor: FRED API (St. Louis Federal Reserve) via FredClient.
    """

    KEYWORD_INDICATOR_MAP = {
        "inflation": "CPI",
        "cpi": "CPI",
        "consumer price": "CPI",
        "fed funds": "FedRate",
        "fed rate": "FedRate",
        "federal funds": "FedRate",
    }

    DEFAULT_INDICATORS = ["FedRate"]

    def __init__(self, indicators: Optional[list] = None, **kwargs):
        super().__init__(**kwargs)
        self.indicators = indicators or list(self.DEFAULT_INDICATORS)
        self.fred_client = FredClient()

    def get_topic_type(self) -> str:
        return "Economy"

    def get_anchor_value(self) -> Optional[float]:
        if self.indicators:
            return self.fred_client.get_latest_value(self.indicators[0])
        return None

    def extract_strike(self, text: str, anchor_val: float) -> Optional[float]:
        return extract_economy_strike(text, anchor_val)

    def get_search_aliases(self) -> list:
        aliases = set(self.KEYWORD_INDICATOR_MAP.keys())
        aliases.update(name.lower() for name in self.fred_client.FRED_SERIES_MAP.keys())
        return list(aliases)

    def hunt(self, skip_ids: list = None, add_cooldown_func=None) -> Optional[MarketData]:
        if skip_ids is None:
            skip_ids = []

        print(f"[EconomyHunter] Starting hunt: {len(self.indicators)} indicators, {len(skip_ids)} skipped")

        for indicator in self.indicators:
            anchor_val = self.fred_client.get_latest_value(indicator)
            if anchor_val is None:
                print(f"[EconomyHunter] No anchor for {indicator}, skipping")
                continue

            found = self._scan_polymarket(
                anchor_val,
                indicator,
                skip_ids=skip_ids,
                add_cooldown_func=add_cooldown_func,
            )
            if found:
                print(f"[EconomyHunter] Found market: {found.market_name}")
                return found

        print("[EconomyHunter] No markets found")
        return None

    def get_live_truth(self, market: MarketData) -> Optional[float]:
        if not market or not market.asset_type.startswith("Economy::"):
            return None
        try:
            indicator = market.asset_type.split("::", 1)[1]
            return self.fred_client.get_latest_value(indicator)
        except Exception as e:
            print(f"[EconomyHunter] get_live_truth error: {e}")
            return None

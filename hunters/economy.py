"""
EconomyHunter: Hunts Polymarket for economic indicator prediction markets.

Uses FredClient for economic data and PolymarketClient for market discovery.
"""

from datetime import datetime
from typing import Optional

from core.models import MarketData
from parsers.economy import extract_economy_strike
from clients.fred import FredClient
from clients.polymarket import PolymarketClient

from .base import BasePolymarketHunter


class EconomyHunter(BasePolymarketHunter):
    """Hunt markets related to economic indicators (Fed Rate, CPI, Unemployment, etc.).

    Anchor: FRED API (St. Louis Federal Reserve) via FredClient
    Uses PolymarketClient for market discovery.
    """

    # Keyword -> canonical indicator mapping (synonyms)
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
        """Initialize EconomyHunter with clients.

        Args:
            indicators: List of indicators to hunt for.
                       Friendly names like "FedRate", "CPI" (matched against FRED)
            **kwargs: Passed to parent (e.g., polymarket_base)
        """
        super().__init__(**kwargs)
        self.indicators = indicators or list(self.DEFAULT_INDICATORS)
        self.fred_client = FredClient()
        self.polymarket_client = PolymarketClient()

    def get_topic_type(self) -> str:
        return "Economy"

    def get_anchor_value(self) -> Optional[float]:
        """Fetch the current value of the primary economic indicator from FRED."""
        if self.indicators:
            return self.fred_client.get_latest_value(self.indicators[0])
        return None

    def extract_strike(self, text: str, anchor_val: float) -> Optional[float]:
        """Extract economic indicator strike price using parser."""
        return extract_economy_strike(text, anchor_val)

    def get_search_aliases(self) -> list:
        """Return all keywords/aliases that may appear in economy-related events."""
        aliases = set()
        # synonyms map keys are already lowercase
        aliases.update(self.KEYWORD_INDICATOR_MAP.keys())
        # also include the canonical names
        aliases.update(name.lower() for name in self.fred_client.FRED_SERIES_MAP.keys())
        return list(aliases)

    def hunt(self, skip_ids: list = None, add_cooldown_func=None) -> Optional[MarketData]:
        """Hunt for an economy market.

        Attempts each configured indicator in order.

        Args:
            skip_ids: List of market_ids to skip (in cooldown). Defaults to [].

        Returns:
            MarketData object or None.
        """
        if skip_ids is None:
            skip_ids = []

        print(
            f"[EconomyHunter] {datetime.now().isoformat()} - Starting hunt for {len(self.indicators)} indicators (skipping {len(skip_ids)} cooldown markets)"
        )

        for indicator in self.indicators:
            print(f"[EconomyHunter] Trying indicator: {indicator}")

            # Get anchor value
            anchor_val = self.fred_client.get_latest_value(indicator)
            if anchor_val is None:
                print(f"[EconomyHunter] Failed to fetch anchor for {indicator}, skipping")
                continue

            # Scan Polymarket using generic scanner
            found = self._scan_polymarket(
                anchor_val,
                indicator,
                skip_ids=skip_ids,
                add_cooldown_func=add_cooldown_func,
            )

            if found:
                print(f"[EconomyHunter] Found market for {indicator}: {found.market_id}")
                return found

        print("[EconomyHunter] No economy markets found after trying all indicators")
        return None

    def get_live_truth(self, market: MarketData) -> Optional[float]:
        """Fetch live economic indicator value from FRED API for the given market.

        Args:
            market: MarketData object with asset_type like "Economy::FedRate"

        Returns:
            Current indicator value or None on error.
        """
        if not market:
            return None

        try:
            asset_type = market.asset_type
            if not asset_type.startswith("Economy::"):
                return None

            # Extract indicator (e.g., "FedRate" from "Economy::FedRate")
            indicator = asset_type.split("::", 1)[1]
            return self.fred_client.get_latest_value(indicator)
        except Exception as e:
            print(f"[EconomyHunter] get_live_truth error: {e}")
            return None

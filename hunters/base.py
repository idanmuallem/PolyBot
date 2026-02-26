"""
Base abstract class for all market hunters.

A Hunter is responsible for searching Polymarket for markets that match
a specific asset domain (Crypto, Weather, Economy) and returning the best match.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseHunter(ABC):
    """Abstract base class for all market hunters.

    Each hunter specializes in a specific domain and must implement hunt()
    to return market data or None if no suitable market is found.

    Selection Strategy:
    - Price floor: Ignore markets priced below $0.18 (too illiquid/risky)
    - Optimization: Among valid markets, select the one with highest volume/liquidity
    - Strike validation: Strict ratio check (0.2 < spot_ratio < 5.0) to avoid absurd strikes
    """

    PRICE_FLOOR = 0.10  # Minimum acceptable market price (no longshots below this)
    PRICE_CEILING = 0.85  # Maximum acceptable market price (ignore markets > 0.85)
    MIN_VOLUME = 250  # Minimum volume required to consider a market active
    STRIKE_RATIO_MIN = 0.2  # Min ratio of strike to anchor (0.2x = 5:1 downside max)
    STRIKE_RATIO_MAX = 2.0  # Max ratio of strike to anchor (2x = allow moderately far strikes)

    POLYMARKET_BASE = "https://gamma-api.polymarket.com/events"

    def __init__(self, polymarket_base: str = POLYMARKET_BASE):
        """Initialize the hunter.

        Args:
            polymarket_base: Base URL for Polymarket API
        """
        self.polymarket_base = polymarket_base

    @abstractmethod
    def hunt(self, skip_ids: list = None) -> Optional[Dict[str, Any]]:
        """Hunt for a market matching this hunter's domain.
        
        The skip_ids parameter allows the engine to exclude markets currently in cooldown,
        forcing the hunter to explore alternative opportunities (e.g., Bitcoin 70k instead of 68k).

        Args:
            skip_ids: List of market_ids to skip (in cooldown). If None, defaults to [].

        Returns:
            A dict with keys:
                - market_id: str (token ID from Polymarket)
                - asset_type: str (e.g., "Crypto::BTCUSDT", "Weather::Miami", "Economy::FedRate")
                - strike_price: float (the strike/benchmark value)
                - question: str (the market question)
                - anchor_url: str (URL of the anchor data source)
                - initial_price: float (current Polymarket price)
            Or None if no suitable market is found.
        """
        pass

    @abstractmethod
    def get_anchor_value(self) -> Optional[float]:
        """Fetch the current anchor value from the external API.

        Returns:
            The anchor value (e.g., BTC price, temperature, Fed rate) or None on failure.
        """
        pass

    @abstractmethod
    def get_live_truth(self, market: Dict[str, Any]) -> Optional[float]:
        """Fetch the live/current value for a specific market.

        This method handles all API calls and data fetching for a given market.
        The engine delegates to this method rather than doing fetches internally.

        Args:
            market: Market dict (returned from hunt()) with keys like:
                   - asset_type: "Crypto::BTCUSDT", "Weather::Miami", "Economy::FedRate"
                   - strike_price: threshold value
                   - question: market question

        Returns:
            The current live value (e.g., spot price, temperature, Fed rate) or None on error.
        """
        pass

    def get_topic_type(self) -> str:
        """Return the topic type identifier for this hunter.

        Examples: "Crypto", "Weather", "Economy"
        """
        raise NotImplementedError("Subclasses should override get_topic_type()")

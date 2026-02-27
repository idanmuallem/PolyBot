"""
Base abstract class for all market hunters.

A Hunter is responsible for searching Polymarket for markets that match
a specific asset domain (Crypto, Weather, Economy) and returning the best match.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import json
from curl_cffi import requests as crequests
from core.models import MarketData


class BaseHunter(ABC):
    """Minimal abstract base class for all market hunters.

    Subclasses must implement:
      - hunt()
      - get_anchor_value()
      - get_live_truth()

    This class does *not* prescribe any Polymarket-specific behavior.  For
    hunters that scan Polymarket events/markets the derived
    :class:`BasePolymarketHunter` provides shared utilities.
    """

    POLYMARKET_BASE = "https://gamma-api.polymarket.com/events"

    def __init__(self, polymarket_base: str = POLYMARKET_BASE):
        """Initialize the hunter.

        Args:
            polymarket_base: Base URL for Polymarket API
        """
        self.polymarket_base = polymarket_base


class BasePolymarketHunter(BaseHunter):
    """Shared functionality for hunters that query the Polymarket API.

    Implements pagination, volume/pricing filters and the basic market loop.
    Concrete subclasses need only provide a few abstract hooks for domain-
    specific behaviour:

    * :meth:`extract_strike` – find a numerical strike inside arbitrary text
    * :meth:`get_search_aliases` – list of alias keywords to match in titles/slugs
    * :meth:`get_live_truth` – fetch the current value for the selected market
    """

    # selection constants moved out of the generic base so they can be tuned
    # independently of non-Polymarket hunters.
    PRICE_FLOOR = 0.10
    PRICE_CEILING = 0.85
    MIN_VOLUME = 250
    STRIKE_RATIO_MIN = 0.2
    STRIKE_RATIO_MAX = 2.0

    def _scan_polymarket(
        self,
        anchor: float,
        keyword: str,
        skip_ids: list = None,
        max_pages: int = 5,
        required_keywords: list = None,
    ) -> Optional[MarketData]:
        """Generic Polymarket scanner used by all derived hunters.

        This method embodies the common "template" for browsing the public
        Polymarket events API and applying a sequence of filters.  Concrete
        subclasses only have to supply a few hooks (see :py:meth:`extract_strike`,
        :py:meth:`get_search_aliases`, :py:meth:`_resolve_keyword`) and can pass
        additional constraints such as ``required_keywords`` when they need to
        impose more than the usual alias-based match (e.g. WeatherHunter needs to
        ensure the location string appears in the event as well as a generic
        weather keyword).

        Args:
            anchor: Current anchor value (price, temperature, indicator, etc.)
            keyword: Primary search term placed in the ``query`` parameter.
            skip_ids: Market IDs to ignore (cooldown cache from engine).
            max_pages: Maximum pages to traverse.
            required_keywords: A list of words that must also appear (anywhere) in
                the event title/slug.  This is useful when the ``keyword`` is
                too broad by itself; for example the location string in
                ``WeatherHunter`` must be present even if a generic keyword
                like "weather" is matched.

        Returns:
            Best MarketData object meeting the criteria or ``None`` if nothing found.
        """
        if skip_ids is None:
            skip_ids = []
        if required_keywords is None:
            required_keywords = []

        best_market = None
        highest_volume = 0.0

        aliases = [k.lower() for k in self.get_search_aliases()]
        # ensure keyword itself is considered
        if keyword.lower() not in aliases:
            aliases.append(keyword.lower())

        for page in range(max_pages):
            params = {
                "active": "true",
                "closed": "false",
                "limit": 100,
                "offset": page * 100,
                "query": keyword,
                "order": "volume",
                "ascending": "false",
            }
            try:
                resp = crequests.get(
                    self.polymarket_base,
                    params=params,
                    impersonate="chrome120",
                    timeout=15,
                )
                if resp.status_code != 200:
                    break
                events = resp.json()
                if not events:
                    break

                for event in events:
                    title = event.get("title", "").lower()
                    slug = event.get("slug", "").lower()

                    # enforce any required keywords first
                    if required_keywords:
                        if not all(
                            (kw.lower() in title or kw.lower() in slug)
                            for kw in required_keywords
                        ):
                            continue

                    # must match at least one alias in title or slug and remember
                    # which one triggered the hit.
                    matched_alias = None
                    for alias in aliases:
                        if alias in title or alias in slug:
                            matched_alias = alias
                            break
                    if not matched_alias:
                        continue

                    for market in event.get("markets", []):
                        if market.get("closed"):
                            continue

                        # extract token id early so we can honour skip_ids
                        tokens = market.get("clobTokenIds")
                        if isinstance(tokens, str):
                            try:
                                tokens = json.loads(tokens)
                            except Exception:
                                tokens = None

                        if not (isinstance(tokens, list) and tokens):
                            continue
                        market_id = str(tokens[0]).strip()
                        if market_id in skip_ids:
                            continue

                        # price filters - try multiple field names for compatibility
                        current_price = float(market.get("lastTradePrice") or market.get("last_price") or market.get("mid_price") or 0)
                        # Debug: show available price fields
                        if not current_price:
                            price_fields = {k: v for k, v in market.items() if 'price' in k.lower()}
                            if price_fields:
                                print(f"[{self.__class__.__name__}] Available price fields: {price_fields}")
                        print(f"[{self.__class__.__name__}] Market {market_id} has price: {current_price} (lastTradePrice={market.get('lastTradePrice')})")
                        if current_price < self.PRICE_FLOOR or current_price > self.PRICE_CEILING:
                            print(f"[{self.__class__.__name__}] Price {current_price} outside bounds [{self.PRICE_FLOOR}, {self.PRICE_CEILING}]")
                            continue

                        # allow subclass to adjust anchor/keyword if needed
                        anchor, keyword = self._resolve_keyword(
                            anchor, event, market, keyword, matched_alias
                        )

                        # build searchable text
                        full_text = (
                            f"{event.get('title','')} {market.get('groupItemTitle','')} "
                            f"{market.get('title','')} {market.get('question','')}"
                        ).strip()

                        valid_strike = self.extract_strike(full_text, anchor)
                        if valid_strike is None:
                            continue

                        # volume requirement
                        volume = float(market.get("volume", 0) or 0)
                        if volume == 0:
                            volume = float(market.get("liquidity", 0) or 0)
                        if volume == 0:
                            volume = float(market.get("tradingVolume", 0) or 0)

                        if volume < self.MIN_VOLUME:
                            continue

                        if volume > highest_volume:
                            highest_volume = volume
                            market_name = (
                                f"{event.get('title','')} - {market.get('groupItemTitle','')}".strip()
                            )
                            if not market_name.strip(" -"):
                                market_name = market.get("question")

                            best_market = MarketData(
                                market_id=market_id,
                                asset_type=f"{self.get_topic_type()}::{keyword}",
                                strike_price=valid_strike,
                                question=market.get("question"),
                                market_name=market_name,
                                initial_price=current_price,
                                volume=volume,
                            )
                            print(f"[{self.__class__.__name__}] Created MarketData: id={market_id}, initial_price={current_price}, strike={valid_strike}, volume={volume}")

            except Exception as e:
                print(f"[{self.__class__.__name__}] Scan error on page {page}: {e}")
                break

        return best_market

    # abstract hooks that concrete hunters must implement
    @abstractmethod
    def extract_strike(self, text: str, anchor: float) -> Optional[float]:
        pass

    @abstractmethod
    def get_live_truth(self, market: MarketData) -> Optional[float]:
        pass

    @abstractmethod
    def get_search_aliases(self) -> list:
        """Return aliases used when matching events/titles for this hunter."""
        pass

    @abstractmethod
    def hunt(self, skip_ids: list = None) -> Optional[MarketData]:
        """Hunt for a market matching this hunter's domain.
        
        The skip_ids parameter allows the engine to exclude markets currently in cooldown,
        forcing the hunter to explore alternative opportunities (e.g., Bitcoin 70k instead of 68k).

        Args:
            skip_ids: List of market_ids to skip (in cooldown). If None, defaults to [].

        Returns:
            A MarketData object with market details, or None if no suitable market is found.
        """
        pass

    @abstractmethod
    def get_anchor_value(self) -> Optional[float]:
        """Fetch the current anchor value from the external API.

        Returns:
            The anchor value (e.g., BTC price, temperature, Fed rate) or None on failure.
        """
        pass

    def get_topic_type(self) -> str:
        """Return the topic type identifier for this hunter.

        Examples: "Crypto", "Weather", "Economy"
        """
        raise NotImplementedError("Subclasses should override get_topic_type()")

    # ------------------------------------------------------------------
    # extension hooks
    # ------------------------------------------------------------------
    def _resolve_keyword(
        self,
        anchor: float,
        event: Dict[str, Any],
        market: Dict[str, Any],
        current_keyword: str,
        matched_alias: str,
    ) -> tuple[float, str]:
        """Optional hook called during scanning when an alias matches.

        Sometimes the alias that triggered the match may imply a different
        "canonical" keyword or require an anchor value other than the one
        originally passed to :meth:`_scan_polymarket`.  The default
        implementation simply returns ``(anchor, current_keyword)`` unchanged.

        Subclasses may override this method to perform more advanced mapping
        (see :class:`EconomyHunter` for an example).

        Args:
            anchor: the anchor value that was provided to ``_scan_polymarket``.
            event: the raw event dict from Gamma API.
            market: the raw market dict inside the event.
            current_keyword: the keyword argument passed to ``_scan_polymarket``.
            matched_alias: the alias string that satisfied the title/slug check.

        Returns:
            A pair ``(new_anchor, new_keyword)`` that will be used going
            forward in the scan loop.  Returning ``(anchor, current_keyword)``
            leaves the values unchanged.
        """
        # by default, do nothing
        return anchor, current_keyword

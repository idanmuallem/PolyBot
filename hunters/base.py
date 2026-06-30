"""
Base classes for all market hunters.

A Hunter discovers Polymarket markets matching a specific asset domain
(Crypto, Weather, Economy) and surfaces the best candidate for evaluation.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import json
from curl_cffi import requests as crequests
from core.models import MarketData, PRICE_FLOOR, PRICE_CEILING


class BaseHunter(ABC):
    """Abstract interface every hunter must satisfy."""

    @abstractmethod
    def hunt(self, skip_ids: list = None, add_cooldown_func=None) -> Optional[MarketData]:
        """Return the best matching market, or None if nothing qualifies.

        Args:
            skip_ids: Market IDs currently in cooldown — these are skipped.
            add_cooldown_func: Callback to register a rejected market_id.
        """

    @abstractmethod
    def get_anchor_value(self) -> Optional[float]:
        """Return the current real-world reference value (price, rate, temp, …)."""

    @abstractmethod
    def get_live_truth(self, market: MarketData) -> Optional[float]:
        """Return the live external value relevant to *market*."""

    @abstractmethod
    def get_topic_type(self) -> str:
        """Short domain identifier, e.g. 'Crypto', 'Weather', 'Economy'."""


class BasePolymarketHunter(BaseHunter):
    """Shared scanning logic for hunters that query the Polymarket Gamma API.

    Concrete subclasses supply three domain-specific hooks:
      - :meth:`extract_strike`     – parse a numeric strike from arbitrary text
      - :meth:`get_search_aliases` – keywords used to match event titles/slugs
      - :meth:`_resolve_keyword`   – optional anchor/keyword adjustment per match
    """

    POLYMARKET_BASE = "https://gamma-api.polymarket.com/events"

    PRICE_FLOOR = PRICE_FLOOR
    PRICE_CEILING = PRICE_CEILING
    MIN_VOLUME = 50_000

    def __init__(self, polymarket_base: str = POLYMARKET_BASE):
        self.polymarket_base = polymarket_base

    def _scan_polymarket(
        self,
        anchor: float,
        keyword: str,
        skip_ids: list = None,
        max_pages: int = 5,
        required_keywords: list = None,
        add_cooldown_func=None,
    ) -> Optional[MarketData]:
        """Paginate the Polymarket events API and return the highest-volume match.

        Args:
            anchor: Real-world reference value used for strike validation.
            keyword: Primary query term sent to the API.
            skip_ids: Market IDs to ignore (cooldown cache).
            max_pages: Page budget for the scan.
            required_keywords: Extra strings that must appear in event title/slug.
            add_cooldown_func: Callback to register rejected market IDs.
        """
        if skip_ids is None:
            skip_ids = []
        if required_keywords is None:
            required_keywords = []

        aliases = [k.lower() for k in self.get_search_aliases()]
        if keyword.lower() not in aliases:
            aliases.append(keyword.lower())

        best_market = None
        highest_volume = 0.0
        tag = self.__class__.__name__

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

                    if required_keywords and not all(
                        kw.lower() in title or kw.lower() in slug
                        for kw in required_keywords
                    ):
                        continue

                    matched_alias = next(
                        (a for a in aliases if a in title or a in slug), None
                    )
                    if not matched_alias:
                        continue

                    for market in event.get("markets", []):
                        if market.get("closed"):
                            continue

                        tokens = market.get("clobTokenIds")
                        if isinstance(tokens, str):
                            try:
                                tokens = json.loads(tokens)
                            except Exception:
                                tokens = None
                        if not (isinstance(tokens, list) and tokens):
                            continue

                        market_id = str(tokens[0]).strip()
                        no_market_id = str(tokens[1]).strip() if len(tokens) > 1 else None

                        if market_id in skip_ids:
                            print(f"[{tag}] SKIP cooldown | id={market_id}")
                            continue

                        current_price = float(
                            market.get("lastTradePrice")
                            or market.get("last_price")
                            or market.get("mid_price")
                            or 0
                        )
                        if current_price < self.PRICE_FLOOR or current_price > self.PRICE_CEILING:
                            if add_cooldown_func:
                                add_cooldown_func(market_id)
                            continue

                        anchor, keyword = self._resolve_keyword(
                            anchor, event, market, keyword, matched_alias
                        )

                        full_text = " ".join(filter(None, [
                            event.get("title"),
                            market.get("groupItemTitle"),
                            market.get("title"),
                            market.get("question"),
                        ]))

                        valid_strike = self.extract_strike(full_text, anchor)
                        if valid_strike is None:
                            print(f"[{tag}] REJECT no-strike | {full_text[:80]} | id={market_id}")
                            if add_cooldown_func:
                                add_cooldown_func(market_id)
                            continue

                        volume = float(
                            market.get("volume")
                            or market.get("liquidity")
                            or market.get("tradingVolume")
                            or 0
                        )
                        if volume < self.MIN_VOLUME:
                            if add_cooldown_func:
                                add_cooldown_func(market_id)
                            continue

                        if volume > highest_volume:
                            highest_volume = volume
                            market_name = (
                                f"{event.get('title', '')} - {market.get('groupItemTitle', '')}".strip(" -")
                                or market.get("question", "unknown")
                            )
                            expiry_date = (
                                market.get("endDateIso")
                                or market.get("endDate")
                                or event.get("endDate")
                                or None
                            )
                            best_market = MarketData(
                                market_id=market_id,
                                asset_type=f"{self.get_topic_type()}::{keyword}",
                                strike_price=valid_strike,
                                question=market.get("question"),
                                market_name=market_name,
                                initial_price=current_price,
                                volume=volume,
                                expiry_date=expiry_date,
                                no_market_id=no_market_id,
                            )
                            print(
                                f"[{tag}] SELECT | {market_name} | "
                                f"id={market_id} | price={current_price:.3f} | "
                                f"strike={valid_strike} | volume={volume:,.0f}"
                            )

            except Exception as e:
                print(f"[{tag}] Scan error on page {page}: {str(e).encode('ascii', errors='replace').decode()}")
                break

        return best_market

    @abstractmethod
    def extract_strike(self, text: str, anchor: float) -> Optional[float]:
        """Parse a numeric strike value from *text* given the current *anchor*."""

    @abstractmethod
    def get_search_aliases(self) -> list:
        """Keywords matched against event titles and slugs during scanning."""

    def _resolve_keyword(
        self,
        anchor: float,
        event: Dict[str, Any],
        market: Dict[str, Any],
        current_keyword: str,
        matched_alias: str,
    ) -> tuple[float, str]:
        """Hook called when an alias matches; returns an (anchor, keyword) pair.

        Override to remap aliases to canonical keywords or fetch a domain-specific
        anchor. Default implementation returns the inputs unchanged.
        """
        return anchor, current_keyword

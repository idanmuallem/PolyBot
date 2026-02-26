import json
from typing import List, Dict, Any

from curl_cffi import requests as crequests

class PolymarketClient:
    """Wrapper around the Polymarket gamma‑api for event searching."""

    BASE_URL = "https://gamma-api.polymarket.com/events"

    def search_events(
        self,
        query: str,
        limit: int = 100,
        offset: int = 0,
        order: str = "volume",
        ascending: str = "false",
    ) -> List[Dict[str, Any]]:
        """Fetch a page of events matching the query.

        The caller is responsible for paginating by incrementing ``offset``.
        The parameters mirror the original hunters' usage so that the
        ``order``/``ascending`` pair is preserved exactly.
        """
        params = {
            "active": "true",
            "closed": "false",
            "limit": limit,
            "offset": offset,
            "query": query,
            "order": order,
            "ascending": ascending,
        }
        resp = crequests.get(
            self.BASE_URL,
            params=params,
            impersonate="chrome120",
            timeout=15,
        )
        if resp.status_code != 200:
            return []
        events = resp.json()
        if not events:
            return []
        return events

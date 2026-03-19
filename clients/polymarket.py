import json
from typing import List, Dict, Any

from curl_cffi import requests as crequests

try:
    from py_clob_client.client import ClobClient  # type: ignore[reportMissingImports]
    from py_clob_client.clob_types import AssetType, BalanceAllowanceParams  # type: ignore[reportMissingImports]
    CLOB_IMPORT_OK = True
except Exception:
    ClobClient = Any  # type: ignore
    AssetType = Any  # type: ignore
    BalanceAllowanceParams = Any  # type: ignore
    CLOB_IMPORT_OK = False

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

    def get_proxy_balance(self, proxy_address: str, private_key: str) -> float:
        """Fetch proxy wallet USDC balance from Polymarket CLOB API.

        Raises an exception when credentials are missing or a balance cannot be
        resolved, so callers can show an explicit connection error state.
        """
        if not proxy_address or not private_key:
            raise ValueError("Missing POLY_ADDRESS and/or POLYGON_PRIVATE_KEY")
        if not CLOB_IMPORT_OK:
            raise RuntimeError("py-clob-client is not available")

        temp_client = ClobClient(
            host="https://clob.polymarket.com",
            chain_id=137,
            key=private_key,
            funder=proxy_address,
            signature_type=1,
        )
        if hasattr(temp_client, "create_or_derive_api_key"):
            creds = temp_client.create_or_derive_api_key()
        else:
            creds = temp_client.create_or_derive_api_creds()

        client = ClobClient(
            host="https://clob.polymarket.com",
            chain_id=137,
            key=private_key,
            creds=creds,
            funder=proxy_address,
            signature_type=1,
        )

        if hasattr(client, "get_balance_allowance"):
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            resp = client.get_balance_allowance(params=params)
            if isinstance(resp, dict):
                balance_section = resp.get("balance") if isinstance(resp.get("balance"), dict) else resp
                for key in ("usdc", "USDC", "available", "amount", "balance"):
                    if key in balance_section:
                        raw_balance = float(balance_section[key])
                        # Polymarket balance responses are commonly 6-decimal fixed-point integers.
                        return raw_balance / 1_000_000.0 if raw_balance > 1_000_000 else raw_balance

        if hasattr(client, "get_balance"):
            resp = client.get_balance()
            if isinstance(resp, dict):
                for key in ("usdc", "USDC", "available_balance", "available", "amount", "balance"):
                    if key in resp:
                        return float(resp[key])

        raise RuntimeError("Unable to resolve balance from Polymarket API")

    def get_balance(self, proxy_address: str, private_key: str) -> float:
        """Compatibility alias for callers expecting get_balance()."""
        return self.get_proxy_balance(proxy_address=proxy_address, private_key=private_key)

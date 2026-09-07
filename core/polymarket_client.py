"""PolymarketClient: thin HTTP wrapper for Gamma API event search and CLOB
balance fetching.

Split out of polymarket.py (see git history) because that module also pulls
in PolymarketScannerHunter's dependencies — hunters.crypto (ccxt, ~41MB RSS)
and brains.base -> brains.pricing_engine (scipy, ~58.6MB RSS) — at module
scope. PolymarketClient itself has no dependency on either; it only needs
curl_cffi and py_clob_client. Any caller that just wants the HTTP/balance
client (e.g. ui/dashboard.py's live-balance fetch) should import it from
here instead of from polymarket.py, so it doesn't pay for the hunting/pricing
import chain it never uses.
"""

from typing import Any, Dict, List

from curl_cffi import requests as crequests

try:
    from py_clob_client.client import ClobClient          # type: ignore[reportMissingImports]
    from py_clob_client.clob_types import AssetType, BalanceAllowanceParams  # type: ignore
    CLOB_IMPORT_OK = True
except Exception:
    ClobClient = Any           # type: ignore
    AssetType = Any            # type: ignore
    BalanceAllowanceParams = Any  # type: ignore
    CLOB_IMPORT_OK = False


class PolymarketClient:
    """Wrapper around the Polymarket gamma-api for event searching."""

    BASE_URL = "https://gamma-api.polymarket.com/events"

    def get_multi_outcome_events(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Fetch active events with no keyword filter, for strategies (e.g.
        EventSumStrategy) that need to see whole events — including all of
        their outcome markets — rather than search by a single keyword like
        the domain hunters do.
        """
        params = {
            "active": "true",
            "closed": "false",
            "limit": limit,
            "offset": offset,
            "order": "volume",
            "ascending": "false",
        }
        resp = crequests.get(self.BASE_URL, params=params, impersonate="chrome120", timeout=15)
        if resp.status_code != 200:
            return []
        events = resp.json()
        return events if events else []

    def get_proxy_balance(self, proxy_address: str, private_key: str) -> float:
        """Fetch proxy wallet USDC balance from Polymarket CLOB API.

        Raises an exception when credentials are missing or a balance cannot be resolved.
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
        creds = (
            temp_client.create_or_derive_api_key()
            if hasattr(temp_client, "create_or_derive_api_key")
            else temp_client.create_or_derive_api_creds()
        )

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
                        raw = float(balance_section[key])
                        return raw / 1_000_000.0 if raw > 1_000_000 else raw

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

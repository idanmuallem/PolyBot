"""
Polymarket module: API client + market scanner/coordinator.

- PolymarketClient   : thin HTTP wrapper for Gamma API event search and CLOB balance fetching.
- PolymarketScannerHunter : coordinator that runs discovery and pricing passes across hunters.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

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

from brains.base import calculate_tte
from core.models import MarketData
from hunters.crypto import CryptoHunter


# ---------------------------------------------------------------------------
# PolymarketClient
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# PolymarketScannerHunter
# ---------------------------------------------------------------------------

class PolymarketScannerHunter:
    """Coordinator hunter that runs one complete discovery + pricing pass."""

    _COOLDOWN_SECONDS = 120

    def __init__(self, bridge, executor, config, hunters: Optional[list] = None):
        self.bridge = bridge
        self.executor = executor
        self.config = config
        self.hunters = hunters or [CryptoHunter()]
        self.min_ev = float(config.min_ev)
        self.seen_markets: dict = {}

    # ------------------------------------------------------------------
    # Cooldown cache
    # ------------------------------------------------------------------

    # TODO(log-noise): [CACHE]/[SCANNER] fire on every loop tick per hunter
    # and aren't part of the DB-backed structured log flow — consider gating
    # behind a log-level flag instead of always-on stdout prints.
    def _get_active_seen_ids(self) -> List[str]:
        now = time.time()
        expired = [mid for mid, ts in self.seen_markets.items() if now - ts >= self._COOLDOWN_SECONDS]
        for mid in expired:
            del self.seen_markets[mid]
        print(f"[CACHE] Active cooldowns: {len(self.seen_markets)} | Expired: {len(expired)}")
        return list(self.seen_markets.keys())

    def add_to_cooldown(self, market_id: str):
        if market_id:
            self.seen_markets[str(market_id)] = time.time()
            print(f"[CACHE] Added {market_id} to cooldown.")


    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def get_active_markets(self, log_func) -> Tuple[Optional[MarketData], Optional[object]]:
        """Run one discovery pass across all hunters with TTE safety filters."""
        skip_ids = self._get_active_seen_ids()
        min_tte_days = float(self.config.min_tte_minutes) / (24.0 * 60.0)

        for hunter in self.hunters:
            hunter_name = hunter.__class__.__name__
            print(f"[SCANNER] Trying {hunter_name}... (skipping {len(skip_ids)} cooldown markets)")
            candidate_market = hunter.hunt(
                skip_ids=skip_ids,
                add_cooldown_func=self.add_to_cooldown,
            )

            if not candidate_market:
                continue

            expiry_hint = (
                getattr(candidate_market, "expiry_date", None)
                or getattr(candidate_market, "question", None)
                or getattr(candidate_market, "market_name", None)
            )
            tte_days = calculate_tte(expiry_hint)

            if tte_days < min_tte_days or tte_days > float(self.config.max_tte_days):
                log_func(
                    "FILTERED",
                    candidate_market.asset_type,
                    candidate_market.market_id,
                    {
                        "market_name": candidate_market.market_name,
                        "reason": "TTE out of bounds",
                        "tte_days": round(tte_days, 4),
                        "min_tte_minutes": self.config.min_tte_minutes,
                        "max_tte_days": self.config.max_tte_days,
                    },
                )
                self.add_to_cooldown(candidate_market.market_id)
                continue

            return candidate_market, hunter

        return None, None


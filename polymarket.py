"""
Polymarket module: market scanner/coordinator.

- PolymarketScannerHunter : coordinator that runs discovery and pricing passes across hunters.

PolymarketClient (the HTTP wrapper for Gamma API event search and CLOB
balance fetching) moved to core/polymarket_client.py — it doesn't need
CryptoHunter/scipy-pricing, so callers that only want it (e.g. the dashboard)
shouldn't have to import this whole hunting/pricing module to get it. Import
it from core.polymarket_client instead.
"""

import time
from typing import List, Optional, Tuple

from brains.base import calculate_tte
from core.models import MarketData
from hunters.crypto import CryptoHunter


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


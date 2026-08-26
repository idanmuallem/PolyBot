"""Paper trading adapter — wraps pm_trader.Engine for PolyBot integration.

Translates between PolyBot's MarketData/Position types and the paper
trading engine's slug-based API.  Best-effort: if the Engine fails,
methods return graceful defaults so the pipeline continues.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.models import Position

logger = logging.getLogger(__name__)

# Lazy import — if polymarket-paper-trader isn't installed, paper trading
# degrades to log-only (the existing DRY-RUN behavior).
try:
    from pm_trader.engine import Engine as PaperEngine
    PAPER_ENGINE_AVAILABLE = True
except ImportError:
    PaperEngine = None
    PAPER_ENGINE_AVAILABLE = False


class PaperAdapter:
    """Adapter between PolyBot execution and the pm_trader paper engine.

    Maintains a token_id → (slug, outcome) mapping so sell_position()
    can resolve the slug the Engine needs.
    """

    def __init__(self, data_dir: Optional[str] = None, initial_balance: Optional[float] = None):
        self.engine: Optional[object] = None
        self._token_map: Dict[str, Tuple[str, str]] = {}  # token_id → (slug, outcome)
        self._token_map_path: Optional[Path] = None

        # Positions cache — Engine.get_portfolio() makes one live HTTP call
        # per open position, and get_positions() can be called many times
        # per pipeline tick (see PortfolioManager._refresh_portfolio callers),
        # so cache briefly rather than refetch on every call.
        self._positions_cache: List[Position] = []
        self._positions_cache_at: float = 0.0
        self._positions_cache_ttl: float = 30.0  # seconds

        if not PAPER_ENGINE_AVAILABLE:
            logger.warning("[PAPER] polymarket-paper-trader not installed — paper fills disabled")
            return

        if data_dir is None:
            # Co-locate with trades.db. Default matches ui/dashboard.py's
            # TRADES_DB_PATH default (/app/trades.db) so this resolves to
            # /app/paper_trading regardless of process CWD — the deploy
            # volume mount targets that exact path (see deploy.yml).
            data_dir = os.getenv("TRADES_DB_PATH", "/app/trades.db")
            data_dir = str(Path(data_dir).parent / "paper_trading")

        if initial_balance is None:
            initial_balance = float(os.getenv("PAPER_BALANCE_USD", "1000.0"))

        try:
            path = Path(data_dir)
            path.mkdir(parents=True, exist_ok=True)
            self.engine = PaperEngine(path)

            # Init account if not already initialized
            try:
                self.engine.get_account()
            except Exception:
                self.engine.init_account(initial_balance)
                logger.info(f"[PAPER] Initialized paper account with ${initial_balance:.2f}")

            # token_id → (slug, outcome) isn't tracked by the Engine itself,
            # so persist it alongside the Engine's own data dir and reload it
            # here — otherwise a process restart loses the mapping and paper
            # positions opened before the restart become unsellable.
            self._token_map_path = path / "token_map.json"
            self._load_token_map()

            logger.info(f"[PAPER] Paper trading engine ready at {path}")
        except Exception as exc:
            logger.error(f"[PAPER] Failed to initialize paper engine: {exc}")
            self.engine = None

    def _load_token_map(self):
        """Load persisted token_id → (slug, outcome) mapping from disk."""
        if self._token_map_path and self._token_map_path.exists():
            try:
                with open(self._token_map_path, "r") as f:
                    raw = json.load(f)
                # JSON stores tuples as lists, convert back
                self._token_map = {k: tuple(v) for k, v in raw.items()}
                logger.info(f"[PAPER] Loaded {len(self._token_map)} token mappings from disk")
            except Exception as exc:
                logger.warning(f"[PAPER] Failed to load token map: {exc}")
                self._token_map = {}

    def _save_token_map(self):
        """Persist the token_id → (slug, outcome) mapping to disk."""
        if self._token_map_path:
            try:
                with open(self._token_map_path, "w") as f:
                    json.dump(self._token_map, f)
            except Exception as exc:
                logger.warning(f"[PAPER] Failed to save token map: {exc}")

    @property
    def is_ready(self) -> bool:
        return self.engine is not None

    def execute_buy(
        self,
        slug: Optional[str],
        condition_id: Optional[str],
        token_id: str,
        side: str,
        amount_usd: float,
        no_token_id: Optional[str] = None,
    ) -> bool:
        """Record a paper buy. Returns True if the fill succeeded."""
        if not self.is_ready:
            return False

        # Resolve the slug or condition_id for the Engine
        slug_or_id = slug or condition_id
        if not slug_or_id:
            logger.warning(f"[PAPER] No slug or condition_id for token {token_id} — skipping paper fill")
            return False

        outcome = side.lower()  # "YES" → "yes", "NO" → "no"

        try:
            result = self.engine.buy(slug_or_id, outcome, amount_usd)

            # Record the token mapping so sell_position can find this later
            execution_token = token_id
            if outcome == "no" and no_token_id:
                execution_token = no_token_id
            self._token_map[execution_token] = (slug_or_id, outcome)
            self._save_token_map()

            logger.info(
                f"[PAPER] BUY {outcome.upper()} ${amount_usd:.2f} → "
                f"{result.trade.shares:.4f} shares @ ${result.trade.avg_price:.4f} "
                f"(fee=${result.trade.fee:.4f}, slippage={result.trade.slippage:.1f}bps)"
            )
            return True
        except Exception as exc:
            logger.warning(f"[PAPER] Buy failed for {slug_or_id} {outcome}: {exc}")
            return False

    def execute_sell(self, token_id: str, shares: float) -> bool:
        """Record a paper sell. Returns True if the fill succeeded."""
        if not self.is_ready:
            return False

        mapping = self._token_map.get(token_id)
        if not mapping:
            # Try to find it from the engine's open positions
            mapping = self._find_mapping_from_positions(token_id)
            if not mapping:
                logger.warning(f"[PAPER] No slug mapping for token {token_id} — skipping paper sell")
                return False

        slug_or_id, outcome = mapping

        try:
            result = self.engine.sell(slug_or_id, outcome, shares)
            logger.info(
                f"[PAPER] SELL {outcome.upper()} {shares:.4f} shares → "
                f"${result.trade.amount_usd:.2f} @ ${result.trade.avg_price:.4f}"
            )
            return True
        except Exception as exc:
            logger.warning(f"[PAPER] Sell failed for {slug_or_id} {outcome}: {exc}")
            return False

    def get_positions(self) -> List[Position]:
        """Return paper positions mapped to PolyBot's Position dataclass.

        Cached briefly: Engine.get_portfolio() makes one live HTTP call per
        open position, and this can be called several times per pipeline
        tick — see PortfolioManager._refresh_portfolio's callers.
        """
        if not self.is_ready:
            return []

        now = time.time()
        if (now - self._positions_cache_at) < self._positions_cache_ttl:
            return self._positions_cache

        try:
            portfolio = self.engine.get_portfolio()
        except Exception as exc:
            logger.warning(f"[PAPER] Failed to fetch portfolio: {exc}")
            return self._positions_cache  # stale data beats an empty list for TP/SL

        positions = []
        for p in portfolio:
            shares = float(p.get("shares", 0))
            if shares <= 0:
                continue

            avg_entry = float(p.get("avg_entry_price", 0))
            live_price = float(p.get("live_price", 0))
            current_value = float(p.get("current_value", 0))
            slug = str(p.get("market_slug", ""))
            outcome = str(p.get("outcome", ""))

            pnl_ratio = (
                (live_price - avg_entry) / avg_entry
                if avg_entry > 0 else 0.0
            )

            # Resolve token_id from our mapping (reverse lookup)
            token_id = self._find_token_for_slug(slug, outcome)

            positions.append(Position(
                market_id=slug,
                token_id=token_id or slug,
                initial_price=avg_entry,
                current_price=live_price,
                shares=shares,
                value=current_value,
                pnl_ratio=pnl_ratio,
                side=outcome.upper(),
                live_ev=pnl_ratio,
            ))

        self._positions_cache = positions
        self._positions_cache_at = now
        return positions

    def get_cash_balance(self) -> float:
        """Return available cash in the paper account."""
        if not self.is_ready:
            return float(os.getenv("PAPER_BALANCE_USD", "1000.0"))

        try:
            balance_info = self.engine.get_balance()
            return float(balance_info.get("cash", 0.0))
        except Exception as exc:
            logger.warning(f"[PAPER] Failed to fetch balance: {exc}")
            return 0.0

    def get_total_value(self) -> float:
        """Return total account value (cash + positions)."""
        if not self.is_ready:
            return float(os.getenv("PAPER_BALANCE_USD", "1000.0"))

        try:
            balance_info = self.engine.get_balance()
            return float(balance_info.get("total_value", 0.0))
        except Exception as exc:
            logger.warning(f"[PAPER] Failed to fetch total value: {exc}")
            return 0.0

    def resolve_closed_markets(self) -> int:
        """Resolve any paper positions in markets that have closed. Returns count resolved."""
        if not self.is_ready:
            return 0

        try:
            results = self.engine.resolve_all()
            if results:
                for r in results:
                    logger.info(
                        f"[PAPER] RESOLVED {r.position.market_slug} "
                        f"{r.position.outcome} → payout=${r.payout:.2f}"
                    )
            return len(results)
        except Exception as exc:
            logger.warning(f"[PAPER] resolve_all failed: {exc}")
            return 0

    def _find_mapping_from_positions(self, token_id: str) -> Optional[Tuple[str, str]]:
        """Try to recover a mapping from the persisted token map on disk.

        If the in-memory map was lost (e.g. cleared, or a fresh instance
        that hasn't loaded yet) but the file on disk still has it, reload
        and look up directly — this must not depend on _token_map already
        containing the entry it's trying to recover (that would just search
        the same empty map it's meant to repair).
        """
        if not self.is_ready:
            return None

        if not self._token_map and self._token_map_path:
            self._load_token_map()

        return self._token_map.get(token_id)

    def _find_token_for_slug(self, slug: str, outcome: str) -> Optional[str]:
        """Reverse lookup: find token_id for a given slug + outcome."""
        for tok, (s, o) in self._token_map.items():
            if s == slug and o == outcome:
                return tok
        return None

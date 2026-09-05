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

# Rate-limit the "no slug/condition_id" skip warning per token_id so a
# market missing these fields (a genuine edge case, not the common case)
# doesn't flood the logs every time the pipeline re-evaluates it.
_SKIP_WARNING_INTERVAL_SECONDS = 3600.0  # once per hour, per token


class PaperAdapter:
    """Adapter between PolyBot execution and the pm_trader paper engine.

    Maintains a token_id → (slug, outcome) mapping so sell_position()
    can resolve the slug the Engine needs.
    """

    def __init__(self, data_dir: Optional[str] = None, initial_balance: Optional[float] = None):
        self.engine: Optional[object] = None
        self._token_map: Dict[str, Tuple[str, str]] = {}  # token_id → (slug, outcome)
        self._token_map_path: Optional[Path] = None
        self._skip_warned_at: Dict[str, float] = {}  # token_id → last "skipping paper fill" warning time

        # token_ids opened through place_limit_buy() — arbitrage's sole paper
        # entry path (see the "Limit orders" section below; execute_buy is
        # the crypto/model path's only entry point). Lets resolve_closed_markets()
        # identify which open positions are arbitrage legs without needing to
        # know anything about asset_type/strategy tagging, which lives outside
        # this class entirely (see ENABLE_ARBITRAGE kill switch).
        self._arbitrage_tokens: set = set()

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
        """Load persisted token_id → (slug, outcome) mapping (plus the
        arbitrage-tagging set) from disk."""
        if self._token_map_path and self._token_map_path.exists():
            try:
                with open(self._token_map_path, "r") as f:
                    raw = json.load(f)
                if isinstance(raw, dict) and "tokens" in raw:
                    # Current format.
                    self._token_map = {k: tuple(v) for k, v in raw.get("tokens", {}).items()}
                    self._arbitrage_tokens = set(raw.get("arbitrage_tokens", []))
                else:
                    # Legacy format written before arbitrage-tagging existed:
                    # a flat token_id -> [slug, outcome] dict with no tagging
                    # info. Any arbitrage legs it holds are untagged until
                    # their next place_limit_buy() call re-adds them — the
                    # next _save_token_map() upgrades the file either way.
                    self._token_map = {k: tuple(v) for k, v in raw.items()}
                    self._arbitrage_tokens = set()
                logger.info(f"[PAPER] Loaded {len(self._token_map)} token mappings from disk")
            except Exception as exc:
                logger.warning(f"[PAPER] Failed to load token map: {exc}")
                self._token_map = {}
                self._arbitrage_tokens = set()

    def _save_token_map(self):
        """Persist the token_id → (slug, outcome) mapping and the
        arbitrage-tagging set to disk."""
        if self._token_map_path:
            try:
                with open(self._token_map_path, "w") as f:
                    json.dump({
                        "tokens": self._token_map,
                        "arbitrage_tokens": sorted(self._arbitrage_tokens),
                    }, f)
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
            now = time.time()
            last_warned = self._skip_warned_at.get(token_id, 0.0)
            if now - last_warned >= _SKIP_WARNING_INTERVAL_SECONDS:
                self._skip_warned_at[token_id] = now
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
            # get_positions() caches for _positions_cache_ttl seconds (30s) —
            # without invalidating here, a caller that reads positions right
            # after this sell (e.g. PortfolioManager.manage_portfolio()'s
            # trailing _refresh_portfolio()) still gets the pre-sell snapshot
            # and re-attempts to sell the same, already-closed position on
            # every tick until the cache naturally expires. Forcing the next
            # get_positions() call to hit the engine fresh closes that gap.
            self._positions_cache_at = 0.0
            return True
        except Exception as exc:
            logger.warning(f"[PAPER] Sell failed for {slug_or_id} {outcome}: {exc}")
            return False

    # ------------------------------------------------------------------
    # Limit orders — used by arbitrage-group legs instead of the market-
    # order path above (execute_buy walks the book at whatever price it
    # takes to fill the full dollar amount; a limit order caps the price
    # paid instead, at the cost of possibly not filling at all). Mirrors
    # execute_buy/execute_sell's same error-handling and honesty contract:
    # best-effort, never raises out, returns a falsy value on any failure.
    # ------------------------------------------------------------------

    def place_limit_buy(
        self,
        slug: Optional[str],
        condition_id: Optional[str],
        token_id: str,
        side: str,
        amount_usd: float,
        limit_price: float,
        no_token_id: Optional[str] = None,
    ) -> Optional[dict]:
        """Place a GTC limit buy for one leg. Returns the pm_trader order
        dict (carries market_condition_id/outcome, needed for the caller's
        later position-shares polling) on success, None if it couldn't even
        be placed — same honesty contract as execute_buy.
        """
        if not self.is_ready:
            return None

        slug_or_id = slug or condition_id
        if not slug_or_id:
            now = time.time()
            last_warned = self._skip_warned_at.get(token_id, 0.0)
            if now - last_warned >= _SKIP_WARNING_INTERVAL_SECONDS:
                self._skip_warned_at[token_id] = now
                logger.warning(f"[PAPER] No slug or condition_id for token {token_id} — skipping limit order")
            return None

        outcome = side.lower()

        try:
            order = self.engine.place_limit_order(
                slug_or_id, outcome, "buy", amount_usd, limit_price, order_type="gtc",
            )

            # Record the token mapping so a subsequent sell can find this,
            # same as execute_buy — harmless if the order never fills (an
            # unused mapping entry costs nothing; only positions that
            # genuinely exist ever get looked up through it).
            execution_token = token_id
            if outcome == "no" and no_token_id:
                execution_token = no_token_id
            self._token_map[execution_token] = (slug_or_id, outcome)
            # This is arbitrage's only paper-fill path (see the module-level
            # comment above) — tag it so resolve_closed_markets() can skip
            # it entirely while ENABLE_ARBITRAGE is off.
            self._arbitrage_tokens.add(execution_token)
            self._save_token_map()

            logger.info(
                f"[PAPER] LIMIT BUY {outcome.upper()} ${amount_usd:.2f} @ "
                f"<= ${limit_price:.4f} — order id {order.get('id')}"
            )
            return order
        except Exception as exc:
            logger.warning(f"[PAPER] Limit buy placement failed for {slug_or_id} {outcome}: {exc}")
            return None

    def check_pending_limit_orders(self) -> list:
        """Advance every pending limit order one tick against the live
        order book (fills at-or-better than each order's limit price).
        Best-effort: an engine failure here must not crash the poller.
        """
        if not self.is_ready:
            return []
        try:
            return self.engine.check_orders()
        except Exception as exc:
            logger.warning(f"[PAPER] check_orders failed: {exc}")
            return []

    def get_position_shares(self, condition_id: str, outcome: str) -> float:
        """Current shares held for one (condition_id, outcome), read fresh
        from the engine's own ledger — NOT get_positions()'s 30s cache,
        since the limit-order poll loop needs up-to-date reads within a
        much shorter window than that cache is meant to serve.
        """
        if not self.is_ready:
            return 0.0
        try:
            position = self.engine.db.get_position(condition_id, outcome)
        except Exception as exc:
            logger.warning(f"[PAPER] get_position failed for {condition_id}/{outcome}: {exc}")
            return 0.0
        return float(position.shares) if position else 0.0

    def cancel_limit_order(self, order_id: int) -> bool:
        """Cancel one still-pending limit order. Returns True if a pending
        order was found and cancelled, False otherwise (including if it had
        already filled, expired, or the engine call itself failed)."""
        if not self.is_ready:
            return False
        try:
            return self.engine.cancel_limit_order(order_id) is not None
        except Exception as exc:
            logger.warning(f"[PAPER] Cancel failed for limit order {order_id}: {exc}")
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

            # Resolve token_id from our mapping (reverse lookup). A miss
            # here means the engine's own ledger (persisted independently
            # and, per the position genuinely showing up in
            # engine.get_portfolio(), definitely holding real shares) has
            # outlived this adapter's _token_map entry for it — most likely
            # a token_map.json persistence gap around a process restart
            # (see execute_buy's _save_token_map() call). Falling back to
            # the slug as a fake token_id used to make this position look
            # normal everywhere downstream, while actually being permanently
            # unsellable: execute_sell()'s own _token_map lookup can never
            # match a slug (only real token_ids are ever stored as keys),
            # so every exit attempt failed silently forever. Surface it as
            # an explicitly broken id instead — obvious in logs/dashboard,
            # and guaranteed not to collide with a real token_id — so this
            # gets noticed rather than mistaken for an ordinary position.
            token_id = self._find_token_for_slug(slug, outcome)
            if not token_id:
                now_ts = time.time()
                last_warned = self._skip_warned_at.get(f"unmapped:{slug}:{outcome}", 0.0)
                if now_ts - last_warned >= _SKIP_WARNING_INTERVAL_SECONDS:
                    self._skip_warned_at[f"unmapped:{slug}:{outcome}"] = now_ts
                    logger.warning(
                        f"[PAPER] Open position {slug}/{outcome} has no _token_map "
                        "entry — it cannot be sold until this is resolved (see "
                        "trading/paper_adapter.py's get_positions())"
                    )
                token_id = f"__unmapped__:{slug}:{outcome}"

            positions.append(Position(
                market_id=slug,
                token_id=token_id,
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

    def resolve_closed_markets(self, resolve_arbitrage: bool = True, log_func=None) -> int:
        """Resolve any paper positions in markets that have closed. Returns count resolved.

        resolve_arbitrage=False is the ENABLE_ARBITRAGE kill switch: while
        arbitrage is disabled, no arbitrage-specific position handling may
        run at all — not even resolving an already-open arbitrage leg —
        regardless of what's sitting in the paper ledger. Markets whose only
        tracked position came from place_limit_buy() (arbitrage's sole
        paper-fill path — see self._arbitrage_tokens) are skipped entirely
        in that case; non-arbitrage (crypto) positions resolve exactly as
        they always have, either way.

        log_func, when given, gets one "EXPIRED" hunt_history row per
        resolved position (see ui/data_manager.log_event) — this used to be
        a stdout-only logger.info, invisible to the dashboard, so a market
        expiring/resolving never showed up as a closed trade anywhere.
        Payload shape mirrors PortfolioManager._exit_position()'s exit rows
        (price/initial_price/shares/sold) so it flows through the same
        get_trade_stats()/get_closed_trade_deltas() aggregation — "price"
        here is the effective per-share payout (1.0 for a winning share,
        0.0 for a losing one), not a market quote.
        """
        if not self.is_ready:
            return 0

        try:
            if resolve_arbitrage:
                results = self.engine.resolve_all()
            else:
                results = self._resolve_non_arbitrage_closed_markets()
            if results:
                for r in results:
                    logger.info(
                        f"[PAPER] RESOLVED {r.position.market_slug} "
                        f"{r.position.outcome} → payout=${r.payout:.2f}"
                    )
                    if log_func is not None:
                        self._log_resolved_position(r, log_func)
            return len(results)
        except Exception as exc:
            logger.warning(f"[PAPER] resolve_all failed: {exc}")
            return 0

    def _log_resolved_position(self, resolve_result, log_func) -> None:
        position = resolve_result.position
        shares = float(position.shares or 0.0)
        payout_price = (float(resolve_result.payout) / shares) if shares else 0.0

        token_id = self._find_token_for_slug(position.market_slug, position.outcome)
        if not token_id:
            # Same "make an unmappable position visible instead of silently
            # dropping it" fallback get_positions() uses above.
            token_id = f"__unmapped__:{position.market_slug}:{position.outcome}"

        try:
            log_func("EXPIRED", "Portfolio", token_id, {
                "shares": shares,
                "price": round(payout_price, 4),
                "initial_price": round(float(position.avg_entry_price or 0.0), 4),
                "side": str(position.outcome or "").upper(),
                "market_name": position.market_question or position.market_slug,
                "payout": round(float(resolve_result.payout), 2),
                "realized_pnl": round(float(position.realized_pnl or 0.0), 4),
                "sold": True,
            })
        except Exception as exc:
            logger.warning(f"[PAPER] Failed to log resolved position {token_id}: {exc}")

    def _resolve_non_arbitrage_closed_markets(self) -> list:
        """Mirrors Engine.resolve_all()'s own closed-market loop, but skips
        any market whose open position is tagged arbitrage (see
        resolve_closed_markets()'s resolve_arbitrage=False path). Reaches
        into engine.db/engine.api the same way get_position_shares() above
        already does — there's no narrower public Engine method for
        "resolve all except these slugs".
        """
        arbitrage_slugs = {
            self._token_map[tok][0]
            for tok in self._arbitrage_tokens
            if tok in self._token_map
        }

        positions = self.engine.db.get_open_positions()
        seen_markets = set()
        results = []
        for pos in positions:
            if pos.market_condition_id in seen_markets:
                continue
            if pos.market_slug in arbitrage_slugs:
                continue
            try:
                market = self.engine.api.get_market(pos.market_slug)
                if market.closed:
                    seen_markets.add(pos.market_condition_id)
                    results.extend(self.engine.resolve_market(pos.market_slug))
            except Exception:
                continue  # transient — retry on next call, same as resolve_all()
        return results

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

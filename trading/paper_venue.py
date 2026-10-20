"""PaperVenue: the Venue adapter for paper trading, wrapping pm_trader.Engine.

Translates between the Venue port (MarketRef + side, token ids, PolyBot
Position) and pm_trader's slug/outcome API. Best-effort: an engine failure
is reported through each method's return value, never raised.

Construction needs a working pm_trader. Its data directory and starting
balance are passed in (see PaperVenue.for_wallet()); this module reads no
environment variables.

Thread affinity: pm_trader.Engine opens its sqlite3 connection with the
library default check_same_thread=True and offers no override, so every
call on a PaperVenue must come from the thread that first used it (in
production, the engine loop's thread). Do not hand its methods to
asyncio.to_thread() or another worker thread.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from core.models import Position
from trading.venue import (
    LimitOrder, MarketRef, Resolution, log_paper_buy, log_paper_sell, normalize_side, report_resolution,
)

logger = logging.getLogger(__name__)

# Rate-limit the "no slug/condition_id" skip warning per token_id so a
# market missing these fields (a genuine edge case, not the common case)
# doesn't flood the logs every time the pipeline re-evaluates it.
_SKIP_WARNING_INTERVAL_SECONDS = 3600.0  # once per hour, per token


def paper_data_dir(trades_db_path: str) -> Path:
    """Where a wallet's paper engine lives: next to its trades.db. With the
    production default (/app/trades.db) that's /app/paper_trading, which the
    deploy volume mount targets (see deploy.yml)."""
    return Path(trades_db_path).parent / "paper_trading"


class PaperVenue:
    """Venue adapter over pm_trader.Engine.

    Maintains a token_id -> (slug, outcome) map, persisted next to the
    engine's data, so sell() can find the slug the engine needs, including
    after a restart.
    """

    name = "paper"

    def __init__(self, data_dir: Path, starting_balance: float, market_api: Any = None):
        """Open (or create) the paper account in *data_dir*.

        starting_balance only applies when the account is first created.
        market_api replaces pm_trader's network client (tests).
        Raises if pm_trader is unavailable or the engine can't open.
        """
        from pm_trader.engine import Engine

        path = Path(data_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.engine = Engine(path)
        if market_api is not None:
            self.engine.api = market_api

        try:
            self.engine.get_account()
        except Exception:
            self.engine.init_account(float(starting_balance))
            logger.info(f"[PAPER] Initialized paper account with ${float(starting_balance):.2f}")

        self._token_map: Dict[str, Tuple[str, str]] = {}  # token_id -> (slug, outcome)
        # token_ids opened through place_limit_buy() -- arbitrage's sole paper
        # entry path. Lets resolve_closed_markets() skip arbitrage legs while
        # ENABLE_ARBITRAGE is off without knowing about strategy tagging.
        self._arbitrage_tokens: set = set()
        self._skip_warned_at: Dict[str, float] = {}

        # Engine.get_portfolio() makes one live HTTP call per open position,
        # and open_positions() can be called many times per tick, so cache
        # briefly. Every mutation below invalidates it (phantom-sell fix).
        self._positions_cache: List[Position] = []
        self._positions_cache_at: float = 0.0
        self._positions_cache_ttl: float = 30.0  # seconds

        self._token_map_path = path / "token_map.json"
        self._load_token_map()
        logger.info(f"[PAPER] Paper trading engine ready at {path}")

    @classmethod
    def for_wallet(cls, ctx) -> "PaperVenue":
        """The paper venue for one wallet: data next to the wallet's
        trades.db, starting balance from its trading config."""
        return cls(
            data_dir=paper_data_dir(ctx.db_path),
            starting_balance=float(ctx.config.paper_balance_usd),
        )

    # ------------------------------------------------------------------
    # Venue port
    # ------------------------------------------------------------------

    def cash(self) -> float:
        try:
            return float(self.engine.get_account().cash)
        except Exception as exc:
            logger.warning(f"[PAPER] Failed to fetch balance: {exc}")
            return 0.0

    def open_positions(self) -> List[Position]:
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
            pnl_ratio = (live_price - avg_entry) / avg_entry if avg_entry > 0 else 0.0

            positions.append(Position(
                market_id=slug,
                token_id=self._token_for_position(slug, outcome),
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

    def buy(
        self, market: MarketRef, side: str, usd: float,
        price: Optional[float] = None, log_func: Optional[Callable] = None,
    ) -> bool:
        log_paper_buy(log_func, market.token_for(side), normalize_side(side), usd, price)
        slug_or_id = self._slug_or_id(market, "skipping paper fill")
        if not slug_or_id:
            return False
        outcome = normalize_side(side).lower()

        try:
            result = self.engine.buy(slug_or_id, outcome, float(usd))
        except Exception as exc:
            logger.warning(f"[PAPER] Buy failed for {slug_or_id} {outcome}: {exc}")
            return False

        self._remember_token(market.token_for(side), slug_or_id, outcome)
        self._invalidate_positions()
        logger.info(
            f"[PAPER] BUY {outcome.upper()} ${float(usd):.2f} -> "
            f"{result.trade.shares:.4f} shares @ ${result.trade.avg_price:.4f} "
            f"(fee=${result.trade.fee:.4f}, slippage={result.trade.slippage:.1f}bps)"
        )
        return True

    def sell(
        self, token_id: str, shares: float,
        price: Optional[float] = None, log_func: Optional[Callable] = None,
    ) -> bool:
        log_paper_sell(log_func, str(token_id), shares, price)
        mapping = self._token_map.get(str(token_id))
        if not mapping:
            logger.warning(f"[PAPER] No slug mapping for token {token_id} -- skipping paper sell")
            return False
        slug_or_id, outcome = mapping

        try:
            result = self.engine.sell(slug_or_id, outcome, float(shares))
        except Exception as exc:
            logger.warning(f"[PAPER] Sell failed for {slug_or_id} {outcome}: {exc}")
            return False

        # Without this, a caller reading positions right after the sale
        # (PortfolioManager's trailing refresh) sees the pre-sale snapshot
        # and sells the same, already-closed position again every tick.
        self._invalidate_positions()
        logger.info(
            f"[PAPER] SELL {outcome.upper()} {float(shares):.4f} shares -> "
            f"${result.trade.amount_usd:.2f} @ ${result.trade.avg_price:.4f}"
        )
        return True

    # Limit orders: arbitrage legs use these instead of buy(). buy() walks
    # the book at whatever price fills the full amount; a limit order caps
    # the price paid, at the cost of possibly not filling at all.

    def place_limit_buy(
        self, market: MarketRef, side: str, usd: float, limit_price: float,
    ) -> Optional[LimitOrder]:
        slug_or_id = self._slug_or_id(market, "skipping limit order")
        if not slug_or_id:
            return None
        outcome = normalize_side(side).lower()

        try:
            order = self.engine.place_limit_order(
                slug_or_id, outcome, "buy", float(usd), float(limit_price), order_type="gtc",
            )
        except Exception as exc:
            logger.warning(f"[PAPER] Limit buy placement failed for {slug_or_id} {outcome}: {exc}")
            return None

        token = market.token_for(side)
        # Recorded even if the order never fills: an unused mapping costs
        # nothing; only positions that exist are ever looked up through it.
        self._arbitrage_tokens.add(token)
        self._remember_token(token, slug_or_id, outcome)
        logger.info(
            f"[PAPER] LIMIT BUY {outcome.upper()} ${float(usd):.2f} @ "
            f"<= ${float(limit_price):.4f} -- order id {order.get('id')}"
        )
        return LimitOrder(id=order.get("id"), token_id=token, details=dict(order))

    def poll_limit_orders(self, orders: Sequence[LimitOrder]) -> Dict[Any, float]:
        """One check_orders() pass (pm_trader fills with fill-and-kill
        semantics: the first check that finds liquidity within the limit
        fills and closes the order), then the shares held per order's
        token, read fresh from the engine's ledger rather than the cache."""
        try:
            if self.engine.check_orders():
                self._invalidate_positions()
        except Exception as exc:
            logger.warning(f"[PAPER] check_orders failed: {exc}")

        filled: Dict[Any, float] = {}
        for order in orders:
            condition_id = order.details.get("market_condition_id")
            outcome = order.details.get("outcome")
            try:
                position = self.engine.db.get_position(condition_id, outcome)
                filled[order.id] = float(position.shares) if position and not position.is_resolved else 0.0
            except Exception as exc:
                logger.warning(f"[PAPER] get_position failed for {condition_id}/{outcome}: {exc}")
                filled[order.id] = 0.0
        return filled

    def cancel_limit_order(self, order: LimitOrder) -> bool:
        try:
            return self.engine.cancel_limit_order(order.id) is not None
        except Exception as exc:
            logger.warning(f"[PAPER] Cancel failed for limit order {order.id}: {exc}")
            return False

    def resolve_closed_markets(
        self,
        resolve_arbitrage: bool = True,
        log_func: Optional[Callable] = None,
        on_resolved: Optional[Callable[[dict], Any]] = None,
    ) -> int:
        """Settle paper positions in closed markets. Returns count resolved.

        resolve_arbitrage=False is the ENABLE_ARBITRAGE kill switch: markets
        whose position came from place_limit_buy() are skipped entirely.
        Each settled position goes to log_func/on_resolved via
        report_resolution() (trading/venue.py).
        """
        try:
            # pm_trader zeroes a position's shares when it resolves it, so
            # snapshot them first.
            shares_before = {
                (p.market_condition_id, p.outcome): float(p.shares)
                for p in self.engine.db.get_open_positions()
            }
            if resolve_arbitrage:
                results = self.engine.resolve_all()
            else:
                results = self._resolve_non_arbitrage_closed_markets()
        except Exception as exc:
            logger.warning(f"[PAPER] resolve_all failed: {exc}")
            return 0

        if results:
            self._invalidate_positions()
        for r in results:
            logger.info(
                f"[PAPER] RESOLVED {r.position.market_slug} "
                f"{r.position.outcome} -> payout=${r.payout:.2f}"
            )
            report_resolution(self._resolution(r, shares_before), log_func, on_resolved)
        return len(results)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _slug_or_id(self, market: MarketRef, what: str) -> Optional[str]:
        slug_or_id = market.slug or market.condition_id
        if not slug_or_id:
            key = market.yes_token_id
            now = time.time()
            if now - self._skip_warned_at.get(key, 0.0) >= _SKIP_WARNING_INTERVAL_SECONDS:
                self._skip_warned_at[key] = now
                logger.warning(f"[PAPER] No slug or condition_id for token {key} -- {what}")
        return slug_or_id

    def _invalidate_positions(self) -> None:
        self._positions_cache_at = 0.0

    def _remember_token(self, token_id: str, slug_or_id: str, outcome: str) -> None:
        self._token_map[str(token_id)] = (slug_or_id, outcome)
        self._save_token_map()

    def _token_for_position(self, slug: str, outcome: str) -> str:
        """Token id for an engine position, via the token map.

        A miss means the engine's own ledger has outlived this venue's map
        entry (most likely a token_map.json persistence gap around a
        restart). Return an explicitly broken id rather than the slug, so
        the position is visibly unsellable instead of looking normal while
        every sell attempt silently fails.
        """
        for tok, (s, o) in self._token_map.items():
            if o == outcome and s == slug:
                return tok

        warn_key = f"unmapped:{slug}:{outcome}"
        now = time.time()
        if now - self._skip_warned_at.get(warn_key, 0.0) >= _SKIP_WARNING_INTERVAL_SECONDS:
            self._skip_warned_at[warn_key] = now
            logger.warning(
                f"[PAPER] Open position {slug}/{outcome} has no _token_map "
                "entry -- it cannot be sold until this is resolved (see "
                "PaperVenue._token_for_position())"
            )
        return f"__unmapped__:{slug}:{outcome}"

    def _resolution(self, resolve_result, shares_before: dict) -> Resolution:
        position = resolve_result.position
        shares = shares_before.get(
            (position.market_condition_id, position.outcome), float(position.shares or 0.0),
        )
        return Resolution(
            token_id=self._token_for_position(position.market_slug, position.outcome),
            side=str(position.outcome or "").upper(),
            shares=shares,
            payout_price=(float(resolve_result.payout) / shares) if shares else 0.0,
            entry_price=float(position.avg_entry_price or 0.0),
            market_name=position.market_question or position.market_slug,
            realized_pnl=float(position.realized_pnl or 0.0),
        )

    def _resolve_non_arbitrage_closed_markets(self) -> list:
        """Engine.resolve_all()'s closed-market loop, minus any market whose
        position is tagged arbitrage. Reaches into engine.db/engine.api:
        there's no narrower public Engine method for this."""
        arbitrage_slugs = {
            self._token_map[tok][0]
            for tok in self._arbitrage_tokens
            if tok in self._token_map
        }

        seen_markets = set()
        results = []
        for pos in self.engine.db.get_open_positions():
            if pos.market_condition_id in seen_markets:
                continue
            if pos.market_slug in arbitrage_slugs or pos.market_condition_id in arbitrage_slugs:
                continue
            try:
                market = self.engine.api.get_market(pos.market_slug)
                if market.closed:
                    seen_markets.add(pos.market_condition_id)
                    results.extend(self.engine.resolve_market(pos.market_slug))
            except Exception:
                continue  # transient -- retry on next call, same as resolve_all()
        return results

    def _load_token_map(self) -> None:
        if not self._token_map_path.exists():
            return
        try:
            with open(self._token_map_path, "r") as f:
                raw = json.load(f)
            if isinstance(raw, dict) and "tokens" in raw:
                self._token_map = {k: tuple(v) for k, v in raw.get("tokens", {}).items()}
                self._arbitrage_tokens = set(raw.get("arbitrage_tokens", []))
            else:
                # Legacy format (before arbitrage tagging): flat token -> [slug, outcome].
                self._token_map = {k: tuple(v) for k, v in raw.items()}
                self._arbitrage_tokens = set()
            logger.info(f"[PAPER] Loaded {len(self._token_map)} token mappings from disk")
        except Exception as exc:
            logger.warning(f"[PAPER] Failed to load token map: {exc}")
            self._token_map = {}
            self._arbitrage_tokens = set()

    def _save_token_map(self) -> None:
        try:
            with open(self._token_map_path, "w") as f:
                json.dump({
                    "tokens": self._token_map,
                    "arbitrage_tokens": sorted(self._arbitrage_tokens),
                }, f)
        except Exception as exc:
            logger.warning(f"[PAPER] Failed to save token map: {exc}")

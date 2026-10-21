"""Venue: the one port for "talk to the market".

Callers name a market (MarketRef) and a side ("YES"/"NO"); the Venue owns
turning that into the token actually traded. That YES->NO translation lives
here, in MarketRef.token_for() and side_price(), and nowhere else.

Adapters:
    - PaperVenue (trading/paper_venue.py) wraps pm_trader.
    - ClobVenue (trading/clob_venue.py) wraps py_clob_client + the Data API.
    - FakeVenue (trading/fake_venue.py) is an in-memory venue for tests.
PaperVenue and FakeVenue pass the shared contract suite in
tests/unit/test_venue_contract.py; ClobVenue is tested against a stub client.

Each adapter has a name, "paper" or "live": which one the engine trades on
is the trading mode.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence

from core.models import Position

logger = logging.getLogger(__name__)


def normalize_side(side: Optional[str]) -> str:
    """"YES" or "NO" (anything not NO counts as YES, matching legacy callers)."""
    return "NO" if str(side or "YES").strip().upper() == "NO" else "YES"


def side_price(yes_price: float, side: Optional[str]) -> float:
    """Price of one share of *side*, given the market's YES price."""
    if normalize_side(side) == "NO":
        return max(1e-6, 1.0 - float(yes_price))
    return float(yes_price)


@dataclass(frozen=True)
class MarketRef:
    """What a Venue needs to identify one binary market.

    yes_token_id is the market's canonical (YES) token. no_token_id is the
    NO outcome's token, when known. condition_id/slug are what the paper
    engine resolves markets by.
    """

    yes_token_id: str
    no_token_id: Optional[str] = None
    condition_id: Optional[str] = None
    slug: Optional[str] = None

    def token_for(self, side: Optional[str]) -> str:
        """The token a *side* order trades. A NO order with no known NO
        token falls back to the YES token id (legacy behaviour)."""
        if normalize_side(side) == "NO" and self.no_token_id:
            return str(self.no_token_id)
        return str(self.yes_token_id)

    @classmethod
    def from_market(cls, market: Any) -> "MarketRef":
        """From a MarketData (or anything shaped like one)."""
        return cls(
            yes_token_id=str(market.market_id),
            no_token_id=_opt_str(getattr(market, "no_market_id", None)),
            condition_id=_opt_str(getattr(market, "condition_id", None)),
            slug=_opt_str(getattr(market, "slug", None)),
        )

    @classmethod
    def from_leg(cls, leg: Mapping[str, Any]) -> "MarketRef":
        """From an arbitrage-leg dict ({token_id, no_token_id, condition_id, slug, ...})."""
        return cls(
            yes_token_id=str(leg["token_id"]),
            no_token_id=_opt_str(leg.get("no_token_id")),
            condition_id=_opt_str(leg.get("condition_id")),
            slug=_opt_str(leg.get("slug")),
        )


def _opt_str(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None
    return str(value)


@dataclass(frozen=True)
class LimitOrder:
    """A resting limit order as handed back by Venue.place_limit_buy().

    id is unique per venue; token_id is the token the order buys. details
    is adapter-private data needed to poll the order later.
    """

    id: Any
    token_id: str
    details: Dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class FillReasons:
    """The reasons an arbitrage basket's rows give, per venue. Live keeps the
    CLOB path's historic names (the hunt_history backfill reads
    `all_legs_filled` as a live fill); paper keeps its limit-order ones."""

    not_filled: str = "limit_order_not_filled"
    all_filled: str = "all_legs_filled_via_limit_order"


@dataclass(frozen=True)
class Resolution:
    """One position settled by its market closing."""

    token_id: str
    side: str
    shares: float
    payout_price: float      # 1.0 for a winning share, 0.0 for a losing one
    entry_price: float
    market_name: str
    realized_pnl: float

    @property
    def payout(self) -> float:
        return self.shares * self.payout_price


def report_resolution(
    resolution: Resolution,
    log_func: Optional[Callable] = None,
    on_resolved: Optional[Callable[[dict], Any]] = None,
) -> None:
    """Hand one settled position to the engine's callbacks. Neither callback
    failing may stop the venue settling the rest.

    log_func gets an "EXPIRED" hunt_history row shaped like
    PortfolioManager._exit_position()'s exit rows ("price" is the per-share
    payout). on_resolved gets the record the engine writes to the Ledger.
    """
    if log_func is not None:
        try:
            log_func("EXPIRED", "Portfolio", resolution.token_id, {
                "shares": resolution.shares,
                "price": round(resolution.payout_price, 4),
                "initial_price": round(resolution.entry_price, 4),
                "side": resolution.side,
                "market_name": resolution.market_name,
                "payout": round(resolution.payout, 2),
                "realized_pnl": round(resolution.realized_pnl, 4),
                "sold": True,
            })
        except Exception as exc:
            logger.warning(f"Failed to log resolved position {resolution.token_id}: {exc}")
    if on_resolved is not None:
        try:
            on_resolved({
                "token_id": resolution.token_id,
                "shares": resolution.shares,
                "payout_price": resolution.payout_price,
                "entry_price": resolution.entry_price,
                "side": resolution.side,
                "market_name": resolution.market_name,
            })
        except Exception as exc:
            logger.warning(f"on_resolved failed for {resolution.token_id}: {exc}")


def log_placement_failed(
    log_func: Optional[Callable], token_id: str, reason: str, **detail: Any,
) -> None:
    """The EXECUTION-ERROR row a venue writes when a limit order can't be placed."""
    if log_func is not None:
        log_func("EXECUTION-ERROR", "Portfolio", token_id, {"reason": reason, **detail})


def whole_cent_shares(usd: float, price: Optional[float]) -> Optional[float]:
    """Shares *usd* buys at *price*, floored to whole cents of a share (so
    never over budget); None without a price. The one share-rounding rule:
    every venue, the executor and the pipeline size orders with it."""
    if not price or float(price) <= 0:
        return None
    return math.floor((float(usd) / float(price)) * 100.0) / 100.0


def log_paper_buy(
    log_func: Optional[Callable], token_id: str, side: str, usd: float, price: Optional[float],
) -> None:
    """The DRY-RUN row a paper venue writes for every buy it attempts."""
    shares = whole_cent_shares(usd, price)
    print(f"[DRY-RUN] Would buy {shares} of {token_id} ({side}) for ${usd}.")
    if log_func is not None:
        log_func("DRY-RUN", "Portfolio", token_id, {
            "price": price, "shares": shares, "bet_amount_usd": usd, "side": side,
        })


def log_paper_sell(log_func: Optional[Callable], token_id: str, shares: float, price: Optional[float]) -> None:
    """The DRY-RUN-SELL row a paper venue writes for every sale it attempts."""
    msg = f"[DRY-RUN] Would SELL {shares} of {token_id} at ${price}"
    print(msg)
    if log_func is not None:
        log_func("DRY-RUN-SELL", "Portfolio", token_id, {"message": msg, "price": price, "shares": shares})


class Venue(Protocol):
    """The trading engine's view of a market venue.

    Every method is best-effort: none raises on an ordinary venue failure;
    each reports it through its return value instead.

    buy/sell take the caller's per-share *price* for the side traded (live
    orders are limit orders at that price; paper fills ignore it) and an
    optional *log_func* for the venue's own order rows (DRY-RUN for paper,
    LIVE-TRADE / SELL for live). Those rows use "Portfolio" as asset type.
    The same goes for limit orders: a venue logs its own placement failure
    (EXECUTION-ERROR) and, if it has any, its cancel rows.
    """

    name: str  # "paper" or "live"
    fill_reasons: FillReasons  # reasons for arbitrage basket rows

    def cash(self) -> float:
        """Deployable cash."""

    def open_positions(self) -> List[Position]:
        """Currently open positions, reflecting every buy/sell made so far."""

    def buy(
        self, market: MarketRef, side: str, usd: float,
        price: Optional[float] = None, log_func: Optional[Callable] = None,
    ) -> bool:
        """Buy *usd* worth of *side*. True only if the order landed."""

    def sell(
        self, token_id: str, shares: float,
        price: Optional[float] = None, log_func: Optional[Callable] = None,
    ) -> bool:
        """Sell *shares* of *token_id*. True only if the sale landed."""

    def place_limit_buy(
        self, market: MarketRef, side: str, usd: float, limit_price: float,
        shares: Optional[float] = None, log_func: Optional[Callable] = None,
    ) -> Optional[LimitOrder]:
        """Place a resting limit buy of *usd* (or *shares*, where the venue
        orders by share count) at *limit_price*. None if it couldn't be
        placed; the venue logs why (EXECUTION-ERROR) to log_func."""

    def poll_limit_orders(self, orders: Sequence[LimitOrder]) -> Dict[Any, float]:
        """Advance pending limit orders against the book; return shares held
        per order id for the token each order buys."""

    def cancel_limit_order(self, order: LimitOrder, log_func: Optional[Callable] = None) -> bool:
        """Cancel a still-pending order. False if it wasn't pending."""

    def resolve_closed_markets(
        self,
        resolve_arbitrage: bool = True,
        log_func: Optional[Callable] = None,
        on_resolved: Optional[Callable[[dict], Any]] = None,
    ) -> int:
        """Settle open positions in closed markets; return how many.
        resolve_arbitrage=False leaves limit-order (arbitrage) positions
        alone. Each settled position goes through report_resolution()."""

"""
Trade execution engine: order sizing, gating and logging on top of a Venue.

The executor trades on one Venue at a time (trading/venue.py): a PaperVenue
in paper mode, a ClobVenue in live mode. It never branches on the mode
itself; dry_run only reports (and switches) which Venue is active.
"""

import asyncio
import logging
import math
import time
from typing import Optional, Dict, Any, Callable, List

from core.models import MarketData, Position, PRICE_FLOOR, PRICE_CEILING
from core.trading_config import TradingConfig
from trading.venue import LimitOrder, MarketRef, Venue, normalize_side, side_price


def _with_asset_type(log_func: Callable, asset_type: str) -> Callable:
    """log_func, with the venue's order rows filed under *asset_type*."""
    def _log(level, _asset_type, token_id, payload):
        log_func(level, asset_type, token_id, payload)
    return _log


def _venue_name(venue: Any) -> str:
    return getattr(venue, "name", "paper")


class TradeExecutor:
    """Gates, sizes and places trades on the active Venue.

    venue is the Venue traded on now. standby_venue, if given, is the other
    mode's Venue: setting dry_run swaps to it (the runtime paper <-> live
    switch). A switch to a mode with no Venue is refused, so live trading
    without a usable CLOB client never happens.
    """

    def __init__(self, config: TradingConfig, venue: Venue, standby_venue: Optional[Venue] = None):
        self.config = config
        self.trade_count_today = 0
        self.trades_by_strategy: Dict[str, int] = {}
        self.venue = venue
        self.standby_venue = standby_venue

    @property
    def dry_run(self) -> bool:
        """True while trading on a paper venue."""
        return _venue_name(self.venue) != "live"

    @dry_run.setter
    def dry_run(self, value: bool) -> None:
        wanted = "paper" if value else "live"
        if _venue_name(self.venue) == wanted:
            return
        if self.standby_venue is not None and _venue_name(self.standby_venue) == wanted:
            self.venue, self.standby_venue = self.standby_venue, self.venue
            print(f"[MODE] Now trading on the {wanted} venue")
            return
        logging.warning(f"[MODE] No {wanted} venue available -- staying on {_venue_name(self.venue)}")

    def get_balance(self) -> float:
        """Deployable cash on the active venue."""
        return float(self.venue.cash())

    def get_open_positions(self) -> List[Position]:
        """Open positions on the active venue."""
        return self.venue.open_positions()

    def execute_trade(
        self,
        token_id: str,
        current_poly_price: float,
        shares: float,
        bet_amount: float,
        asset_type: str,
        side: str,
        no_token_id: Optional[str],
        log_func: Callable,
        condition_id: Optional[str] = None,
        slug: Optional[str] = None,
    ) -> bool:
        """Buy *bet_amount* of *side* on the active venue.

        token_id/current_poly_price are the market's YES token and price;
        side picks which outcome is bought. True only if the order landed:
        callers spend budget and trade count, and assume a position exists,
        on True.
        """
        market = MarketRef(
            yes_token_id=str(token_id), no_token_id=no_token_id or None,
            condition_id=condition_id or None, slug=slug or None,
        )
        execution_side = normalize_side(side)
        try:
            return bool(self.venue.buy(
                market, execution_side, bet_amount,
                price=side_price(current_poly_price, execution_side),
                log_func=_with_asset_type(log_func, asset_type),
            ))
        except Exception as exc:
            logging.warning(f"[VENUE] Buy failed (non-fatal to pipeline): {exc}")
            return False

    def sell_position(self, token_id: str, shares: float, price: float, log_func: Callable) -> bool:
        """Sell an existing position on the active venue. True only if the
        sale landed: PortfolioManager._exit_position() credits cash on True,
        so a sale that kept failing used to be "sold" every tick (6a06f35)."""
        try:
            return bool(self.venue.sell(token_id, shares, price=price, log_func=log_func))
        except Exception as exc:
            logging.warning(f"[VENUE] Sell failed (non-fatal to pipeline): {exc}")
            return False

    # ------------------------------------------------------------------
    # Arbitrage group execution — limit orders + fill timeout (Phase 3)
    # ------------------------------------------------------------------

    def _record_strategy_trade(self, strategy_tag: str):
        self.trade_count_today += 1
        self.trades_by_strategy[strategy_tag] = self.trades_by_strategy.get(strategy_tag, 0) + 1

    async def _execute_limit_arbitrage_group(
        self, legs: List[Dict[str, Any]], timeout_seconds: float, log_func: Callable,
        strategy_tag: str, asset_type: str, group_id: Optional[str],
    ) -> Dict[str, Any]:
        """Arbitrage execution via limit orders on self.venue: submit every
        leg, poll with a bounded async loop, collect fills as they land,
        cancel/void whatever's left unfilled at the timeout.

        The limit price is the raw scan-time leg["price"], no premium above
        it, on both venues, so paper's fill rate isn't systematically more
        lenient than live's for the same signal.

        The paper engine fills limit orders fill-and-kill per poll -- the
        first poll that finds ANY liquidity within a leg's limit price fills
        it (fully or partially) and closes that order -- so there polling
        mainly gives legs that haven't found ANY liquidity yet a chance for
        the book to move favorably before the timeout. On the CLOB, polling
        reads each resting order's matched size.
        """
        # On paper, polling advances every pending order on the venue, not
        # just this group's -- harmless since arbitrage legs are the only
        # limit orders placed, and groups are processed one at a time.
        orders: Dict[str, Optional[LimitOrder]] = {}

        for leg in legs:
            token_id = str(leg["token_id"])
            price = float(leg["price"])
            shares = float(leg["shares"])
            order = self.venue.place_limit_buy(
                MarketRef.from_leg(leg),
                str(leg.get("side", "YES")),
                float(leg.get("bet_amount_usd", shares * price)),
                limit_price=price,
            )
            orders[token_id] = order
            if order is None:
                log_func("EXECUTION-ERROR", asset_type, token_id, {
                    "reason": "limit_order_placement_failed",
                    "group_id": group_id,
                })

        fills: Dict[str, float] = {tid: 0.0 for tid in orders}
        poll_interval = max(0.01, min(1.0, timeout_seconds / 10.0)) if timeout_seconds > 0 else 0.0
        deadline = time.monotonic() + timeout_seconds

        placed = [order for order in orders.values() if order is not None]
        while True:
            filled_by_order = self.venue.poll_limit_orders(placed)
            for token_id, order in orders.items():
                if order is not None:
                    fills[token_id] = float(filled_by_order.get(order.id, 0.0))
            if fills and all(filled > 0 for filled in fills.values()):
                break
            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(poll_interval)

        unfilled = [tid for tid, filled in fills.items() if filled <= 0]

        if unfilled:
            # Cancel every order -- cancelling one that already filled is a
            # harmless no-op (it doesn't retract the fill).
            for order in placed:
                self.venue.cancel_limit_order(order)

            for token_id in unfilled:
                log_func("REJECTED", asset_type, token_id, {
                    "reason": "limit_order_not_filled",
                    "group_id": group_id,
                    "timeout_seconds": timeout_seconds,
                })

            log_func("REJECTED", asset_type, group_id or "group", {
                "reason": "partial_fill_no_arb",
                "group_id": group_id,
                "unfilled": unfilled,
                "fills": fills,
            })

            # Shares that did fill before the group was voided are real,
            # unretractable trades (cancel only stops further fills) —
            # they still count against this strategy's daily trade count.
            for token_id, filled in fills.items():
                if filled > 0:
                    self._record_strategy_trade(strategy_tag)

            return {"success": False, "arb_sets": 0, "fills": fills, "unfilled": unfilled, "surplus": {}}

        arb_sets = min(fills.values())
        surplus = {tid: round(filled - arb_sets, 4) for tid, filled in fills.items() if filled > arb_sets}

        for _token_id in fills:
            self._record_strategy_trade(strategy_tag)

        log_func("ARBITRAGE-FILL", asset_type, group_id or "group", {
            "reason": "all_legs_filled_via_limit_order",
            "group_id": group_id,
            "fills": fills,
            "arb_sets": arb_sets,
            "surplus": surplus,
        })

        return {"success": True, "arb_sets": arb_sets, "fills": fills, "unfilled": [], "surplus": surplus}

    async def execute_arbitrage_group(
        self,
        legs: List[Dict[str, Any]],
        timeout_seconds: float,
        log_func: Callable,
        strategy_tag: str = "arbitrage",
    ) -> Dict[str, Any]:
        """Place limit orders for every leg of an arbitrage group, wait up
        to timeout_seconds for fills, then settle the group:

        - Every leg got >=1 fill: keep everything. The guaranteed
          arbitrage position size ("arb_sets") is the minimum fill count
          across all legs; surplus shares on the other legs are bonus
          directional exposure, not part of the guaranteed structure.
        - Any leg got 0 fills: no arbitrage exists. Every leg's resting
          order is cancelled and the zero-fill leg(s) are reported.

        *legs* is `[{token_id, price, shares, side, ...}, ...]` — optional
        per-leg keys `bet_amount_usd`, `asset_type`, `no_token_id`, and
        `group_id` (read from the first leg) enrich logging.

        Legs are limit orders on the active venue (paper or live), capping
        the price paid per leg instead of walking the book as deep as it
        takes (real trade data showed 100% of this strategy's cost overhead
        was slippage from that walk).
        """
        if not legs:
            return {"success": False, "arb_sets": 0, "fills": {}, "unfilled": [], "surplus": {}}

        asset_type = str(legs[0].get("asset_type") or "Arbitrage")
        group_id = legs[0].get("group_id")

        return await self._execute_limit_arbitrage_group(
            legs, float(timeout_seconds), log_func, strategy_tag, asset_type, group_id,
        )

    def _strategy_max_daily_trades(self, strategy_tag: str) -> int:
        """Per-strategy daily trade cap (see core/trading_config.py).
        Unrecognized strategy tags fall back to the global max_daily_trades."""
        if strategy_tag == "arbitrage":
            return int(getattr(self.config, "arbitrage_max_daily_trades", self.config.max_daily_trades))
        if strategy_tag == "crypto":
            return int(getattr(self.config, "crypto_max_daily_trades", self.config.max_daily_trades))
        return int(self.config.max_daily_trades)

    def evaluate_and_execute(
        self,
        market: MarketData,
        fair_value: float,
        ev: float,
        current_poly_price: float,
        bet_amount_usd: float,
        side: str,
        log_func: Callable,
        strategy_tag: str = "crypto",
    ) -> bool:
        """Gate on EV + daily limit + price bounds, then fire the order."""
        asset_type = market.asset_type
        token_id = market.market_id
        execution_side = normalize_side(side)

        execution_token_id = MarketRef.from_market(market).token_for(execution_side)
        execution_price = side_price(current_poly_price, execution_side)
        execution_fair_value = float(fair_value)
        if execution_side == "NO":
            execution_fair_value = 1.0 - float(fair_value)

        if ev < self.config.min_ev:
            return False

        strategy_max_daily_trades = self._strategy_max_daily_trades(strategy_tag)
        if self.trades_by_strategy.get(strategy_tag, 0) >= strategy_max_daily_trades:
            log_func("RISK", asset_type, token_id,
                     f"Daily trade limit ({strategy_max_daily_trades}) reached for {strategy_tag}")
            return False

        # Belt-and-suspenders hard ceiling: the check above only compares
        # against this strategy's own allocation. Per-strategy caps
        # (crypto_max_daily_trades, arbitrage_max_daily_trades) can be
        # configured to sum higher than the account-wide MAX_DAILY_TRADES —
        # TradingConfig only warns about that, it never clamps it — so this
        # makes the global cap an actual ceiling no strategy can bypass via
        # its own generous per-strategy bucket.
        if self.trade_count_today >= int(self.config.max_daily_trades):
            log_func("RISK", asset_type, token_id,
                     f"Global daily trade limit ({self.config.max_daily_trades}) reached")
            return False

        if not self._validate_market(market, log_func):
            return False

        if execution_price < PRICE_FLOOR or execution_price > PRICE_CEILING:
            log_func("EXECUTION", asset_type, execution_token_id, {
                "reason": "entry price out of bounds",
                "side": execution_side,
                "execution_price": round(execution_price, 4),
                "price_floor": PRICE_FLOOR,
                "price_ceiling": PRICE_CEILING,
            })
            return False

        if execution_price <= 0:
            log_func("EXECUTION", asset_type, execution_token_id,
                     f"Invalid market price for execution: {execution_price}")
            return False

        shares = math.floor((bet_amount_usd / execution_price) * 100.0) / 100.0
        if shares <= 0:
            log_func("EXECUTION", asset_type, execution_token_id,
                     f"Calculated zero shares for bet_amount_usd={bet_amount_usd}")
            return False

        log_func("AUTO-TRADE", asset_type, execution_token_id, {
            "market_price": execution_price,
            "fair_value": execution_fair_value,
            "ev": round(ev, 4),
            "bet_amount_usd": bet_amount_usd,
            "shares": shares,
            "side": execution_side,
        })

        executed = self.execute_trade(
            token_id=token_id,
            current_poly_price=float(current_poly_price),
            shares=shares,
            bet_amount=bet_amount_usd,
            asset_type=asset_type,
            side=execution_side,
            no_token_id=getattr(market, "no_market_id", None),
            log_func=log_func,
            condition_id=getattr(market, "condition_id", None),
            slug=getattr(market, "slug", None),
        )
        if executed:
            self.trade_count_today += 1
            self.trades_by_strategy[strategy_tag] = self.trades_by_strategy.get(strategy_tag, 0) + 1
        return bool(executed)

    def _validate_market(self, market: MarketData, log_func: Callable) -> bool:
        for field_name, field_value in [
            ("market_id", market.market_id),
            ("asset_type", market.asset_type),
            ("strike_price", market.strike_price),
            ("question", market.question),
        ]:
            if field_value is None:
                log_func("VALIDATE", "Market", "Unknown", f"Missing {field_name}")
                return False
        return True

    def reset_daily_count(self):
        self.trade_count_today = 0
        self.trades_by_strategy = {}

"""FakeVenue: an in-memory Venue for tests.

Behaves like PaperVenue through the Venue port (see the shared contract in
tests/unit/test_venue_contract.py): fills at the listed price with no fees
or slippage, refuses what it can't fill, and settles closed markets.

Exchange controls for tests: list_market(), set_price(), close_market().
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from core.models import Position
from trading.venue import (
    LimitOrder, MarketRef, Resolution, log_paper_buy, log_paper_sell, normalize_side, report_resolution,
)

MIN_ORDER_USD = 1.0  # same floor pm_trader enforces


@dataclass
class _Holding:
    market: MarketRef
    side: str
    shares: float
    cost: float
    arbitrage: bool = False


class FakeVenue:
    name = "paper"

    def __init__(self, cash: float = 1000.0) -> None:
        self._cash = float(cash)
        self._prices: Dict[str, float] = {}           # token -> price per share
        self._markets: Dict[str, MarketRef] = {}      # token -> market
        self._winners: Dict[MarketRef, str] = {}      # closed market -> winning side
        self._holdings: Dict[str, _Holding] = {}      # token -> holding
        self._pending: Dict[int, dict] = {}           # order id -> order
        self._order_ids = itertools.count(1)

        # Every buy/sell that landed, in order: (op, token_id, shares, usd).
        self.fills: List[tuple] = []

    # -- exchange controls (tests only) -------------------------------------

    def list_market(self, market: MarketRef, yes_price: float) -> None:
        self.set_price(market, "YES", yes_price)
        self.set_price(market, "NO", round(1.0 - float(yes_price), 6))

    def set_price(self, market: MarketRef, side: str, price: float) -> None:
        token = market.token_for(side)
        self._prices[token] = float(price)
        self._markets[token] = market

    def close_market(self, market: MarketRef, winning_side: str) -> None:
        self._winners[market] = normalize_side(winning_side)

    # -- Venue port ---------------------------------------------------------

    def cash(self) -> float:
        return self._cash

    def open_positions(self) -> List[Position]:
        positions = []
        for token, h in self._holdings.items():
            avg = h.cost / h.shares if h.shares else 0.0
            price = self._prices.get(token, avg)
            pnl_ratio = (price - avg) / avg if avg > 0 else 0.0
            positions.append(Position(
                market_id=h.market.condition_id or h.market.slug or h.market.yes_token_id,
                token_id=token,
                initial_price=avg,
                current_price=price,
                shares=h.shares,
                value=h.shares * price,
                pnl_ratio=pnl_ratio,
                side=h.side,
                live_ev=pnl_ratio,
            ))
        return positions

    def buy(self, market: MarketRef, side: str, usd: float, price=None, log_func=None) -> bool:
        log_paper_buy(log_func, market.token_for(side), normalize_side(side), usd, price)
        return self._fill_buy(market, side, float(usd), max_price=None, arbitrage=False) > 0

    def sell(self, token_id: str, shares: float, price=None, log_func=None) -> bool:
        log_paper_sell(log_func, str(token_id), shares, price)
        holding = self._holdings.get(str(token_id))
        price = self._prices.get(str(token_id))
        shares = float(shares)
        if holding is None or price is None or shares <= 0 or shares > holding.shares + 1e-9:
            return False
        if self._market_closed(holding.market):
            return False
        proceeds = shares * price
        self._cash += proceeds
        remaining = holding.shares - shares
        if remaining <= 1e-9:
            del self._holdings[str(token_id)]
        else:
            holding.cost *= remaining / holding.shares
            holding.shares = remaining
        self.fills.append(("sell", str(token_id), shares, proceeds))
        return True

    def place_limit_buy(
        self, market: MarketRef, side: str, usd: float, limit_price: float,
    ) -> Optional[LimitOrder]:
        if float(usd) < MIN_ORDER_USD or not (0 < float(limit_price) < 1):
            return None
        order_id = next(self._order_ids)
        token = market.token_for(side)
        self._pending[order_id] = {
            "market": market, "side": normalize_side(side),
            "usd": float(usd), "limit_price": float(limit_price),
        }
        return LimitOrder(id=order_id, token_id=token)

    def poll_limit_orders(self, orders: Sequence[LimitOrder]) -> Dict[Any, float]:
        for order_id, pending in list(self._pending.items()):
            filled = self._fill_buy(
                pending["market"], pending["side"], pending["usd"],
                max_price=pending["limit_price"], arbitrage=True,
            )
            if filled > 0:
                del self._pending[order_id]
        return {
            order.id: (self._holdings[order.token_id].shares if order.token_id in self._holdings else 0.0)
            for order in orders
        }

    def cancel_limit_order(self, order: LimitOrder) -> bool:
        return self._pending.pop(order.id, None) is not None

    def resolve_closed_markets(
        self,
        resolve_arbitrage: bool = True,
        log_func: Optional[Callable] = None,
        on_resolved: Optional[Callable[[dict], Any]] = None,
    ) -> int:
        resolved = 0
        for token, holding in list(self._holdings.items()):
            winner = self._winners.get(holding.market)
            if winner is None or (holding.arbitrage and not resolve_arbitrage):
                continue
            payout_price = 1.0 if holding.side == winner else 0.0
            payout = holding.shares * payout_price
            self._cash += payout
            del self._holdings[token]
            resolved += 1
            report_resolution(Resolution(
                token_id=token,
                side=holding.side,
                shares=holding.shares,
                payout_price=payout_price,
                entry_price=holding.cost / holding.shares if holding.shares else 0.0,
                market_name=holding.market.slug or token,
                realized_pnl=payout - holding.cost,
            ), log_func, on_resolved)
        return resolved

    # -- internals ----------------------------------------------------------

    def _market_closed(self, market: MarketRef) -> bool:
        return market in self._winners

    def _fill_buy(
        self, market: MarketRef, side: str, usd: float, max_price: Optional[float], arbitrage: bool,
    ) -> float:
        token = market.token_for(side)
        price = self._prices.get(token)
        if price is None or price <= 0 or self._market_closed(market):
            return 0.0
        if max_price is not None and price > max_price:
            return 0.0
        if usd < MIN_ORDER_USD or usd > self._cash + 1e-9:
            return 0.0
        shares = usd / price
        self._cash -= usd
        holding = self._holdings.get(token)
        if holding is None:
            self._holdings[token] = _Holding(
                market=market, side=normalize_side(side), shares=shares, cost=usd, arbitrage=arbitrage,
            )
        else:
            holding.shares += shares
            holding.cost += usd
            holding.arbitrage = holding.arbitrage or arbitrage
        self.fills.append(("buy", token, shares, usd))
        return shares

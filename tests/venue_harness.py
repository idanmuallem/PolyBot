"""Test harness for Venue adapters.

Each harness exposes the same "exchange" controls a test needs to set up a
scenario, whatever venue is under test:

    list_market(ref, yes_price)       -- make a binary market tradeable
    set_price(ref, side, price)       -- move one side's price
    close_market(ref, winning_side)   -- close the market with a winner

PmTraderMarketStub stands in for pm_trader's network client
(PolymarketClient), so PaperVenue runs a real pm_trader Engine and sqlite
ledger in a temp dir with no network access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from trading.venue import MarketRef, normalize_side

_BOOK_DEPTH = 1_000_000.0


class PmTraderMarketStub:
    """Duck-types pm_trader.api.PolymarketClient for the calls Engine makes."""

    def __init__(self) -> None:
        # key (slug or condition_id) -> market record
        self._markets: Dict[str, dict] = {}
        self._token_prices: Dict[str, float] = {}

    # -- harness controls ----------------------------------------------------

    def list_market(self, ref: MarketRef, yes_price: float) -> None:
        record = {
            "condition_id": ref.condition_id or f"cond-{ref.yes_token_id}",
            "slug": ref.slug or f"slug-{ref.yes_token_id}",
            "yes_token": ref.yes_token_id,
            "no_token": ref.no_token_id or f"{ref.yes_token_id}-no",
            "closed": False,
            "winner": None,
        }
        self._markets[record["slug"]] = record
        self._markets[record["condition_id"]] = record
        self._token_prices[record["yes_token"]] = float(yes_price)
        self._token_prices[record["no_token"]] = round(1.0 - float(yes_price), 6)

    def set_price(self, ref: MarketRef, side: str, price: float) -> None:
        record = self._record(ref)
        token = record["no_token"] if normalize_side(side) == "NO" else record["yes_token"]
        self._token_prices[token] = float(price)

    def close_market(self, ref: MarketRef, winning_side: str) -> None:
        record = self._record(ref)
        record["closed"] = True
        record["winner"] = normalize_side(winning_side)

    def _record(self, ref: MarketRef) -> dict:
        return self._markets[ref.slug or ref.condition_id or f"slug-{ref.yes_token_id}"]

    # -- PolymarketClient surface --------------------------------------------

    def get_market(self, slug_or_id: str):
        from pm_trader.models import Market, MarketNotFoundError

        record = self._markets.get(slug_or_id)
        if record is None:
            raise MarketNotFoundError(slug_or_id)
        if record["closed"]:
            yes_won = record["winner"] == "YES"
            prices = [1.0 if yes_won else 0.0, 0.0 if yes_won else 1.0]
        else:
            prices = [self._token_prices[record["yes_token"]], self._token_prices[record["no_token"]]]
        return Market(
            condition_id=record["condition_id"],
            slug=record["slug"],
            question=f"Question for {record['slug']}?",
            description="",
            outcomes=["Yes", "No"],
            outcome_prices=prices,
            tokens=[
                {"outcome": "Yes", "token_id": record["yes_token"]},
                {"outcome": "No", "token_id": record["no_token"]},
            ],
            active=not record["closed"],
            closed=record["closed"],
        )

    def get_order_book(self, token_id: str):
        from pm_trader.models import OrderBook, OrderBookLevel

        price = self._token_prices.get(token_id)
        if price is None:
            return OrderBook()
        level = OrderBookLevel(price=price, size=_BOOK_DEPTH)
        return OrderBook(bids=[OrderBookLevel(price=price, size=_BOOK_DEPTH)], asks=[level])

    def get_midpoint(self, token_id: str) -> float:
        return float(self._token_prices.get(token_id, 0.0))

    def get_fee_rate(self, token_id: str) -> int:
        return 0

    def get_tick_size(self, token_id: str) -> float:
        return 0.01

    def close(self) -> None:
        pass


def make_paper_venue(data_dir: Path, starting_balance: float) -> Tuple[object, PmTraderMarketStub]:
    """A real PaperVenue (pm_trader + sqlite in data_dir) on a stubbed market."""
    from trading.paper_venue import PaperVenue

    stub = PmTraderMarketStub()
    venue = PaperVenue(data_dir=data_dir, starting_balance=starting_balance, market_api=stub)
    return venue, stub


def open_book_venue(cash: float = 1000.0):
    """A FakeVenue where every order finds liquidity: a market buy fills at
    the caller's price and a limit order at its limit price, for any
    market, as if each were listed there. For tests about what the engine
    does around a fill, not about the fill itself."""
    from trading.fake_venue import FakeVenue

    class OpenBookVenue(FakeVenue):
        def buy(self, market, side, usd, price=None, log_func=None):
            if price and market.token_for(side) not in self._prices:
                self.set_price(market, side, price)
            return super().buy(market, side, usd, price=price, log_func=log_func)

        def place_limit_buy(self, market, side, usd, limit_price, shares=None, log_func=None):
            self.set_price(market, side, limit_price)
            return super().place_limit_buy(market, side, usd, limit_price, shares=shares, log_func=log_func)

    return OpenBookVenue(cash=cash)

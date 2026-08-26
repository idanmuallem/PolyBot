"""EventSumStrategy: intra-event sum arbitrage.

For a Polymarket event with N mutually exclusive outcomes (e.g. "Who will
win the election?" with candidates A, B, C, ...), exactly one outcome
resolves YES, so the fair-value constraint is:

    sum(YES_price_i for i in 1..N) == 1.00

In practice this drifts due to spreads and mispricing. When outcomes are
underpriced (sum meaningfully below 1.00 after fees), buying YES on every
outcome costs less than the $1.00 guaranteed payout — a profit regardless of
which outcome actually wins:

    Cost   = sum(price_i)   < 1.00
    Payout = 1.00  (exactly one outcome settles YES)
    Profit = 1.00 - sum(price_i)  > 0

Adapted from oracle3's EventSumArbStrategy
(oracle3/strategy/contrib/event_sum_arb_strategy.py) -
https://github.com/YichengYang-Ethan/oracle3, licensed under the Apache
License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Two
differences from that implementation:

- oracle3 streams live order-book events and accumulates per-market asks
  incrementally; PolyBot polls, so scan() is handed one Gamma API event
  snapshot per call and evaluates it directly.
- Only the underpriced/buy-YES case is implemented here. The symmetric
  overpriced/sell-both case (sum > 1.00) is exclusivity arbitrage — a
  separate strategy (trading/strategies/exclusivity.py, not yet built).
"""

import json
from typing import Any, Dict, List

from core.models import MarketData
from core.trading_config import TradingConfig
from hunters.crypto import CryptoHunter

from .base import Strategy, StrategySignal

# Conservative per-leg fee estimate (0.5%), matching oracle3's default.
_DEFAULT_FEE_RATE = 0.005


class EventSumStrategy(Strategy):
    """Buys YES on every outcome of a multi-outcome event when those
    outcomes' YES prices sum to meaningfully less than $1.00.

    Parameters
    ----------
    config:
        WalletContext-style TradingConfig (injected, not read from globals/env).
    min_edge:
        Minimum guaranteed profit per $1.00 basket, after estimated fees,
        to trigger a trade (default 0.02 = 2%).
    trade_size_usd:
        Dollar amount bet on *each* leg. Total outlay per opportunity is
        roughly trade_size_usd * n_outcomes, so this is deliberately small
        by default.
    min_outcomes:
        Skip events with fewer than this many priced, open outcomes (default 2
        — below that there's no "sum" to arbitrage).
    max_outcomes:
        Skip events with more than this many outcomes (default 20) — very
        large multi-outcome events (e.g. 100+ candidates) turn one arb into
        an impractical number of separate on-chain orders.
    fee_rate:
        Estimated per-leg fee rate, subtracted once per leg from the edge.
    """

    strategy_type = "arbitrage"

    def __init__(
        self,
        config: TradingConfig,
        min_edge: float = 0.02,
        trade_size_usd: float = 1.0,
        min_outcomes: int = 2,
        max_outcomes: int = 20,
        fee_rate: float = _DEFAULT_FEE_RATE,
    ):
        super().__init__(config)
        self.min_edge = float(min_edge)
        self.trade_size_usd = float(trade_size_usd)
        self.min_outcomes = int(min_outcomes)
        self.max_outcomes = int(max_outcomes)
        self.fee_rate = float(fee_rate)

    async def scan(self, markets: List[Dict[str, Any]]) -> List[StrategySignal]:
        """*markets* is a list of raw Polymarket Gamma API event payloads
        (each with a nested "markets" list — one entry per outcome)."""
        signals: List[StrategySignal] = []
        for event in self._ordered_events(markets):
            if isinstance(event, dict):
                signals.extend(self._scan_event(event))
        return signals

    def _ordered_events(self, markets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Crypto-first scan order (Phase 4): crypto event groups are
        higher-volume/more liquid than the general catalog, so they're
        scanned — and so spend the shared arbitrage budget — before
        everything else. The pipeline processes signals in the order scan()
        returns them, group by group, against a single finite budget, so
        putting crypto events first here is sufficient to give them budget
        priority without any change downstream.

        config.arbitrage_crypto_first=False restores the original
        single-pass order, for A/B comparison during dry-run.
        """
        markets = list(markets or [])
        if not bool(getattr(self.config, "arbitrage_crypto_first", True)):
            return markets

        crypto_events = [e for e in markets if isinstance(e, dict) and self._is_crypto_event(e)]
        other_events = [e for e in markets if not (isinstance(e, dict) and self._is_crypto_event(e))]
        return crypto_events + other_events

    @staticmethod
    def _is_crypto_event(event: Dict[str, Any]) -> bool:
        """True if the event's title/slug matches a crypto keyword.

        Uses CryptoHunter.get_search_aliases() as the single source of
        truth for the keyword list, rather than duplicating it here.
        """
        haystack = " ".join(
            str(event.get(field) or "") for field in ("title", "slug")
        ).lower()
        return any(alias in haystack for alias in CryptoHunter.get_search_aliases())

    def _scan_event(self, event: Dict[str, Any]) -> List[StrategySignal]:
        legs = self._extract_legs(event)
        n = len(legs)
        if n < self.min_outcomes or n > self.max_outcomes:
            return []

        sum_yes = sum(price for _, price in legs)
        edge = 1.0 - sum_yes - (self.fee_rate * n)

        if edge < self.min_edge:
            return []

        event_id = str(event.get("id") or event.get("slug") or event.get("title") or "unknown")
        event_title = str(event.get("title") or event_id)
        group_id = f"event_sum:{event_id}"
        reasoning = (
            f"event_sum: {event_title[:60]!r} sum_yes={sum_yes:.4f} "
            f"n={n} edge={edge:.4f} (after {self.fee_rate:.3f}/leg fees)"
        )

        return [
            StrategySignal(
                market=market_data,
                side="YES",
                price=price,
                bet_amount_usd=self.trade_size_usd,
                strategy_type=self.strategy_type,
                reasoning=reasoning,
                group_id=group_id,
                edge=edge,
            )
            for market_data, price in legs
        ]

    @staticmethod
    def _extract_legs(event: Dict[str, Any]) -> List[tuple]:
        """Return [(MarketData, yes_price), ...] for every open, priced
        outcome market in *event*.

        Deliberately does NOT apply core.models.PRICE_FLOOR/PRICE_CEILING —
        those bounds exist to avoid illiquid tails on single-market
        probability trades, but a multi-outcome event's whole point is that
        most legs sit well outside that band (e.g. 0.02, 0.05, 0.15, ...)
        while still being exactly what the arb needs to buy.
        """
        legs = []
        event_title = str(event.get("title") or "")

        for market in event.get("markets", []) or []:
            if market.get("closed"):
                continue

            tokens = market.get("clobTokenIds")
            if isinstance(tokens, str):
                try:
                    tokens = json.loads(tokens)
                except Exception:
                    tokens = None
            if not (isinstance(tokens, list) and tokens):
                continue

            price = float(
                market.get("lastTradePrice")
                or market.get("last_price")
                or market.get("mid_price")
                or 0.0
            )
            # A resolved/degenerate price (<=0 or >=1) can't be a live leg.
            if price <= 0.0 or price >= 1.0:
                continue

            market_id = str(tokens[0]).strip()
            if not market_id:
                continue
            no_market_id = str(tokens[1]).strip() if len(tokens) > 1 else None

            outcome_name = str(market.get("groupItemTitle") or market.get("question") or "?")
            market_name = f"{event_title} - {outcome_name}".strip(" -")

            market_data = MarketData(
                market_id=market_id,
                asset_type="Arbitrage::EventSum",
                strike_price=0.0,
                question=str(market.get("question") or outcome_name),
                market_name=market_name,
                initial_price=price,
                volume=float(market.get("volume") or market.get("liquidity") or 0.0),
                expiry_date=(
                    market.get("endDateIso") or market.get("endDate") or event.get("endDate")
                ),
                no_market_id=no_market_id,
                condition_id=market.get("conditionId") or market.get("condition_id"),
                slug=market.get("slug") or event.get("slug"),
            )
            legs.append((market_data, price))

        return legs

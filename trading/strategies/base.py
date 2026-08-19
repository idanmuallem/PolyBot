"""Base class for constraint-based ("arbitrage") strategies.

A Strategy finds risk-free or near-risk-free trades from the arithmetic of
market prices alone — no probability estimate, no brain, no PricingEngine.
The canonical example (event_sum.py) is: N mutually exclusive outcome prices
must sum to $1.00; if they sum to less, buying all of them is a guaranteed
profit regardless of which outcome wins.

Adapted from oracle3's Strategy/process_event pattern
(oracle3/strategy/strategy.py, oracle3/strategy/contrib/*) -
https://github.com/YichengYang-Ethan/oracle3, licensed under the Apache
License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0) - for
PolyBot's poll-based scanner: instead of reacting to a stream of order-book
events, scan() is handed one snapshot of raw Polymarket event payloads per
call and returns every leg that should be traded right now.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List

from core.models import MarketData
from core.trading_config import TradingConfig


@dataclass
class StrategySignal:
    """One executable leg of a strategy-driven trade.

    A single arbitrage opportunity usually emits several StrategySignals —
    one per outcome/leg — that share `group_id` and `edge`. The pipeline
    executes every signal in a group together (see decision_pipeline.py's
    _stage_strategy_scan).
    """

    market: MarketData
    side: str              # "YES" or "NO"
    price: float            # price this leg is bought at
    bet_amount_usd: float
    strategy_type: str      # e.g. "arbitrage" — separates strategy P&L from model-driven trades in logs
    reasoning: str
    group_id: str            # ties every leg of the same opportunity together
    edge: float                # the guaranteed/structural edge for the whole group (same for every leg in it)


class Strategy(ABC):
    """Base class for all constraint-based strategies.

    Config is injected through the constructor (WalletContext pattern) — no
    globals, no env vars. Strategy-specific tuning knobs (min edge, trade
    size, ...) are separate constructor kwargs with sensible defaults, same
    as oracle3's contrib strategies.
    """

    strategy_type: str = "strategy"

    def __init__(self, config: TradingConfig):
        self.config = config

    @abstractmethod
    async def scan(self, markets: List[Any]) -> List[StrategySignal]:
        """Scan *markets* and return every leg that should be traded now.

        *markets* is strategy-specific input (event_sum expects raw
        Polymarket Gamma API event payloads, each with a nested "markets"
        list — one entry per outcome).
        """
        raise NotImplementedError

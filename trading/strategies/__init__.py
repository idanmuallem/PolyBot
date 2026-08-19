"""Constraint-based strategies: model-free trades that don't need a probability
estimate at all — just an arithmetic mispricing in market prices.

Distinct from brains/ (which estimate a probability and route it through
PricingEngine): strategies look for structural guarantees like "these N
outcome prices must sum to $1.00" and trade the arithmetic directly.
"""

from .base import Strategy, StrategySignal
from .event_sum import EventSumStrategy

__all__ = ["Strategy", "StrategySignal", "EventSumStrategy"]

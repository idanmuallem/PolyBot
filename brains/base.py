"""Abstract base class and shared utilities for all pricing brains.

Template Method pattern: evaluate() orchestrates the pipeline while subclasses
implement _calculate_probability() with domain-specific models.
"""

from abc import ABC, abstractmethod
import re
from datetime import datetime, timezone
from core.trading_config import DEFAULT_MIN_EV
from core.models import MarketData, TradeSignal


def calculate_tte(expiry_date) -> float:
    """Return time-to-expiry in days. Defaults to 30 days on parse failure."""
    default_days = 30.0
    if expiry_date is None:
        return default_days

    now = datetime.now(timezone.utc)

    if isinstance(expiry_date, datetime):
        target = expiry_date if expiry_date.tzinfo else expiry_date.replace(tzinfo=timezone.utc)
        return max(0.0, (target - now).total_seconds() / 86400.0)

    if isinstance(expiry_date, (int, float)):
        try:
            target = datetime.fromtimestamp(float(expiry_date), tz=timezone.utc)
            return max(0.0, (target - now).total_seconds() / 86400.0)
        except Exception:
            return default_days

    text = str(expiry_date).strip()
    if not text:
        return default_days

    try:
        normalized = text.replace("Z", "+00:00")
        target = datetime.fromisoformat(normalized)
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        return max(0.0, (target - now).total_seconds() / 86400.0)
    except Exception:
        pass

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y", "%b %d, %Y", "%B %d, %Y"):
        try:
            target = datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            return max(0.0, (target - now).total_seconds() / 86400.0)
        except Exception:
            continue

    match = re.search(r"(20\d{2}[-/]\d{1,2}[-/]\d{1,2})", text)
    if match:
        try:
            target = datetime.strptime(
                match.group(1).replace("/", "-"), "%Y-%m-%d"
            ).replace(tzinfo=timezone.utc)
            return max(0.0, (target - now).total_seconds() / 86400.0)
        except Exception:
            pass

    return default_days


class BaseBrain(ABC):
    """Abstract base for fair value calculation brains.

    Subclasses implement _calculate_probability(); evaluate() handles EV,
    Kelly sizing, and tradability on top of that probability.
    """

    @abstractmethod
    def _calculate_probability(self, market: MarketData, live_truth: float) -> float:
        """Return a probability in [0.0, 1.0] for *market* given *live_truth*.

        *live_truth* is usually a bare spot/anchor value, but a hunter may pass
        a richer data package (e.g. CryptoHunter's CCXTDataClient dict) — see
        the concrete brain for what it accepts.
        """

    def get_raw_probability(self, market: MarketData, live_truth: float) -> float:
        """Return this brain's raw probability estimate (p_true), clamped to [0, 1].

        This is the brain's unadjusted opinion — the "raw" input that
        PricingEngine.compute_edge() Wang-adjusts into a market-consistent
        fair value. evaluate() below still uses it directly for the legacy
        (pre-Wang) EV/Kelly path.
        """
        return max(0.0, min(1.0, self._calculate_probability(market, live_truth)))

    def evaluate(
        self,
        market: MarketData,
        live_truth: float,
        min_ev: float = DEFAULT_MIN_EV,
    ) -> TradeSignal:
        """Compute fair value, EV, Kelly size, and tradability for *market*."""
        fair_value = self.get_raw_probability(market, live_truth)

        expected_value = (
            (fair_value - market.initial_price) / market.initial_price
            if market.initial_price > 0 else 0.0
        )

        kelly_size = max(0.0, min(0.05, self._calculate_kelly(fair_value, market.initial_price)))
        is_tradable = expected_value >= min_ev and kelly_size > 0.0

        return TradeSignal(
            fair_value=fair_value,
            expected_value=expected_value,
            kelly_size=kelly_size,
            is_tradable=is_tradable,
        )

    @staticmethod
    def _calculate_kelly(fair_value: float, market_price: float) -> float:
        """Kelly criterion: (fair * (b+1) - 1) / b, where b = 1/price - 1."""
        if market_price <= 0 or market_price >= 1:
            return 0.0
        try:
            b = (1.0 / market_price) - 1.0
            if b <= 0:
                return 0.0
            return (fair_value * (b + 1.0) - 1.0) / b
        except (ValueError, ZeroDivisionError):
            return 0.0

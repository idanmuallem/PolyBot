"""
Base abstract class for all pricing brains.

A Brain is responsible for computing fair value probabilities
for a specific asset domain using domain-specific models.
Uses Template Method pattern: evaluate() orchestrates the workflow,
while subclasses implement _calculate_probability().
"""

from abc import ABC, abstractmethod
from models import MarketData, TradeSignal


class BaseBrain(ABC):
    """Abstract base class for all fair value calculation brains.

    Implements Template Method pattern:
    - evaluate() is the main public interface (handles EV, Kelly, tradability)
    - _calculate_probability() is the extension point (domain-specific logic)
    """

    @abstractmethod
    def _calculate_probability(
        self,
        market: MarketData,
        live_truth: float
    ) -> float:
        """Calculate the probability for a market.

        This is the extension point for subclasses to implement
        their domain-specific probability calculation.

        Args:
            market: MarketData object containing market details
            live_truth: The current market value or reference value
                       (e.g., BTC price, current temperature, Fed rate)

        Returns:
            A probability in [0.0, 1.0] representing the fair value.
        """
        pass

    def evaluate(
        self,
        market: MarketData,
        live_truth: float,
        min_ev: float = 0.15
    ) -> TradeSignal:
        """Evaluate a market and generate a trade signal.

        Template Method: orchestrates fair value calculation,
        EV computation, Kelly sizing, and tradability assessment.

        Args:
            market: MarketData object with market details
            live_truth: Current reference value
            min_ev: Minimum EV threshold required for tradability

        Returns:
            TradeSignal with fair_value, expected_value, kelly_size, is_tradable
        """
        # Step 1: Calculate fair value (probability)
        fair_value = self._calculate_probability(market, live_truth)
        fair_value = max(0.0, min(1.0, fair_value))  # Clamp [0, 1]

        # Step 2: Calculate Expected Value
        # EV = (fair_value - market_price) / market_price
        if market.initial_price <= 0:
            expected_value = 0.0
        else:
            expected_value = (fair_value - market.initial_price) / market.initial_price

        # Step 3: Calculate Kelly criterion size
        # kelly = (fair * (b + 1) - 1) / b, where b = (1/p) - 1
        kelly_size = self._calculate_kelly(fair_value, market.initial_price)
        kelly_size = max(0.0, min(0.05, kelly_size))  # Clamp [0, 0.05]

        # Step 4: Determine tradability
        is_tradable = (expected_value >= min_ev) and (kelly_size > 0.0)

        # Step 5: Return trade signal
        return TradeSignal(
            fair_value=fair_value,
            expected_value=expected_value,
            kelly_size=kelly_size,
            is_tradable=is_tradable
        )

    @staticmethod
    def _calculate_kelly(fair_value: float, market_price: float) -> float:
        """Calculate Kelly criterion bet size.

        kelly = (fair_value * (b + 1) - 1) / b
        where b = (1 / market_price) - 1

        Args:
            fair_value: Probability in [0.0, 1.0]
            market_price: Current market price in [0.0, 1.0]

        Returns:
            Kelly size (fraction of bankroll)
        """
        if market_price <= 0 or market_price >= 1:
            return 0.0

        try:
            b = (1.0 / market_price) - 1.0
            if b <= 0:
                return 0.0
            kelly = (fair_value * (b + 1.0) - 1.0) / b
            return kelly
        except (ValueError, ZeroDivisionError):
            return 0.0

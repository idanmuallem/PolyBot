"""
Base abstract class for all pricing brains.

A Brain is responsible for computing fair value probabilities
for a specific asset domain using domain-specific models.
"""

from abc import ABC, abstractmethod


class BaseBrain(ABC):
    """Abstract base class for all fair value calculation brains.

    Each brain specializes in a specific domain and uses domain-appropriate
    mathematical models to calculate probabilities.
    """

    @abstractmethod
    def get_fair_value(
        self,
        live_truth: float,
        strike: float,
        days_left: float,
        **kwargs
    ) -> float:
        """Calculate the fair value (probability) for a market.

        Args:
            live_truth: The current market value or reference value
                       (e.g., BTC price, current temperature, Fed rate)
            strike: The strike/threshold value from the market question
            days_left: Days until market expiry
            **kwargs: Domain-specific parameters (volatility, std dev, etc.)

        Returns:
            A probability in [0.0, 1.0] representing the fair value.
        """
        pass

    def get_topic_type(self) -> str:
        """Return the domain identifier for this brain.

        Examples: "Crypto", "Weather", "Economy"
        """
        raise NotImplementedError("Subclasses should override get_topic_type()")

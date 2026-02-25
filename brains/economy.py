"""
EconomyBrain: Fair value calculation for economic indicator prediction markets.

Uses a normal distribution model with historical volatility scaling.
"""

import math
from scipy.stats import norm

from .base import BaseBrain


class EconomyBrain(BaseBrain):
    """Calculate fair value for economic indicator prediction markets.

    Model: Normal distribution with historical volatility.
    Factors: Current indicator value, strike value, time to expiry, volatility.
    """

    DEFAULT_HIST_VOLATILITY = {
        "FedRate": 0.5,        # Fed Funds Rate volatility
        "CPI": 0.3,            # CPI volatility
        "Unemployment": 0.2,   # Unemployment rate volatility
        "GDP": 0.5,            # GDP growth volatility
    }

    def __init__(self, hist_volatilities: dict = None):
        """Initialize EconomyBrain.

        Args:
            hist_volatilities: Dict mapping indicators to historical vols.
                              (default uses DEFAULT_HIST_VOLATILITY)
        """
        self.hist_volatilities = hist_volatilities or dict(self.DEFAULT_HIST_VOLATILITY)

    def get_topic_type(self) -> str:
        return "Economy"

    def get_volatility_for_indicator(self, indicator: str) -> float:
        """Get the historical volatility for a given indicator.

        Args:
            indicator: Indicator name (e.g., "FedRate", "CPI")

        Returns:
            Historical volatility (in appropriate units)
        """
        return self.hist_volatilities.get(indicator, 0.5)

    def get_fair_value(
        self,
        live_truth: float,
        strike: float,
        days_left: float,
        indicator: str = "FedRate",
        **kwargs
    ) -> float:
        """Calculate fair value using historical volatility model.

        Args:
            live_truth: Current indicator value (e.g., Fed rate = 5.25)
            strike: Strike/threshold value from the market
            days_left: Days until market expiry
            indicator: Indicator name (used to lookup volatility)
            **kwargs: Override hist_vol with 'hist_vol' kwarg if provided

        Returns:
            Probability (CDF value) in [0.0, 1.0]
        """
        # Allow override via kwarg
        hist_vol = kwargs.get("hist_vol")
        if hist_vol is None:
            hist_vol = self.get_volatility_for_indicator(indicator)

        return self._calculate_prob(live_truth, strike, days_left, hist_vol)

    @staticmethod
    def _calculate_prob(
        current_val: float,
        strike_val: float,
        time_to_expiry_days: float,
        hist_vol: float = 0.5
    ) -> float:
        """Calculate probability using normal model with historical volatility.

        Assumes the indicator follows a normal distribution with volatility
        scaled by sqrt(time).

        Args:
            current_val: Current indicator value
            strike_val: Strike/threshold value
            time_to_expiry_days: Time until expiry in days
            hist_vol: Historical volatility (same units as current_val)

        Returns:
            Probability in [0.0, 1.0]
        """
        # Handle edge cases
        if time_to_expiry_days <= 0:
            return 1.0 if current_val > strike_val else 0.0

        if hist_vol <= 0:
            return 1.0 if current_val > strike_val else 0.0

        # Historical vol scaled by sqrt(time)
        time_as_fraction_of_year = max(1e-6, time_to_expiry_days / 365.0)
        stdev = hist_vol * math.sqrt(time_as_fraction_of_year)

        if stdev <= 0:
            return 1.0 if current_val > strike_val else 0.0

        # Z-score
        try:
            z = (current_val - strike_val) / stdev
        except (ValueError, ZeroDivisionError):
            return 0.5

        # Return CDF: P(value > strike)
        return float(norm.cdf(z))

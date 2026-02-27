"""
EconomyBrain: Fair value calculation for economic indicator prediction markets.

Uses a normal distribution model with historical volatility scaling.
"""

import math
from scipy.stats import norm
from core.models import MarketData

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

    def get_volatility_for_indicator(self, indicator: str) -> float:
        """Get the historical volatility for a given indicator.

        Args:
            indicator: Indicator name (e.g., "FedRate", "CPI")

        Returns:
            Historical volatility (in appropriate units)
        """
        return self.hist_volatilities.get(indicator, 0.5)

    def _calculate_probability(
        self,
        market: MarketData,
        live_truth: float
    ) -> float:
        """Calculate probability using historical volatility model.

        Args:
            market: MarketData object with strike_price and other details
            live_truth: Current indicator value (e.g., Fed rate = 5.25)

        Returns:
            Probability (CDF value) in [0.0, 1.0]
        """
        # Extract volatility for the indicator (FedRate, CPI, etc.)
        hist_vol = self.get_volatility_for_indicator(market.asset_type)

        # Estimate days to expiry (default to 365 days if not specified)
        days_to_expiry = 365.0

        return self._calculate_prob(
            live_truth,
            market.strike_price,
            days_to_expiry,
            hist_vol
        )

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

"""
CryptoBrain: Fair value calculation for cryptocurrency markets.

Uses a Black-Scholes style CDF approach with annualized volatility.
"""

import math
from scipy.stats import norm
from models import MarketData

from .base import BaseBrain


class CryptoBrain(BaseBrain):
    """Calculate fair value for cryptocurrency prediction markets.

    Model: Log-normal distribution with annualized volatility.
    Factors: Current price, strike price, time to expiry, volatility.
    """

    # Default volatilities (annualized) by proxy- these are representative
    DEFAULT_VOLATILITIES = {
        "BTC": 0.5,   # Bitcoin: 50% annualized volatility
        "ETH": 0.7,   # Ethereum: 70% annualized volatility
        "SOL": 0.9,   # Solana: 90% annualized volatility
    }

    def __init__(self, volatilities: dict = None):
        """Initialize CryptoBrain.

        Args:
            volatilities: Dict mapping symbol prefixes to annualized vols.
                         (default uses DEFAULT_VOLATILITIES)
        """
        self.volatilities = volatilities or dict(self.DEFAULT_VOLATILITIES)

    def get_volatility_for_symbol(self, symbol: str) -> float:
        """Get the annualized volatility for a given symbol.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT", "ETHUSDT")

        Returns:
            Annualized volatility (e.g., 0.5 = 50%)
        """
        symbol_upper = symbol.upper()
        for key, vol in self.volatilities.items():
            if key.upper() in symbol_upper or symbol_upper.startswith(key.upper()):
                return vol
        # Default fallback
        return 0.6

    def _calculate_probability(
        self,
        market: MarketData,
        live_truth: float
    ) -> float:
        """Calculate probability using Black-Scholes-style CDF.

        Args:
            market: MarketData object with strike_price and other details
            live_truth: Current spot price (e.g., BTC/USDT)

        Returns:
            Probability (CDF value) in [0.0, 1.0]
        """
        # Extract volatility for the asset type (BTC, ETH, etc.)
        vol = self.get_volatility_for_symbol(market.asset_type)

        # Estimate days to expiry (default to 30 days if not specified)
        # For now, we'll use a reasonable default
        days_to_expiry = 30.0

        return self._calculate_prob(
            live_truth,
            market.strike_price,
            days_to_expiry,
            vol
        )

    @staticmethod
    def _calculate_prob(
        current_price: float,
        strike_price: float,
        time_to_expiry_days: float,
        volatility: float = 0.5
    ) -> float:
        """Black-Scholes style probability calculation.

        Uses log-normal CDF to compute P(price > strike) at expiry.

        Args:
            current_price: Current spot price
            strike_price: Strike/threshold price
            time_to_expiry_days: Time until expiry in days
            volatility: Annualized volatility (e.g., 0.5 = 50%)

        Returns:
            Probability in [0.0, 1.0]
        """
        # Handle edge cases
        if time_to_expiry_days <= 0:
            return 1.0 if current_price > strike_price else 0.0

        if strike_price <= 0:
            return 1.0

        if current_price <= 0:
            return 0.0

        # Annualized volatility scaled by sqrt(time)
        time_as_fraction_of_year = max(1e-6, time_to_expiry_days / 365.0)
        stdev = volatility * math.sqrt(time_as_fraction_of_year)

        if stdev <= 0:
            return 1.0 if current_price > strike_price else 0.0

        # Black-Scholes d2 term: log price ratio adjusted for drift
        try:
            d2 = (
                math.log(current_price / strike_price) - 0.5 * stdev * stdev
            ) / stdev
        except (ValueError, ZeroDivisionError):
            return 0.5

        # Return CDF at d2
        return float(norm.cdf(d2))

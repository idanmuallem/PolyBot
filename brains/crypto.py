"""
HybridCryptoBrain: Time-aware fair value calculation for cryptocurrency markets.

Uses a Black-Scholes style CDF approach with annualized volatility.
"""
import math
import numpy as np
from scipy.stats import norm
from core.models import MarketData

from .base import BaseBrain, calculate_tte


class HybridCryptoBrain(BaseBrain):
    """Calculate fair value for cryptocurrency prediction markets.

    Model switcher by TTE:
    - TTE < 1 day: short-term technical/trend model
    - 1 <= TTE < 30 days: Black-Scholes style model
    - TTE >= 30 days: Heston characteristic function with FFT pricing
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
        self.last_model_used = "standard_bs"

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
        """Calculate probability using TTE-aware model switching.

        Args:
            market: MarketData object with strike_price and other details
            live_truth: Current spot price (e.g., BTC/USDT)

        Returns:
            Probability (CDF value) in [0.0, 1.0]
        """
        # Extract volatility for the asset type (BTC, ETH, etc.)
        vol = self.get_volatility_for_symbol(market.asset_type)

        base_prob = self.evaluate_fair_value(
            market=market,
            live_truth=live_truth,
            volatility=vol,
        )

        question = str(getattr(market, "market_name", "") or getattr(market, "question", "")).lower()
        invert_keywords = ["↓", "below", "under", "less", "down", "lower"]

        if any(kw in question for kw in invert_keywords):
            base_prob = 1.0 - base_prob

        return base_prob

    def evaluate_fair_value(self, market: MarketData, live_truth: float, volatility: float) -> float:
        """Select pricing model based on time-to-expiry (TTE) with safe fallback.

        If the selected primary model fails, we explicitly fall back to
        Black-Scholes.
        """
        # DYNAMIC VOLATILITY SKEW (FAT TAIL ADJUSTMENT)
        # Inflates volatility the further the strike is from the current price.
        strike = float(getattr(market, "strike_price", 0.0) or 0.0)
        if strike > 0 and live_truth > 0:
            distance_penalty = abs(math.log(live_truth / strike))
            volatility = volatility * (1.0 + distance_penalty)

        tte_days = calculate_tte(getattr(market, "expiry_date", None))

        try:
            if tte_days < 1.0:
                self.last_model_used = "short_term"
                fair_value = self._price_short_term(live_truth, market.strike_price)
            elif tte_days < 30.0:
                self.last_model_used = "standard_bs"
                fair_value = self._price_standard_bs(live_truth, market.strike_price, tte_days, volatility)
            else:
                self.last_model_used = "heston_fft"
                fair_value = self._price_heston_fft(live_truth, market.strike_price, tte_days, volatility)
        except Exception:
            self.last_model_used = "black_scholes_fallback"
            fair_value = self._price_black_scholes(market, live_truth)

        return float(fair_value)

    def _price_black_scholes(self, market: MarketData, live_truth: float) -> float:
        """Safe Black-Scholes fallback used when primary model output is unstable."""
        strike_price = float(getattr(market, "strike_price", 0.0) or 0.0)
        tte_days = calculate_tte(getattr(market, "expiry_date", None))
        volatility = self.get_volatility_for_symbol(str(getattr(market, "asset_type", "")))
        return self._price_standard_bs(float(live_truth), strike_price, float(tte_days), float(volatility))

    def _price_short_term(self, current_price: float, strike_price: float) -> float:
        """Simple short-term trend/technical approximation.

        Uses normalized distance between spot and strike and a smooth sigmoid map.
        """
        if strike_price <= 0 or current_price <= 0:
            return 0.5

        distance = (current_price - strike_price) / max(strike_price, 1e-9)
        score = 5.0 * distance
        prob = 1.0 / (1.0 + math.exp(-score))
        return float(max(0.0, min(1.0, prob)))

    def _price_standard_bs(
        self,
        current_price: float,
        strike_price: float,
        time_to_expiry_days: float,
        volatility: float,
    ) -> float:
        return self._calculate_prob(current_price, strike_price, time_to_expiry_days, volatility)

    def _price_heston_fft(
        self,
        current_price: float,
        strike_price: float,
        time_to_expiry_days: float,
        volatility: float,
    ) -> float:
        """Approximate Heston pricing via Carr-Madan style FFT call valuation.

        Returns call probability proxy by normalizing call value with spot.
        Falls back to Black-Scholes if numerical issues occur.
        """
        if current_price <= 0 or strike_price <= 0 or time_to_expiry_days <= 0:
            return 1.0 if current_price > strike_price else 0.0

        try:
            t = max(1e-6, time_to_expiry_days / 365.0)
            s0 = float(current_price)
            k = float(strike_price)

            # Heston params (stable defaults)
            kappa = 2.0
            theta = max(1e-6, volatility * volatility)
            sigma_v = 0.6
            rho = -0.5
            v0 = theta
            r = 0.0

            alpha = 1.5
            n = 1024
            eta = 0.25
            lambd = 2.0 * math.pi / (n * eta)
            b = 0.5 * n * lambd

            u = np.arange(n) * eta
            iu = 1j * u

            def heston_cf(phi):
                d = np.sqrt((rho * sigma_v * iu - kappa) ** 2 + sigma_v * sigma_v * (iu + phi * phi))
                g = (kappa - rho * sigma_v * iu - d) / (kappa - rho * sigma_v * iu + d)
                exp_dt = np.exp(-d * t)
                c = r * iu * t + (kappa * theta / (sigma_v * sigma_v)) * (
                    (kappa - rho * sigma_v * iu - d) * t - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g))
                )
                d_term = ((kappa - rho * sigma_v * iu - d) / (sigma_v * sigma_v)) * ((1.0 - exp_dt) / (1.0 - g * exp_dt))
                return np.exp(c + d_term * v0 + iu * np.log(s0))

            numerator = np.exp(-r * t) * heston_cf(u - (alpha + 1.0) * 1j)
            denominator = alpha * alpha + alpha - u * u + 1j * (2.0 * alpha + 1.0) * u
            psi = numerator / denominator

            # Simpson weights
            weights = np.ones(n)
            weights[0] = 1.0
            weights[1::2] = 4.0
            weights[2::2] = 2.0
            weights[-1] = 1.0
            weights = weights / 3.0

            fft_input = np.exp(1j * b * u) * psi * eta * weights
            fft_values = np.fft.fft(fft_input).real

            k_grid = -b + np.arange(n) * lambd
            strikes = np.exp(k_grid)
            calls = np.exp(-alpha * k_grid) / math.pi * fft_values

            call_interp = np.interp(k, strikes, calls)
            prob_proxy = max(0.0, min(1.0, call_interp / max(s0, 1e-9)))
            return float(prob_proxy)
        except Exception:
            return self._price_standard_bs(current_price, strike_price, time_to_expiry_days, volatility)

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



# Backward compatibility for existing imports/factory usage
CryptoBrain = HybridCryptoBrain

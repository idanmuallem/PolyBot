"""
HybridCryptoBrain: Time-aware fair value calculation for cryptocurrency markets.

Uses a Black-Scholes style CDF approach with annualized volatility.
"""
import math
from typing import Optional, Union

from scipy.stats import norm
from core.models import MarketData

from .base import BaseBrain, calculate_tte


class HybridCryptoBrain(BaseBrain):
    """Calculate fair value for cryptocurrency prediction markets.

    Model switcher by TTE:
    - TTE < 1 day: short-term technical/trend model
    - TTE >= 1 day: Black-Scholes style model

    A Heston/Carr-Madan FFT model previously handled TTE >= 90 days, but it
    was dropped: even after fixing its numerical degeneracy (it was
    returning near-zero probabilities on real markets - see git history for
    the full investigation), its output (a normalized vanilla-call value,
    C(K)/S0) is not the same quantity as P(S_T > K) - it's systematically
    and substantially different from the Black-Scholes CDF probability on
    real inputs (e.g. 0.117 vs 0.374 on the same BTC market), which means
    Wang Transform + market-blending downstream were being calibrated
    against a wrong input. Black-Scholes is used for every TTE now; nothing
    in this codebase's usage (binary yes/no strikes, not path-dependent
    payoffs) actually needs vol-of-vol/mean-reversion modeling badly enough
    to justify reintroducing that risk without first computing an actual
    digital/CDF probability from the characteristic function (the Π2
    formula from Heston 1993), not a normalized call price.
    """

    # Default volatilities (annualized) by proxy- these are representative
    DEFAULT_VOLATILITIES = {
        "BTC": 0.5,   # Bitcoin: 50% annualized volatility
        "ETH": 0.7,   # Ethereum: 70% annualized volatility
        "SOL": 0.9,   # Solana: 90% annualized volatility
    }

    def __init__(self, volatilities: dict = None):
        """Initialize HybridCryptoBrain.

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

    @staticmethod
    def _unpack_live_truth(live_truth: Union[float, dict]) -> tuple:
        """Accept either a bare spot price or CCXTDataClient's enriched dict.

        Returns (spot_price, enriched_dict_or_None).
        """
        if isinstance(live_truth, dict):
            return float(live_truth.get("spot_price") or 0.0), live_truth
        return float(live_truth), None

    def _select_volatility(self, market: MarketData, enriched: Optional[dict]) -> float:
        """Prefer CCXT realized volatility (matched to the contract's time
        horizon) over the hardcoded per-symbol default, when available."""
        if enriched:
            tte_days = calculate_tte(getattr(market, "expiry_date", None))
            if tte_days < 30.0:
                realized = enriched.get("vol_30d")
            elif tte_days < 60.0:
                realized = enriched.get("vol_60d")
            else:
                realized = enriched.get("vol_90d")
            if realized:
                return float(realized)
        return self.get_volatility_for_symbol(market.asset_type)

    def _calculate_probability(
        self,
        market: MarketData,
        live_truth: Union[float, dict],
    ) -> float:
        """Calculate probability using TTE-aware model switching.

        Args:
            market: MarketData object with strike_price and other details
            live_truth: Current spot price (float), or CCXTDataClient's
                enriched data package (dict with "spot_price", "vol_30d",
                "vol_60d", "vol_90d", ...). Either form is accepted so
                existing callers/tests that pass a bare spot price keep working.

        Returns:
            Probability (CDF value) in [0.0, 1.0]
        """
        spot_price, enriched = self._unpack_live_truth(live_truth)
        vol = self._select_volatility(market, enriched)

        base_prob = self.evaluate_fair_value(
            market=market,
            live_truth=spot_price,
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
            else:
                self.last_model_used = "standard_bs"
                fair_value = self._price_standard_bs(live_truth, market.strike_price, tte_days, volatility)
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

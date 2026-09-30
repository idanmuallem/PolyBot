"""
HybridCryptoBrain: Time-aware fair value calculation for cryptocurrency markets.

Uses a Black-Scholes style CDF approach with annualized volatility.
"""
import math
from typing import Optional, Union

from scipy.stats import norm
from core.models import MarketData

from .base import BaseBrain, calculate_tte


# Resolution-type classifier — decides whether a market resolves on first
# touch (any point before the deadline) or only at expiry (terminal price).
#
# Verified against every resolution-criteria string actually observed on
# Polymarket's BTC/ETH price-threshold markets (see the "What price will
# Bitcoin/Ethereum hit in [month/year]?" series and its "all time high by"
# sibling) — all of them state, in slightly varying wording, that the
# market resolves the moment a 1-minute Binance candle's High/Low crosses
# the listed price. Requiring several signals to co-occur (a candle/
# monitoring reference, an explicit "final ... price" trigger, and a High/
# Low + price mention) keeps this from false-positiving on an unrelated
# description while still tolerating the wording drift already seen across
# markets (different date phrasing, "immediately resolve" vs "resolve",
# encoding-mangled quote characters).
#
# No expiry-style crypto market has been observed yet to test a true
# negative against real data — every crypto market reachable by the
# hunter's current filters happens to be touch-style (see the market-
# discovery trace). Anything that doesn't match this pattern — including
# a genuinely different, not-yet-seen expiry-style wording — falls back to
# "unknown", which routes to the pre-existing (correct-for-expiry) terminal
# Black-Scholes path. That's the safe default: a market this classifier
# fails to recognize gets today's existing behavior, never a worse one.
_TOUCH_RESOLUTION_MARKERS = ("1 minute candle", "1-minute candle")


def classify_resolution_type(description: str) -> str:
    """Return "touch" if *description* states first-touch/barrier
    resolution, else "unknown" (caller should treat as expiry/terminal)."""
    text = (description or "").lower()
    has_candle = any(marker in text for marker in _TOUCH_RESOLUTION_MARKERS)
    has_final_trigger = "has a final" in text
    has_high_low_price = ("high" in text or "low" in text) and "price" in text
    if has_candle and has_final_trigger and has_high_low_price:
        return "touch"
    return "unknown"


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

        # The first-passage path (see evaluate_fair_value/_price_first_passage)
        # already returns P(the stated condition holds) directly — it picks
        # up-touch vs down-touch from where the strike sits relative to spot,
        # not from question wording. "P(touches up)" and "P(touches down)"
        # aren't complements of each other the way terminal's "ends above" /
        # "ends below" are, so applying the keyword-driven inversion below
        # to it would silently invert an already-correct answer.
        if self.last_model_used == "first_passage":
            return base_prob

        question = str(getattr(market, "market_name", "") or getattr(market, "question", "")).lower()
        # Downward-direction keywords: a market asking whether price falls to /
        # below a level answers the complement of the brain's "ends above"
        # probability, so invert. "dip" (as in "dip to $X") is a downward
        # phrasing that was previously missing - without it, a terminal-path
        # "dip to $X" market was priced in the wrong direction. Note this whole
        # block is skipped for the first-passage path above (which gets
        # direction from strike-vs-spot geometry, not wording), so this only
        # affects the terminal Black-Scholes path (sub-1-day TTE, or
        # expiry/unknown-classified markets).
        invert_keywords = ["↓", "below", "under", "less", "down", "lower", "dip"]

        if any(kw in question for kw in invert_keywords):
            base_prob = 1.0 - base_prob

        return base_prob

    def evaluate_fair_value(self, market: MarketData, live_truth: float, volatility: float) -> float:
        """Select pricing model based on time-to-expiry (TTE) and, for
        TTE >= 1 day, the market's actual resolution mechanism — with safe
        fallback.

        Polymarket's BTC/ETH price-threshold markets resolve on first touch
        (any 1-minute candle crossing the strike before the deadline), not
        on the terminal price at expiry — confirmed directly from each
        market's own resolution-criteria text (see classify_resolution_type
        above). Plain Black-Scholes answers "where does price end up",
        which is the wrong question for a market that pays out the moment
        price is EVER seen past the strike; a first-passage/barrier
        probability answers the right one. Markets this classifier can't
        positively identify as touch-style (including a genuinely
        different, not-yet-seen expiry-style market) keep using the
        existing terminal calculation — see classify_resolution_type's
        docstring for why that's the safe default.

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
            elif classify_resolution_type(getattr(market, "description", "")) == "touch":
                self.last_model_used = "first_passage"
                fair_value = self._price_first_passage(live_truth, market.strike_price, tte_days, volatility)
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

    def _price_first_passage(
        self,
        current_price: float,
        strike_price: float,
        time_to_expiry_days: float,
        volatility: float,
    ) -> float:
        """First-passage (barrier) probability: P(price ever crosses
        strike_price before expiry), for touch-resolving markets — see
        evaluate_fair_value's docstring for why this differs from
        _price_standard_bs's terminal P(price > strike at expiry).

        Direction (crossing up through the strike, vs. down through it) is
        taken directly from whether the strike sits above or below the
        current price — not from the question's wording — so this is
        correct regardless of "reach $X" vs "dip to $X" phrasing, and
        doesn't depend on a keyword match the way the terminal path's
        up/down inversion does. _calculate_probability skips that
        keyword-based inversion whenever this path was used (see there):
        "P(touches up)" and "P(touches down)" aren't complements of each
        other the way "ends above K" / "ends below K" are, so this method
        returns P(the stated condition holds) directly, already correctly
        signed — inverting it again would silently be wrong.

        Closed-form reflection-principle solution for a driftless
        geometric Brownian motion's running maximum/minimum (Shreve,
        Stochastic Calculus for Finance II, ch. 7) — the same zero-drift
        convention _calculate_prob already uses (its d2 formula has no
        risk-free-rate term), so this needs no new free parameter. Verified
        numerically against Monte Carlo simulation of the same process
        before landing this.

        This is the continuous-monitoring solution; these markets actually
        resolve off discrete 1-minute candles. The standard discrete-
        monitoring correction (Broadie-Glasserman-Kou) shifts the effective
        barrier by volatility * 0.5826 * sqrt(dt) in log-space — with dt on
        the order of 1 minute against strikes/TTEs measured in days to
        months, that shift is on the order of 1e-4 to 1e-3 in log-price,
        utterly negligible next to the barrier distances actually seen
        here (0.01-1.0+). Not applied — the continuous formula is already
        an excellent approximation at this monitoring frequency.
        """
        if strike_price <= 0 or current_price <= 0:
            return 0.5
        if time_to_expiry_days <= 0:
            return 1.0 if current_price >= strike_price else 0.0

        t_years = time_to_expiry_days / 365.0
        # Zero log-drift, matching _calculate_prob's own convention (no r term).
        nu = -0.5 * volatility * volatility
        sigma_sqrt_t = volatility * math.sqrt(t_years)
        if sigma_sqrt_t <= 0:
            return 1.0 if current_price >= strike_price else 0.0

        if strike_price >= current_price:
            # Barrier above spot: P(running max ever >= strike).
            b = math.log(strike_price / current_price)
            if b <= 0:
                return 1.0
            term1 = norm.cdf((nu * t_years - b) / sigma_sqrt_t)
            term2 = math.exp(2.0 * nu * b / (volatility * volatility)) * norm.cdf(
                (-nu * t_years - b) / sigma_sqrt_t
            )
        else:
            # Barrier below spot: P(running min ever <= strike).
            b = math.log(current_price / strike_price)
            if b <= 0:
                return 1.0
            term1 = norm.cdf((-nu * t_years - b) / sigma_sqrt_t)
            term2 = math.exp(-2.0 * nu * b / (volatility * volatility)) * norm.cdf(
                (nu * t_years - b) / sigma_sqrt_t
            )

        return float(max(0.0, min(1.0, term1 + term2)))

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

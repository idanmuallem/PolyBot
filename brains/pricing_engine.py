"""PricingEngine: Wang Transform fair-value pricing for prediction markets.

The hierarchical lambda formula, its coefficients, and the pooled prior
below are adapted from oracle3 (oracle3/pricing/fair_value.py,
oracle3/pricing/wang_mle.py) - https://github.com/YichengYang-Ethan/oracle3,
licensed under the Apache License, Version 2.0
(http://www.apache.org/licenses/LICENSE-2.0).

Adapted from oracle3's calibrated Wang Transform (Yang, 2026), fit on 291K+
Polymarket/Kalshi contracts. Existing brains still produce a raw probability
(p_true) from their domain models (crypto vol, weather forecasts, economic
releases) - they no longer compute EV directly. This engine takes that raw
probability and Wang-adjusts it into a market-consistent fair value, then
compares it against the live market price to compute tradeable edge.

The core transform is a probit-space constant shift:

    p_market = Phi(Phi^-1(p_true) + lambda)

where Phi is the standard normal CDF and lambda is a risk-premium parameter.
lambda > 0 means the market systematically overprices relative to the raw
probability (the well-documented favorite-longshot bias).

lambda is estimated per-contract via oracle3's hierarchical model (Table 3,
N=13,274 Polymarket contracts) when volume and time-to-expiry are known:

    lambda_i = 0.259 - 0.072*ln(1+V) + 0.143*ln(1+D) - 0.477*|p_true - 0.5|

Falls back to oracle3's pooled prior (291K contracts, lambda = 0.183) when
that metadata isn't available.
"""
import math

from scipy.stats import norm

# Hierarchical coefficients (oracle3 pricing/fair_value.py, Yang 2026 Table 3)
_HIER_CONSTANT = 0.259
_HIER_LN_VOLUME = -0.072
_HIER_LN_DURATION = 0.143
_HIER_EXTREMITY = -0.477

_EPS = 1e-6  # clamp probabilities before Phi^-1 to avoid +/-inf


def _clamp_prob(p: float) -> float:
    return max(_EPS, min(1.0 - _EPS, p))


def wang_transform(p_true: float, lam: float) -> float:
    """Flat-lambda Wang Transform: Phi(Phi^-1(p_true) + lam).

    Used directly by BaseBrain.evaluate() (brains/base.py) for entry-side
    pricing, where lam is a fixed risk-aversion constant (config.wang_lambda)
    rather than PricingEngine's hierarchical, metadata-driven lambda below.
    lam < 0 pulls p_true toward 0.5 (risk-averse); lam > 0 pushes it away;
    lam == 0.0 is an exact passthrough.

    The shift shrinks as p_true approaches 0 or 1 (see wang_fair_value's
    delta_lambda) - by itself this transform cannot fully correct a raw
    probability that's already saturated near an extreme. That's what
    evaluate()'s market-blending step (after this one) is for.
    """
    if lam == 0.0:
        return float(p_true)
    p = _clamp_prob(p_true)
    return float(norm.cdf(norm.ppf(p) + lam))


class PricingEngine:
    """Wang-adjusts a brain's raw probability into a tradeable fair value.

    Config is injected through the constructor (no globals, no env vars),
    consistent with WalletContext.
    """

    def __init__(self, base_lambda: float = 0.183):
        # oracle3's pooled prior across 291K contracts - the fallback lambda
        # when volume/days_to_expiry aren't available for the hierarchical
        # adjustment below.
        self.base_lambda = base_lambda

    def _lambda_for(self, p_true: float, volume: float = None, days_to_expiry: float = None) -> float:
        """Hierarchical lambda when volume+expiry are known, else the pooled prior."""
        if volume is None or days_to_expiry is None:
            return self.base_lambda

        lam = _HIER_CONSTANT
        lam += _HIER_LN_VOLUME * math.log(1 + max(0.0, volume))
        lam += _HIER_LN_DURATION * math.log(1 + max(0.0, days_to_expiry))
        lam += _HIER_EXTREMITY * abs(p_true - 0.5)
        return lam

    def wang_fair_value(self, p_true: float, volume: float = None, days_to_expiry: float = None) -> dict:
        """Wang-adjust a raw probability into the market-consistent fair value.

        Args:
            p_true: raw probability from a brain's domain model, in [0, 1].
            volume: market trading volume (USD), if known.
            days_to_expiry: time to contract expiry in days, if known.

        Returns:
            dict with:
                fair_value: Phi(Phi^-1(p_true) + lambda) - what the market
                    should trade at once the risk premium is priced in.
                lambda_used: the risk-premium parameter applied.
                edge: fair_value - p_true, the risk premium baked into the price.
                delta_lambda: dp/dlambda at this point (model Greek - price
                    sensitivity to the risk-premium parameter).
        """
        p = _clamp_prob(p_true)
        lam = self._lambda_for(p, volume, days_to_expiry)

        z = norm.ppf(p) + lam
        fair_value = float(norm.cdf(z))

        return {
            "fair_value": fair_value,
            "lambda_used": lam,
            "edge": fair_value - p,
            "delta_lambda": float(norm.pdf(z)),
        }

    def compute_edge(
        self,
        p_true: float,
        market_price: float,
        volume: float = None,
        days_to_expiry: float = None,
    ) -> dict:
        """Compare the Wang-adjusted fair value against the live market price.

        Args:
            p_true: raw probability from a brain's domain model.
            market_price: current market price for the contract.
            volume: market trading volume (USD), if known.
            days_to_expiry: time to contract expiry in days, if known.

        Returns:
            dict with:
                edge: fair_value - market_price. Positive means the market
                    is underpriced relative to the model (buy YES); negative
                    means overpriced (buy NO).
                direction: "YES", "NO", or "NONE" (no edge).
                confidence: [0, 1] blend of metadata completeness and edge
                    magnitude.
                fair_value / lambda_used: passthrough from wang_fair_value().
        """
        result = self.wang_fair_value(p_true, volume, days_to_expiry)
        fair_value = result["fair_value"]
        price = _clamp_prob(market_price)

        edge = fair_value - price
        if edge > 0:
            direction = "YES"
        elif edge < 0:
            direction = "NO"
        else:
            direction = "NONE"

        data_quality = 1.0 if (volume is not None and days_to_expiry is not None) else 0.5
        confidence = max(0.0, min(1.0, data_quality * min(1.0, abs(edge) / 0.05)))

        return {
            "edge": edge,
            "direction": direction,
            "confidence": confidence,
            "fair_value": fair_value,
            "lambda_used": result["lambda_used"],
        }

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

NOTE: `PricingEngine` is the EXIT-side mechanism (trading/risk_manager.py);
it inlines its own probit shift in `wang_fair_value()` and does not call the
standalone `wang_transform()` below.

Entry-side pricing history (BaseBrain.evaluate()):
  1. originally wang_transform() (probit shift toward 0.5) - had a crossover
     bug that pushed the real output range AWAY from 0.5;
  2. then logit_shrink() (proportional shrink toward 0.5) - fixed the
     monotonicity bug but still manufactured a spurious YES edge on cheap
     markets even when the brain agreed with the market (A1_TRADE_RATE_FINDINGS.md);
  3. now market_anchored_shrink() - interpolates between market and brain in
     logit space, so agreement yields zero edge and entry_k is a true
     "trust the brain vs defer to market" dial.
wang_transform() and logit_shrink() are retained only as reference/test
implementations documenting (1) and (2); no production path calls them.
See PHASE1_FINDINGS.md, A1_TRADE_RATE_FINDINGS.md and PROGRESS.md.
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

    REFERENCE / TEST-ONLY as of the entry-side redesign below - no
    production path calls this function any more (the exit-side PricingEngine
    inlines its own probit shift; entry-side uses logit_shrink()). Retained as
    the reference implementation of the bug that motivated logit_shrink(): a
    constant probit-space shift is only "toward 0.5" on one side of a lambda-
    dependent crossover point. For lam < 0, inputs with p_true > Phi(-lam)
    actually get pushed AWAY from 0.5, not toward it - see
    test_pricing_engine.py's characterization test and
    scripts/phase3_validation.py.

    lam < 0 pulls p_true toward 0.5 for most of [0, 1] (risk-averse); lam > 0
    pushes it away for most of [0, 1]; lam == 0.0 is an exact passthrough.
    Neither claim holds near the crossover point noted above.
    """
    if lam == 0.0:
        return float(p_true)
    p = _clamp_prob(p_true)
    return float(norm.cdf(norm.ppf(p) + lam))


def logit_shrink(p_true: float, k: float) -> float:
    """Proportional shrink toward 0.5 in log-odds (logit) space.

        p' = sigmoid(logit(p_true) * k),  k in [0, 1]

    Replaces the old probit-space Wang shift (wang_transform() above) for
    entry-side pricing (BaseBrain.evaluate(), config.entry_k). Unlike that
    transform, this one is monotonic toward 0.5 for EVERY p_true in (0, 1)
    and every k in [0, 1], by construction - there is no crossover point
    where it reverses direction. See test_pricing_engine.py's property test
    for the proof-by-sweep.

    k = 1.0 is an exact passthrough (p_true unchanged). k = 0.0 collapses
    every input to exactly 0.5 (total distrust of the raw probability).
    k in between shrinks proportionally: p_true = 0.3 with k = 0.5 lands
    halfway (in logit space) between 0.3 and 0.5.
    """
    if k >= 1.0:
        return float(p_true)
    if k <= 0.0:
        return 0.5
    p = _clamp_prob(p_true)
    logit_p = math.log(p / (1.0 - p))
    shrunk_logit = logit_p * k
    return float(1.0 / (1.0 + math.exp(-shrunk_logit)))


def market_anchored_shrink(p_true: float, market_price: float, k: float) -> float:
    """Trust-weighted interpolation between the market price and the brain's
    raw probability, in log-odds (logit) space:

        post = sigmoid( logit(market) + k * (logit(p_true) - logit(market)) )

    This is the entry-side mechanism as of the market-anchored redesign
    (BaseBrain.evaluate(), config.entry_k). It replaces BOTH the old
    shrink-toward-0.5 step (logit_shrink) AND the separate model_weight
    market blend - the interpolation IS the blend, with correct semantics:

        k = 0.0  -> post = market_price   (defer fully to the market; the
                                           brain is ignored)
        k = 1.0  -> post = p_true         (trust the brain fully)
        k in between -> trust that fraction of the brain's divergence from
                        the market, in logit space.

    Key property that fixes the shrink-toward-0.5 bug: when the brain AGREES
    with the market (p_true == market_price), post == market_price for every
    k, so no edge is manufactured. logit_shrink pulled toward 0.5 instead,
    which on a cheap market lifted the blended value above the price and
    manufactured a spurious YES edge even on agreement (see
    A1_TRADE_RATE_FINDINGS.md). This function is monotonic in p_true and
    always lands between market_price and p_true - see
    test_pricing_engine.py's property sweep.

    k is clamped to [0, 1] defensively (config validates it too).
    """
    k = max(0.0, min(1.0, float(k)))
    if k <= 0.0:
        return _clamp_prob(market_price)
    if k >= 1.0:
        return _clamp_prob(p_true)
    p = _clamp_prob(p_true)
    m = _clamp_prob(market_price)
    logit_p = math.log(p / (1.0 - p))
    logit_m = math.log(m / (1.0 - m))
    blended_logit = logit_m + k * (logit_p - logit_m)
    return float(1.0 / (1.0 + math.exp(-blended_logit)))


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

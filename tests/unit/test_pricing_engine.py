import math

import numpy as np
import pytest
from scipy.stats import norm

from brains.pricing_engine import (
    PricingEngine,
    _clamp_prob,
    logit_shrink,
    market_anchored_shrink,
    wang_transform,
)


# ── wang_fair_value: known values ───────────────────────────────────────────

def test_known_wang_output_zero_lambda_is_identity():
    # lambda = 0 -> Phi(Phi^-1(p) + 0) = p exactly
    engine = PricingEngine(base_lambda=0.0)
    result = engine.wang_fair_value(0.5)
    assert result["fair_value"] == pytest.approx(0.5, abs=1e-9)
    assert result["lambda_used"] == 0.0
    assert result["edge"] == pytest.approx(0.0, abs=1e-9)


def test_known_wang_output_pooled_prior():
    # No volume/expiry given -> falls back to base_lambda (pooled prior).
    engine = PricingEngine(base_lambda=0.183)
    result = engine.wang_fair_value(0.5)
    expected = norm.cdf(norm.ppf(0.5) + 0.183)
    assert result["fair_value"] == pytest.approx(expected, abs=1e-9)
    assert result["lambda_used"] == 0.183
    # Positive lambda -> systematic overpricing above the raw probability.
    assert result["fair_value"] > 0.5


def test_known_wang_output_hierarchical():
    # Hand-computed against oracle3's Table 3 hierarchical formula:
    # lambda = 0.259 - 0.072*ln(1+V) + 0.143*ln(1+D) - 0.477*|p-0.5|
    engine = PricingEngine()
    p_true, volume, days = 0.6, 5000.0, 10.0
    result = engine.wang_fair_value(p_true, volume=volume, days_to_expiry=days)

    expected_lambda = (
        0.259
        - 0.072 * math.log(1 + volume)
        + 0.143 * math.log(1 + days)
        - 0.477 * abs(p_true - 0.5)
    )
    expected_fair_value = norm.cdf(norm.ppf(p_true) + expected_lambda)

    assert result["lambda_used"] == pytest.approx(expected_lambda, abs=1e-9)
    assert result["fair_value"] == pytest.approx(expected_fair_value, abs=1e-9)
    assert result["edge"] == pytest.approx(expected_fair_value - p_true, abs=1e-9)


def test_negative_lambda_underprices():
    engine = PricingEngine(base_lambda=-0.2)
    for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
        result = engine.wang_fair_value(p)
        assert result["fair_value"] < p


def test_positive_lambda_overprices():
    engine = PricingEngine(base_lambda=0.2)
    for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
        result = engine.wang_fair_value(p)
        assert result["fair_value"] > p


# ── Edge cases ───────────────────────────────────────────────────────────────

def test_p_true_zero_does_not_crash():
    engine = PricingEngine()
    result = engine.wang_fair_value(0.0)
    assert 0.0 <= result["fair_value"] <= 1.0
    assert math.isfinite(result["lambda_used"])


def test_p_true_one_does_not_crash():
    engine = PricingEngine()
    result = engine.wang_fair_value(1.0)
    assert 0.0 <= result["fair_value"] <= 1.0
    assert math.isfinite(result["lambda_used"])


def test_very_high_volume_stays_in_unit_interval():
    engine = PricingEngine()
    result = engine.wang_fair_value(0.5, volume=1_000_000_000.0, days_to_expiry=30.0)
    assert 0.0 <= result["fair_value"] <= 1.0


def test_very_low_volume_stays_in_unit_interval():
    engine = PricingEngine()
    result = engine.wang_fair_value(0.5, volume=0.0, days_to_expiry=30.0)
    assert 0.0 <= result["fair_value"] <= 1.0


def test_clamp_prob_bounds():
    assert _clamp_prob(0.0) > 0.0
    assert _clamp_prob(1.0) < 1.0
    assert _clamp_prob(0.5) == 0.5
    assert _clamp_prob(-5.0) > 0.0
    assert _clamp_prob(5.0) < 1.0


# ── Hierarchical lambda varies with metadata ────────────────────────────────

def test_higher_volume_lowers_lambda():
    # Coefficient on ln(1+V) is negative: more liquid markets have a
    # smaller risk premium (already competed away).
    engine = PricingEngine()
    lam_low_vol = engine.wang_fair_value(0.6, volume=100.0, days_to_expiry=10.0)["lambda_used"]
    lam_high_vol = engine.wang_fair_value(0.6, volume=100_000.0, days_to_expiry=10.0)["lambda_used"]
    assert lam_high_vol < lam_low_vol


def test_longer_duration_raises_lambda():
    # Coefficient on ln(1+D) is positive: longer time horizons carry more
    # risk premium.
    engine = PricingEngine()
    lam_short = engine.wang_fair_value(0.6, volume=1000.0, days_to_expiry=1.0)["lambda_used"]
    lam_long = engine.wang_fair_value(0.6, volume=1000.0, days_to_expiry=365.0)["lambda_used"]
    assert lam_long > lam_short


def test_extremity_lowers_lambda():
    # Coefficient on |p-0.5| is negative: near-coinflip markets carry the
    # highest premium; extreme probabilities carry less.
    engine = PricingEngine()
    lam_coinflip = engine.wang_fair_value(0.5, volume=1000.0, days_to_expiry=10.0)["lambda_used"]
    lam_extreme = engine.wang_fair_value(0.95, volume=1000.0, days_to_expiry=10.0)["lambda_used"]
    assert lam_extreme < lam_coinflip


def test_missing_metadata_falls_back_to_base_lambda():
    engine = PricingEngine(base_lambda=0.1)
    result_no_volume = engine.wang_fair_value(0.6, volume=None, days_to_expiry=10.0)
    result_no_days = engine.wang_fair_value(0.6, volume=1000.0, days_to_expiry=None)
    result_neither = engine.wang_fair_value(0.6)
    assert result_no_volume["lambda_used"] == 0.1
    assert result_no_days["lambda_used"] == 0.1
    assert result_neither["lambda_used"] == 0.1


# ── Greeks ───────────────────────────────────────────────────────────────────

def test_delta_lambda_is_positive_pdf():
    engine = PricingEngine(base_lambda=0.15)
    result = engine.wang_fair_value(0.5)
    assert result["delta_lambda"] > 0.0

    # dp/dlambda = phi(Phi^-1(p) + lambda)
    expected = norm.pdf(norm.ppf(0.5) + 0.15)
    assert result["delta_lambda"] == pytest.approx(expected, abs=1e-9)


def test_delta_lambda_maximized_near_coinflip():
    # Sensitivity to lambda should be higher near p=0.5 than at the extremes.
    engine = PricingEngine(base_lambda=0.15)
    delta_mid = engine.wang_fair_value(0.5)["delta_lambda"]
    delta_extreme = engine.wang_fair_value(0.95)["delta_lambda"]
    assert delta_mid > delta_extreme


# ── compute_edge ─────────────────────────────────────────────────────────────

def test_compute_edge_underpriced_market_is_yes():
    engine = PricingEngine(base_lambda=0.2)
    # fair_value > p_true (positive lambda) and market trades at p_true,
    # so the market is underpriced relative to fair value -> buy YES.
    result = engine.compute_edge(p_true=0.5, market_price=0.5)
    assert result["edge"] > 0
    assert result["direction"] == "YES"


def test_compute_edge_overpriced_market_is_no():
    engine = PricingEngine(base_lambda=0.2)
    fair_value = engine.wang_fair_value(0.5)["fair_value"]
    result = engine.compute_edge(p_true=0.5, market_price=fair_value + 0.05)
    assert result["edge"] < 0
    assert result["direction"] == "NO"


def test_compute_edge_no_edge_when_price_matches_fair_value():
    engine = PricingEngine(base_lambda=0.2)
    fair_value = engine.wang_fair_value(0.5)["fair_value"]
    result = engine.compute_edge(p_true=0.5, market_price=fair_value)
    assert result["edge"] == pytest.approx(0.0, abs=1e-9)
    assert result["direction"] == "NONE"


def test_compute_edge_confidence_in_unit_interval():
    engine = PricingEngine(base_lambda=0.2)
    for market_price in [0.1, 0.3, 0.5, 0.7, 0.9]:
        result = engine.compute_edge(0.5, market_price, volume=5000.0, days_to_expiry=10.0)
        assert 0.0 <= result["confidence"] <= 1.0


def test_compute_edge_confidence_lower_without_metadata():
    engine = PricingEngine(base_lambda=0.2)
    with_meta = engine.compute_edge(0.5, 0.3, volume=5000.0, days_to_expiry=10.0)
    without_meta = engine.compute_edge(0.5, 0.3)
    assert with_meta["confidence"] >= without_meta["confidence"]


# ── wang_transform (flat-lambda, used directly by BaseBrain.evaluate()) ──────

def test_wang_transform_reduces_high_probability():
    # Raw 0.95 with lambda=-0.75 shaves off overconfidence on deep-ITM markets.
    fair = wang_transform(0.95, -0.75)
    assert fair < 0.90


def test_wang_transform_penalizes_uncertainty_more():
    # Near-coinflip probabilities move more, proportionally, than
    # already-extreme ones - the transform's shift is largest near p=0.5
    # and shrinks toward 0/1 (see delta_lambda above).
    drop_mid = 0.50 - wang_transform(0.50, -0.75)
    drop_high = 0.90 - wang_transform(0.90, -0.75)
    assert (drop_mid / 0.50) > (drop_high / 0.90)


def test_wang_lambda_zero_is_passthrough():
    assert wang_transform(0.70, 0.0) == 0.70


# ── wang_transform: characterization of the crossover-point bug ────────────
#
# This documents the actual bug that motivated logit_shrink() (see below):
# a constant probit-space shift is only "toward 0.5" on one side of a
# lambda-dependent crossover point. wang_transform() itself is UNCHANGED and
# still used on the exit side (PricingEngine, trading/risk_manager.py, whose
# lambda is a real fitted hierarchical model, not a flat guess - see
# brains/pricing_engine.py's module docstring for why entry and exit no
# longer share a design). These tests exist purely to document, with a
# runnable proof, the specific failure that entry-side pricing no longer has.

def test_wang_transform_crossover_bug_at_entry_lambda():
    # lambda=-0.75 was the crypto brain's old entry-side constant. Its
    # crossover point (the p above which the shift flips from "toward 0.5"
    # to "away from 0.5") sits around p ~= 0.62-0.65. Every raw probability
    # the crypto brain has ever actually produced sat in 0.25-0.45 - i.e.
    # entirely on the wrong side of the crossover - so in practice this
    # transform moved every real input AWAY from 0.5, the opposite of its
    # own docstring's claim ("pulls toward 0.5"). See PHASE1_FINDINGS.md.
    for p in (0.25, 0.30, 0.35, 0.40, 0.45):
        shifted = wang_transform(p, -0.75)
        assert shifted < p, (
            f"expected wang_transform({p}, -0.75) to move further from 0.5 "
            f"(the documented bug), got {shifted} which moved toward 0.5"
        )

    # By contrast, for p above the crossover (empirically ~0.62-0.65 for
    # this lambda), the same lambda genuinely moves the result closer to
    # 0.5 than p was, exactly as documented - which is what made the bug
    # easy to miss: the transform behaves correctly for inputs the crypto
    # brain never actually produces. Note this can still overshoot past 0.5
    # to the other side (e.g. p=0.70 -> ~0.41): "moved toward 0.5" and
    # "landed between p and 0.5" are not the same claim, and only the
    # former is documented/promised.
    for p in (0.70, 0.80, 0.90):
        shifted = wang_transform(p, -0.75)
        assert abs(shifted - 0.5) < abs(p - 0.5), (
            f"expected wang_transform({p}, -0.75) to move closer to 0.5, "
            f"got {shifted}"
        )


# ── logit_shrink: property-based monotonicity proof ─────────────────────────
#
# The actual regression test for the crossover bug above. logit_shrink must
# be monotonic toward 0.5 for EVERY p in (0, 1) and EVERY k in [0, 1] - not
# just the range the crypto brain happens to produce today - because that
# was exactly the blind spot that let the old bug ship unnoticed.

_P_GRID = np.concatenate([
    np.array([1e-6, 1e-4, 1e-3]),
    np.linspace(0.01, 0.99, 99),
    np.array([1 - 1e-3, 1 - 1e-4, 1 - 1e-6]),
])
_K_GRID = np.linspace(0.0, 1.0, 21)


def test_logit_shrink_monotonic_toward_half_full_sweep():
    for k in _K_GRID:
        for p in _P_GRID:
            shrunk = logit_shrink(float(p), float(k))
            assert 0.0 <= shrunk <= 1.0, f"out of range at p={p}, k={k}: {shrunk}"
            if p < 0.5:
                assert p - 1e-9 <= shrunk <= 0.5 + 1e-9, (
                    f"p={p} k={k}: expected p <= shrunk <= 0.5, got {shrunk}"
                )
            elif p > 0.5:
                assert 0.5 - 1e-9 <= shrunk <= p + 1e-9, (
                    f"p={p} k={k}: expected 0.5 <= shrunk <= p, got {shrunk}"
                )


def test_logit_shrink_half_is_fixed_point():
    for k in _K_GRID:
        assert logit_shrink(0.5, float(k)) == pytest.approx(0.5, abs=1e-9)


def test_logit_shrink_k_one_is_identity():
    for p in _P_GRID:
        assert logit_shrink(float(p), 1.0) == pytest.approx(float(p), abs=1e-9)


def test_logit_shrink_k_zero_is_always_half():
    for p in _P_GRID:
        assert logit_shrink(float(p), 0.0) == pytest.approx(0.5, abs=1e-9)


def test_logit_shrink_order_preserving_in_p():
    # For a fixed k, shrinking must not reorder inputs (no crossing lines).
    for k in (0.0, 0.25, 0.5, 0.75, 1.0):
        outputs = [logit_shrink(float(p), k) for p in _P_GRID]
        assert all(a <= b + 1e-9 for a, b in zip(outputs, outputs[1:])), (
            f"logit_shrink not order-preserving at k={k}"
        )


def test_logit_shrink_never_reproduces_crossover_bug():
    # The direct contrast with test_wang_transform_crossover_bug_at_entry_lambda
    # above: over the same p range where the old transform moved values away
    # from 0.5, logit_shrink always moves them toward (or leaves them at) 0.5.
    for p in (0.25, 0.30, 0.35, 0.40, 0.45):
        for k in (0.1, 0.3, 0.5, 0.7, 0.9):
            shrunk = logit_shrink(p, k)
            assert shrunk >= p, f"p={p} k={k}: expected shrunk >= p (toward 0.5), got {shrunk}"


# ── market_anchored_shrink: the current entry-side mechanism ────────────────
#
# post = sigmoid(logit(market) + k*(logit(brain) - logit(market))). Properties
# that MUST hold for every brain, market in (0,1) and k in [0,1]:
#   - k=0 -> market; k=1 -> brain
#   - brain == market -> market (no manufactured edge; the A1 bug fix)
#   - post always lands between market and brain (inclusive)
#   - monotonic non-decreasing in brain for fixed market, k

_MP_GRID = np.linspace(0.02, 0.98, 25)
_BRAIN_GRID = np.concatenate([[1e-4], np.linspace(0.02, 0.98, 25), [1 - 1e-4]])
_K_GRID2 = np.linspace(0.0, 1.0, 11)


def test_market_anchored_k0_is_market_k1_is_brain():
    for m in _MP_GRID:
        for b in _BRAIN_GRID:
            assert market_anchored_shrink(float(b), float(m), 0.0) == pytest.approx(float(m), abs=1e-9)
            assert market_anchored_shrink(float(b), float(m), 1.0) == pytest.approx(float(b), abs=1e-9)


def test_market_anchored_agreement_is_fixed_point():
    # brain == market -> post == market, for every k. This is the property the
    # old shrink-toward-0.5 design violated (it manufactured edge on agreement).
    for m in _MP_GRID:
        for k in _K_GRID2:
            assert market_anchored_shrink(float(m), float(m), float(k)) == pytest.approx(float(m), abs=1e-9)


def test_market_anchored_lands_between_market_and_brain():
    for m in _MP_GRID:
        for b in _BRAIN_GRID:
            for k in _K_GRID2:
                post = market_anchored_shrink(float(b), float(m), float(k))
                lo, hi = sorted((float(m), float(b)))
                assert lo - 1e-9 <= post <= hi + 1e-9, f"m={m} b={b} k={k}: {post}"


def test_market_anchored_monotonic_in_brain():
    for m in (0.10, 0.30, 0.50, 0.80):
        for k in (0.2, 0.5, 0.9):
            outs = [market_anchored_shrink(float(b), m, k) for b in _BRAIN_GRID]
            assert all(a <= b + 1e-9 for a, b in zip(outs, outs[1:])), f"m={m} k={k}"


def test_market_anchored_k_clamped():
    # k outside [0,1] is clamped, not extrapolated.
    assert market_anchored_shrink(0.8, 0.3, 5.0) == pytest.approx(market_anchored_shrink(0.8, 0.3, 1.0))
    assert market_anchored_shrink(0.8, 0.3, -2.0) == pytest.approx(market_anchored_shrink(0.8, 0.3, 0.0))


def test_market_anchored_no_cheap_market_manufactured_edge():
    # The A1 regression, at the function level: on a cheap market, agreement
    # yields exactly the market price (zero edge), where logit_shrink+blend
    # previously produced a value well above it.
    assert market_anchored_shrink(0.10, 0.10, 0.5) == pytest.approx(0.10, abs=1e-9)
    assert market_anchored_shrink(0.05, 0.05, 0.5) == pytest.approx(0.05, abs=1e-9)

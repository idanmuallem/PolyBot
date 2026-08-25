import math

import pytest
from scipy.stats import norm

from brains.pricing_engine import PricingEngine, _clamp_prob, wang_transform


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

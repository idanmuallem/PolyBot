from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from freezegun import freeze_time

from brains.base import calculate_tte, BaseBrain
from brains.crypto import HybridCryptoBrain
from brains.pricing_engine import logit_shrink, wang_transform
from core.models import MarketData


def _make_market(price=0.50, strike=100_000.0, expiry="2026-12-31", asset_type="Crypto::BTCUSDT"):
    return MarketData(
        market_id="tok1",
        asset_type=asset_type,
        strike_price=strike,
        question="Will BTC exceed $100,000?",
        market_name="Bitcoin - Will BTC exceed $100,000?",
        initial_price=price,
        volume=500_000.0,
        expiry_date=expiry,
    )


# ── calculate_tte ─────────────────────────────────────────────────────────────

@freeze_time("2026-06-01T00:00:00+00:00")
def test_future_iso_string():
    result = calculate_tte("2026-12-31T00:00:00+00:00")
    assert abs(result - 213.0) < 1.0


@freeze_time("2026-06-01T00:00:00+00:00")
def test_past_date_returns_zero():
    result = calculate_tte("2025-01-01")
    assert result == 0.0


def test_none_returns_default_30():
    assert calculate_tte(None) == 30.0


def test_malformed_string_returns_default():
    assert calculate_tte("sometime-next-year") == 30.0


@freeze_time("2026-06-01T00:00:00+00:00")
def test_unix_timestamp_parsing():
    # 2027-01-01 00:00:00 UTC as unix timestamp
    ts = int(datetime(2027, 1, 1, tzinfo=timezone.utc).timestamp())
    result = calculate_tte(ts)
    assert result > 0


# ── BaseBrain._calculate_kelly ─────────────────────────────────────────────────

def test_kelly_positive_edge():
    result = BaseBrain._calculate_kelly(0.7, 0.5)
    assert result > 0


def test_kelly_returns_negative_when_fair_below_market():
    # _calculate_kelly itself returns negative; evaluate() clips it to 0
    result = BaseBrain._calculate_kelly(0.3, 0.5)
    assert result < 0


def test_kelly_zero_at_money():
    assert BaseBrain._calculate_kelly(0.5, 0.5) == 0.0


def test_kelly_zero_at_zero_price():
    assert BaseBrain._calculate_kelly(0.7, 0.0) == 0.0


def test_kelly_zero_at_unit_price():
    assert BaseBrain._calculate_kelly(0.7, 1.0) == 0.0


# ── BaseBrain.evaluate ─────────────────────────────────────────────────────────

def test_tradable_above_min_ev():
    # entry_k=1.0 (passthrough - no shrink), model_weight=1.0 isolate the
    # EV/Kelly/min_ev gate from the shrink/blend calibration layers (covered
    # separately in test_pricing_engine.py) so post_prob == the raw
    # probability here. Note entry_k=1.0 is the passthrough value for
    # logit_shrink, unlike the old wang_lambda=0.0 passthrough convention.
    brain = HybridCryptoBrain()
    market = _make_market(price=0.50)
    with patch.object(brain, "_calculate_probability", return_value=0.80):
        signal = brain.evaluate(
            market, 95_000.0, min_ev=0.30,
            entry_k=1.0, model_weight=1.0,
        )
    # EV = (0.80 - 0.50) / 0.50 = 0.60 > 0.30
    assert signal.is_tradable is True
    assert signal.kelly_size > 0
    assert signal.post_prob == pytest.approx(0.80)


def test_not_tradable_below_min_ev():
    brain = HybridCryptoBrain()
    market = _make_market(price=0.50)
    with patch.object(brain, "_calculate_probability", return_value=0.55):
        signal = brain.evaluate(market, 95_000.0, min_ev=0.30, entry_k=1.0, model_weight=1.0)
    # EV = (0.55 - 0.50) / 0.50 = 0.10 < 0.30
    assert signal.is_tradable is False


def test_raw_probability_clamped_to_unit_interval():
    # get_raw_probability() clamps the brain's raw output before Wang/blend/
    # cap ever see it - post_prob itself is no longer a raw passthrough
    # (see brains/pricing_engine.py's wang_transform + market-blending in
    # evaluate()), so the clamp is verified on pre_prob directly.
    brain = HybridCryptoBrain()
    market = _make_market(price=0.50)
    with patch.object(brain, "_calculate_probability", return_value=1.5):
        signal = brain.evaluate(market, 95_000.0)
    assert signal.pre_prob == 1.0


def test_raw_probability_clamped_below_zero():
    brain = HybridCryptoBrain()
    market = _make_market(price=0.50)
    with patch.object(brain, "_calculate_probability", return_value=-0.3):
        signal = brain.evaluate(market, 95_000.0)
    assert signal.pre_prob == 0.0


# ── model_weight validation ─────────────────────────────────────────────────

def test_model_weight_above_one_is_clamped():
    # model_weight=1.5 should clamp to 1.0 (trust model only) rather than
    # overshoot past the shrunk value.
    brain = HybridCryptoBrain()
    market = _make_market(price=0.50)
    with patch.object(brain, "_calculate_probability", return_value=0.80):
        signal = brain.evaluate(
            market, 95_000.0, entry_k=1.0, model_weight=1.5,
        )
    assert signal.post_prob == pytest.approx(0.80)


def test_model_weight_below_zero_is_clamped():
    # model_weight=-0.5 should clamp to 0.0 (trust market only).
    brain = HybridCryptoBrain()
    market = _make_market(price=0.50)
    with patch.object(brain, "_calculate_probability", return_value=0.80):
        signal = brain.evaluate(
            market, 95_000.0, entry_k=1.0, model_weight=-0.5,
        )
    assert signal.post_prob == pytest.approx(0.50)


# ── Logit shrink + market blending, applied through evaluate() ─────────────
#
# Replaces the old flat-lambda Wang Transform (see brains/pricing_engine.py's
# wang_transform() docstring and test_pricing_engine.py's characterization
# test for the bug this fixed: a constant probit-space shift is only
# "toward 0.5" on one side of a lambda-dependent crossover point). Some
# properties the old tests checked genuinely don't hold for the new
# mechanism and aren't reproduced below - see test_shrink_zero_shift_at_half
# for why "shrink is largest near 0.5" (true of the old transform) is the
# opposite of how logit_shrink behaves.

def test_shrink_reduces_high_probability():
    # Raw 0.95 with entry_k=0.5 pulls it toward 0.5. model_weight=1.0
    # isolates the shrink step from blending.
    brain = HybridCryptoBrain()
    market = _make_market(price=0.50)
    with patch.object(brain, "_calculate_probability", return_value=0.95):
        signal = brain.evaluate(
            market, 95_000.0, entry_k=0.5, model_weight=1.0,
        )
    assert signal.post_prob < 0.95


def test_shrink_reduces_low_probability_toward_half():
    # THE ACTUAL BUG REGRESSION TEST. 0.30 sits in the crypto brain's real
    # historical output range (0.25-0.45) - the exact range where the old
    # flat wang_lambda=-0.75 transform moved every single observed value
    # AWAY from 0.5 instead of toward it (see PHASE1_FINDINGS.md). The new
    # mechanism must move it TOWARD 0.5 (i.e. up, since 0.30 < 0.5) for
    # this input, unlike the old one.
    brain = HybridCryptoBrain()
    market = _make_market(price=0.50)
    with patch.object(brain, "_calculate_probability", return_value=0.30):
        signal = brain.evaluate(
            market, 95_000.0, entry_k=0.5, model_weight=1.0,
        )
    assert 0.30 < signal.post_prob < 0.50
    # Contrast with the old transform on the same input, to document the bug
    # this replaces (not asserted against production code - just evidence).
    old_wang_fair = wang_transform(0.30, -0.75)
    assert old_wang_fair < 0.30  # old transform moved this AWAY from 0.5


def test_shrink_zero_shift_at_half():
    # logit(0.5) == 0, so any k leaves 0.5 exactly at 0.5 - there is no
    # "distortion" to apply to a genuine coin flip. This is why "shrink is
    # largest near 0.5" (a real property of the old probit-shift Wang
    # Transform) does NOT hold for logit_shrink: the old transform's
    # constant additive shift produced its biggest absolute move at 0.5,
    # while logit_shrink's proportional shift produces none there and grows
    # as p moves toward the extremes.
    brain = HybridCryptoBrain()
    market = _make_market(price=0.50)
    with patch.object(brain, "_calculate_probability", return_value=0.50):
        signal = brain.evaluate(
            market, 95_000.0, entry_k=0.5, model_weight=1.0,
        )
    assert signal.post_prob == pytest.approx(0.50)


def test_shrink_is_proportional_in_logit_space():
    # The defining invariant of logit_shrink: logit(post) / logit(pre) == k
    # for every pre != 0.5, regardless of which side of 0.5 pre sits on.
    import math

    def _logit(p):
        return math.log(p / (1.0 - p))

    brain = HybridCryptoBrain()
    market = _make_market(price=0.50)
    k = 0.5

    for raw in (0.10, 0.30, 0.70, 0.90):
        with patch.object(brain, "_calculate_probability", return_value=raw):
            signal = brain.evaluate(
                market, 95_000.0, entry_k=k, model_weight=1.0,
            )
        assert _logit(signal.post_prob) == pytest.approx(_logit(raw) * k, rel=1e-6)


def test_entry_k_one_is_passthrough():
    # entry_k=1.0 is the passthrough value for logit_shrink (unlike the old
    # wang_lambda=0.0 passthrough convention - see decision_pipeline.py's
    # legacy-mode comment for why the two use opposite "off" values). With
    # model_weight=1.0 (no blending either), post_prob should equal the
    # raw probability.
    brain = HybridCryptoBrain()
    market = _make_market(price=0.50)
    with patch.object(brain, "_calculate_probability", return_value=0.70):
        signal = brain.evaluate(
            market, 95_000.0, entry_k=1.0, model_weight=1.0,
        )
    assert signal.post_prob == pytest.approx(0.70)


def test_entry_k_zero_collapses_to_half():
    # entry_k=0.0 is total distrust - every input collapses to exactly 0.5,
    # regardless of the raw probability. model_weight=1.0 isolates this
    # from blending.
    brain = HybridCryptoBrain()
    market = _make_market(price=0.50)
    with patch.object(brain, "_calculate_probability", return_value=0.95):
        signal = brain.evaluate(
            market, 95_000.0, entry_k=0.0, model_weight=1.0,
        )
    assert signal.post_prob == pytest.approx(0.50)


def test_blending_pulls_fair_toward_market():
    # wang_fair (raw=0.95, entry_k=0.5) sits well above the market price
    # of 0.50. With model_weight=0.40, the blended post_prob should land
    # strictly between wang_fair and the market price, closer to the
    # market (60% weight) than to the model (40% weight).
    brain = HybridCryptoBrain()
    market = _make_market(price=0.50)
    with patch.object(brain, "_calculate_probability", return_value=0.95):
        signal = brain.evaluate(
            market, 95_000.0, entry_k=0.5, model_weight=0.40,
        )
    wang_fair = logit_shrink(0.95, 0.5)
    expected_blended = 0.40 * wang_fair + 0.60 * 0.50
    assert signal.post_prob == pytest.approx(expected_blended)
    assert 0.50 < signal.post_prob < wang_fair
    assert abs(signal.post_prob - 0.50) < abs(signal.post_prob - wang_fair)


def test_escape_hatch_reproduces_old_behavior():
    # entry_k=1.0 + model_weight=1.0 disables both calibration layers,
    # reproducing pre-calibration behavior exactly: post_prob == raw
    # probability, regardless of market price.
    brain = HybridCryptoBrain()
    market = _make_market(price=0.50)
    with patch.object(brain, "_calculate_probability", return_value=0.80):
        signal = brain.evaluate(
            market, 95_000.0, entry_k=1.0, model_weight=1.0,
        )
    assert signal.post_prob == pytest.approx(0.80)
    assert signal.post_prob == pytest.approx(signal.pre_prob)


def test_raw_fair_value_preserved():
    # signal.pre_prob always holds the brain's unadjusted model output,
    # even when shrink + blending move post_prob far away from it.
    brain = HybridCryptoBrain()
    market = _make_market(price=0.50)
    with patch.object(brain, "_calculate_probability", return_value=0.95):
        signal = brain.evaluate(market, 95_000.0, entry_k=0.5, model_weight=0.40)

    assert signal.pre_prob == pytest.approx(0.95)
    assert signal.post_prob != pytest.approx(0.95)

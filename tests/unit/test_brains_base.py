from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from freezegun import freeze_time

from brains.base import calculate_tte, BaseBrain
from brains.crypto import HybridCryptoBrain
from brains.pricing_engine import wang_transform
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
    # wang_lambda=0.0, model_weight=1.0 isolate the EV/Kelly/min_ev gate
    # from the Wang/blend calibration layers (covered separately in
    # test_pricing_engine.py) so post_prob == the raw probability here.
    brain = HybridCryptoBrain()
    market = _make_market(price=0.50)
    with patch.object(brain, "_calculate_probability", return_value=0.80):
        signal = brain.evaluate(
            market, 95_000.0, min_ev=0.30,
            wang_lambda=0.0, model_weight=1.0, max_disagreement_ratio=float("inf"),
        )
    # EV = (0.80 - 0.50) / 0.50 = 0.60 > 0.30
    assert signal.is_tradable is True
    assert signal.kelly_size > 0
    assert signal.post_prob == pytest.approx(0.80)


def test_not_tradable_below_min_ev():
    brain = HybridCryptoBrain()
    market = _make_market(price=0.50)
    with patch.object(brain, "_calculate_probability", return_value=0.55):
        signal = brain.evaluate(market, 95_000.0, min_ev=0.30, wang_lambda=0.0, model_weight=1.0)
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
    # overshoot past the Wang-adjusted value.
    brain = HybridCryptoBrain()
    market = _make_market(price=0.50)
    with patch.object(brain, "_calculate_probability", return_value=0.80):
        signal = brain.evaluate(
            market, 95_000.0, wang_lambda=0.0, model_weight=1.5,
            max_disagreement_ratio=float("inf"),
        )
    assert signal.post_prob == pytest.approx(0.80)


def test_model_weight_below_zero_is_clamped():
    # model_weight=-0.5 should clamp to 0.0 (trust market only).
    brain = HybridCryptoBrain()
    market = _make_market(price=0.50)
    with patch.object(brain, "_calculate_probability", return_value=0.80):
        signal = brain.evaluate(
            market, 95_000.0, wang_lambda=0.0, model_weight=-0.5,
            max_disagreement_ratio=float("inf"),
        )
    assert signal.post_prob == pytest.approx(0.50)


# ── Wang Transform + market blending, applied through evaluate() ───────────

def test_wang_transform_reduces_high_probability():
    # Raw 0.95 with lambda=-0.75 shaves off overconfidence on deep-ITM
    # markets. model_weight=1.0 isolates the Wang step from blending.
    brain = HybridCryptoBrain()
    market = _make_market(price=0.50)
    with patch.object(brain, "_calculate_probability", return_value=0.95):
        signal = brain.evaluate(
            market, 95_000.0, wang_lambda=-0.75, model_weight=1.0,
            max_disagreement_ratio=float("inf"),
        )
    assert signal.post_prob < 0.90


def test_wang_transform_penalizes_uncertainty_more():
    # Near-coinflip probabilities move more, proportionally, than
    # already-extreme ones - the Wang shift is largest near p=0.5 and
    # shrinks toward 0/1.
    brain = HybridCryptoBrain()
    market = _make_market(price=0.50)
    kwargs = dict(wang_lambda=-0.75, model_weight=1.0, max_disagreement_ratio=float("inf"))

    with patch.object(brain, "_calculate_probability", return_value=0.50):
        signal_mid = brain.evaluate(market, 95_000.0, **kwargs)
    with patch.object(brain, "_calculate_probability", return_value=0.90):
        signal_high = brain.evaluate(market, 95_000.0, **kwargs)

    drop_mid = 0.50 - signal_mid.post_prob
    drop_high = 0.90 - signal_high.post_prob
    assert (drop_mid / 0.50) > (drop_high / 0.90)


def test_wang_lambda_zero_is_passthrough():
    # lambda=0.0 disables the Wang step entirely; with model_weight=1.0
    # (no blending either), post_prob should equal the raw probability.
    brain = HybridCryptoBrain()
    market = _make_market(price=0.50)
    with patch.object(brain, "_calculate_probability", return_value=0.70):
        signal = brain.evaluate(
            market, 95_000.0, wang_lambda=0.0, model_weight=1.0,
            max_disagreement_ratio=float("inf"),
        )
    assert signal.post_prob == pytest.approx(0.70)


def test_blending_pulls_fair_toward_market():
    # wang_fair (raw=0.95, lambda=-0.75) sits well above the market price
    # of 0.50. With model_weight=0.40, the blended post_prob should land
    # strictly between wang_fair and the market price, closer to the
    # market (60% weight) than to the model (40% weight).
    brain = HybridCryptoBrain()
    market = _make_market(price=0.50)
    with patch.object(brain, "_calculate_probability", return_value=0.95):
        signal = brain.evaluate(
            market, 95_000.0, wang_lambda=-0.75, model_weight=0.40,
            max_disagreement_ratio=float("inf"),
        )
    wang_fair = wang_transform(0.95, -0.75)
    expected_blended = 0.40 * wang_fair + 0.60 * 0.50
    assert signal.post_prob == pytest.approx(expected_blended)
    assert 0.50 < signal.post_prob < wang_fair
    assert abs(signal.post_prob - 0.50) < abs(signal.post_prob - wang_fair)


def test_escape_hatch_reproduces_old_behavior():
    # wang_lambda=0.0 + model_weight=1.0 disables both calibration layers,
    # reproducing pre-calibration behavior exactly: post_prob == raw
    # probability, regardless of market price.
    brain = HybridCryptoBrain()
    market = _make_market(price=0.50)
    with patch.object(brain, "_calculate_probability", return_value=0.80):
        signal = brain.evaluate(
            market, 95_000.0, wang_lambda=0.0, model_weight=1.0,
            max_disagreement_ratio=float("inf"),
        )
    assert signal.post_prob == pytest.approx(0.80)
    assert signal.post_prob == pytest.approx(signal.pre_prob)


def test_raw_fair_value_preserved():
    # signal.pre_prob always holds the brain's unadjusted model output,
    # even when Wang + blending move post_prob far away from it.
    brain = HybridCryptoBrain()
    market = _make_market(price=0.50)
    with patch.object(brain, "_calculate_probability", return_value=0.95):
        signal = brain.evaluate(market, 95_000.0, wang_lambda=-0.75, model_weight=0.40)

    assert signal.pre_prob == pytest.approx(0.95)
    assert signal.post_prob != pytest.approx(0.95)

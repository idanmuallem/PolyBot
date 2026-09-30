from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from freezegun import freeze_time

from brains.base import calculate_tte, BaseBrain
from brains.crypto import HybridCryptoBrain
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
    # entry_k=1.0 (trust the brain fully -> post_prob == raw probability)
    # isolates the EV/Kelly/min_ev gate from the market-anchored trust blend
    # (covered separately in test_pricing_engine.py).
    brain = HybridCryptoBrain()
    market = _make_market(price=0.50)
    with patch.object(brain, "_calculate_probability", return_value=0.80):
        signal = brain.evaluate(
            market, 95_000.0, min_ev=0.30, entry_k=1.0,
        )
    # EV = (0.80 - 0.50) / 0.50 = 0.60 > 0.30
    assert signal.is_tradable is True
    assert signal.kelly_size > 0
    assert signal.post_prob == pytest.approx(0.80)


def test_not_tradable_below_min_ev():
    brain = HybridCryptoBrain()
    market = _make_market(price=0.50)
    with patch.object(brain, "_calculate_probability", return_value=0.55):
        signal = brain.evaluate(market, 95_000.0, min_ev=0.30, entry_k=1.0)
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


# ── Market-anchored entry pricing, applied through evaluate() ──────────────
#
# post = sigmoid(logit(market) + entry_k*(logit(brain) - logit(market))):
# a single trust-weighted interpolation between the market price and the
# brain's raw probability (brains/pricing_engine.market_anchored_shrink).
# entry_k=0 -> market, entry_k=1 -> brain, and brain==market -> post==market
# for every k. This replaced a shrink-toward-0.5 step + a separate
# model_weight blend; model_weight is no longer a parameter of evaluate().

def test_agreement_yields_no_edge():
    # THE KEY REGRESSION TEST for the market-anchored redesign: when the brain
    # agrees with the market, post_prob == market and there is NO edge, at any
    # entry_k. The prior shrink-toward-0.5 design manufactured a spurious YES
    # edge here on cheap markets (see A1_TRADE_RATE_FINDINGS.md).
    brain = HybridCryptoBrain()
    market = _make_market(price=0.10)
    for k in (0.0, 0.3, 0.5, 0.8, 1.0):
        with patch.object(brain, "_calculate_probability", return_value=0.10):
            signal = brain.evaluate(market, 95_000.0, entry_k=k)
        assert signal.post_prob == pytest.approx(0.10, abs=1e-9), f"k={k}"
        # EV on both sides is ~0, so nothing is tradable at a real min_ev.
        assert signal.is_tradable is False, f"k={k}"


def test_cheap_market_agreement_not_tradable():
    # Direct contrast with the old bug: market=0.10, brain=0.10 used to yield
    # post_prob=0.16 -> ev_yes=+0.60 (tradable). Market-anchored yields
    # post_prob=0.10 -> ev_yes=0 (not tradable).
    brain = HybridCryptoBrain()
    market = _make_market(price=0.10)
    with patch.object(brain, "_calculate_probability", return_value=0.10):
        signal = brain.evaluate(market, 95_000.0, entry_k=0.5, min_ev=0.30)
    assert signal.post_prob == pytest.approx(0.10, abs=1e-9)
    assert signal.is_tradable is False


def test_entry_k_zero_defers_to_market():
    # entry_k=0.0 -> the brain is ignored entirely, post_prob == market price,
    # regardless of how far the raw probability diverges.
    brain = HybridCryptoBrain()
    market = _make_market(price=0.30)
    with patch.object(brain, "_calculate_probability", return_value=0.95):
        signal = brain.evaluate(market, 95_000.0, entry_k=0.0)
    assert signal.post_prob == pytest.approx(0.30, abs=1e-9)


def test_entry_k_one_trusts_brain():
    # entry_k=1.0 -> the market is ignored, post_prob == raw brain probability.
    brain = HybridCryptoBrain()
    market = _make_market(price=0.30)
    with patch.object(brain, "_calculate_probability", return_value=0.70):
        signal = brain.evaluate(market, 95_000.0, entry_k=1.0)
    assert signal.post_prob == pytest.approx(0.70, abs=1e-9)


def test_partial_trust_lands_between_market_and_brain():
    # 0 < entry_k < 1 -> post_prob sits strictly between market and brain, on
    # the brain's side of the market.
    brain = HybridCryptoBrain()
    market = _make_market(price=0.20)
    with patch.object(brain, "_calculate_probability", return_value=0.60):
        signal = brain.evaluate(market, 95_000.0, entry_k=0.5)
    assert 0.20 < signal.post_prob < 0.60


def test_interpolation_is_linear_in_logit_space():
    # The defining invariant: logit(post) == logit(market) + k*(logit(brain)
    # - logit(market)), for divergent brain/market on either side.
    import math

    def _logit(p):
        return math.log(p / (1.0 - p))

    brain = HybridCryptoBrain()
    k = 0.5
    for mkt, raw in ((0.20, 0.60), (0.40, 0.10), (0.15, 0.80)):
        market = _make_market(price=mkt)
        with patch.object(brain, "_calculate_probability", return_value=raw):
            signal = brain.evaluate(market, 95_000.0, entry_k=k)
        expected = _logit(mkt) + k * (_logit(raw) - _logit(mkt))
        assert _logit(signal.post_prob) == pytest.approx(expected, rel=1e-6)


def test_raw_fair_value_preserved():
    # signal.pre_prob always holds the brain's unadjusted model output, even
    # when the market-anchored blend moves post_prob away from it.
    brain = HybridCryptoBrain()
    market = _make_market(price=0.30)
    with patch.object(brain, "_calculate_probability", return_value=0.95):
        signal = brain.evaluate(market, 95_000.0, entry_k=0.5)
    assert signal.pre_prob == pytest.approx(0.95)
    assert signal.post_prob != pytest.approx(0.95)
    # wang_fair_value now equals post_prob (no separate pre-blend value).
    assert signal.wang_fair_value == pytest.approx(signal.post_prob)

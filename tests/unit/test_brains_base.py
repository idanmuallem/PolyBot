from datetime import datetime, timezone, timedelta
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
    brain = HybridCryptoBrain()
    market = _make_market(price=0.50)
    with patch.object(brain, "_calculate_probability", return_value=0.80):
        signal = brain.evaluate(market, 95_000.0, min_ev=0.30)
    # EV = (0.80 - 0.50) / 0.50 = 0.60 > 0.30
    assert signal.is_tradable is True
    assert signal.kelly_size > 0
    assert signal.fair_value == pytest.approx(0.80)


def test_not_tradable_below_min_ev():
    brain = HybridCryptoBrain()
    market = _make_market(price=0.50)
    with patch.object(brain, "_calculate_probability", return_value=0.55):
        signal = brain.evaluate(market, 95_000.0, min_ev=0.30)
    # EV = (0.55 - 0.50) / 0.50 = 0.10 < 0.30
    assert signal.is_tradable is False


def test_fair_value_clamped_to_unit_interval():
    brain = HybridCryptoBrain()
    market = _make_market(price=0.50)
    with patch.object(brain, "_calculate_probability", return_value=1.5):
        signal = brain.evaluate(market, 95_000.0)
    assert signal.fair_value == 1.0


def test_fair_value_clamped_below_zero():
    brain = HybridCryptoBrain()
    market = _make_market(price=0.50)
    with patch.object(brain, "_calculate_probability", return_value=-0.3):
        signal = brain.evaluate(market, 95_000.0)
    assert signal.fair_value == 0.0

from datetime import datetime, timezone, timedelta

import pytest
from freezegun import freeze_time

from brains.crypto import HybridCryptoBrain
from core.models import MarketData


def _make_market(
    price=0.50,
    strike=100_000.0,
    expiry="2026-12-31",
    market_name=None,
    asset_type="Crypto::BTCUSDT",
):
    return MarketData(
        market_id="tok1",
        asset_type=asset_type,
        strike_price=strike,
        question="Will BTC exceed $100,000?",
        market_name=market_name or "Bitcoin - Will BTC exceed $100,000?",
        initial_price=price,
        volume=500_000.0,
        expiry_date=expiry,
    )


# ── _calculate_prob (Black-Scholes) ──────────────────────────────────────────

def test_atm_prob_near_half():
    brain = HybridCryptoBrain()
    prob = brain._calculate_prob(100_000, 100_000, 365, 0.5)
    assert 0.4 < prob < 0.5  # slightly below 0.5 due to -0.5σ²t drift term


def test_deep_itm_prob_near_one():
    brain = HybridCryptoBrain()
    prob = brain._calculate_prob(200_000, 100_000, 30, 0.5)
    assert prob > 0.95


def test_deep_otm_prob_near_zero():
    brain = HybridCryptoBrain()
    prob = brain._calculate_prob(50_000, 100_000, 30, 0.5)
    assert prob < 0.05


def test_zero_tte_spot_above_strike():
    brain = HybridCryptoBrain()
    assert brain._calculate_prob(110_000, 100_000, 0, 0.5) == 1.0


def test_zero_tte_spot_below_strike():
    brain = HybridCryptoBrain()
    assert brain._calculate_prob(90_000, 100_000, 0, 0.5) == 0.0


# ── Model selector via evaluate_fair_value ───────────────────────────────────

@freeze_time("2026-06-02T00:00:00+00:00")
def test_selects_short_term_under_1_day():
    brain = HybridCryptoBrain()
    expiry = (datetime(2026, 6, 2, tzinfo=timezone.utc) + timedelta(hours=12)).isoformat()
    market = _make_market(expiry=expiry)
    brain._calculate_probability(market, 95_000.0)
    assert brain.last_model_used == "short_term"


@freeze_time("2026-06-02T00:00:00+00:00")
def test_selects_standard_bs_1_to_90_days():
    brain = HybridCryptoBrain()
    expiry = (datetime(2026, 6, 2, tzinfo=timezone.utc) + timedelta(days=10)).isoformat()
    market = _make_market(expiry=expiry)
    brain._calculate_probability(market, 95_000.0)
    assert brain.last_model_used == "standard_bs"


@freeze_time("2026-06-02T00:00:00+00:00")
def test_selects_standard_bs_at_60_days():
    # 60 days used to route to Heston (old 30-day cutoff); Heston was
    # dropped entirely (its output wasn't actually a probability - see
    # HybridCryptoBrain's docstring), so every TTE >= 1 day now stays on BS.
    brain = HybridCryptoBrain()
    expiry = (datetime(2026, 6, 2, tzinfo=timezone.utc) + timedelta(days=60)).isoformat()
    market = _make_market(expiry=expiry)
    brain._calculate_probability(market, 95_000.0)
    assert brain.last_model_used == "standard_bs"


@freeze_time("2026-06-02T00:00:00+00:00")
def test_selects_standard_bs_above_90_days():
    # 120 days used to route to Heston (old 90-day cutoff, before Heston
    # was dropped entirely). Long-dated markets stay on Black-Scholes too.
    brain = HybridCryptoBrain()
    expiry = (datetime(2026, 6, 2, tzinfo=timezone.utc) + timedelta(days=120)).isoformat()
    market = _make_market(expiry=expiry)
    brain._calculate_probability(market, 95_000.0)
    assert brain.last_model_used == "standard_bs"


# ── Inversion keywords ───────────────────────────────────────────────────────

@freeze_time("2026-06-02T00:00:00+00:00")
def test_inversion_keyword_flips_probability():
    brain_above = HybridCryptoBrain()
    brain_below = HybridCryptoBrain()
    expiry = (datetime(2026, 6, 2, tzinfo=timezone.utc) + timedelta(days=10)).isoformat()
    m_above = _make_market(expiry=expiry, market_name="Bitcoin Will BTC exceed $100,000?")
    m_below = _make_market(expiry=expiry, market_name="Bitcoin Will BTC trade below $100,000?")
    prob_above = brain_above._calculate_probability(m_above, 95_000.0)
    prob_below = brain_below._calculate_probability(m_below, 95_000.0)
    assert abs(prob_above - (1.0 - prob_below)) < 0.01


# ── _price_short_term ─────────────────────────────────────────────────────────

def test_sigmoid_mid_at_equal_prices():
    brain = HybridCryptoBrain()
    result = brain._price_short_term(100_000.0, 100_000.0)
    assert abs(result - 0.5) < 0.01


def test_sigmoid_above_strike_favors_yes():
    brain = HybridCryptoBrain()
    result = brain._price_short_term(110_000.0, 100_000.0)
    assert result > 0.5


def test_sigmoid_zero_current_price_returns_half():
    brain = HybridCryptoBrain()
    assert brain._price_short_term(0, 100_000.0) == 0.5


def test_sigmoid_zero_strike_price_returns_half():
    brain = HybridCryptoBrain()
    assert brain._price_short_term(100_000.0, 0) == 0.5


# ── Enriched live_truth (CCXTDataClient dict) ────────────────────────────────

def test_calculate_probability_accepts_enriched_dict():
    brain = HybridCryptoBrain()
    market = _make_market()
    enriched = {
        "spot_price": 95_000.0,
        "vol_30d": 0.4,
        "vol_60d": 0.45,
        "vol_90d": 0.5,
        "funding_rate": 0.0001,
        "volume_24h": 1_000_000.0,
        "spread": 0.001,
    }
    # Should not raise, and should match the float-only call using the same spot price.
    prob_dict = brain._calculate_probability(market, enriched)
    prob_float = brain._calculate_probability(market, 95_000.0)
    assert 0.0 <= prob_dict <= 1.0
    # They can differ (dict path uses realized vol instead of the hardcoded
    # default), but both must be valid probabilities computed from the same spot.
    assert isinstance(prob_float, float)


def test_select_volatility_prefers_realized_vol_when_enriched():
    brain = HybridCryptoBrain()
    market = _make_market(expiry="2026-06-15")  # short horizon -> vol_30d
    enriched = {"spot_price": 95_000.0, "vol_30d": 0.42, "vol_60d": 0.5, "vol_90d": 0.6}
    with freeze_time("2026-06-01T00:00:00+00:00"):
        vol = brain._select_volatility(market, enriched)
    assert vol == pytest.approx(0.42)


def test_select_volatility_falls_back_without_enriched_data():
    brain = HybridCryptoBrain()
    market = _make_market(asset_type="Crypto::BTCUSDT")
    assert brain._select_volatility(market, None) == brain.get_volatility_for_symbol("Crypto::BTCUSDT")


def test_unpack_live_truth_float_vs_dict():
    spot, enriched = HybridCryptoBrain._unpack_live_truth(95_000.0)
    assert spot == 95_000.0 and enriched is None

    spot, enriched = HybridCryptoBrain._unpack_live_truth({"spot_price": 95_000.0, "vol_30d": 0.4})
    assert spot == 95_000.0 and enriched == {"spot_price": 95_000.0, "vol_30d": 0.4}

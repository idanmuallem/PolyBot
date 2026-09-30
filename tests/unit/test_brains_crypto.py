import math
from datetime import datetime, timezone, timedelta

import pytest
from freezegun import freeze_time
from scipy.stats import norm

from brains.crypto import HybridCryptoBrain, classify_resolution_type
from core.models import MarketData

# Real resolution-criteria text pulled from Polymarket's Gamma API for the
# markets this bot has actually traded (see decision_pipeline diagnosis) -
# used to validate classify_resolution_type against real wording, not a
# hand-crafted approximation of it.
TOUCH_DESCRIPTION_SAMPLES = [
    # BTC reach $90,000 by Dec 31 2026
    'This market will immediately resolve to "Yes" if any Binance 1 minute '
    'candle for Bitcoin (BTC/USDT) between the creation of this market and '
    'December 31, 2026, 23:59 in the ET timezone has a final "High" price '
    'equal to or greater than the price specified in the title. Otherwise, '
    'this market will resolve to "No."',
    # BTC reach $85,000 by Dec 31 2026 (slightly different wording)
    'This market will resolve to "Yes" if any Binance 1 minute candle for '
    'Bitcoin (BTC/USDT) between the creation of this market through 11:59 '
    'PM ET on the last day of the year specified in the title has a final '
    '"High" price equal to or greater than the price specified in the '
    'title. Otherwise, this market will resolve to "No≈".',
    # ETH dip to $2,250 by Dec 31 2026 (High-or-Low wording, mangled quotes)
    'This market will resolve to ≈Yes≈ if any Binance 1-minute '
    'candle for Ethereum (ETH/USDT) from the creation of this market '
    'through 11:59 PM ET on the last day of the year specified in the '
    'title has a final "High" or "Low" price equal to or beyond (above '
    'for ≈ High Prices, below for ≈ Low Prices) the listed price.',
    # BTC dip to $75,000 in September (monthly ladder, Low-price wording)
    'This market will immediately resolve to "Yes" if any Binance 1 minute '
    'candle for BTC/USDT during the month specified in the title (from '
    '00:00 AM ET on the first day to 11:59 PM ET on the last), has a '
    'final Low price equal to or lower than the price specified in the '
    'title. Otherwise, this market will resolve to "No."',
    # "all time high by" sibling market (running-max wording)
    'This market will resolve to "Yes" if any Binance 1 minute candle for '
    'BTC/USDT between 16 December \'25 10:30 and 11:59PM ET on the date '
    'specified in the title has a final "High" price that is higher than '
    'any previous Binance 1 minute candle\'s "High" price on any prior '
    'date. Otherwise, this market will resolve to "No".',
]

# Same GBM zero-log-drift first-passage formula, computed independently here
# (not imported from brains/crypto.py) so the test can't pass just because
# it shares a bug with the implementation.
def _reference_first_passage(spot, strike, tte_days, vol):
    t = tte_days / 365.0
    nu = -0.5 * vol * vol
    s = vol * math.sqrt(t)
    if strike >= spot:
        b = math.log(strike / spot)
        return norm.cdf((nu * t - b) / s) + math.exp(2 * nu * b / vol**2) * norm.cdf((-nu * t - b) / s)
    b = math.log(spot / strike)
    return norm.cdf((-nu * t - b) / s) + math.exp(-2 * nu * b / vol**2) * norm.cdf((nu * t - b) / s)


def _make_market(
    price=0.50,
    strike=100_000.0,
    expiry="2026-12-31",
    market_name=None,
    asset_type="Crypto::BTCUSDT",
    description="",
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
        description=description,
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


@freeze_time("2026-06-02T00:00:00+00:00")
def test_dip_phrasing_triggers_inversion():
    # "dip to $X" is a downward phrasing that was previously missing from
    # invert_keywords, so a terminal-path "dip to" market was priced in the
    # wrong direction. It should now invert like "below" does: prob for a
    # "dip to" market ≈ 1 - prob for the equivalent "exceed" market.
    # (Terminal path: default description="" classifies as non-touch.)
    brain_above = HybridCryptoBrain()
    brain_dip = HybridCryptoBrain()
    expiry = (datetime(2026, 6, 2, tzinfo=timezone.utc) + timedelta(days=10)).isoformat()
    m_above = _make_market(expiry=expiry, market_name="Bitcoin Will BTC exceed $100,000?")
    m_dip = _make_market(expiry=expiry, market_name="Will Bitcoin dip to $100,000?")
    prob_above = brain_above._calculate_probability(m_above, 95_000.0)
    prob_dip = brain_dip._calculate_probability(m_dip, 95_000.0)
    # Confirm the dip market actually took the terminal (invertible) path,
    # not first-passage (which would skip inversion and break this contract).
    assert brain_dip.last_model_used != "first_passage"
    assert abs(prob_above - (1.0 - prob_dip)) < 0.01


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


# ── classify_resolution_type ──────────────────────────────────────────────────

@pytest.mark.parametrize("description", TOUCH_DESCRIPTION_SAMPLES)
def test_classify_resolution_type_recognizes_real_touch_wording(description):
    assert classify_resolution_type(description) == "touch"


@pytest.mark.parametrize("description", [
    "",
    None,
    "This market will resolve to Yes if candidate X wins the election on "
    "November 3rd based on the AP call.",
    "Will Bitcoin be above $100,000 on December 31, 2026 at 11:59pm ET, "
    "based on the Coinbase spot price at that exact time?",
])
def test_classify_resolution_type_defaults_to_unknown_for_non_touch_text(description):
    assert classify_resolution_type(description) == "unknown"


# ── _price_first_passage ──────────────────────────────────────────────────────

def test_first_passage_matches_independent_reference_implementation():
    brain = HybridCryptoBrain()
    cases = [
        (79214.0, 90000.0, 115.15, 0.41),
        (79214.0, 82500.0, 23.15, 0.49),
        (78708.0, 75000.0, 22.78, 0.49),  # barrier below spot
        (2473.0, 2250.0, 114.75, 0.58),   # barrier below spot
    ]
    for spot, strike, tte, vol in cases:
        got = brain._price_first_passage(spot, strike, tte, vol)
        want = _reference_first_passage(spot, strike, tte, vol)
        assert got == pytest.approx(want, abs=1e-9)


def test_first_passage_exceeds_terminal_probability():
    """A first-passage (ever-touches) probability must be >= the terminal
    (ends-above) probability for the same inputs - touching is strictly
    easier than ending there. This is the core miscalibration the backtest
    found: terminal systematically underestimates touch-resolving markets."""
    brain = HybridCryptoBrain()
    spot, strike, tte, vol = 79214.0, 90000.0, 115.15, 0.5
    touch_prob = brain._price_first_passage(spot, strike, tte, vol)
    terminal_prob = brain._calculate_prob(spot, strike, tte, vol)
    assert touch_prob > terminal_prob


def test_first_passage_barrier_exactly_at_spot_returns_one():
    """strike == spot is the only reachable "already there" case - direction
    is auto-selected from strike-vs-spot, so a barrier strictly on the far
    side of spot from where it started always routes to the other branch
    (see _price_first_passage's docstring) rather than reaching this
    boundary check with a negative distance."""
    brain = HybridCryptoBrain()
    assert brain._price_first_passage(90_000.0, 90_000.0, 30, 0.5) == 1.0


def test_first_passage_closer_barrier_is_more_likely_touched():
    brain = HybridCryptoBrain()
    near = brain._price_first_passage(90_000.0, 95_000.0, 30, 0.5)
    far = brain._price_first_passage(90_000.0, 150_000.0, 30, 0.5)
    assert near > far


def test_first_passage_zero_tte_matches_terminal_at_the_boundary():
    brain = HybridCryptoBrain()
    assert brain._price_first_passage(110_000.0, 100_000.0, 0, 0.5) == 1.0
    assert brain._price_first_passage(90_000.0, 100_000.0, 0, 0.5) == 0.0


def test_first_passage_invalid_inputs_return_half():
    brain = HybridCryptoBrain()
    assert brain._price_first_passage(0.0, 100_000.0, 30, 0.5) == 0.5
    assert brain._price_first_passage(100_000.0, 0.0, 30, 0.5) == 0.5


# ── Resolution-type routing (evaluate_fair_value / _calculate_probability) ────

@freeze_time("2026-06-02T00:00:00+00:00")
def test_touch_description_routes_to_first_passage():
    brain = HybridCryptoBrain()
    expiry = (datetime(2026, 6, 2, tzinfo=timezone.utc) + timedelta(days=30)).isoformat()
    market = _make_market(expiry=expiry, description=TOUCH_DESCRIPTION_SAMPLES[0])
    brain._calculate_probability(market, 95_000.0)
    assert brain.last_model_used == "first_passage"


@freeze_time("2026-06-02T00:00:00+00:00")
def test_missing_description_keeps_existing_terminal_path():
    """No description (the default) must fall back to today's existing
    behavior, never a different one - this is the safe-default contract
    classify_resolution_type documents."""
    brain = HybridCryptoBrain()
    expiry = (datetime(2026, 6, 2, tzinfo=timezone.utc) + timedelta(days=30)).isoformat()
    market = _make_market(expiry=expiry, description="")
    brain._calculate_probability(market, 95_000.0)
    assert brain.last_model_used == "standard_bs"


@freeze_time("2026-06-02T00:00:00+00:00")
def test_touch_path_is_not_double_inverted_for_down_phrased_question():
    """A 'dip to' / down-phrased touch market must NOT also go through the
    keyword-based inversion _calculate_probability applies to the terminal
    path - _price_first_passage already picks touch direction from strike-
    vs-spot and returns P(stated condition) directly. Regression guard for
    exactly the bug this would silently reintroduce: whether or not an
    invert keyword ("below") appears in the market name must make zero
    difference to a first_passage-routed result - only the terminal path
    is supposed to be keyword-sensitive."""
    brain_with_keyword = HybridCryptoBrain()
    brain_without_keyword = HybridCryptoBrain()
    expiry = (datetime(2026, 6, 2, tzinfo=timezone.utc) + timedelta(days=115)).isoformat()
    market_with_keyword = _make_market(
        strike=2250.0, expiry=expiry,
        market_name="Ethereum - Will ETH trade below $2,250?",  # contains "below"
        description=TOUCH_DESCRIPTION_SAMPLES[2],
    )
    market_without_keyword = _make_market(
        strike=2250.0, expiry=expiry,
        market_name="Ethereum - ETH $2,250 threshold market",  # no invert keyword
        description=TOUCH_DESCRIPTION_SAMPLES[2],
    )
    prob_with = brain_with_keyword._calculate_probability(market_with_keyword, 2473.0)
    prob_without = brain_without_keyword._calculate_probability(market_without_keyword, 2473.0)

    assert brain_with_keyword.last_model_used == "first_passage"
    assert brain_without_keyword.last_model_used == "first_passage"
    assert prob_with == pytest.approx(prob_without, abs=1e-9)
    # Sanity: spot (2473) is above the 2250 barrier the strike represents,
    # so P(ever dips to 2250) should be well under certainty, not near 1
    # (which is what a wrongly-double-inverted "P(never dips)" could also
    # coincidentally look like at other parameter values - the exact
    # cross-check above is the real regression guard, this is a sanity nudge).
    assert 0.0 < prob_with < 1.0

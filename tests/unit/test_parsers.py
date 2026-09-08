from hunters.parsers import extract_crypto_strike, extract_economy_strike


# ── extract_crypto_strike ─────────────────────────────────────────────────────

def test_basic_dollar_strike():
    result = extract_crypto_strike("Will BTC exceed $100,000 by December?", 95_000.0)
    assert result == 100_000.0


def test_million_suffix_expansion():
    result = extract_crypto_strike("Will BTC reach 100M?", 95_000_000.0)
    assert result == 100_000_000.0


def test_year_stripping_does_not_produce_strike():
    # "2026" is stripped before number extraction; remaining number $95000 is returned
    result = extract_crypto_strike("Will BTC hit $95000 by 2026?", 95_000.0)
    assert result == 95_000.0


def test_anchor_ratio_rejects_far_outlier():
    # 999999999 / 95000 >> 50 → filtered, leaving 95000
    result = extract_crypto_strike("Token ID 999999999 will exceed $95000", 95_000.0)
    assert result == 95_000.0


def test_empty_text_returns_none():
    assert extract_crypto_strike("", 95_000.0) is None


def test_none_text_returns_none():
    assert extract_crypto_strike(None, 95_000.0) is None


def test_no_numbers_returns_none():
    assert extract_crypto_strike("Will bitcoin outperform gold?", 95_000.0) is None


def test_closest_candidate_selected():
    # Two valid numbers [80000, 100000]; anchor=95000 → closest is 100000
    result = extract_crypto_strike("Will BTC be above $80000 or $100000?", 95_000.0)
    assert result == 100_000.0


# ── extract_economy_strike ────────────────────────────────────────────────────

def test_bps_conversion():
    # 425 bps → 4.25, which is close to anchor 4.25
    result = extract_economy_strike("Will Fed rate exceed 425 bps?", 4.25)
    assert result == 4.25


def test_percent_extraction():
    result = extract_economy_strike("Will CPI exceed 3.5%?", 3.5)
    assert result == 3.5


def test_window_filter_rejects_far_values():
    # 99 is more than 5 units from anchor 4.25
    result = extract_economy_strike("Will rate exceed 99%?", 4.25)
    assert result is None


def test_economy_window_accepts_close_value():
    result = extract_economy_strike("Will rate fall to 4.0%?", 4.25)
    assert result == 4.0

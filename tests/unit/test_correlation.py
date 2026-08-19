import pytest

from core.trading_config import TradingConfig
from trading.correlation import CorrelationTracker


def _tracker():
    return CorrelationTracker(config=TradingConfig())


# ── pairwise_correlation ──────────────────────────────────────────────────────

def test_same_market_is_fully_correlated():
    tracker = _tracker()
    assert tracker.pairwise_correlation("Crypto::BTCUSDT", "Crypto::BTCUSDT") == 1.0


def test_same_crypto_asset_different_case_fully_correlated():
    tracker = _tracker()
    # e.g. two BTCUSDT markets discovered via different symbol strings
    assert tracker.pairwise_correlation("Crypto::BTC", "Crypto::BTCUSDT") == 1.0


def test_related_crypto_assets_use_static_lookup():
    tracker = _tracker()
    assert tracker.pairwise_correlation("Crypto::BTCUSDT", "Crypto::ETHUSDT") == pytest.approx(0.85)
    assert tracker.pairwise_correlation("Crypto::ETHUSDT", "Crypto::BTCUSDT") == pytest.approx(0.85)  # symmetric


def test_unrelated_crypto_asset_falls_back_to_default():
    tracker = _tracker()
    # No entry for DOGE -> falls back to the same-category default.
    assert tracker.pairwise_correlation("Crypto::DOGEUSDT", "Crypto::BTCUSDT") == pytest.approx(0.30)


def test_different_category_is_uncorrelated():
    tracker = _tracker()
    assert tracker.pairwise_correlation("Crypto::BTCUSDT", "Weather::Miami") == 0.0


def test_same_non_crypto_category_uses_moderate_default():
    tracker = _tracker()
    assert tracker.pairwise_correlation("Weather::Miami", "Weather::NYC") == pytest.approx(0.30)


def test_empty_asset_type_is_uncorrelated():
    tracker = _tracker()
    assert tracker.pairwise_correlation("", "Crypto::BTCUSDT") == 0.0
    assert tracker.pairwise_correlation("Crypto::BTCUSDT", "") == 0.0


# ── exposure_for_new_position ─────────────────────────────────────────────────

def test_exposure_zero_with_empty_book():
    tracker = _tracker()
    assert tracker.exposure_for_new_position("Crypto::BTCUSDT", []) == 0.0


def test_exposure_averages_across_open_book():
    tracker = _tracker()
    exposure = tracker.exposure_for_new_position(
        "Crypto::BTCUSDT", ["Crypto::BTCUSDT", "Weather::Miami"]
    )
    # avg(1.0, 0.0) == 0.5
    assert exposure == pytest.approx(0.5)


def test_exposure_high_with_duplicate_book():
    tracker = _tracker()
    exposure = tracker.exposure_for_new_position(
        "Crypto::BTCUSDT", ["Crypto::BTCUSDT", "Crypto::BTCUSDT"]
    )
    assert exposure == 1.0


# ── correlation_matrix ────────────────────────────────────────────────────────

def test_correlation_matrix_includes_diagonal():
    tracker = _tracker()
    matrix = tracker.correlation_matrix(["Crypto::BTCUSDT", "Crypto::ETHUSDT"])
    assert matrix[("Crypto::BTCUSDT", "Crypto::BTCUSDT")] == 1.0
    assert matrix[("Crypto::ETHUSDT", "Crypto::ETHUSDT")] == 1.0


def test_correlation_matrix_is_symmetric():
    tracker = _tracker()
    matrix = tracker.correlation_matrix(["Crypto::BTCUSDT", "Crypto::ETHUSDT", "Weather::Miami"])
    for a in ("Crypto::BTCUSDT", "Crypto::ETHUSDT", "Weather::Miami"):
        for b in ("Crypto::BTCUSDT", "Crypto::ETHUSDT", "Weather::Miami"):
            assert matrix[(a, b)] == pytest.approx(matrix[(b, a)])


def test_correlation_matrix_size():
    tracker = _tracker()
    assets = ["Crypto::BTCUSDT", "Crypto::ETHUSDT", "Crypto::SOLUSDT"]
    matrix = tracker.correlation_matrix(assets)
    assert len(matrix) == len(assets) * len(assets)

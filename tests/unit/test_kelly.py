"""The Kelly formula: kelly_optimal = edge / odds, scaled by model
confidence and kelly_fraction, in dollars of bankroll. Caps (max bet,
budget) are the Account's, not the formula's."""
import pytest

from trading.kelly import kelly_bet_size


def _size(edge, odds, confidence=1.0, kelly_fraction=0.25, bankroll_usd=1000.0):
    return kelly_bet_size(edge=edge, odds=odds, confidence=confidence,
                          kelly_fraction=kelly_fraction, bankroll_usd=bankroll_usd)


def test_worked_example():
    # 0.01 / 0.99 x 0.25 x $1000
    assert _size(0.01, 0.99) == pytest.approx(2.5253, abs=1e-4)


def test_not_capped_at_any_dollar_ceiling():
    # 0.30 / 0.60 x 0.25 x $1000
    assert _size(0.30, 0.60) == pytest.approx(125.0)


def test_zero_for_non_positive_edge_or_odds():
    assert _size(0.0, 0.5) == 0.0
    assert _size(-0.1, 0.5) == 0.0
    assert _size(0.3, 0.0) == 0.0
    assert _size(0.3, -0.2) == 0.0


def test_scales_with_confidence_clamped_to_unit_interval():
    assert _size(0.01, 0.99, confidence=0.5) == pytest.approx(_size(0.01, 0.99) / 2)
    assert _size(0.01, 0.99, confidence=5.0) == pytest.approx(_size(0.01, 0.99))
    assert _size(0.3, 0.6, confidence=-1.0) == 0.0


def test_scales_with_kelly_fraction():
    assert _size(0.01, 0.99, kelly_fraction=0.5) == pytest.approx(_size(0.01, 0.99) * 2)

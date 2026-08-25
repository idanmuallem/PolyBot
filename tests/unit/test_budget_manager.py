import pytest

from core.bridge import DataBridge
from core.trading_config import TradingConfig
from trading.budget_manager import BudgetManager


def _make_manager(
    initial_balance=100.0, daily_limit=15.0, min_balance=5.0, bankroll=1000.0,
    max_bet_size_usd=3.0, kelly_fraction=0.25,
):
    bridge = DataBridge()
    bridge.current_balance = initial_balance
    config = TradingConfig(
        bankroll_usd=bankroll,
        daily_limit_usd=daily_limit,
        # Per-strategy limits default to $50 each (core/trading_config.py);
        # pin both to daily_limit here so pre-existing tests below — which
        # exercise the default strategy_tag="crypto" and were written before
        # per-strategy budgets existed — see the same ceiling as before.
        crypto_daily_limit_usd=daily_limit,
        arbitrage_daily_limit_usd=daily_limit,
        min_trading_balance=min_balance,
        max_bet_size_usd=max_bet_size_usd,
        kelly_fraction=kelly_fraction,
        dry_run=True,
    )
    mgr = BudgetManager(bridge=bridge, config=config, initial_balance=initial_balance)
    return mgr, bridge


def test_under_daily_limit():
    mgr, _ = _make_manager()
    # 0.001 * 1000 = $1.0 desired; remaining = $15.0 → bet = $1.0
    bet, ok = mgr.check_and_cap_bet(0.001)
    assert ok is True
    assert bet == 1.0


def test_at_daily_limit_returns_zero():
    mgr, _ = _make_manager()
    mgr.spent_by_strategy["crypto"] = 15.0
    bet, ok = mgr.check_and_cap_bet(0.001)
    assert ok is False
    assert bet == 0.0


def test_caps_at_remaining_budget():
    mgr, _ = _make_manager()
    mgr.spent_by_strategy["crypto"] = 12.0  # $3.0 remaining
    # 0.01 * 1000 = $10.0 desired, but only $3.0 left
    bet, ok = mgr.check_and_cap_bet(0.01)
    assert ok is True
    assert bet == 3.0


def test_record_trade_increments_spent():
    mgr, bridge = _make_manager(initial_balance=50.0)
    mgr.record_trade(2.5)
    assert mgr.total_spent_today == 2.5
    assert bridge.daily_spend == 2.5


def test_record_trade_decrements_bridge_balance():
    mgr, bridge = _make_manager(initial_balance=50.0)
    mgr.record_trade(2.5)
    # _sync_bridge sets current_balance = max(base_balance - spent, 0)
    assert bridge.current_balance == 47.5


def test_watch_only_when_below_minimum():
    mgr, bridge = _make_manager(initial_balance=3.0, min_balance=5.0)
    assert mgr.watch_only is True
    assert bridge.watch_only is True


def test_not_watch_only_above_minimum():
    mgr, bridge = _make_manager(initial_balance=10.0, min_balance=5.0)
    assert mgr.watch_only is False
    assert bridge.watch_only is False


def test_kelly_times_bankroll_capped_by_daily_limit():
    mgr, _ = _make_manager()
    # 0.02 * 1000 = $20 desired, but daily_limit=15 → capped to $15
    bet, ok = mgr.check_and_cap_bet(0.02)
    assert ok is True
    assert bet == 15.0


def test_remaining_budget_decreases_after_record():
    mgr, _ = _make_manager()
    assert mgr.get_remaining_budget() == 15.0
    mgr.record_trade(5.0)
    assert mgr.get_remaining_budget() == 10.0


# ── compute_kelly_bet_size ────────────────────────────────────────────────────

def test_compute_kelly_bet_size_clamped_to_max_bet_size():
    mgr, _ = _make_manager(bankroll=1000.0, max_bet_size_usd=3.0, kelly_fraction=0.25)
    # kelly_raw = 0.30/0.60 = 0.5, scaled = 0.125, dollar = $125 -> clamped
    bet = mgr.compute_kelly_bet_size(edge=0.30, odds=0.60, confidence=1.0)
    assert bet == 3.0


def test_compute_kelly_bet_size_below_ceiling_is_not_clamped():
    mgr, _ = _make_manager(bankroll=1000.0, max_bet_size_usd=100.0, kelly_fraction=0.25)
    bet = mgr.compute_kelly_bet_size(edge=0.01, odds=0.99, confidence=1.0)
    expected = (0.01 / 0.99) * 0.25 * 1000.0
    assert bet == pytest.approx(expected)
    assert 0.0 < bet < 100.0


def test_compute_kelly_bet_size_zero_for_non_positive_edge():
    mgr, _ = _make_manager()
    assert mgr.compute_kelly_bet_size(edge=0.0, odds=0.5) == 0.0
    assert mgr.compute_kelly_bet_size(edge=-0.1, odds=0.5) == 0.0


def test_compute_kelly_bet_size_zero_for_non_positive_odds():
    mgr, _ = _make_manager()
    assert mgr.compute_kelly_bet_size(edge=0.3, odds=0.0) == 0.0
    assert mgr.compute_kelly_bet_size(edge=0.3, odds=-0.2) == 0.0


def test_compute_kelly_bet_size_scales_with_confidence():
    mgr, _ = _make_manager(bankroll=1000.0, max_bet_size_usd=100.0)
    full_conf = mgr.compute_kelly_bet_size(edge=0.01, odds=0.99, confidence=1.0)
    half_conf = mgr.compute_kelly_bet_size(edge=0.01, odds=0.99, confidence=0.5)
    assert half_conf == pytest.approx(full_conf * 0.5)


def test_compute_kelly_bet_size_confidence_clamped_to_unit_interval():
    mgr, _ = _make_manager(bankroll=1000.0, max_bet_size_usd=100.0)
    over_conf = mgr.compute_kelly_bet_size(edge=0.01, odds=0.99, confidence=5.0)
    full_conf = mgr.compute_kelly_bet_size(edge=0.01, odds=0.99, confidence=1.0)
    assert over_conf == pytest.approx(full_conf)

    negative_conf = mgr.compute_kelly_bet_size(edge=0.3, odds=0.6, confidence=-1.0)
    assert negative_conf == 0.0


def test_compute_kelly_bet_size_scales_with_kelly_fraction():
    quarter, _ = _make_manager(bankroll=1000.0, max_bet_size_usd=100.0, kelly_fraction=0.25)
    half, _ = _make_manager(bankroll=1000.0, max_bet_size_usd=100.0, kelly_fraction=0.50)
    bet_quarter = quarter.compute_kelly_bet_size(edge=0.01, odds=0.99, confidence=1.0)
    bet_half = half.compute_kelly_bet_size(edge=0.01, odds=0.99, confidence=1.0)
    assert bet_half == pytest.approx(bet_quarter * 2.0)


def test_kelly_fraction_defaults_to_quarter_kelly():
    mgr, _ = _make_manager()
    assert mgr.kelly_fraction == 0.25


# ── cap_to_remaining_budget ────────────────────────────────────────────────────

def test_cap_to_remaining_budget_under_limit():
    mgr, _ = _make_manager()
    bet, ok = mgr.cap_to_remaining_budget(1.0)
    assert ok is True
    assert bet == 1.0


def test_cap_to_remaining_budget_caps_at_remaining():
    mgr, _ = _make_manager()
    mgr.spent_by_strategy["crypto"] = 12.0  # $3 remaining
    bet, ok = mgr.cap_to_remaining_budget(10.0)
    assert ok is True
    assert bet == 3.0


def test_cap_to_remaining_budget_zero_at_limit():
    mgr, _ = _make_manager()
    mgr.spent_by_strategy["crypto"] = 15.0
    bet, ok = mgr.cap_to_remaining_budget(1.0)
    assert ok is False
    assert bet == 0.0


# ── Per-strategy budget isolation (Phase 1) ───────────────────────────────────

def test_strategy_budgets_are_independent():
    mgr, _ = _make_manager(daily_limit=15.0)
    mgr.record_trade(10.0, strategy_tag="arbitrage")

    # Spending on arbitrage doesn't touch crypto's own $15 allocation.
    assert mgr.get_remaining_budget("crypto") == 15.0
    assert mgr.get_remaining_budget("arbitrage") == 5.0


def test_strategy_limit_caps_bet():
    mgr, _ = _make_manager(daily_limit=15.0)
    mgr.spent_by_strategy["arbitrage"] = 15.0  # arbitrage strategy at its limit

    bet, ok = mgr.check_and_cap_bet(0.001, strategy_tag="arbitrage")
    assert ok is False
    assert bet == 0.0

    # crypto's own allocation is untouched.
    bet, ok = mgr.check_and_cap_bet(0.001, strategy_tag="crypto")
    assert ok is True
    assert bet == 1.0


def test_total_spent_is_sum_of_strategies():
    mgr, _ = _make_manager()
    mgr.record_trade(2.5, strategy_tag="crypto")
    mgr.record_trade(4.0, strategy_tag="arbitrage")

    assert mgr.total_spent_today == pytest.approx(6.5)
    assert mgr.total_spent_today == pytest.approx(
        sum(mgr.spent_by_strategy.values())
    )


def test_daily_reset_clears_all_strategies():
    mgr, bridge = _make_manager()
    mgr.record_trade(2.5, strategy_tag="crypto")
    mgr.record_trade(4.0, strategy_tag="arbitrage")
    assert mgr.total_spent_today > 0.0

    # Mirrors what SequentialTradingPipeline._reset_daily_if_needed() does.
    mgr.total_spent_today = 0.0
    mgr.spent_by_strategy = {}
    mgr.trades_by_strategy = {}

    assert mgr.spent_by_strategy == {}
    assert mgr.trades_by_strategy == {}
    assert mgr.get_remaining_budget("crypto") == 15.0
    assert mgr.get_remaining_budget("arbitrage") == 15.0

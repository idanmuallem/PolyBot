import pytest

from core.bridge import DataBridge
from core.trading_config import TradingConfig
from trading.budget_manager import BudgetManager


def _make_manager(
    initial_balance=100.0, daily_limit=15.0, min_balance=5.0, bankroll=1000.0,
    max_bet_size_usd=3.0, kelly_fraction=0.25, global_limit=None,
):
    bridge = DataBridge()
    bridge.current_balance = initial_balance
    config = TradingConfig(
        bankroll_usd=bankroll,
        # global_limit defaults to daily_limit (pinned equal to the
        # per-strategy limits below) so pre-existing tests — which exercise
        # the default strategy_tag="crypto" and were written before
        # per-strategy budgets existed — see the same ceiling as before.
        # Pass global_limit explicitly to test the global cap as a distinct,
        # tighter (or looser) ceiling than any one strategy's own bucket.
        daily_limit_usd=global_limit if global_limit is not None else daily_limit,
        crypto_daily_limit_usd=daily_limit,
        arbitrage_daily_limit_usd=daily_limit,
        min_trading_balance=min_balance,
        max_bet_size_usd=max_bet_size_usd,
        kelly_fraction=kelly_fraction,
        trading_mode="dry_run",
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


# ── Real-trade clustering regression (see investigation: all 7 live trades
# clamped to a flat $25, then a flat $100, regardless of EV 0.35-0.78 —
# because quarter-Kelly (0.25) at bankroll_usd=10000 produces raw suggestions
# of $670-$850 for every trade that clears MIN_EV=0.35, dwarfing any
# reasonable dollar cap) ──────────────────────────────────────────────────────

def test_quarter_kelly_at_prod_bankroll_clamps_every_real_historical_trade():
    """Reproduces the clustering bug: at kelly_fraction=0.25 (the old
    default) and bankroll_usd=10000, a real logged trade (price=0.49,
    ev=0.3527, side=NO -> edge=0.172823, odds=0.51) raw-sizes to ~$847,
    which clamps to max_bet_size_usd regardless of what the cap is set to.
    This is the mechanism that made every trade cost exactly $25, then
    exactly $100 -- the cap, not Kelly, was setting every trade's size."""
    mgr, _ = _make_manager(bankroll=10000.0, max_bet_size_usd=100.0, kelly_fraction=0.25)
    bet = mgr.compute_kelly_bet_size(edge=0.172823, odds=0.51, confidence=1.0)
    assert bet == 100.0  # clamped, same as every real trade under this config


def test_realistic_kelly_fraction_produces_unclamped_size_that_varies_across_real_trades():
    """At kelly_fraction=0.015 (the calibrated fix) and bankroll_usd=10000,
    real historical edge/odds pairs should size well clear of both the old
    flat $25 and the new flat $100 -- proof the dollar amount is actually
    coming from Kelly's edge/odds inputs, not just hitting the ceiling."""
    mgr, _ = _make_manager(bankroll=10000.0, max_bet_size_usd=100.0, kelly_fraction=0.015)

    # price=0.49, ev=0.3527, side=NO -> edge=0.172823, odds=0.51
    trade_a = mgr.compute_kelly_bet_size(edge=0.172823, odds=0.51, confidence=1.0)
    # price=0.42, ev=0.3719, side=NO -> edge=0.156198, odds=0.58 (the
    # smallest raw_fraction of the 7 real trades logged so far)
    trade_b = mgr.compute_kelly_bet_size(edge=0.156198, odds=0.58, confidence=1.0)

    for bet in (trade_a, trade_b):
        assert bet != 25.0
        assert bet != 100.0
        assert 0.0 < bet < 100.0

    assert trade_a == pytest.approx(50.83, abs=0.01)
    assert trade_b == pytest.approx(40.40, abs=0.01)
    # Real variation between two real trades, not identical flat sizing.
    assert trade_a != pytest.approx(trade_b)


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
    # global_limit generous enough that it isn't the binding constraint —
    # this test is about per-strategy buckets not stepping on each other's
    # own allocation; the global cap itself is covered separately below.
    mgr, _ = _make_manager(daily_limit=15.0, global_limit=100.0)
    mgr.record_trade(10.0, strategy_tag="arbitrage")

    # Spending on arbitrage doesn't touch crypto's own $15 allocation.
    assert mgr.get_remaining_budget("crypto") == 15.0
    assert mgr.get_remaining_budget("arbitrage") == 5.0


def test_global_daily_limit_caps_every_strategy_regardless_of_own_allocation():
    """Belt-and-suspenders: per-strategy limits (crypto_daily_limit_usd +
    arbitrage_daily_limit_usd) can be configured to sum higher than the
    account-wide daily_limit_usd — TradingConfig only warns about that, it
    never clamps it — so get_remaining_budget() must cap every strategy's
    remaining budget by what's actually left of the global ceiling too."""
    mgr, _ = _make_manager(daily_limit=15.0, global_limit=15.0)
    mgr.record_trade(10.0, strategy_tag="arbitrage")

    # arbitrage's own $15 allocation has $5 left, same as the global
    # ceiling's $5 left ($15 global - $10 spent) — both give the same
    # answer here, so this alone doesn't prove the global cap is applied.
    assert mgr.get_remaining_budget("arbitrage") == 5.0

    # crypto's own $15 allocation is untouched, but only $5 of the $15
    # global ceiling remains — crypto must be capped to that $5, not its
    # own unspent $15 allocation.
    assert mgr.get_remaining_budget("crypto") == 5.0


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

from core.bridge import DataBridge
from core.trading_config import TradingConfig
from trading.budget_manager import BudgetManager


def _make_manager(initial_balance=100.0, daily_limit=15.0, min_balance=5.0, bankroll=1000.0):
    bridge = DataBridge()
    bridge.current_balance = initial_balance
    config = TradingConfig(
        bankroll_usd=bankroll,
        daily_limit_usd=daily_limit,
        min_trading_balance=min_balance,
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
    mgr.total_spent_today = 15.0
    bet, ok = mgr.check_and_cap_bet(0.001)
    assert ok is False
    assert bet == 0.0


def test_caps_at_remaining_budget():
    mgr, _ = _make_manager()
    mgr.total_spent_today = 12.0  # $3.0 remaining
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

"""Account budget: reserve(strategy, usd) -> Approved | Refused, the daily
and per-strategy spend limits, one trade counter (global + per strategy),
the daily reset and the max-bet ceiling. Model ("crypto") and arbitrage
trades go through the same calls."""
from datetime import datetime, timezone

import pytest

from core.bridge import DataBridge
from core.trading_config import TradingConfig
from trading.account import Account, Approved, Refused
from trading.fake_venue import FakeVenue


def _config(**overrides):
    values = dict(
        bankroll_usd=1000.0, daily_limit_usd=15.0, crypto_daily_limit_usd=15.0,
        arbitrage_daily_limit_usd=15.0, max_bet_size_usd=3.0,
        max_daily_trades=10, crypto_max_daily_trades=10, arbitrage_max_daily_trades=10,
        min_trading_balance=5.0, trading_mode="dry_run",
    )
    values.update(overrides)
    return TradingConfig(**values)


def _account(bridge=None, cash=100.0, **config):
    return Account(
        active_venue=lambda: FakeVenue(cash=cash),
        bridge=bridge if bridge is not None else DataBridge(),
        config=_config(**config),
    )


def test_reserve_caps_a_bet_at_max_bet_size():
    decision = _account().reserve("crypto", 10.0)

    assert isinstance(decision, Approved)
    assert decision.usd == pytest.approx(3.0)


# ── Daily spend limits ───────────────────────────────────────────────────────

def test_reserve_caps_a_bet_at_what_is_left_of_the_strategy_budget():
    account = _account()
    account.record_fill("crypto", 12.0)                    # $3 of $15 left

    decision = account.reserve("crypto", 2.5)
    assert decision.usd == pytest.approx(2.5)
    account.record_fill("crypto", 2.5)                     # $0.50 left
    assert account.reserve("crypto", 3.0).usd == pytest.approx(0.5)


def test_reserve_refuses_once_the_strategy_budget_is_spent():
    account = _account()
    account.record_fill("crypto", 15.0)

    decision = account.reserve("crypto", 1.0)

    assert isinstance(decision, Refused)
    assert not decision
    assert decision.reason == "daily_limit_reached"
    assert decision.detail["strategy_spent_today"] == pytest.approx(15.0)


def test_strategy_budgets_are_isolated():
    """Arbitrage spending its own allocation leaves crypto's untouched."""
    account = _account(daily_limit_usd=100.0)
    account.record_fill("arbitrage", 15.0)

    assert isinstance(account.reserve("arbitrage", 1.0), Refused)
    assert account.reserve("crypto", 1.0).usd == pytest.approx(1.0)


def test_global_daily_limit_caps_every_strategy():
    """Per-strategy limits may sum past daily_limit_usd (TradingConfig only
    warns), so the global limit is still a hard ceiling for each."""
    account = _account(daily_limit_usd=15.0, crypto_daily_limit_usd=15.0,
                       arbitrage_daily_limit_usd=15.0, max_bet_size_usd=100.0)
    account.record_fill("arbitrage", 10.0)

    assert account.reserve("crypto", 15.0).usd == pytest.approx(5.0)
    assert account.reserve("arbitrage", 15.0).usd == pytest.approx(5.0)


def test_spent_today_is_published_in_the_snapshot_and_bridge():
    bridge = DataBridge()
    account = _account(bridge=bridge)
    account.record_fill("crypto", 2.5)
    account.record_fill("arbitrage", 4.0)

    assert account.snapshot().spent_today == pytest.approx(6.5)
    assert bridge.spent_today == pytest.approx(6.5)


# ── Daily trade counts ───────────────────────────────────────────────────────

def test_reserve_refuses_once_the_strategy_trade_count_is_reached():
    account = _account(crypto_max_daily_trades=2, daily_limit_usd=100.0, crypto_daily_limit_usd=100.0)
    account.record_fill("crypto", 1.0)
    account.record_fill("crypto", 1.0)

    decision = account.reserve("crypto", 1.0)

    assert isinstance(decision, Refused)
    assert decision.reason == "daily_trade_limit_would_be_exceeded"
    assert decision.detail["strategy_trades_today"] == 2


def test_global_trade_count_caps_a_generous_strategy_allowance():
    """A per-strategy allowance of 200 trades/day can't get past the
    account-wide max_daily_trades."""
    account = _account(max_daily_trades=2, crypto_max_daily_trades=200,
                       daily_limit_usd=100.0, crypto_daily_limit_usd=100.0)
    account.record_fill("arbitrage", 1.0)
    account.record_fill("crypto", 1.0)

    decision = account.reserve("crypto", 1.0)

    assert isinstance(decision, Refused)
    assert decision.reason == "global_daily_trade_limit_would_be_exceeded"
    assert decision.detail["trades_today"] == 2


def test_a_basket_reserves_all_its_trades_at_once():
    """An arbitrage basket of n legs is n trades: refused whole if they
    don't all fit."""
    account = _account(arbitrage_max_daily_trades=3, daily_limit_usd=100.0,
                       arbitrage_daily_limit_usd=100.0)
    account.record_fill("arbitrage", 1.0)

    assert isinstance(account.reserve("arbitrage", 2.0, trades=2), Approved)
    refused = account.reserve("arbitrage", 3.0, trades=3)
    assert isinstance(refused, Refused)
    assert refused.reason == "daily_trade_limit_would_be_exceeded"


def test_max_bet_ceiling_applies_per_trade():
    account = _account(daily_limit_usd=100.0, arbitrage_daily_limit_usd=100.0)

    assert account.reserve("arbitrage", 4.0, trades=4).usd == pytest.approx(4.0)
    assert account.reserve("arbitrage", 20.0, trades=4).usd == pytest.approx(12.0)


# ── Daily reset ──────────────────────────────────────────────────────────────

def test_daily_reset_clears_spend_and_trade_counts_on_a_new_utc_day():
    bridge = DataBridge()
    account = _account(bridge=bridge, crypto_max_daily_trades=1)
    day1 = datetime(2026, 10, 21, 23, 0, tzinfo=timezone.utc)
    account.daily_reset(day1)
    account.record_fill("crypto", 15.0)

    assert account.daily_reset(day1.replace(hour=23, minute=59)) is False
    assert isinstance(account.reserve("crypto", 1.0), Refused)

    assert account.daily_reset(datetime(2026, 10, 22, 0, 1, tzinfo=timezone.utc)) is True
    assert account.reserve("crypto", 1.0).usd == pytest.approx(1.0)
    assert account.snapshot().spent_today == 0.0
    assert bridge.spent_today == 0.0


# ── Watch-only ───────────────────────────────────────────────────────────────

def test_watch_only_while_cash_is_below_the_minimum_trading_balance():
    bridge = DataBridge()
    assert _account(bridge=bridge, cash=4.0).refresh().watch_only is True
    assert bridge.watch_only is True
    assert _account(bridge=bridge, cash=100.0).refresh().watch_only is False
    assert bridge.watch_only is False


# ── Kelly clamp clustering regression (de2c1af) ──────────────────────────────
# All 7 live trades were sized a flat $25, then a flat $100, whatever their
# EV (0.35-0.78): quarter-Kelly at bankroll_usd=10000 raw-sizes every trade
# that clears MIN_EV to $670-$850, so the max-bet ceiling, not Kelly, set
# every trade's size.

def _kelly_reserve(kelly_fraction, edge, odds):
    from trading.kelly import kelly_bet_size
    account = _account(bankroll_usd=10000.0, max_bet_size_usd=100.0,
                       daily_limit_usd=10000.0, crypto_daily_limit_usd=10000.0)
    proposed = kelly_bet_size(edge=edge, odds=odds, confidence=1.0,
                              kelly_fraction=kelly_fraction, bankroll_usd=10000.0)
    return account.reserve("crypto", proposed).usd


def test_quarter_kelly_at_prod_bankroll_clamps_every_real_historical_trade():
    # price=0.49, ev=0.3527, side=NO -> edge=0.172823, odds=0.51 raw-sizes to ~$847
    assert _kelly_reserve(0.25, edge=0.172823, odds=0.51) == 100.0


def test_realistic_kelly_fraction_sizes_vary_with_edge_below_the_ceiling():
    trade_a = _kelly_reserve(0.015, edge=0.172823, odds=0.51)
    # price=0.42, ev=0.3719, side=NO -> edge=0.156198, odds=0.58
    trade_b = _kelly_reserve(0.015, edge=0.156198, odds=0.58)

    assert trade_a == pytest.approx(50.83, abs=0.01)
    assert trade_b == pytest.approx(40.40, abs=0.01)


# ── Restart: today's spend and trade counts come back from the Ledger ────────

def _restarted_account(ledger, bridge=None, **config):
    """A fresh Account over a Ledger that already holds today's entries,
    as after an engine restart."""
    return Account(
        active_venue=lambda: FakeVenue(cash=100.0),
        bridge=bridge if bridge is not None else DataBridge(),
        ledger=ledger,
        config=_config(**config),
    )


def _spend(ledger, strategy, usd, at):
    ledger.record_entry(token_id=f"{strategy}-{at}", condition_id=None, side="YES",
                        shares=usd / 0.5, price=0.5, venue="paper", strategy=strategy, at=at)


def test_a_restarted_account_resumes_todays_spend_from_the_ledger(tmp_path):
    from trading.ledger import Ledger
    ledger = Ledger(str(tmp_path / "trades.db"))
    now = datetime(2026, 10, 21, 15, 0, tzinfo=timezone.utc)
    _spend(ledger, "crypto", 12.0, "2026-10-21 09:00:00")
    _spend(ledger, "arbitrage", 1.0, "2026-10-21 10:00:00")
    _spend(ledger, "crypto", 9.0, "2026-10-20 23:59:59")         # yesterday (UTC)

    bridge = DataBridge()
    account = _restarted_account(ledger, bridge=bridge)
    account.daily_reset(now)

    assert account.snapshot().spent_today == pytest.approx(13.0)
    assert bridge.spent_today == pytest.approx(13.0)
    # $15 global limit, $13 spent: $2 left, whatever the strategy asks for.
    assert account.reserve("crypto", 3.0).usd == pytest.approx(2.0)
    account.record_fill("crypto", 2.0)
    assert isinstance(account.reserve("crypto", 1.0), Refused)


def test_a_restarted_account_resumes_todays_trade_counts_from_the_ledger(tmp_path):
    from trading.ledger import Ledger
    ledger = Ledger(str(tmp_path / "trades.db"))
    _spend(ledger, "crypto", 1.0, "2026-10-21 09:00:00")
    _spend(ledger, "arbitrage", 1.0, "2026-10-21 09:30:00")

    account = _restarted_account(ledger, crypto_max_daily_trades=1, max_daily_trades=3)
    account.daily_reset(datetime(2026, 10, 21, 15, 0, tzinfo=timezone.utc))

    assert account.reserve("crypto", 1.0).reason == "daily_trade_limit_would_be_exceeded"
    assert account.reserve("arbitrage", 2.0, trades=2).reason == "global_daily_trade_limit_would_be_exceeded"


def test_the_first_refresh_keeps_the_restored_spend(tmp_path):
    from trading.ledger import Ledger
    ledger = Ledger(str(tmp_path / "trades.db"))
    _spend(ledger, "crypto", 6.0, "2026-10-21 09:00:00")
    bridge = DataBridge()
    account = _restarted_account(ledger, bridge=bridge)
    account.daily_reset(datetime(2026, 10, 21, 15, 0, tzinfo=timezone.utc))

    account.refresh()

    assert bridge.spent_today == pytest.approx(6.0)

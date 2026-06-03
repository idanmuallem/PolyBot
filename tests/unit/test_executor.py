import os
from unittest.mock import patch, MagicMock

import pytest

from core.models import MarketData
from core.trading_config import TradingConfig


def _dry_config(**overrides):
    defaults = dict(
        dry_run=True,
        min_ev=0.30,
        max_daily_trades=10,
        bankroll_usd=1000.0,
        daily_limit_usd=15.0,
        max_bet_size_usd=3.0,
        private_key="",
        proxy_address="",
    )
    defaults.update(overrides)
    return TradingConfig(**defaults)


def _make_executor(**config_overrides):
    """Create a TradeExecutor with a controlled TradingConfig (no real CLOB)."""
    from trading.executor import TradeExecutor
    config = _dry_config(**config_overrides)
    with patch("trading.executor.TradingConfig.from_env", return_value=config):
        executor = TradeExecutor()
    return executor


def _valid_market():
    return MarketData(
        market_id="tok1",
        asset_type="Crypto::BTCUSDT",
        strike_price=100_000.0,
        question="Will BTC exceed $100,000?",
        market_name="BTC test market",
        initial_price=0.50,
        volume=500_000.0,
        no_market_id="tok1_no",
    )


# ── Bug #1 regression: NameError on position_size ────────────────────────────

def test_no_name_error_on_auto_trade_log():
    """After the fix, evaluate_and_execute must not raise NameError."""
    executor = _make_executor(min_ev=0.10)
    market = _valid_market()
    log_calls = []

    result = executor.evaluate_and_execute(
        market=market,
        fair_value=0.75,
        ev=0.50,
        current_poly_price=0.50,
        bet_amount_usd=2.0,
        side="YES",
        log_func=lambda level, *a, **kw: log_calls.append(level),
    )

    assert result is True
    assert "AUTO-TRADE" in log_calls
    assert "DRY-RUN" in log_calls


# ── Bug #2 regression: EV threshold boundary ─────────────────────────────────

def test_ev_at_threshold_is_allowed():
    """After fix (strict <), ev == min_ev should NOT be rejected."""
    executor = _make_executor(min_ev=0.30)
    market = _valid_market()

    result = executor.evaluate_and_execute(
        market=market,
        fair_value=0.65,
        ev=0.30,  # exactly at threshold
        current_poly_price=0.50,
        bet_amount_usd=2.0,
        side="YES",
        log_func=lambda *a, **kw: None,
    )
    assert result is True


def test_ev_below_threshold_is_rejected():
    executor = _make_executor(min_ev=0.30)
    market = _valid_market()

    result = executor.evaluate_and_execute(
        market=market,
        fair_value=0.60,
        ev=0.29,
        current_poly_price=0.50,
        bet_amount_usd=2.0,
        side="YES",
        log_func=lambda *a, **kw: None,
    )
    assert result is False


# ── Daily trade limit ─────────────────────────────────────────────────────────

def test_daily_limit_blocks_trade():
    executor = _make_executor(max_daily_trades=10)
    executor.trade_count_today = 10  # already at limit
    market = _valid_market()

    log_calls = []
    result = executor.evaluate_and_execute(
        market=market,
        fair_value=0.75,
        ev=0.50,
        current_poly_price=0.50,
        bet_amount_usd=2.0,
        side="YES",
        log_func=lambda level, *a, **kw: log_calls.append(level),
    )
    assert result is False
    assert "RISK" in log_calls


# ── Price bounds ──────────────────────────────────────────────────────────────

def test_price_below_floor_rejected():
    executor = _make_executor()
    market = _valid_market()

    log_calls = []
    result = executor.evaluate_and_execute(
        market=market,
        fair_value=0.50,
        ev=0.50,
        current_poly_price=0.20,  # below PRICE_FLOOR=0.30
        bet_amount_usd=2.0,
        side="YES",
        log_func=lambda level, *a, **kw: log_calls.append(level),
    )
    assert result is False
    assert "EXECUTION" in log_calls


def test_price_above_ceiling_rejected():
    executor = _make_executor()
    market = _valid_market()

    log_calls = []
    result = executor.evaluate_and_execute(
        market=market,
        fair_value=0.90,
        ev=0.50,
        current_poly_price=0.90,  # above PRICE_CEILING=0.85
        bet_amount_usd=2.0,
        side="YES",
        log_func=lambda level, *a, **kw: log_calls.append(level),
    )
    assert result is False
    assert "EXECUTION" in log_calls


def test_no_side_flips_price_to_complement():
    """For a NO trade, execution_price = 1 - price_yes."""
    executor = _make_executor()
    market = _valid_market()

    log_calls = []
    # price_yes=0.82 → execution_price_no = 1-0.82 = 0.18 < PRICE_FLOOR → rejected
    result = executor.evaluate_and_execute(
        market=market,
        fair_value=0.20,
        ev=0.50,
        current_poly_price=0.82,
        bet_amount_usd=2.0,
        side="NO",
        log_func=lambda level, *a, **kw: log_calls.append(level),
    )
    assert result is False


def test_no_side_uses_no_token_id():
    """When side=NO and no_market_id is set, the NO token is used in the log."""
    executor = _make_executor()
    market = _valid_market()  # has no_market_id="tok1_no"

    log_calls = []
    # price_yes=0.50 → execution_price_no=0.50 (in bounds)
    executor.evaluate_and_execute(
        market=market,
        fair_value=0.40,
        ev=0.50,
        current_poly_price=0.50,
        bet_amount_usd=2.0,
        side="NO",
        log_func=lambda level, asset, token, *a, **kw: log_calls.append((level, token)),
    )
    dry_run_entries = [(l, t) for l, t in log_calls if l == "DRY-RUN"]
    assert dry_run_entries, "DRY-RUN log not found"
    assert dry_run_entries[0][1] == "tok1_no"


# ── Dry-run mode ──────────────────────────────────────────────────────────────

def test_dry_run_returns_true_and_logs():
    executor = _make_executor()
    market = _valid_market()
    log_calls = []

    result = executor.evaluate_and_execute(
        market=market,
        fair_value=0.75,
        ev=0.50,
        current_poly_price=0.50,
        bet_amount_usd=2.0,
        side="YES",
        log_func=lambda level, *a, **kw: log_calls.append(level),
    )
    assert result is True
    assert "DRY-RUN" in log_calls


def test_trade_count_increments_on_success():
    executor = _make_executor()
    assert executor.trade_count_today == 0
    market = _valid_market()

    executor.evaluate_and_execute(
        market=market,
        fair_value=0.75,
        ev=0.50,
        current_poly_price=0.50,
        bet_amount_usd=2.0,
        side="YES",
        log_func=lambda *a, **kw: None,
    )
    assert executor.trade_count_today == 1


# ── _is_valid_order_response ──────────────────────────────────────────────────

def test_valid_response_with_order_id():
    from trading.executor import TradeExecutor
    assert TradeExecutor._is_valid_order_response({"orderID": "abc123"}) is True


def test_response_with_error_field_is_invalid():
    from trading.executor import TradeExecutor
    assert TradeExecutor._is_valid_order_response({"error": "insufficient_funds"}) is False


def test_none_response_is_invalid():
    from trading.executor import TradeExecutor
    assert TradeExecutor._is_valid_order_response(None) is False


# ── get_open_positions: live_ev matches pnl_ratio ────────────────────────────

def test_get_open_positions_live_ev_normalization_at_boundary():
    """A 1% gain (pnl_ratio=0.01) must produce live_ev=0.01."""
    executor = _make_executor()
    import requests as _req
    from unittest.mock import patch as _patch
    raw_position = {
        "asset": "tok1",
        "avgPrice": "0.40",
        "currentPrice": "0.404",   # +1% from 0.40
        "currentValue": "4.04",
        "size": "10",
    }
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = [raw_position]

    with _patch("trading.executor.requests.get", return_value=mock_resp), \
         _patch.object(executor, "_resolve_positions_user_address", return_value="0xabc"):
        positions = executor.get_open_positions()

    assert positions, "expected one position"
    assert abs(positions[0].live_ev - 0.01) < 0.001, (
        f"live_ev should be ~0.01, got {positions[0].live_ev}"
    )


# ── get_balance in dry-run ────────────────────────────────────────────────────

def test_get_balance_dry_run_returns_env_default(monkeypatch):
    monkeypatch.setenv("PAPER_BALANCE_USD", "500.0")
    executor = _make_executor()
    assert executor.get_balance() == 500.0

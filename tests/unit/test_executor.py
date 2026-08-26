import asyncio
from unittest.mock import MagicMock, patch

from core.models import MarketData
from core.trading_config import TradingConfig


def _dry_config(**overrides):
    defaults = dict(
        dry_run=True,
        min_ev=0.30,
        max_daily_trades=10,
        # Per-strategy trade caps default to 50 each (core/trading_config.py);
        # pin the "crypto" tag's cap to max_daily_trades here so pre-existing
        # tests below — which exercise the default strategy_tag="crypto" and
        # were written before per-strategy tracking existed — see the same
        # ceiling as before.
        crypto_max_daily_trades=10,
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
    return TradeExecutor(config=config)


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
    executor.trades_by_strategy["crypto"] = 10  # already at limit
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


def test_global_daily_limit_blocks_trade_even_under_strategy_allowance():
    """Belt-and-suspenders: a generous per-strategy crypto_max_daily_trades
    must not let trades continue past the account-wide max_daily_trades
    ceiling."""
    executor = _make_executor(max_daily_trades=2, crypto_max_daily_trades=200)
    executor.trade_count_today = 2  # at the global ceiling, strategy bucket untouched
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
    # This test targets the live Data-API positions path specifically (field
    # normalization), not the paper adapter — force the paper adapter off so
    # get_open_positions() falls through to the mocked HTTP call below.
    executor.paper = None
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

def test_get_balance_dry_run_returns_configured_paper_balance():
    executor = _make_executor(paper_balance_usd=500.0)
    # This test targets the config-driven fallback used when no paper
    # adapter is available, not the paper engine's own tracked balance.
    executor.paper = None
    assert executor.get_balance() == 500.0


# ── Paper trading adapter integration (best-effort, must not break dry-run) ──

def test_dry_run_with_paper_adapter_still_returns_true():
    """Paper adapter failures must not break the DRY-RUN return value."""
    executor = _make_executor()  # dry_run=True

    # Inject a paper adapter whose execute_buy always raises
    mock_paper = MagicMock()
    mock_paper.execute_buy.side_effect = RuntimeError("Engine exploded")
    executor.paper = mock_paper

    market = _valid_market()
    log_calls = []

    # Even if paper adapter raises, execute_trade should still return True
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


# ── execute_arbitrage_group (Phase 3: limit order timeout + partial fills) ──

def _live_executor(**config_overrides):
    """A TradeExecutor wired for the "live" branch of execute_arbitrage_group
    without a real ClobClient — client is a bare MagicMock so _submit_order/
    _get_order_filled_shares/cancel can be patched directly."""
    executor = _make_executor(dry_run=False, **config_overrides)
    executor.client = MagicMock()
    return executor


def _mock_order_placement(fills_by_token: dict):
    """Returns (fake_submit_order, fake_get_filled) matched by token_id via
    a deterministic order_id, so tests can simulate arbitrary per-leg fills
    without a real order book."""
    def fake_submit_order(token_id, price, side, size):
        return {"orderID": f"order_{token_id}"}

    def fake_get_filled(order_id):
        token_id = order_id.replace("order_", "")
        return fills_by_token[token_id]

    return fake_submit_order, fake_get_filled


def test_execute_arbitrage_group_all_legs_fill_returns_success():
    executor = _live_executor()
    legs = [
        {"token_id": "A", "price": 0.30, "shares": 3.0, "side": "YES"},
        {"token_id": "B", "price": 0.40, "shares": 2.0, "side": "YES"},
    ]
    fake_submit, fake_get_filled = _mock_order_placement({"A": 3.0, "B": 2.0})

    with patch.object(executor, "_submit_order", side_effect=fake_submit), \
         patch.object(executor, "_get_order_filled_shares", side_effect=fake_get_filled):
        result = asyncio.run(executor.execute_arbitrage_group(
            legs=legs, timeout_seconds=0.05, log_func=lambda *a, **kw: None,
        ))

    assert result["success"] is True
    assert result["arb_sets"] == 2.0  # min(3.0, 2.0)
    assert result["unfilled"] == []
    assert result["fills"] == {"A": 3.0, "B": 2.0}


def test_execute_arbitrage_group_zero_fill_leg_cancels_remaining():
    executor = _live_executor()
    legs = [
        {"token_id": "A", "price": 0.30, "shares": 3.0, "side": "YES"},
        {"token_id": "B", "price": 0.40, "shares": 2.0, "side": "YES"},
    ]
    fake_submit, fake_get_filled = _mock_order_placement({"A": 3.0, "B": 0.0})

    with patch.object(executor, "_submit_order", side_effect=fake_submit), \
         patch.object(executor, "_get_order_filled_shares", side_effect=fake_get_filled):
        result = asyncio.run(executor.execute_arbitrage_group(
            legs=legs, timeout_seconds=0.05, log_func=lambda *a, **kw: None,
        ))

    assert result["success"] is False
    assert result["arb_sets"] == 0
    assert result["unfilled"] == ["B"]
    # Both legs' resting orders get cancelled — no arbitrage without leg B.
    assert executor.client.cancel.call_count == 2
    cancelled_order_ids = {call.args[0] for call in executor.client.cancel.call_args_list}
    assert cancelled_order_ids == {"order_A", "order_B"}


def test_execute_arbitrage_group_uneven_fills_computes_surplus():
    executor = _live_executor()
    legs = [
        {"token_id": "A", "price": 0.30, "shares": 3.0, "side": "YES"},
        {"token_id": "B", "price": 0.30, "shares": 10.0, "side": "YES"},
        {"token_id": "C", "price": 0.30, "shares": 5.0, "side": "YES"},
    ]
    fake_submit, fake_get_filled = _mock_order_placement({"A": 3.0, "B": 10.0, "C": 5.0})

    with patch.object(executor, "_submit_order", side_effect=fake_submit), \
         patch.object(executor, "_get_order_filled_shares", side_effect=fake_get_filled):
        result = asyncio.run(executor.execute_arbitrage_group(
            legs=legs, timeout_seconds=0.05, log_func=lambda *a, **kw: None,
        ))

    assert result["success"] is True
    assert result["arb_sets"] == 3.0
    assert result["surplus"] == {"B": 7.0, "C": 2.0}


def test_execute_arbitrage_group_dry_run_simulates_full_fill_no_real_orders():
    executor = _make_executor()  # default dry_run=True, client stays None
    assert executor.client is None

    legs = [
        {"token_id": "A", "price": 0.30, "shares": 3.0, "side": "YES", "bet_amount_usd": 1.0},
        {"token_id": "B", "price": 0.40, "shares": 2.0, "side": "YES", "bet_amount_usd": 1.0},
    ]
    log_calls = []
    result = asyncio.run(executor.execute_arbitrage_group(
        legs=legs, timeout_seconds=60.0,
        log_func=lambda level, *a, **kw: log_calls.append(level),
    ))

    assert result["success"] is True
    assert result["fills"] == {"A": 3.0, "B": 2.0}
    assert result["arb_sets"] == 2.0
    assert result["surplus"] == {"A": 1.0}
    # Simulated, not real: no client, and the per-leg DRY-RUN path fired
    # (same as execute_trade()'s existing dry-run behavior) plus a clear
    # group-level structure log.
    assert executor.client is None
    assert log_calls.count("DRY-RUN") == 2
    assert "ARBITRAGE-FILL" in log_calls

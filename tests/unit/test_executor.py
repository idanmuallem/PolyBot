import asyncio

import pytest
from unittest.mock import MagicMock, patch

from core.models import MarketData
from core.trading_config import TradingConfig


def _dry_config(**overrides):
    defaults = dict(
        trading_mode="dry_run",
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


def _make_executor(venue=None, **config_overrides):
    """A TradeExecutor with a controlled TradingConfig (no real CLOB).
    venue is the paper Venue (e.g. a FakeVenue); None is log-only dry-run."""
    from trading.executor import TradeExecutor
    return TradeExecutor(config=_dry_config(**config_overrides), venue=venue)


def _listed_venue(market_ref=None, yes_price=0.50, cash=1000.0):
    """A FakeVenue with tok1/tok1_no (see _valid_market) tradeable."""
    from trading.fake_venue import FakeVenue
    from trading.venue import MarketRef
    venue = FakeVenue(cash=cash)
    venue.list_market(market_ref or MarketRef(yes_token_id="tok1", no_token_id="tok1_no"), yes_price)
    return venue


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
    executor = _make_executor(trading_mode="live_run")
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


# ── Construction ──────────────────────────────────────────────────────────────

def test_executor_constructs_without_pm_trader_or_filesystem(tmp_path, monkeypatch):
    """The executor never builds a paper venue itself: constructing one in
    dry-run must not import pm_trader or write anything to disk."""
    import sys
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRADES_DB_PATH", str(tmp_path / "trades.db"))
    monkeypatch.setitem(sys.modules, "pm_trader", None)
    monkeypatch.setitem(sys.modules, "pm_trader.engine", None)

    executor = _make_executor()

    assert executor.venue is None
    assert list(tmp_path.iterdir()) == []


# ── Dry-run reads cash and positions from the Venue ───────────────────────────

def test_get_balance_dry_run_reads_venue_cash():
    venue = _listed_venue(cash=640.0)
    assert _make_executor(venue=venue, paper_balance_usd=500.0).get_balance() == 640.0


def test_get_balance_dry_run_without_venue_returns_configured_paper_balance():
    assert _make_executor(paper_balance_usd=500.0).get_balance() == 500.0


def test_get_open_positions_dry_run_reads_venue_positions():
    from trading.venue import MarketRef
    venue = _listed_venue(yes_price=0.40)
    venue.buy(MarketRef(yes_token_id="tok1", no_token_id="tok1_no"), "NO", 6.0)

    positions = _make_executor(venue=venue).get_open_positions()

    assert [(p.token_id, p.side) for p in positions] == [("tok1_no", "NO")]


# ── Dry-run buys go through the Venue ─────────────────────────────────────────

def _evaluate(executor, side="YES", price=0.50, bet=2.0, log_calls=None):
    sink = log_calls if log_calls is not None else []
    return executor.evaluate_and_execute(
        market=_valid_market(),
        fair_value=0.75,
        ev=0.50,
        current_poly_price=price,
        bet_amount_usd=bet,
        side=side,
        log_func=lambda level, *a, **kw: sink.append(level),
    )


def test_dry_run_yes_buy_fills_on_the_venue():
    venue = _listed_venue(yes_price=0.50)
    executor = _make_executor(venue=venue)

    assert _evaluate(executor, side="YES", bet=2.0) is True

    assert venue.fills == [("buy", "tok1", pytest.approx(4.0), 2.0)]
    assert executor.trade_count_today == 1


def test_dry_run_no_buy_trades_the_no_token_through_the_venue():
    """YES->NO translation happens behind the port: the executor passes the
    market and side; the NO token is what gets bought."""
    venue = _listed_venue(yes_price=0.40)
    executor = _make_executor(venue=venue)
    log_rows = []

    result = executor.evaluate_and_execute(
        market=_valid_market(), fair_value=0.30, ev=0.50, current_poly_price=0.40,
        bet_amount_usd=3.0, side="NO",
        log_func=lambda level, asset, token, payload: log_rows.append((level, token, payload)),
    )

    assert result is True
    assert venue.fills == [("buy", "tok1_no", pytest.approx(5.0), 3.0)]
    dry_run = [(token, payload) for level, token, payload in log_rows if level == "DRY-RUN"]
    assert dry_run[0][0] == "tok1_no"
    assert dry_run[0][1]["price"] == pytest.approx(0.60)


def test_dry_run_buy_the_venue_refuses_is_not_reported_as_success():
    """Phantom-trade regression (6a06f35): a fill that never landed must not
    count as a trade -- callers spend budget and trade count on True."""
    from trading.fake_venue import FakeVenue
    executor = _make_executor(venue=FakeVenue(cash=1000.0))  # tok1 not listed: unfillable
    log_calls = []

    assert _evaluate(executor, log_calls=log_calls) is False
    assert executor.trade_count_today == 0
    assert "DRY-RUN" in log_calls  # the attempt itself isn't hidden


class _ExplodingVenue:
    def buy(self, *a, **kw):
        raise RuntimeError("Engine exploded")

    def sell(self, *a, **kw):
        raise RuntimeError("Engine exploded")


def test_dry_run_venue_raising_on_buy_does_not_raise_but_reports_failure():
    executor = _make_executor(venue=_ExplodingVenue())
    assert _evaluate(executor) is False


def test_dry_run_without_venue_is_log_only_and_reports_success():
    """No paper venue at all is the log-only dry-run mode (removed by #17)."""
    assert _evaluate(_make_executor()) is True


# ── Dry-run sells go through the Venue ────────────────────────────────────────

def test_dry_run_sell_closes_the_venue_position():
    from trading.venue import MarketRef
    venue = _listed_venue(yes_price=0.50)
    venue.buy(MarketRef(yes_token_id="tok1", no_token_id="tok1_no"), "YES", 5.0)
    executor = _make_executor(venue=venue)

    assert executor.sell_position("tok1", 10.0, 0.50, log_func=lambda *a, **kw: None) is True
    assert executor.get_open_positions() == []


def test_dry_run_sell_the_venue_refuses_is_not_reported_as_success():
    """Phantom-sell regression (6a06f35): PortfolioManager credits cash and
    treats the position as closed on True alone, so a sale that never
    landed must report False -- every tick, not just the first."""
    venue = _listed_venue()
    executor = _make_executor(venue=venue)
    log_calls = []

    for _ in range(3):
        result = executor.sell_position(
            token_id="stuck_tok", shares=1000.0, price=0.0005,
            log_func=lambda level, *a, **kw: log_calls.append(level),
        )
        assert result is False
    assert log_calls.count("DRY-RUN-SELL") == 3  # attempts still logged
    assert venue.cash() == 1000.0


def test_dry_run_venue_raising_on_sell_is_not_reported_as_success():
    executor = _make_executor(venue=_ExplodingVenue())
    assert executor.sell_position("stuck_tok", 1000.0, 0.0005, log_func=lambda *a, **kw: None) is False


def test_dry_run_sell_without_venue_is_log_only_and_reports_success():
    assert _make_executor().sell_position("tok1", 10.0, 0.5, log_func=lambda *a, **kw: None) is True


# ── execute_arbitrage_group (Phase 3: limit order timeout + partial fills) ──

def _live_executor(**config_overrides):
    """A TradeExecutor wired for the "live" branch of execute_arbitrage_group
    without a real ClobClient — client is a bare MagicMock so _submit_order/
    _get_order_filled_shares/cancel can be patched directly."""
    executor = _make_executor(trading_mode="live_run", **config_overrides)
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
    executor = _make_executor()  # default trading_mode="dry_run", client stays None
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


# ── Paper arbitrage: limit orders through the Venue ─────────────────────────
#
# Paper-mode arbitrage places limit orders (capping the price paid per leg)
# on the Venue rather than walking the book. A FakeVenue stands in for the
# paper engine: listed legs fill at their listed price, unlisted legs never
# fill.

def _arb_venue(listed: dict, cash=1000.0):
    from trading.fake_venue import FakeVenue
    from trading.venue import MarketRef
    venue = FakeVenue(cash=cash)
    for token_id, price in listed.items():
        venue.set_price(MarketRef(yes_token_id=token_id), "YES", price)
    return venue


def _leg(token_id, price, usd):
    return {"token_id": token_id, "price": price, "shares": round(usd / price, 2),
            "side": "YES", "bet_amount_usd": usd}


def _run_arb(executor, legs, timeout=0.05):
    return asyncio.run(executor.execute_arbitrage_group(
        legs=legs, timeout_seconds=timeout, log_func=lambda *a, **kw: None,
    ))


def test_paper_limit_arbitrage_group_all_legs_fill_returns_success():
    venue = _arb_venue({"A": 0.30, "B": 0.40})
    executor = _make_executor(venue=venue)

    result = _run_arb(executor, [_leg("A", 0.30, 3.0), _leg("B", 0.40, 2.0)])

    assert result["success"] is True
    assert result["arb_sets"] == pytest.approx(5.0)
    assert result["unfilled"] == []
    assert result["fills"] == {"A": pytest.approx(10.0), "B": pytest.approx(5.0)}
    assert executor.trades_by_strategy["arbitrage"] == 2


def test_paper_limit_arbitrage_limit_price_is_the_raw_leg_price():
    """No premium above the scan-time leg price -- the same price the live
    path submits at, so paper doesn't fill more easily than live would."""
    venue = _arb_venue({"A": 0.31})  # one tick above the leg price
    executor = _make_executor(venue=venue)

    result = _run_arb(executor, [_leg("A", 0.30, 3.0)])

    assert result["success"] is False
    assert result["unfilled"] == ["A"]


def test_paper_limit_arbitrage_group_never_filled_leg_voids_group_and_cancels():
    from trading.venue import MarketRef
    venue = _arb_venue({"A": 0.30})  # B never listed: its limit never fills
    executor = _make_executor(venue=venue)

    result = _run_arb(executor, [_leg("A", 0.30, 3.0), _leg("B", 0.40, 2.0)])

    assert result["success"] is False
    assert result["arb_sets"] == 0
    assert result["unfilled"] == ["B"]
    assert result["fills"]["A"] == pytest.approx(10.0)  # the real fill is preserved
    assert result["fills"]["B"] == 0.0
    # B's resting order was cancelled: a later price move can't fill it.
    venue.set_price(MarketRef(yes_token_id="B"), "YES", 0.40)
    venue.poll_limit_orders([])
    assert [p.token_id for p in venue.open_positions()] == ["A"]


def test_paper_limit_arbitrage_group_uneven_fills_computes_surplus():
    venue = _arb_venue({"A": 0.30, "B": 0.30, "C": 0.30})
    executor = _make_executor(venue=venue)

    result = _run_arb(executor, [_leg("A", 0.30, 1.5), _leg("B", 0.30, 3.0), _leg("C", 0.30, 2.1)])

    assert result["success"] is True
    assert result["arb_sets"] == pytest.approx(5.0)
    assert result["surplus"] == {"B": pytest.approx(5.0), "C": pytest.approx(2.0)}


def test_execute_arbitrage_group_falls_back_to_simulation_without_venue():
    """dry_run with no paper venue at all (log-only mode) must
    still fall back to the log-only simulation, not the limit-order path
    (which has nothing real to place orders against in that mode)."""
    executor = _make_executor()  # trading_mode="dry_run", no venue
    legs = [{"token_id": "A", "price": 0.30, "shares": 3.0, "side": "YES", "bet_amount_usd": 1.0}]

    result = asyncio.run(executor.execute_arbitrage_group(
        legs=legs, timeout_seconds=60.0, log_func=lambda *a, **kw: None,
    ))

    assert result["success"] is True
    assert result["fills"] == {"A": 3.0}

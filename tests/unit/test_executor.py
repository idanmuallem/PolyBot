import asyncio

import pytest

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
    """A TradeExecutor with a controlled TradingConfig, trading on *venue*
    (default: a FakeVenue with tok1/tok1_no listed at 0.50)."""
    from trading.executor import TradeExecutor
    return TradeExecutor(config=_dry_config(**config_overrides), venue=venue or _listed_venue())


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


# ── Construction ──────────────────────────────────────────────────────────────

def test_executor_constructs_without_pm_trader_or_filesystem(tmp_path, monkeypatch):
    """The executor never builds a paper venue itself: constructing one in
    dry-run must not import pm_trader or write anything to disk."""
    import sys
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRADES_DB_PATH", str(tmp_path / "trades.db"))
    monkeypatch.setitem(sys.modules, "pm_trader", None)
    monkeypatch.setitem(sys.modules, "pm_trader.engine", None)

    from trading.fake_venue import FakeVenue
    venue = FakeVenue()
    executor = _make_executor(venue=venue)

    assert executor.venue is venue
    assert list(tmp_path.iterdir()) == []


# ── Dry-run reads cash and positions from the Venue ───────────────────────────

def test_get_balance_dry_run_reads_venue_cash():
    venue = _listed_venue(cash=640.0)
    assert _make_executor(venue=venue, paper_balance_usd=500.0).get_balance() == 640.0


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
    assert "DRY-RUN" in log_calls  # the attempt itself isn't hidden


class _ExplodingVenue:
    name = "paper"

    def buy(self, *a, **kw):
        raise RuntimeError("Engine exploded")

    def sell(self, *a, **kw):
        raise RuntimeError("Engine exploded")


def test_dry_run_venue_raising_on_buy_does_not_raise_but_reports_failure():
    executor = _make_executor(venue=_ExplodingVenue())
    assert _evaluate(executor) is False


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


# ── execute_arbitrage_group (Phase 3: limit order timeout + partial fills) ──

def _live_executor(filled_by_token: dict):
    """A TradeExecutor on a ClobVenue over a stub CLOB client: each leg's
    order id is order_<token>, and get_order reports filled_by_token."""
    from tests.unit.test_clob_venue import StubClobClient
    from trading.clob_venue import ClobVenue
    client = StubClobClient(
        order_response=lambda order: {"orderID": f"order_{order.token_id}"},
        filled={f"order_{token}": shares for token, shares in filled_by_token.items()},
    )
    executor = _make_executor(
        venue=ClobVenue(client=client, proxy_address="0xabc", http_get=None),
        trading_mode="live_run",
    )
    return executor, client


def test_live_arbitrage_group_all_legs_fill_returns_success():
    executor, client = _live_executor({"A": 3.0, "B": 2.0})

    result = _run_arb(executor, [
        {"token_id": "A", "price": 0.30, "shares": 3.0, "side": "YES", "bet_amount_usd": 0.9},
        {"token_id": "B", "price": 0.40, "shares": 2.0, "side": "YES", "bet_amount_usd": 0.8},
    ])

    assert client.orders == [("A", 0.3, "BUY", 3.0), ("B", 0.4, "BUY", 2.0)]
    assert result["success"] is True
    assert result["arb_sets"] == 2.0  # min(3.0, 2.0)
    assert result["unfilled"] == []
    assert result["fills"] == {"A": 3.0, "B": 2.0}


def test_live_arbitrage_group_zero_fill_leg_cancels_every_order():
    executor, client = _live_executor({"A": 3.0, "B": 0.0})

    result = _run_arb(executor, [
        {"token_id": "A", "price": 0.30, "shares": 3.0, "side": "YES", "bet_amount_usd": 0.9},
        {"token_id": "B", "price": 0.40, "shares": 2.0, "side": "YES", "bet_amount_usd": 0.8},
    ])

    assert result["success"] is False
    assert result["arb_sets"] == 0
    assert result["unfilled"] == ["B"]
    # Both legs' resting orders get cancelled -- no arbitrage without leg B.
    assert sorted(client.cancelled) == ["order_A", "order_B"]


def test_live_arbitrage_group_uneven_fills_computes_surplus():
    executor, _ = _live_executor({"A": 3.0, "B": 10.0, "C": 5.0})

    result = _run_arb(executor, [
        {"token_id": "A", "price": 0.30, "shares": 3.0, "side": "YES", "bet_amount_usd": 0.9},
        {"token_id": "B", "price": 0.30, "shares": 10.0, "side": "YES", "bet_amount_usd": 3.0},
        {"token_id": "C", "price": 0.30, "shares": 5.0, "side": "YES", "bet_amount_usd": 1.5},
    ])

    assert result["success"] is True
    assert result["arb_sets"] == 3.0
    assert result["surplus"] == {"B": 7.0, "C": 2.0}


# ── Live buys and sells go through the ClobVenue ─────────────────────────────

def test_live_buy_posts_the_order_and_logs_live_trade_under_the_market_asset_type():
    executor, client = _live_executor({})
    rows = []

    result = executor.evaluate_and_execute(
        market=_valid_market(), fair_value=0.30, ev=0.50, current_poly_price=0.40,
        bet_amount_usd=3.0, side="NO",
        log_func=lambda level, asset, token, payload: rows.append((level, asset, token)),
    )

    assert result is True
    assert client.orders == [("tok1_no", 0.6, "BUY", 5.0)]
    assert ("LIVE-TRADE", "Crypto::BTCUSDT", "tok1_no") in rows


def test_live_rejected_buy_is_not_a_trade():
    executor, client = _live_executor({})
    client.order_response = {"error": "insufficient_funds"}

    assert _evaluate(executor) is False


def test_live_sell_posts_a_sell_at_the_given_price():
    executor, client = _live_executor({})
    rows = []

    assert executor.sell_position("tok1", 10.0, 0.55, log_func=lambda level, *a: rows.append(level)) is True
    assert client.orders == [("tok1", 0.55, "SELL", 10.0)]
    assert rows == ["SELL"]


# ── Mode: the active Venue decides paper vs live ─────────────────────────────

def _clob_venue():
    from tests.unit.test_clob_venue import StubClobClient
    from trading.clob_venue import ClobVenue
    return ClobVenue(client=StubClobClient(balance="42"), proxy_address="0xabc", http_get=None)


def test_dry_run_reflects_the_active_venue():
    from trading.executor import TradeExecutor
    assert _make_executor().dry_run is True
    assert TradeExecutor(_dry_config(trading_mode="live_run"), venue=_clob_venue()).dry_run is False


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


# ── Live arbitrage rows match main's live path ───────────────────────────────

def _run_arb_logged(executor, legs, timeout=0.05):
    rows = []
    asyncio.run(executor.execute_arbitrage_group(
        legs=legs, timeout_seconds=timeout,
        log_func=lambda level, asset, token, payload: rows.append((level, asset, token, payload)),
    ))
    return rows


def _reasons(rows, level):
    return [payload.get("reason") for lvl, _a, _t, payload in rows if lvl == level]


def _live_legs(*specs):
    return [{"token_id": t, "price": p, "shares": s, "side": "YES", "bet_amount_usd": round(p * s, 2),
             "group_id": "g"} for t, p, s in specs]


def test_live_arbitrage_all_legs_filled_reason():
    executor, _ = _live_executor({"A": 3.0, "B": 2.0})

    rows = _run_arb_logged(executor, _live_legs(("A", 0.30, 3.0), ("B", 0.40, 2.0)))

    assert _reasons(rows, "ARBITRAGE-FILL") == ["all_legs_filled"]


def test_live_arbitrage_unfilled_leg_is_insufficient_liquidity_and_cancels_are_logged():
    executor, client = _live_executor({"A": 3.0, "B": 0.0})

    rows = _run_arb_logged(executor, _live_legs(("A", 0.30, 3.0), ("B", 0.40, 2.0)))

    assert "insufficient_liquidity" in _reasons(rows, "REJECTED")
    cancels = [(token, payload) for lvl, _a, token, payload in rows if lvl == "CANCEL"]
    assert sorted(token for token, _ in cancels) == ["order_A", "order_B"]
    assert all(p["reason"] == "partial_fill_no_arb" and p["group_id"] == "g" and "response" in p
               for _, p in cancels)


def test_live_arbitrage_failed_cancel_is_logged_as_cancel_error():
    executor, client = _live_executor({"A": 3.0, "B": 0.0})

    def broken_cancel(order_id):
        raise RuntimeError("clob down")
    client.cancel = broken_cancel

    rows = _run_arb_logged(executor, _live_legs(("A", 0.30, 3.0), ("B", 0.40, 2.0)))

    errors = [payload for lvl, _a, _t, payload in rows if lvl == "CANCEL-ERROR"]
    assert len(errors) == 2
    assert all(p["reason"] == "cancel_failed" and "clob down" in p["error"] for p in errors)


def test_live_arbitrage_placement_failure_logs_the_response():
    executor, client = _live_executor({})
    client.order_response = {"errorMsg": "not enough balance"}

    rows = _run_arb_logged(executor, _live_legs(("A", 0.30, 3.0)))

    errors = [payload for lvl, _a, _t, payload in rows if lvl == "EXECUTION-ERROR"]
    assert errors == [{"reason": "order_placement_failed", "group_id": "g",
                       "response": {"errorMsg": "not enough balance"}}]


def test_live_arbitrage_orders_the_legs_share_count():
    """The leg's own share count is ordered, not one re-derived from its dollars."""
    executor, client = _live_executor({"A": 7.0})

    _run_arb_logged(executor, [
        {"token_id": "A", "price": 0.30, "shares": 7.0, "side": "YES", "bet_amount_usd": 3.0},
    ])

    assert client.orders == [("A", 0.3, "BUY", 7.0)]


def test_paper_arbitrage_keeps_its_own_reasons():
    venue = _arb_venue({"A": 0.30})
    rows = _run_arb_logged(_make_executor(venue=venue), [_leg("A", 0.30, 3.0), _leg("B", 0.40, 2.0)])

    assert "limit_order_not_filled" in _reasons(rows, "REJECTED")
    assert not [r for r in rows if r[0] in ("CANCEL", "CANCEL-ERROR")]

    venue = _arb_venue({"A": 0.30})
    rows = _run_arb_logged(_make_executor(venue=venue), [_leg("A", 0.30, 3.0)])
    assert _reasons(rows, "ARBITRAGE-FILL") == ["all_legs_filled_via_limit_order"]

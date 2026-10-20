"""One-time backfill of hunt_history trade rows into the Ledger (spec #12,
ticket #16).

Each test seeds a pre-Ledger trades database with hunt_history rows shaped
the way the engine logged them (Python-repr payloads, as log_event writes,
and JSON), runs database init, and asserts on what the Ledger reports
through its public reads."""
import json
import sqlite3

import pytest

import ui.data_manager as dm
from trading.ledger import Ledger


def _seed(db_path, rows):
    """A trades DB from before the Ledger existed: hunt_history only.
    rows: (timestamp, level, asset_type, token_id, payload). A dict payload
    is stored as its Python repr (what log_event writes)."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS hunt_history (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "timestamp TEXT, level TEXT, asset_type TEXT, token_id TEXT, payload TEXT)"
        )
        conn.executemany(
            "INSERT INTO hunt_history (timestamp, level, asset_type, token_id, payload) "
            "VALUES (?, ?, ?, ?, ?)",
            [(ts, lvl, at, tok, p if isinstance(p, str) else str(p)) for ts, lvl, at, tok, p in rows],
        )


def _model_entry_rows(ts, yes_token, exec_token, side, price, shares, bet, level="DRY-RUN",
                      market_name="Will BTC exceed $100k?", asset_type="Crypto::BTCUSDT"):
    """What one executed model trade leaves in hunt_history: AUTO-TRADE
    (intent), the executor's fill row, then the pipeline's TRACK row."""
    exec_payload = {"price": price, "shares": shares, "bet_amount_usd": bet, "side": side}
    if level == "LIVE-TRADE":
        exec_payload = {"success": True, "response": {"orderID": "0xabc"}, "side": side, "price": price}
    return [
        (ts, "AUTO-TRADE", asset_type, exec_token, {
            "market_price": price, "fair_value": 0.6, "ev": 0.3, "bet_amount_usd": bet,
            "shares": shares, "side": side}),
        (ts, level, asset_type, exec_token, exec_payload),
        (ts, "TRACK", asset_type, yes_token, {
            "market_name": market_name, "post_prob": 0.62, "pre_prob": 0.66, "wang_edge": 0.12,
            "side": side, "bet_usd": bet, "executed": True, "total_equity": 990.0,
            "spent_today": bet}),
    ]


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "trades.db")


def test_paper_model_entry_is_backfilled_once_with_its_entry_details(db):
    _seed(db, _model_entry_rows("2026-09-01 10:00:00", "yes1", "yes1", "YES", 0.40, 25.0, 10.0))

    dm.init_db(db)

    ledger = Ledger(db)
    [holding] = ledger.open_positions()
    assert holding.token_id == "yes1"
    assert holding.side == "YES"
    assert holding.venue == "paper"
    assert holding.strategy == "crypto"
    assert holding.shares == pytest.approx(25.0)
    assert holding.entry_price == pytest.approx(0.40)
    assert holding.pre_prob == pytest.approx(0.66)
    assert holding.post_prob == pytest.approx(0.62)
    assert holding.wang_edge == pytest.approx(0.12)
    assert holding.asset_type == "Crypto::BTCUSDT"
    assert holding.market_name == "Will BTC exceed $100k?"
    assert holding.opened_at == "2026-09-01 10:00:00"
    [buy] = ledger.transactions()
    assert buy["kind"] == "BUY"
    assert buy["time"] == "2026-09-01 10:00:00"
    assert buy["amount_usd"] == pytest.approx(10.0)


def test_live_no_side_entry_with_json_payloads_holds_the_no_token(db):
    rows = _model_entry_rows("2026-09-02 11:00:00", "yes2", "no2", "NO", 0.30, 33.33, 10.0,
                             level="LIVE-TRADE")
    _seed(db, [(ts, lvl, at, tok, json.dumps(p)) for ts, lvl, at, tok, p in rows])

    dm.init_db(db)

    ledger = Ledger(db)
    holding = ledger.holding("no2")
    assert holding is not None
    assert holding.side == "NO"
    assert holding.venue == "live"
    assert holding.entry_price == pytest.approx(0.30)
    assert holding.shares == pytest.approx(33.33)  # from AUTO-TRADE: LIVE-TRADE has no shares
    assert ledger.holding("yes2") is None


def test_legacy_paper_trade_row_with_fair_key_is_an_entry(db):
    _seed(db, [
        ("2026-04-01 09:00:00", "PAPER-TRADE", "Crypto::ETHUSDT", "eth1",
         {"price": 0.5, "shares": 20.0, "bet_amount_usd": 10.0, "side": "YES",
          "reason": "No live CLOB client configured"}),
        ("2026-04-01 09:00:00", "TRACK", "Crypto::ETHUSDT", "eth1",
         {"market_name": "ETH up?", "fair": 0.7, "side": "YES", "bet_usd": 10.0, "executed": True}),
    ])

    dm.init_db(db)

    holding = Ledger(db).holding("eth1")
    assert holding.venue == "paper"
    assert holding.post_prob == pytest.approx(0.7)
    assert holding.shares == pytest.approx(20.0)


def test_executor_fill_without_track_is_not_an_entry(db):
    """A DRY-RUN row whose paper fill then failed is followed by no TRACK."""
    rows = _model_entry_rows("2026-09-01 10:00:00", "yes1", "yes1", "YES", 0.40, 25.0, 10.0)
    _seed(db, rows[:2])

    dm.init_db(db)

    assert Ledger(db).transactions() == []


def _arbitrage_rows(ts, group_id, reason="all_legs_filled_via_limit_order"):
    """A two-leg basket: ARBITRAGE-FILL (from the executor), then one
    STRATEGY-LEG per leg (from the pipeline). Leg B's NO token is "legB_no"."""
    return [
        (ts, "ARBITRAGE-FILL", "Arbitrage::EventSum", group_id, {
            "reason": reason, "group_id": group_id,
            "fills": {"legA": 20.0, "legB_no": 25.0}, "arb_sets": 20.0, "surplus": {"legB_no": 5.0}}),
        (ts, "STRATEGY-LEG", "Arbitrage::EventSum", "legA", {
            "group_id": group_id, "market_name": "Leg A", "price": 0.45, "bet_usd": 9.0,
            "shares": 20.0, "side": "YES", "executed": True, "strategy_type": "arbitrage"}),
        (ts, "STRATEGY-LEG", "Arbitrage::EventSum", "legB_no", {
            "group_id": group_id, "market_name": "Leg B", "price": 0.40, "bet_usd": 10.0,
            "shares": 25.0, "side": "NO", "executed": True, "strategy_type": "arbitrage"}),
        (ts, "STRATEGY-LEG", "Arbitrage::EventSum", "legC", {
            "group_id": group_id, "market_name": "Leg C", "price": 0.10, "bet_usd": 2.0,
            "shares": 0.0, "side": "YES", "executed": False, "strategy_type": "arbitrage"}),
    ]


def test_arbitrage_legs_are_backfilled_once_per_filled_leg_with_their_group(db):
    _seed(db, _arbitrage_rows("2026-09-03 12:00:00", "grp1"))

    dm.init_db(db)

    ledger = Ledger(db)
    holdings = {h.token_id: h for h in ledger.open_positions()}
    assert set(holdings) == {"legA", "legB_no"}
    assert holdings["legA"].strategy == "arbitrage"
    assert holdings["legA"].group_id == "grp1"
    assert holdings["legA"].venue == "paper"
    assert holdings["legA"].entry_price == pytest.approx(0.45)
    assert holdings["legB_no"].side == "NO"
    assert holdings["legB_no"].shares == pytest.approx(25.0)
    assert len(ledger.transactions()) == 2


def test_live_arbitrage_fill_tags_its_legs_live(db):
    _seed(db, _arbitrage_rows("2026-09-03 12:00:00", "grp1", reason="all_legs_filled"))

    dm.init_db(db)

    assert {h.venue for h in Ledger(db).open_positions()} == {"live"}


def test_arbitrage_unwind_sale_closes_the_leg(db):
    _seed(db, _arbitrage_rows("2026-09-03 12:00:00", "grp1") + [
        ("2026-09-03 12:00:05", "DRY-RUN-SELL", "Portfolio", "legB_no",
         {"message": "[DRY-RUN] Would SELL", "price": 0.40, "shares": 5.0}),
    ])

    dm.init_db(db)

    ledger = Ledger(db)
    assert ledger.holding("legB_no").shares == pytest.approx(20.0)
    sale = ledger.transactions()[0]
    assert sale["kind"] == "SELL"
    assert sale["reason"] == "ARBITRAGE-UNWIND"
    assert sale["group_id"] == "grp1"


def _sell_rows(ts, token, price, shares, initial_price, reason="TAKE-PROFIT", sold=True,
               level="DRY-RUN-SELL"):
    sell_payload = {"price": price, "shares": shares}
    if level == "DRY-RUN-SELL":
        sell_payload["message"] = f"[DRY-RUN] Would SELL {shares} of {token} at ${price}"
    if level == "SELL":
        sell_payload.update(success=True, response={"orderID": "0xdef"})
    return [
        (ts, level, "Portfolio", token, sell_payload),
        (ts, reason, "Portfolio", token, {
            "threshold": 0.2, "shares": shares, "price": price, "initial_price": initial_price,
            "sold": sold}),
    ]


def test_take_profit_sale_closes_the_position_with_realized_pnl(db):
    _seed(db, _model_entry_rows("2026-09-01 10:00:00", "yes1", "yes1", "YES", 0.40, 25.0, 10.0)
          + _sell_rows("2026-09-01 14:00:00", "yes1", 0.60, 25.0, 0.40))

    dm.init_db(db)

    ledger = Ledger(db)
    assert ledger.open_positions() == []
    assert ledger.realized_pnl() == pytest.approx(5.0)
    sale = ledger.transactions()[0]
    assert (sale["kind"], sale["reason"], sale["time"]) == ("SELL", "TAKE-PROFIT", "2026-09-01 14:00:00")
    assert ledger.trade_stats()["win_rate"] == pytest.approx(100.0)


def test_failed_sell_attempt_is_not_a_sale(db):
    _seed(db, _model_entry_rows("2026-09-01 10:00:00", "yes1", "yes1", "YES", 0.40, 25.0, 10.0)
          + _sell_rows("2026-09-01 14:00:00", "yes1", 0.30, 25.0, 0.40, reason="STOP-LOSS",
                       sold=False))

    dm.init_db(db)

    ledger = Ledger(db)
    assert ledger.holding("yes1").shares == pytest.approx(25.0)
    assert [t["kind"] for t in ledger.transactions()] == ["BUY"]


def test_repeated_phantom_sells_after_a_position_closed_are_skipped(db):
    _seed(db, _model_entry_rows("2026-09-01 10:00:00", "yes1", "yes1", "YES", 0.40, 25.0, 10.0)
          + _sell_rows("2026-09-01 14:00:00", "yes1", 0.60, 25.0, 0.40)
          + _sell_rows("2026-09-01 14:00:30", "yes1", 0.60, 25.0, 0.40)
          + _sell_rows("2026-09-01 14:01:00", "yes1", 0.60, 25.0, 0.40))

    dm.init_db(db)

    ledger = Ledger(db)
    assert [t["kind"] for t in ledger.transactions()] == ["SELL", "BUY"]
    assert ledger.realized_pnl() == pytest.approx(5.0)


def test_live_sell_is_a_live_sale(db):
    rows = _model_entry_rows("2026-09-02 11:00:00", "yes2", "no2", "NO", 0.30, 33.33, 10.0,
                             level="LIVE-TRADE")
    _seed(db, rows + _sell_rows("2026-09-02 15:00:00", "no2", 0.20, 33.33, 0.30,
                                reason="STOP-LOSS", level="SELL"))

    dm.init_db(db)

    sale = Ledger(db).transactions()[0]
    assert (sale["kind"], sale["venue"], sale["reason"], sale["side"]) == ("SELL", "live", "STOP-LOSS", "NO")


def test_paper_resolution_pays_out_and_closes_the_position(db):
    _seed(db, _model_entry_rows("2026-09-01 10:00:00", "yes1", "yes1", "YES", 0.40, 25.0, 10.0) + [
        ("2026-09-05 00:00:00", "EXPIRED", "Portfolio", "yes1", {
            "shares": 0.0, "price": 1.0, "initial_price": 0.4, "side": "YES",
            "market_name": "Will BTC exceed $100k?", "payout": 0.0, "sold": True}),
    ])

    dm.init_db(db)

    ledger = Ledger(db)
    assert ledger.open_positions() == []
    resolved = ledger.transactions()[0]
    assert (resolved["kind"], resolved["venue"]) == ("RESOLVED", "paper")
    # Pre-fix paper rows logged 0 shares; the held 25 shares are what resolved.
    assert ledger.realized_pnl() == pytest.approx(15.0)


def test_live_gave_up_expiry_is_unconfirmed(db):
    rows = _model_entry_rows("2026-09-02 11:00:00", "yes2", "no2", "NO", 0.30, 33.33, 10.0,
                             level="LIVE-TRADE")
    _seed(db, rows + [
        ("2026-09-09 00:00:00", "EXPIRED", "Portfolio", "no2", {
            "shares": 33.33, "price": 0.0, "initial_price": 0.3, "side": "NO",
            "market_name": "cond2", "sold": False, "reason": "Gave up selling"}),
    ])

    dm.init_db(db)

    ledger = Ledger(db)
    assert ledger.open_positions() == []
    expired = ledger.transactions()[0]
    assert (expired["kind"], expired["confirmed"], expired["venue"]) == ("EXPIRED", False, "live")
    assert ledger.realized_pnl() == 0.0


def _full_history():
    return (
        _model_entry_rows("2026-09-01 10:00:00", "yes1", "yes1", "YES", 0.40, 25.0, 10.0)
        + _model_entry_rows("2026-09-02 11:00:00", "yes2", "no2", "NO", 0.30, 33.33, 10.0)
        + _arbitrage_rows("2026-09-03 12:00:00", "grp1")
        + _sell_rows("2026-09-04 14:00:00", "yes1", 0.60, 25.0, 0.40)
    )


def test_backfill_runs_once_across_repeated_inits(db):
    _seed(db, _full_history())

    dm.init_db(db)
    first = Ledger(db).transactions()
    dm.init_db(db)
    dm.init_db(db)

    assert Ledger(db).transactions() == first
    assert len(first) == 5


def test_interrupted_backfill_commits_nothing_and_completes_on_next_init(db, monkeypatch):
    """A crash or redeploy mid-backfill (the old container is stopped right
    after its WAL checkpoint) leaves no partial history, and the next start
    backfills it in full."""
    _seed(db, _full_history())
    calls = {"n": 0}
    real_record_entry = Ledger.record_entry

    def crash_on_third_entry(self, **kw):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("container stopped mid-backfill")
        return real_record_entry(self, **kw)

    monkeypatch.setattr(Ledger, "record_entry", crash_on_third_entry)
    dm.init_db(db)  # must not raise: the process still starts
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    assert Ledger(db).transactions() == []

    monkeypatch.setattr(Ledger, "record_entry", real_record_entry)
    dm.init_db(db)

    assert len(Ledger(db).transactions()) == 5


def test_history_already_in_the_ledger_is_not_backfilled_again(db):
    """A database whose engine already recorded trades into the Ledger: only
    hunt_history rows from before the Ledger's first trade are replayed."""
    _seed(db, _model_entry_rows("2026-09-01 10:00:00", "yes1", "yes1", "YES", 0.40, 25.0, 10.0)
          + _model_entry_rows("2026-10-01 10:00:00", "yes9", "yes9", "YES", 0.50, 20.0, 10.0))
    Ledger(db).record_entry(token_id="yes9", condition_id="c9", side="YES", shares=20.0, price=0.5,
                            venue="paper", strategy="crypto", at="2026-10-01 10:00:00")

    dm.init_db(db)

    assert sorted(t["token_id"] for t in Ledger(db).transactions()) == ["yes1", "yes9"]


def test_fresh_database_backfills_nothing(db):
    dm.init_db(db)
    dm.init_db(db)

    assert Ledger(db).transactions() == []

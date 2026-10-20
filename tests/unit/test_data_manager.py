import json
from datetime import datetime
import sqlite3
from types import SimpleNamespace

import pytest

import ui.data_manager as dm
from core.bridge import DataBridge
from trading.ledger import Ledger


@pytest.fixture
def db(tmp_path):
    """Create and return a fresh test DB path."""
    db_file = str(tmp_path / "test_trades.db")
    dm.init_db(db_file)
    return db_file


# ── init_db ───────────────────────────────────────────────────────────────────

def test_init_db_creates_table(db):
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='hunt_history'"
        ).fetchone()
    assert row is not None

def test_init_db_creates_paper_snapshots_table(db):
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='paper_snapshots'"
        ).fetchone()
    assert row is not None

# ── insert_paper_snapshot ─────────────────────────────────────────────────────

def test_insert_paper_snapshot(db):
    dm.insert_paper_snapshot(db_path=db, cash=100.50, positions_value=50.25)
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT cash, positions_value, total_value FROM paper_snapshots"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 100.50
    assert rows[0][1] == 50.25
    assert rows[0][2] == 150.75

def test_get_paper_snapshots(db):
    dm.insert_paper_snapshot(db_path=db, cash=100.0, positions_value=50.0)
    dm.insert_paper_snapshot(db_path=db, cash=120.0, positions_value=60.0)
    
    # Test ordering and format
    snaps = dm.get_paper_snapshots(db)
    assert len(snaps) == 2
    assert snaps[0]["cash"] == 100.0
    assert snaps[0]["total_value"] == 150.0
    assert snaps[1]["cash"] == 120.0
    assert snaps[1]["total_value"] == 180.0
    
    # Test limit
    snaps_limit = dm.get_paper_snapshots(db, limit=1)
    assert len(snaps_limit) == 1


# ── log_event ─────────────────────────────────────────────────────────────────

def test_log_event_inserts_row(db):
    bridge = DataBridge()
    dm.log_event(bridge, "DRY-RUN", "Crypto::BTC", "tok1", {"ev": 0.5}, db_path=db)

    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT level FROM hunt_history").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "DRY-RUN"


def test_log_event_updates_ev_samples(db):
    bridge = DataBridge()
    dm.log_event(bridge, "TRACK", "Crypto::BTC", "tok1", {"ev": 0.42}, db_path=db)

    assert len(bridge.ev_samples) == 1
    assert bridge.ev_samples[0] == 0.42


def test_log_event_non_dict_payload_does_not_crash(db):
    bridge = DataBridge()
    dm.log_event(bridge, "INFO", "Crypto::BTC", "tok1", "plain string payload", db_path=db)

    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT payload FROM hunt_history").fetchall()
    assert rows[0][0] == "plain string payload"


# ── trade stats / transactions: read from the Ledger ─────────────────────────

def _ledger_with_round_trip(db):
    ledger = Ledger(db)
    ledger.record_entry(
        token_id="tok1", condition_id="c1", side="YES", shares=10.0, price=0.40,
        venue="paper", strategy="crypto", market_name="Will BTC hit 100k?",
    )
    ledger.record_exit(token_id="tok1", shares=10.0, price=0.65, reason="TAKE-PROFIT", venue="paper")
    return ledger


def test_get_trade_stats_empty_db(db):
    stats = dm.get_trade_stats(db_path=db)
    assert stats["total_trades"] == 0
    assert stats["win_rate"] == 0.0


def test_get_trade_stats_reads_the_ledger(db):
    _ledger_with_round_trip(db)

    stats = dm.get_trade_stats(db_path=db)

    assert stats["total_trades"] == 2
    assert stats["win_rate"] == 100.0


def test_trade_log_rows_in_hunt_history_are_not_counted_as_trades(db):
    """hunt_history is diagnostics only: AUTO-TRADE / DRY-RUN / TRACK rows
    logged for one fill must not count — the Ledger's one entry does."""
    bridge = DataBridge()
    for level in ("AUTO-TRADE", "DRY-RUN", "TRACK", "PAPER-TRADE"):
        dm.log_event(bridge, level, "Crypto::BTC", "tok1",
                     {"side": "YES", "price": 0.5, "shares": 2}, db_path=db)
    Ledger(db).record_entry(
        token_id="tok1", condition_id="c1", side="YES", shares=2.0, price=0.5,
        venue="paper", strategy="crypto",
    )

    assert dm.get_trade_stats(db_path=db)["total_trades"] == 1


def test_get_closed_trade_deltas_reads_the_ledger(db):
    _ledger_with_round_trip(db)

    assert dm.get_closed_trade_deltas(db_path=db) == pytest.approx([0.25])


def test_fetch_transactions_empty_db(db):
    df = dm.fetch_transactions(db_path=db)
    assert df.empty
    assert list(df.columns) == dm._TRANSACTIONS_COLUMNS


def test_fetch_transactions_includes_buys_and_sells(db):
    _ledger_with_round_trip(db)

    df = dm.fetch_transactions(db_path=db)

    assert len(df) == 2
    assert set(df["Type"]) == {"BUY", "SELL (Take-Profit)"}
    sell_row = df[df["Type"] == "SELL (Take-Profit)"].iloc[0]
    assert sell_row["P&L ($/share)"] == pytest.approx(0.25)
    assert sell_row["Market Name"] == "Will BTC hit 100k?"
    assert sell_row["Status"] == "Sold"
    assert df[df["Type"] == "BUY"].iloc[0]["Amount ($)"] == pytest.approx(4.0)


def test_fetch_transactions_shows_unconfirmed_expiry(db):
    ledger = Ledger(db)
    ledger.record_entry(token_id="dead", condition_id="c1", side="NO", shares=5.0, price=0.30,
                        venue="live", strategy="crypto")
    ledger.record_resolution(token_id="dead", shares=5.0, payout_price=None, venue="live")

    df = dm.fetch_transactions(db_path=db)

    expiry = df.iloc[0]
    assert expiry["Type"] == "EXPIRED"
    assert expiry["Status"] == "Unconfirmed"
    assert expiry["Side"] == "NO"


# ── process_logs_for_display: Phase 7 analytics columns ──────────────────────

def test_display_table_surfaces_wang_and_strategy_columns(db):
    bridge = DataBridge()
    payload = {
        "market_name": "BTC >100k", "price": 0.40, "post_prob": 0.55, "ev": 0.375,
        "pre_prob": 0.50, "wang_lambda": 0.15, "wang_fair_value": 0.55,
        "wang_edge": 0.15, "strategy_type": "model",
        "kelly_fraction_used": 0.25, "correlation_exposure": 0.3,
    }
    dm.log_event(bridge, "TRACK", "Crypto::BTC", "tok1", payload, db_path=db)

    table = dm.fetch_latest_history(db, limit=10)

    for col in ("Strategy", "Raw Prob", "Entry k", "Wang FV", "Wang Edge", "Kelly Frac", "Correlation"):
        assert col in table.columns

    row = table.iloc[0]
    assert row["Strategy"] == "model"
    assert row["Raw Prob"] == pytest.approx(0.50)
    assert row["Wang Edge"] == pytest.approx(0.15)
    assert row["Kelly Frac"] == pytest.approx(0.25)
    assert row["Correlation"] == pytest.approx(0.3)


def test_display_table_arbitrage_strategy_tag():
    df = __import__("pandas").DataFrame([{
        "timestamp": "2026-01-01 00:00:00", "level": "STRATEGY-LEG", "asset_type": "Arbitrage::EventSum",
        "token_id": "tok1", "payload": json.dumps({"strategy_type": "arbitrage", "edge": 0.05}),
    }])
    table = dm.process_logs_for_display(df)
    assert table.iloc[0]["Strategy"] == "arbitrage"


# ── restore_runtime_state ─────────────────────────────────────────────────────

def test_restore_runtime_state_returns_default_when_empty(db):
    state = dm.restore_runtime_state(db, fallback_starting_balance=500.0)
    assert state["current_balance"] == 500.0
    assert state["source"] == "default"


def test_restore_runtime_state_spent_today_is_todays_ledger_entries(db):
    ledger = Ledger(db)
    common = dict(condition_id="c", side="YES", venue="paper", strategy="crypto")
    ledger.record_entry(token_id="old", shares=10.0, price=0.5, at="2026-10-19 23:59:59", **common)
    ledger.record_entry(token_id="a", shares=20.0, price=0.5, at="2026-10-20 00:00:01", **common)
    ledger.record_entry(token_id="b", shares=8.0, price=0.25, at="2026-10-20 13:00:00",
                        **{**common, "strategy": "arbitrage"})

    state = dm.restore_runtime_state(db, fallback_starting_balance=500.0,
                                     now=datetime(2026, 10, 20, 15, 0, 0))

    assert state["spent_today"] == pytest.approx(12.0)


def test_restore_runtime_state_balances_resume_from_the_last_engine_status(db):
    dm.write_engine_status(db, balance=850.0, starting_balance=1000.0, position_value=120.0,
                           unrealized_pnl=3.0, watch_only=False)

    state = dm.restore_runtime_state(db, fallback_starting_balance=500.0)

    assert state["current_balance"] == 850.0
    assert state["starting_balance"] == 1000.0
    assert state["source"] == "engine_status"


# ── WAL mode ──────────────────────────────────────────────────────────────────

def test_init_db_enables_wal_mode(db):
    with sqlite3.connect(db) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


# ── engine_status: engine writes, dashboard reads ──────────────────────────────

def test_read_engine_status_returns_none_when_absent(db):
    assert dm.read_engine_status(db) is None


def test_write_and_read_engine_status_roundtrip(db):
    dm.write_engine_status(
        db, balance=123.45, starting_balance=100.0,
        position_value=50.0, unrealized_pnl=5.5, watch_only=True,
    )
    status = dm.read_engine_status(db)

    assert status["current_balance"] == pytest.approx(123.45)
    assert status["starting_balance"] == pytest.approx(100.0)
    assert status["open_position_value"] == pytest.approx(50.0)
    assert status["unrealized_pnl"] == pytest.approx(5.5)
    assert status["watch_only"] is True


def test_write_engine_status_overwrites_previous_row(db):
    """One row (id=1), overwritten every tick — not accumulated."""
    dm.write_engine_status(db, balance=100.0, starting_balance=100.0,
                            position_value=0.0, unrealized_pnl=0.0, watch_only=False)
    dm.write_engine_status(db, balance=200.0, starting_balance=100.0,
                            position_value=10.0, unrealized_pnl=1.0, watch_only=True)

    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT COUNT(*) FROM engine_status").fetchone()
    assert rows[0] == 1

    status = dm.read_engine_status(db)
    assert status["current_balance"] == pytest.approx(200.0)
    assert status["watch_only"] is True


# ── engine_control (live_trading_requested): dashboard writes, engine reads ───

def test_read_live_trading_requested_defaults_false_when_absent(db):
    """Safe default: an unattended engine should never live-trade real
    money before an operator (or the config-derived seed) opts in."""
    assert dm.read_live_trading_requested(db) is False


def test_write_and_read_live_trading_requested_roundtrip(db):
    dm.write_live_trading_requested(db, True)
    assert dm.read_live_trading_requested(db) is True

    dm.write_live_trading_requested(db, False)
    assert dm.read_live_trading_requested(db) is False


def test_seed_live_trading_requested_if_absent_only_seeds_once(db):
    dm.seed_live_trading_requested_if_absent(db, True)
    assert dm.read_live_trading_requested(db) is True

    # A later restart's seed call must not clobber an operator's own
    # preference already on record.
    dm.write_live_trading_requested(db, False)
    dm.seed_live_trading_requested_if_absent(db, True)
    assert dm.read_live_trading_requested(db) is False


# ── open_positions: engine writes (wholesale replace), dashboard reads ────────

def _pos(token_id="tok1", market_id="mkt1", side="YES", shares=10.0,
         initial_price=0.40, current_price=0.50, value=5.0, pnl_ratio=0.25):
    return SimpleNamespace(
        token_id=token_id, market_id=market_id, side=side, shares=shares,
        initial_price=initial_price, current_price=current_price,
        value=value, pnl_ratio=pnl_ratio,
    )


def test_read_open_positions_empty_when_none_written(db):
    assert dm.read_open_positions(db) == []


def test_write_and_read_open_positions_roundtrip(db):
    dm.write_open_positions(db, [_pos(token_id="tok1")])
    rows = dm.read_open_positions(db)

    assert len(rows) == 1
    row = rows[0]
    assert row["token_id"] == "tok1"
    assert row["market_id"] == "mkt1"
    assert row["side"] == "YES"
    assert row["shares"] == pytest.approx(10.0)
    assert row["initial_price"] == pytest.approx(0.40)
    assert row["current_price"] == pytest.approx(0.50)
    assert row["pnl_ratio"] == pytest.approx(0.25)


def test_write_open_positions_merges_analytics(db):
    analytics = {"tok1": {"asset_type": "Crypto::BTCUSDT", "entry_wang_edge": 0.10,
                           "current_wang_edge": 0.03, "edge_delta": -0.07}}
    dm.write_open_positions(db, [_pos(token_id="tok1")], position_analytics=analytics)
    row = dm.read_open_positions(db)[0]

    assert row["asset_type"] == "Crypto::BTCUSDT"
    assert row["wang_edge_entry"] == pytest.approx(0.10)
    assert row["wang_edge_now"] == pytest.approx(0.03)
    assert row["wang_edge_delta"] == pytest.approx(-0.07)


def test_write_open_positions_is_a_wholesale_replace(db):
    """Each call replaces the whole table — a closed position must
    disappear, not linger from a previous cycle's write."""
    dm.write_open_positions(db, [_pos(token_id="tok1"), _pos(token_id="tok2")])
    assert len(dm.read_open_positions(db)) == 2

    dm.write_open_positions(db, [_pos(token_id="tok1")])
    rows = dm.read_open_positions(db)
    assert len(rows) == 1
    assert rows[0]["token_id"] == "tok1"


def test_write_open_positions_empty_list_clears_table(db):
    dm.write_open_positions(db, [_pos(token_id="tok1")])
    dm.write_open_positions(db, [])
    assert dm.read_open_positions(db) == []


# ── get_level_counts: replaces bridge.level_counts ─────────────────────────────

def test_get_level_counts_empty_db(db):
    assert dm.get_level_counts(db) == {}


def test_get_level_counts_groups_by_level(db):
    bridge = DataBridge()
    dm.log_event(bridge, "DRY-RUN", "Crypto::BTC", "tok1", {"ev": 0.5}, db_path=db)
    dm.log_event(bridge, "DRY-RUN", "Crypto::BTC", "tok2", {"ev": 0.4}, db_path=db)
    dm.log_event(bridge, "REJECTED", "Crypto::BTC", "tok3", {"reason": "low ev"}, db_path=db)

    counts = dm.get_level_counts(db)
    assert counts["DRY-RUN"] == 2
    assert counts["REJECTED"] == 1


# ── get_position_market_names: Portfolio view labels, from the Ledger ───────

def test_get_position_market_names_reads_open_ledger_holdings(db):
    ledger = Ledger(db)
    ledger.record_entry(token_id="tok1", condition_id="c1", side="YES", shares=10.0, price=0.4,
                        venue="paper", strategy="crypto", market_name="Will BTC hit 100k?")
    ledger.record_entry(token_id="sold", condition_id="c2", side="NO", shares=5.0, price=0.3,
                        venue="paper", strategy="crypto", market_name="Gone")
    ledger.record_exit(token_id="sold", shares=5.0, price=0.5, reason="TAKE-PROFIT", venue="paper")
    bridge = DataBridge()
    dm.log_event(bridge, "TRACK", "Crypto::BTC", "tok2",
                 {"ev": 0.5, "market_name": "Only in the event log"}, db_path=db)

    assert dm.get_position_market_names(db) == {"tok1": "Will BTC hit 100k?"}


# ── get_recent_opportunity_map / get_latest_ev_by_token: the EV-by-market
#    chart (diagnostics, not position data) ────────────────────────────────────

def test_get_recent_opportunity_map_empty_db(db):
    assert dm.get_recent_opportunity_map(db) == {}


def test_get_recent_opportunity_map_keeps_most_recent_per_token(db):
    bridge = DataBridge()
    dm.log_event(bridge, "TRACK", "Crypto::BTC", "tok1",
                 {"ev": 0.30, "market_name": "old name"}, db_path=db)
    dm.log_event(bridge, "TRACK", "Crypto::BTC", "tok1",
                 {"ev": 0.60, "market_name": "new name"}, db_path=db)

    opp_map = dm.get_recent_opportunity_map(db)
    assert opp_map["tok1"]["ev"] == pytest.approx(0.60)
    assert opp_map["tok1"]["market_name"] == "new name"


def test_get_recent_opportunity_map_ignores_payloads_without_ev(db):
    bridge = DataBridge()
    dm.log_event(bridge, "SCAN-SKIP", "Crypto::BTC", "tok1",
                 {"reason": "already_held_simulated"}, db_path=db)

    assert dm.get_recent_opportunity_map(db) == {}


def test_get_latest_ev_by_token_sorted_descending_and_capped(db):
    bridge = DataBridge()
    dm.log_event(bridge, "TRACK", "Crypto::BTC", "tok_low",
                 {"ev": 0.10, "market_name": "Low EV Market"}, db_path=db)
    dm.log_event(bridge, "TRACK", "Crypto::ETH", "tok_high",
                 {"ev": 0.90, "market_name": "High EV Market"}, db_path=db)

    items = dm.get_latest_ev_by_token(db, limit=1)
    assert len(items) == 1
    assert items[0]["token_id"] == "tok_high"
    assert items[0]["market_name"] == "High EV Market"


def test_get_latest_ev_by_token_falls_back_to_token_id_when_no_market_name(db):
    bridge = DataBridge()
    dm.log_event(bridge, "TRACK", "Crypto::BTC", "tok1", {"ev": 0.5}, db_path=db)

    items = dm.get_latest_ev_by_token(db)
    assert items[0]["market_name"] == "tok1"


# ── get_terminal_feed: replaces bridge.terminal_logs ───────────────────────────

def test_get_terminal_feed_empty_db(db):
    assert dm.get_terminal_feed(db) == []


def test_get_terminal_feed_formats_most_recent_first(db):
    bridge = DataBridge()
    dm.log_event(bridge, "REJECTED", "Crypto::BTC", "tok1",
                 {"reason": "EV below threshold", "ev": 0.1}, db_path=db)
    dm.log_event(bridge, "DRY-RUN", "Crypto::ETH", "tok2",
                 {"market_name": "ETH market", "ev": 0.6}, db_path=db)

    lines = dm.get_terminal_feed(db, limit=20)
    assert len(lines) == 2
    # Most recent (DRY-RUN) first.
    assert lines[0].startswith("[DRY-RUN] Crypto::ETH - ETH market")
    assert "ev=0.6" in lines[0]
    assert lines[1].startswith("[REJECTED] Crypto::BTC - EV below threshold")

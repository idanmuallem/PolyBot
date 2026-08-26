import json
import sqlite3

import pytest

import ui.data_manager as dm
from core.bridge import DataBridge


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


def test_log_event_updates_opportunity_map(db):
    bridge = DataBridge()
    payload = {"ev": 0.60, "post_prob": 0.70, "market_name": "BTC >100k"}
    dm.log_event(bridge, "TRACK", "Crypto::BTC", "tok1", payload, db_path=db)

    assert "tok1" in bridge.opportunity_map
    assert bridge.opportunity_map["tok1"]["ev"] == 0.60


def test_log_event_opportunity_map_carries_wang_and_strategy_fields(db):
    bridge = DataBridge()
    payload = {
        "ev": 0.60, "post_prob": 0.70, "market_name": "BTC >100k",
        "pre_prob": 0.55, "wang_lambda": 0.12, "wang_fair_value": 0.62,
        "wang_edge": 0.08, "strategy_type": "model",
        "kelly_fraction_used": 0.25, "correlation_exposure": 0.4,
    }
    dm.log_event(bridge, "TRACK", "Crypto::BTC", "tok1", payload, db_path=db)

    entry = bridge.opportunity_map["tok1"]
    assert entry["pre_prob"] == 0.55
    assert entry["wang_lambda"] == 0.12
    assert entry["wang_fair_value"] == 0.62
    assert entry["wang_edge"] == 0.08
    assert entry["strategy_type"] == "model"
    assert entry["kelly_fraction_used"] == 0.25
    assert entry["correlation_exposure"] == 0.4


def test_log_event_opportunity_map_omits_missing_wang_fields(db):
    bridge = DataBridge()
    dm.log_event(bridge, "TRACK", "Crypto::BTC", "tok1", {"ev": 0.5, "post_prob": 0.5}, db_path=db)

    entry = bridge.opportunity_map["tok1"]
    assert "pre_prob" not in entry
    assert "wang_edge" not in entry


def test_log_event_non_dict_payload_does_not_crash(db):
    bridge = DataBridge()
    dm.log_event(bridge, "INFO", "Crypto::BTC", "tok1", "plain string payload", db_path=db)

    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT payload FROM hunt_history").fetchall()
    assert rows[0][0] == "plain string payload"


# ── get_trade_stats ───────────────────────────────────────────────────────────

def test_get_trade_stats_empty_db(db):
    stats = dm.get_trade_stats(db_path=db)
    assert stats["total_trades"] == 0
    assert stats["win_rate"] == 0.0


def test_get_trade_stats_counts_sides(db):
    bridge = DataBridge()
    dm.log_event(bridge, "DRY-RUN", "Crypto::BTC", "tok_yes", {"side": "YES", "price": 0.5, "shares": 2}, db_path=db)
    dm.log_event(bridge, "DRY-RUN", "Crypto::ETH", "tok_no", {"side": "NO", "price": 0.5, "shares": 2}, db_path=db)

    stats = dm.get_trade_stats(db_path=db)
    assert stats["total_trades"] == 2
    assert stats["total_yes_trades"] == 1
    assert stats["total_no_trades"] == 1


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

    for col in ("Strategy", "Raw Prob", "Wang λ", "Wang FV", "Wang Edge", "Kelly Frac", "Correlation"):
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


def test_restore_runtime_state_picks_most_recent_balance(db):
    bridge = DataBridge()
    # Insert older TRACK row with equity=900
    dm.log_event(bridge, "TRACK", "Crypto::BTC", "tok_old",
                 {"total_equity": 900.0}, db_path=db)
    # Insert newer TRACK row with equity=850
    dm.log_event(bridge, "TRACK", "Crypto::BTC", "tok_new",
                 {"total_equity": 850.0}, db_path=db)

    # restore_runtime_state reads DESC by id → picks most recent first → 850
    state = dm.restore_runtime_state(db, fallback_starting_balance=500.0)
    assert state["current_balance"] == 850.0

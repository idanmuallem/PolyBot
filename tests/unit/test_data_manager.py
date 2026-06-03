import json
import sqlite3

import pytest

import ui.data_manager as dm
from core.bridge import DataBridge


@pytest.fixture(autouse=True)
def reset_db_path(monkeypatch):
    """Reset the module-level DB path cache between tests."""
    monkeypatch.setattr(dm, "_ACTIVE_DB_PATH", None)


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Create and return a fresh test DB path."""
    db_file = str(tmp_path / "test_trades.db")
    monkeypatch.setattr(dm, "_ACTIVE_DB_PATH", None)
    dm.init_db(db_file)
    return db_file


# ── init_db ───────────────────────────────────────────────────────────────────

def test_init_db_creates_table(db):
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='hunt_history'"
        ).fetchone()
    assert row is not None


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
    payload = {"ev": 0.60, "fair": 0.70, "market_name": "BTC >100k"}
    dm.log_event(bridge, "TRACK", "Crypto::BTC", "tok1", payload, db_path=db)

    assert "tok1" in bridge.opportunity_map
    assert bridge.opportunity_map["tok1"]["ev"] == 0.60


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
    # Reset to bypass caching
    monkeypatch_reset_active_path(db)
    # Insert newer TRACK row with equity=850
    dm.log_event(bridge, "TRACK", "Crypto::BTC", "tok_new",
                 {"total_equity": 850.0}, db_path=db)

    # restore_runtime_state reads DESC by id → picks most recent first → 850
    state = dm.restore_runtime_state(db, fallback_starting_balance=500.0)
    assert state["current_balance"] == 850.0


def monkeypatch_reset_active_path(db):
    """Helper: keep _ACTIVE_DB_PATH pointing to the temp DB across log_event calls."""
    dm._ACTIVE_DB_PATH = db

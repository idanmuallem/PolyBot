import json
import sqlite3
from types import SimpleNamespace

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


def test_get_trade_stats_does_not_double_count_auto_trade(db):
    """evaluate_and_execute() logs AUTO-TRADE immediately before calling
    execute_trade(), which logs exactly one of DRY-RUN/PAPER-TRADE/LIVE-TRADE
    for the same trade. AUTO-TRADE must not also be counted, or every real
    trade is counted twice."""
    bridge = DataBridge()
    dm.log_event(bridge, "AUTO-TRADE", "Crypto::BTC", "tok_yes",
                 {"side": "YES", "price": 0.5, "shares": 2}, db_path=db)
    dm.log_event(bridge, "DRY-RUN", "Crypto::BTC", "tok_yes",
                 {"side": "YES", "price": 0.5, "shares": 2}, db_path=db)

    stats = dm.get_trade_stats(db_path=db)
    assert stats["total_trades"] == 1


def test_get_trade_stats_counts_ev_convergence_and_wang_edge_decay_exits(db):
    """EV-CONVERGENCE/WANG-EDGE-DECAY exits aren't P&L-threshold-triggered
    like TAKE-PROFIT/STOP-LOSS, but they're still real closed trades and
    must count toward total_trades/win_rate/avg_win/avg_loss — win/loss for
    these two levels is derived from the realized price move (price vs
    initial_price) since the level itself doesn't imply either, unlike
    TAKE-PROFIT/STOP-LOSS."""
    bridge = DataBridge()
    dm.log_event(bridge, "EV-CONVERGENCE", "Crypto::BTC", "tok1",
                 {"side": "NO", "price": 0.38, "initial_price": 0.43, "shares": 58.14, "sold": True,
                  "estimated_ev": -0.1163}, db_path=db)
    dm.log_event(bridge, "WANG-EDGE-DECAY", "Crypto::ETH", "tok2",
                 {"side": "YES", "price": 0.55, "initial_price": 0.50, "shares": 10, "sold": True,
                  "pnl_ratio": 0.10}, db_path=db)

    stats = dm.get_trade_stats(db_path=db)

    assert stats["total_trades"] == 2
    # EV-CONVERGENCE: price(0.38) < initial_price(0.43) -> a loss.
    # WANG-EDGE-DECAY: price(0.55) > initial_price(0.50) -> a win.
    assert stats["win_rate"] == pytest.approx(50.0)
    assert stats["avg_win"] == pytest.approx(round(0.55 * 10, 2))
    assert stats["avg_loss"] == pytest.approx(round(0.38 * 58.14, 2))


def test_get_trade_stats_excludes_failed_sell_attempts_from_win_loss(db):
    """sell_position() now honestly reports failure (see
    trading/executor.py) rather than the old hardcoded True, so a position
    that can't actually be sold gets retried on every loop tick
    indefinitely, logging a fresh sold: false TAKE-PROFIT/STOP-LOSS row
    each time. price/shares are still present on a failed attempt (they
    describe the position being evaluated, not the sale's outcome), so
    those rows must not be counted toward win_rate/avg_win/avg_loss."""
    bridge = DataBridge()
    for _ in range(5):  # simulates repeated retries against one stuck position
        dm.log_event(bridge, "STOP-LOSS", "Crypto::BTC", "stuck_tok",
                      {"price": 0.0005, "initial_price": 0.001, "shares": 1000.0, "sold": False},
                      db_path=db)
    dm.log_event(bridge, "TAKE-PROFIT", "Crypto::ETH", "tok2",
                 {"price": 0.60, "initial_price": 0.50, "shares": 5, "sold": True}, db_path=db)

    stats = dm.get_trade_stats(db_path=db)

    assert stats["win_rate"] == 100.0
    assert stats["avg_win"] == pytest.approx(3.0)  # 0.60 * 5
    assert stats["avg_loss"] == 0.0


# ── get_closed_trade_deltas ───────────────────────────────────────────────────

def test_get_closed_trade_deltas_empty_db(db):
    assert dm.get_closed_trade_deltas(db_path=db) == []


def test_get_closed_trade_deltas_computes_price_minus_initial_price(db):
    bridge = DataBridge()
    dm.log_event(bridge, "TAKE-PROFIT", "Crypto::BTC", "tok1",
                 {"price": 0.65, "initial_price": 0.50, "shares": 10, "sold": True}, db_path=db)
    dm.log_event(bridge, "STOP-LOSS", "Crypto::ETH", "tok2",
                 {"price": 0.30, "initial_price": 0.45, "shares": 5, "sold": True}, db_path=db)

    deltas = sorted(dm.get_closed_trade_deltas(db_path=db))

    assert deltas == pytest.approx([-0.15, 0.15])


def test_get_closed_trade_deltas_includes_all_exit_reasons(db):
    bridge = DataBridge()
    for level in ("TAKE-PROFIT", "STOP-LOSS", "WANG-EDGE-DECAY", "EV-CONVERGENCE"):
        dm.log_event(bridge, level, "Crypto::BTC", "tok1",
                     {"price": 0.55, "initial_price": 0.50, "sold": True}, db_path=db)

    deltas = dm.get_closed_trade_deltas(db_path=db)

    assert len(deltas) == 4
    assert all(d == pytest.approx(0.05) for d in deltas)


def test_get_closed_trade_deltas_ignores_non_exit_levels(db):
    bridge = DataBridge()
    dm.log_event(bridge, "DRY-RUN", "Crypto::BTC", "tok1",
                 {"price": 0.55, "initial_price": 0.50, "sold": True}, db_path=db)
    dm.log_event(bridge, "TRACK", "Crypto::BTC", "tok1",
                 {"price": 0.55, "initial_price": 0.50, "sold": True}, db_path=db)

    assert dm.get_closed_trade_deltas(db_path=db) == []


def test_get_closed_trade_deltas_skips_rows_missing_initial_price(db):
    """Rows logged before initial_price was added to the exit payload."""
    bridge = DataBridge()
    dm.log_event(bridge, "TAKE-PROFIT", "Crypto::BTC", "tok1",
                 {"price": 0.65, "shares": 10, "sold": True}, db_path=db)  # no initial_price

    assert dm.get_closed_trade_deltas(db_path=db) == []


def test_get_closed_trade_deltas_excludes_failed_sell_attempts(db):
    """sell_position() now honestly reports failure (see
    trading/executor.py) rather than the old hardcoded True, so a position
    that can't actually be sold (e.g. a missing paper-adapter token
    mapping) gets retried on every loop tick indefinitely, logging a fresh
    sold: false row each time. Those must not be counted as real closed
    trades — a stuck position retried for months would otherwise swamp
    this average with duplicates of one delta from a position that never
    actually closed."""
    bridge = DataBridge()
    dm.log_event(bridge, "STOP-LOSS", "Crypto::BTC", "stuck_tok",
                 {"price": 0.0005, "initial_price": 0.001, "shares": 1000.0, "sold": False}, db_path=db)
    dm.log_event(bridge, "TAKE-PROFIT", "Crypto::ETH", "tok2",
                 {"price": 0.60, "initial_price": 0.50, "shares": 5, "sold": True}, db_path=db)

    deltas = dm.get_closed_trade_deltas(db_path=db)

    assert deltas == pytest.approx([0.10])


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


# ── get_recent_opportunity_map / get_latest_ev_by_token: replaces
#    bridge.opportunity_map ─────────────────────────────────────────────────────

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

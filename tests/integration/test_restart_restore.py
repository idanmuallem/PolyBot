"""Restart restore reads the Ledger (spec #12, ticket #16): what the engine
held and spent before a restart or redeploy is what it resumes with."""
import sqlite3
from datetime import datetime

import pytest

import ui.data_manager as dm
from core.bridge import DataBridge
from core.runtime_env import restore_wallet_state
from core.trading_config import TradingConfig
from core.wallet_context import WalletContext
from trading.ledger import Ledger


def _ctx(db_path):
    return WalletContext(wallet_id="w", config=TradingConfig(), bridge=DataBridge(), db_path=db_path)


def test_state_after_restart_matches_state_before(tmp_path):
    db = str(tmp_path / "trades.db")
    today = datetime.now().strftime("%Y-%m-%d")

    # The engine's session before the restart.
    restore_wallet_state(_ctx(db))
    ledger = Ledger(db)
    ledger.record_entry(token_id="no1", condition_id="c1", side="NO", shares=25.0, price=0.40,
                        venue="paper", strategy="crypto", pre_prob=0.3, post_prob=0.35,
                        wang_edge=0.08, at=f"{today} 09:00:00")
    ledger.record_entry(token_id="legA", condition_id="c2", side="YES", shares=10.0, price=0.30,
                        venue="paper", strategy="arbitrage", group_id="g1",
                        at=f"{today} 10:00:00")
    dm.write_engine_status(db, balance=987.0, starting_balance=1000.0, position_value=13.0,
                           unrealized_pnl=0.0, watch_only=False)
    held_before = Ledger(db).open_positions()

    # Restart: a fresh process restores from the database.
    ctx = _ctx(db)
    restore_wallet_state(ctx)

    assert ctx.bridge.current_balance == pytest.approx(987.0)
    assert ctx.bridge.starting_balance == pytest.approx(1000.0)
    assert ctx.bridge.spent_today == pytest.approx(13.0)
    assert ctx.bridge.daily_spend == pytest.approx(13.0)
    assert Ledger(db).open_positions() == held_before
    assert Ledger(db).holding_for("c1", "NO").token_id == "no1"


def test_first_start_after_upgrade_restores_backfilled_history(tmp_path):
    """The first start on a pre-Ledger database backfills hunt_history, so
    restore already sees today's historical entries and open holdings."""
    db = str(tmp_path / "trades.db")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE hunt_history (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, "
            "level TEXT, asset_type TEXT, token_id TEXT, payload TEXT)"
        )
        conn.executemany(
            "INSERT INTO hunt_history (timestamp, level, asset_type, token_id, payload) VALUES (?, ?, ?, ?, ?)",
            [
                (now, "DRY-RUN", "Crypto::BTCUSDT", "yes1",
                 str({"price": 0.4, "shares": 25.0, "bet_amount_usd": 10.0, "side": "YES"})),
                (now, "TRACK", "Crypto::BTCUSDT", "yes1",
                 str({"market_name": "BTC?", "post_prob": 0.6, "side": "YES", "bet_usd": 10.0,
                      "executed": True, "total_equity": 990.0, "spent_today": 10.0})),
            ],
        )

    ctx = _ctx(db)
    restore_wallet_state(ctx)

    assert ctx.bridge.spent_today == pytest.approx(10.0)
    assert Ledger(db).holding("yes1").shares == pytest.approx(25.0)

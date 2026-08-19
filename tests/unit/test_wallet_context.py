"""Unit tests for WalletContext: verifies wallets are fully isolated from each other."""
import json

from core.bridge import DataBridge
from core.trading_config import TradingConfig
from core.wallet_context import WalletContext


def _make_ctx(wallet_id: str, db_path: str, **config_overrides) -> WalletContext:
    return WalletContext(
        wallet_id=wallet_id,
        config=TradingConfig(**config_overrides),
        bridge=DataBridge(wallet_id=wallet_id),
        db_path=db_path,
    )


# ── Isolation between two wallet contexts ────────────────────────────────────

def test_two_contexts_do_not_share_bridge_state(tmp_path):
    ctx_a = _make_ctx("wallet_alpha", str(tmp_path / "alpha" / "trades.db"))
    ctx_b = _make_ctx("wallet_beta", str(tmp_path / "beta" / "trades.db"))

    assert ctx_a.bridge is not ctx_b.bridge

    ctx_a.bridge.current_balance = 111.0
    ctx_a.bridge.terminal_logs.appendleft("alpha-only log line")

    assert ctx_b.bridge.current_balance == 0.0
    assert list(ctx_b.bridge.terminal_logs) == []


def test_two_contexts_use_different_db_paths(tmp_path):
    ctx_a = _make_ctx("wallet_alpha", str(tmp_path / "alpha" / "trades.db"))
    ctx_b = _make_ctx("wallet_beta", str(tmp_path / "beta" / "trades.db"))

    assert ctx_a.db_path != ctx_b.db_path
    assert ctx_a.wallet_id != ctx_b.wallet_id


def test_bridge_carries_its_own_wallet_id():
    ctx = _make_ctx("wallet_alpha", "wallet_alpha/trades.db")
    assert ctx.bridge.wallet_id == "wallet_alpha"


# ── Config loaded from file ───────────────────────────────────────────────────

def test_config_from_file_matches_expected_values(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "private_key": "0xabc",
        "proxy_address": "0xdef",
        "signature_type": 2,
        "dry_run": True,
        "paper_trade_mode": False,
        "min_ev": 0.35,
        "daily_limit_usd": 20.0,
        "max_bet_size_usd": 4.0,
        "bankroll_usd": 2000.0,
        "min_trading_balance": 5.0,
        "take_profit_pct": 0.25,
        "stop_loss_pct": -0.4,
        "min_hold_ev": -0.1,
        "loop_delay_seconds": 3.0,
        "max_daily_trades": 12,
        "min_tte_minutes": 60,
        "max_tte_days": 180,
    }), encoding="utf-8")

    config = TradingConfig.from_file(str(config_path))
    ctx = WalletContext(
        wallet_id="wallet_alpha",
        config=config,
        bridge=DataBridge(wallet_id="wallet_alpha"),
        db_path=str(tmp_path / "trades.db"),
    )

    assert ctx.config.min_ev == 0.35
    assert ctx.config.bankroll_usd == 2000.0
    assert ctx.config.max_daily_trades == 12
    assert ctx.config.private_key == "0xabc"
    assert ctx.config.proxy_address == "0xdef"
    assert ctx.config.dry_run is True


def test_two_contexts_loaded_from_different_files_keep_independent_config(tmp_path):
    config_a_path = tmp_path / "alpha.json"
    config_b_path = tmp_path / "beta.json"
    config_a_path.write_text(json.dumps({"min_ev": 0.10, "bankroll_usd": 500.0}), encoding="utf-8")
    config_b_path.write_text(json.dumps({"min_ev": 0.60, "bankroll_usd": 5000.0}), encoding="utf-8")

    ctx_a = WalletContext(
        wallet_id="wallet_alpha",
        config=TradingConfig.from_file(str(config_a_path)),
        bridge=DataBridge(wallet_id="wallet_alpha"),
        db_path=str(tmp_path / "alpha" / "trades.db"),
    )
    ctx_b = WalletContext(
        wallet_id="wallet_beta",
        config=TradingConfig.from_file(str(config_b_path)),
        bridge=DataBridge(wallet_id="wallet_beta"),
        db_path=str(tmp_path / "beta" / "trades.db"),
    )

    assert ctx_a.config.min_ev == 0.10
    assert ctx_b.config.min_ev == 0.60
    assert ctx_a.config.bankroll_usd == 500.0
    assert ctx_b.config.bankroll_usd == 5000.0


# ── Default status ────────────────────────────────────────────────────────────

def test_new_context_defaults_to_idle_status(tmp_path):
    ctx = _make_ctx("wallet_alpha", str(tmp_path / "trades.db"))
    assert ctx.status == "idle"

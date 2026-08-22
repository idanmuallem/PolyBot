from unittest.mock import MagicMock

import pytest

from core.bridge import DataBridge
from core.models import Position
from core.trading_config import TradingConfig
from core.wallet_context import WalletContext
from trading.risk_manager import PortfolioManager


def _make_pm(take_profit=0.20, stop_loss=-0.50, min_hold_ev=-0.10):
    bridge = DataBridge()
    bridge.current_balance = 50.0
    bridge.current_portfolio = []
    bridge.open_position_value = 0.0
    bridge.open_positions_value = 0.0
    bridge.total_pnl = 0.0
    config = TradingConfig(
        take_profit_pct=take_profit,
        stop_loss_pct=stop_loss,
        min_hold_ev=min_hold_ev,
        dry_run=True,
    )
    ctx = WalletContext(wallet_id="test_wallet", config=config, bridge=bridge, db_path="test_wallet_trades.db")
    executor = MagicMock()
    executor.get_open_positions.return_value = []
    pm = PortfolioManager(bridge=ctx.bridge, executor=executor, config=ctx.config, db_path=ctx.db_path)
    return pm, bridge, executor


def _pos(pnl_ratio=0.0, live_ev=0.0, token_id="tok1", price=0.50, shares=10.0, value=5.0, side="YES"):
    return Position(
        market_id="mkt1",
        token_id=token_id,
        initial_price=0.40,
        current_price=price,
        shares=shares,
        value=value,
        pnl_ratio=pnl_ratio,
        side=side,
        live_ev=live_ev,
    )


# ── manage_portfolio: exit logic ──────────────────────────────────────────────

def test_take_profit_triggered():
    pm, bridge, executor = _make_pm()
    pos = _pos(pnl_ratio=0.25, token_id="take_tok")
    executor.get_open_positions.side_effect = [[pos], []]
    executor.sell_position.return_value = True

    log_levels = []
    pm.manage_portfolio(lambda level, *a, **kw: log_levels.append(level))

    executor.sell_position.assert_called_once()
    assert "TAKE-PROFIT" in log_levels


def test_stop_loss_triggered():
    pm, bridge, executor = _make_pm()
    pos = _pos(pnl_ratio=-0.60, token_id="stop_tok")
    executor.get_open_positions.side_effect = [[pos], []]
    executor.sell_position.return_value = True

    log_levels = []
    pm.manage_portfolio(lambda level, *a, **kw: log_levels.append(level))

    executor.sell_position.assert_called_once()
    assert "STOP-LOSS" in log_levels


def test_ev_convergence_triggered():
    pm, bridge, executor = _make_pm()
    # pnl within bounds but live_ev below min_hold_ev
    pos = _pos(pnl_ratio=0.05, live_ev=-0.15, token_id="ev_tok")
    executor.get_open_positions.side_effect = [[pos], []]
    executor.sell_position.return_value = True

    log_levels = []
    pm.manage_portfolio(lambda level, *a, **kw: log_levels.append(level))

    executor.sell_position.assert_called_once()
    assert "EV-CONVERGENCE" in log_levels


def test_no_exit_within_bounds():
    pm, bridge, executor = _make_pm()
    pos = _pos(pnl_ratio=0.10, live_ev=0.0, token_id="fine_tok")
    executor.get_open_positions.return_value = [pos]

    pm.manage_portfolio(lambda *a, **kw: None)

    executor.sell_position.assert_not_called()


# ── manage_portfolio: Wang edge decay exit ────────────────────────────────────

def test_wang_edge_decay_triggers_exit():
    pm, bridge, executor = _make_pm()
    # pre_prob=0.55 -> Wang fair_value (base_lambda=0.183) ≈ 0.6212;
    # pricing the position exactly at that fair value collapses the edge to 0.
    pos = _pos(pnl_ratio=0.05, live_ev=0.0, token_id="decayed_tok", price=0.6212104251426875)
    bridge.opportunity_map = {"decayed_tok": {"pre_prob": 0.55}}
    executor.get_open_positions.side_effect = [[pos], []]
    executor.sell_position.return_value = True

    log_levels = []
    pm.manage_portfolio(lambda level, *a, **kw: log_levels.append(level))

    executor.sell_position.assert_called_once()
    assert "WANG-EDGE-DECAY" in log_levels


def test_healthy_wang_edge_does_not_trigger_exit():
    pm, bridge, executor = _make_pm()
    # Same pre_prob, but priced well below fair value -> healthy edge.
    pos = _pos(pnl_ratio=0.05, live_ev=0.0, token_id="healthy_tok", price=0.35)
    bridge.opportunity_map = {"healthy_tok": {"pre_prob": 0.55}}
    executor.get_open_positions.return_value = [pos]

    log_levels = []
    pm.manage_portfolio(lambda level, *a, **kw: log_levels.append(level))

    executor.sell_position.assert_not_called()
    assert "WANG-EDGE-DECAY" not in log_levels


def test_wang_edge_decay_skipped_without_raw_probability():
    # No opportunity_map entry and no DB row -> nothing to re-evaluate;
    # falls through to the existing P&L-based checks untouched.
    pm, bridge, executor = _make_pm()
    pos = _pos(pnl_ratio=0.10, live_ev=0.0, token_id="unknown_tok")
    executor.get_open_positions.return_value = [pos]

    log_levels = []
    pm.manage_portfolio(lambda level, *a, **kw: log_levels.append(level))

    executor.sell_position.assert_not_called()
    assert "WANG-EDGE-DECAY" not in log_levels


def test_take_profit_still_wins_over_decayed_edge():
    # Hard limits are checked first: a position hitting take-profit exits as
    # TAKE-PROFIT even though its Wang edge has also collapsed to zero.
    pm, bridge, executor = _make_pm(take_profit=0.20)
    pos = _pos(pnl_ratio=0.25, live_ev=0.0, token_id="tp_tok", price=0.6212104251426875)
    bridge.opportunity_map = {"tp_tok": {"pre_prob": 0.55}}
    executor.get_open_positions.side_effect = [[pos], []]
    executor.sell_position.return_value = True

    log_levels = []
    pm.manage_portfolio(lambda level, *a, **kw: log_levels.append(level))

    executor.sell_position.assert_called_once()
    assert "TAKE-PROFIT" in log_levels
    assert "WANG-EDGE-DECAY" not in log_levels


def test_wang_edge_decay_translates_no_side_probability():
    # pre_prob is always recorded in YES-space; for a NO position the
    # "probability my side wins" is 1 - pre_prob. p=0.30 (YES) ->
    # 0.70 (NO) -> Wang fair_value ≈ 0.7603 for that translated probability;
    # pricing the NO leg exactly there collapses its edge to 0.
    pm, bridge, executor = _make_pm()
    pos = _pos(pnl_ratio=0.05, live_ev=0.0, token_id="no_tok",
               price=0.7603411908017963, side="NO")
    bridge.opportunity_map = {"no_tok": {"pre_prob": 0.30}}
    executor.get_open_positions.side_effect = [[pos], []]
    executor.sell_position.return_value = True

    log_levels = []
    pm.manage_portfolio(lambda level, *a, **kw: log_levels.append(level))

    executor.sell_position.assert_called_once()
    assert "WANG-EDGE-DECAY" in log_levels


def test_wang_decay_threshold_is_half_wang_min_edge():
    pm, _, _ = _make_pm()
    assert pm.wang_decay_threshold == pytest.approx(pm.wang_min_edge / 2.0)


# ── Correlation exposure (trading/correlation.py wiring) ─────────────────────

def test_get_book_asset_types_resolves_via_opportunity_map():
    pm, bridge, executor = _make_pm()
    pos_a = _pos(token_id="tokA", price=0.5)
    pos_b = _pos(token_id="tokB", price=0.5)
    bridge.opportunity_map = {
        "tokA": {"asset_type": "Crypto::BTCUSDT"},
        "tokB": {"asset_type": "Weather::Miami"},
    }
    executor.get_open_positions.return_value = [pos_a, pos_b]

    pm._refresh_portfolio()
    asset_types = pm.get_book_asset_types()

    assert set(asset_types) == {"Crypto::BTCUSDT", "Weather::Miami"}


def test_correlation_exposure_for_averages_against_book():
    pm, bridge, executor = _make_pm()
    pos = _pos(token_id="tokA", price=0.5)
    bridge.opportunity_map = {"tokA": {"asset_type": "Crypto::BTCUSDT"}}
    executor.get_open_positions.return_value = [pos]
    pm._refresh_portfolio()

    # Same market already open -> exposure to another BTC candidate is 1.0.
    assert pm.correlation_exposure_for("Crypto::BTCUSDT") == 1.0
    # Unrelated category -> 0.0.
    assert pm.correlation_exposure_for("Weather::Miami") == 0.0


def test_correlation_exposure_zero_with_no_open_positions():
    pm, bridge, executor = _make_pm()
    executor.get_open_positions.return_value = []
    pm._refresh_portfolio()

    assert pm.correlation_exposure_for("Crypto::BTCUSDT") == 0.0


# ── Position analytics snapshot (dashboard edge-decay tracking) ──────────────

def test_manage_portfolio_stashes_position_analytics():
    pm, bridge, executor = _make_pm()
    pos = _pos(pnl_ratio=0.05, token_id="analytics_tok", price=0.40)
    bridge.opportunity_map = {
        "analytics_tok": {"pre_prob": 0.50, "wang_edge": 0.10, "asset_type": "Crypto::BTCUSDT"},
    }
    executor.get_open_positions.return_value = [pos]

    pm.manage_portfolio(lambda *a, **kw: None)

    snapshot = bridge.position_analytics["analytics_tok"]
    assert snapshot["asset_type"] == "Crypto::BTCUSDT"
    assert snapshot["pre_prob"] == pytest.approx(0.50)
    assert snapshot["entry_wang_edge"] == pytest.approx(0.10)
    assert snapshot["current_wang_edge"] is not None
    assert snapshot["edge_delta"] == pytest.approx(snapshot["current_wang_edge"] - 0.10)


def test_position_analytics_none_without_raw_probability():
    pm, bridge, executor = _make_pm()
    pos = _pos(pnl_ratio=0.05, token_id="unknown_tok", price=0.40)
    executor.get_open_positions.return_value = [pos]

    pm.manage_portfolio(lambda *a, **kw: None)

    snapshot = bridge.position_analytics["unknown_tok"]
    assert snapshot["current_wang_edge"] is None
    assert snapshot["edge_delta"] is None


# ── optimize_for_candidate ────────────────────────────────────────────────────

def test_cull_weak_position():
    pm, bridge, executor = _make_pm()
    # Position: post_prob=0.10, current_price=0.50 → live_ev=-0.80
    # Candidate: ev=0.55; 0.55 >= -0.80 + 0.10 → cull ✓
    pos = _pos(pnl_ratio=0.0, token_id="weak_tok", price=0.50, shares=10.0, value=5.0)
    bridge.opportunity_map = {"weak_tok": {"post_prob": 0.10}}
    executor.get_open_positions.return_value = [pos]
    executor.sell_position.return_value = True

    freed = pm.optimize_for_candidate(
        new_candidate_ev=0.55, min_improvement=0.10, log_func=lambda *a, **kw: None
    )

    assert freed > 0
    executor.sell_position.assert_called_once()


def test_no_cull_competitive_position():
    pm, bridge, executor = _make_pm()
    # Position: post_prob=0.80, current_price=0.50 → live_ev=0.60
    # Candidate: ev=0.55; 0.55 < 0.60 + 0.10=0.70 → don't cull ✓
    pos = _pos(pnl_ratio=0.0, token_id="comp_tok", price=0.50, shares=10.0, value=5.0)
    bridge.opportunity_map = {"comp_tok": {"post_prob": 0.80}}
    executor.get_open_positions.return_value = [pos]

    freed = pm.optimize_for_candidate(
        new_candidate_ev=0.55, min_improvement=0.10, log_func=lambda *a, **kw: None
    )

    assert freed == 0.0
    executor.sell_position.assert_not_called()


# ── free_up_capital ───────────────────────────────────────────────────────────

def test_free_up_capital_sells_weakest_first():
    pm, bridge, executor = _make_pm()
    bridge.current_balance = 0.0  # force liquidation
    pos_strong = _pos(pnl_ratio=-0.10, token_id="strong", price=0.50, shares=10.0, value=5.0)
    pos_weak = _pos(pnl_ratio=-0.30, token_id="weak", price=0.50, shares=10.0, value=5.0)
    executor.get_open_positions.return_value = [pos_strong, pos_weak]
    executor.sell_position.return_value = True

    sold_tokens = []
    pm.free_up_capital(
        required_amount=4.0,
        log_func=lambda level, asset, token, *a, **kw: sold_tokens.append(token),
    )

    # Weakest position (pnl_ratio=-0.30) is sold first
    assert sold_tokens[0] == "weak"

from unittest.mock import MagicMock

from core.bridge import DataBridge
from core.models import Position
from core.trading_config import TradingConfig
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
    executor = MagicMock()
    executor.get_open_positions.return_value = []
    pm = PortfolioManager(bridge=bridge, executor=executor, config=config)
    return pm, bridge, executor


def _pos(pnl_ratio=0.0, live_ev=0.0, token_id="tok1", price=0.50, shares=10.0, value=5.0):
    return Position(
        market_id="mkt1",
        token_id=token_id,
        initial_price=0.40,
        current_price=price,
        shares=shares,
        value=value,
        pnl_ratio=pnl_ratio,
        side="YES",
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


# ── optimize_for_candidate ────────────────────────────────────────────────────

def test_cull_weak_position():
    pm, bridge, executor = _make_pm()
    # Position: fair_value=0.10, current_price=0.50 → live_ev=-0.80
    # Candidate: ev=0.55; 0.55 >= -0.80 + 0.10 → cull ✓
    pos = _pos(pnl_ratio=0.0, token_id="weak_tok", price=0.50, shares=10.0, value=5.0)
    bridge.opportunity_map = {"weak_tok": {"fair_value": 0.10}}
    executor.get_open_positions.return_value = [pos]
    executor.sell_position.return_value = True

    freed = pm.optimize_for_candidate(
        new_candidate_ev=0.55, min_improvement=0.10, log_func=lambda *a, **kw: None
    )

    assert freed > 0
    executor.sell_position.assert_called_once()


def test_no_cull_competitive_position():
    pm, bridge, executor = _make_pm()
    # Position: fair_value=0.80, current_price=0.50 → live_ev=0.60
    # Candidate: ev=0.55; 0.55 < 0.60 + 0.10=0.70 → don't cull ✓
    pos = _pos(pnl_ratio=0.0, token_id="comp_tok", price=0.50, shares=10.0, value=5.0)
    bridge.opportunity_map = {"comp_tok": {"fair_value": 0.80}}
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

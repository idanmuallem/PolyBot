"""Integration: PortfolioManager + BudgetManager working together."""
from unittest.mock import MagicMock

from core.bridge import DataBridge
from core.models import Position
from core.trading_config import TradingConfig
from core.wallet_context import WalletContext
from trading.budget_manager import BudgetManager
from trading.risk_manager import PortfolioManager


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


def _make_components(initial_balance=50.0, take_profit=0.20, stop_loss=-0.50):
    bridge = DataBridge()
    bridge.current_balance = initial_balance
    bridge.current_portfolio = []
    bridge.open_position_value = 0.0
    bridge.open_positions_value = 0.0
    bridge.unrealized_pnl = 0.0

    config = TradingConfig(
        take_profit_pct=take_profit,
        stop_loss_pct=stop_loss,
        min_hold_ev=-0.10,
        trading_mode="dry_run",
        bankroll_usd=1000.0,
        daily_limit_usd=15.0,
    )
    ctx = WalletContext(wallet_id="test_wallet", config=config, bridge=bridge, db_path="test_wallet_trades.db")

    executor = MagicMock()
    executor.get_open_positions.return_value = []
    executor.sell_position.return_value = True

    pm = PortfolioManager(bridge=ctx.bridge, executor=executor, config=ctx.config, db_path=ctx.db_path)
    budget = BudgetManager(bridge=ctx.bridge, config=ctx.config, initial_balance=initial_balance)

    return pm, budget, bridge, executor


# ── Stop-loss exit increases available balance ────────────────────────────────

def test_stop_loss_frees_balance():
    pm, budget, bridge, executor = _make_components(initial_balance=50.0)
    pos = _pos(pnl_ratio=-0.60, token_id="stop_tok", value=5.0)
    executor.get_open_positions.side_effect = [[pos], []]

    initial_balance = bridge.current_balance
    pm.manage_portfolio(lambda *a, **kw: None)

    # _exit_position calls _apply_sale_to_bridge which adds position value to balance
    assert bridge.current_balance > initial_balance


def test_take_profit_exit_increases_balance():
    pm, budget, bridge, executor = _make_components(initial_balance=50.0)
    pos = _pos(pnl_ratio=0.25, token_id="tp_tok", value=8.0)
    executor.get_open_positions.side_effect = [[pos], []]

    initial_balance = bridge.current_balance
    pm.manage_portfolio(lambda *a, **kw: None)

    assert bridge.current_balance > initial_balance


# ── free_up_capital sells weakest first ──────────────────────────────────────

def test_free_up_capital_sells_weakest_first():
    pm, budget, bridge, executor = _make_components(initial_balance=0.0)
    bridge.current_balance = 0.0

    pos_strong = _pos(pnl_ratio=-0.10, token_id="strong", value=5.0)
    pos_weak = _pos(pnl_ratio=-0.30, token_id="weak", value=5.0)
    executor.get_open_positions.return_value = [pos_strong, pos_weak]

    sold_first = []
    pm.free_up_capital(
        required_amount=4.0,
        log_func=lambda level, asset, token, *a, **kw: sold_first.append(token),
    )

    # Weakest (pnl=-30%) sold first
    assert sold_first[0] == "weak"


def test_free_up_capital_stops_when_enough():
    pm, budget, bridge, executor = _make_components(initial_balance=0.0)
    bridge.current_balance = 0.0

    pos_a = _pos(pnl_ratio=-0.30, token_id="pos_a", value=10.0)
    pos_b = _pos(pnl_ratio=-0.20, token_id="pos_b", value=10.0)
    executor.get_open_positions.return_value = [pos_a, pos_b]

    sold_tokens = []
    pm.free_up_capital(
        required_amount=5.0,
        log_func=lambda level, asset, token, *a, **kw: sold_tokens.append(token),
    )

    # First sale gives 10.0 → exceeds 5.0 → stop (only one sell)
    assert executor.sell_position.call_count == 1
    assert sold_tokens == ["pos_a"]


# ── BudgetManager + PortfolioManager do not double-count ─────────────────────

def test_budget_spend_and_portfolio_manage_independent():
    pm, budget, bridge, executor = _make_components(initial_balance=100.0)

    # Spend $5 via budget_manager
    budget.record_trade(5.0)
    assert budget.total_spent_today == 5.0
    assert bridge.daily_spend == 5.0

    # manage_portfolio with no positions → no change to spent
    pm.manage_portfolio(lambda *a, **kw: None)
    assert budget.total_spent_today == 5.0  # unchanged

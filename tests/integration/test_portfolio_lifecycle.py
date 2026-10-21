"""Integration: PortfolioManager exits on a FakeVenue, with cash read back
through the Account (the single cash writer, which also keeps the daily
budget)."""
import pytest

from core.bridge import DataBridge
from core.trading_config import TradingConfig
from trading.account import Account
from trading.executor import TradeExecutor
from trading.fake_venue import FakeVenue
from trading.risk_manager import PortfolioManager
from trading.venue import MarketRef


def _market(name):
    return MarketRef(yes_token_id=name, no_token_id=f"{name}-no", condition_id=f"c-{name}", slug=name)


def _make_components(cash=50.0, take_profit=0.20, stop_loss=-0.50):
    venue = FakeVenue(cash=cash)
    bridge = DataBridge()
    config = TradingConfig(
        take_profit_pct=take_profit,
        stop_loss_pct=stop_loss,
        min_hold_ev=-0.10,
        trading_mode="dry_run",
        bankroll_usd=1000.0,
        daily_limit_usd=15.0,
    )
    executor = TradeExecutor(config=config, venue=venue)
    account = Account(active_venue=lambda: executor.venue, bridge=bridge, config=config)
    pm = PortfolioManager(bridge=bridge, executor=executor, config=config, account=account)
    return pm, account, venue


def _hold(venue, name, usd=10.0, entry=0.50, now=None):
    """Buy `usd` of `name` YES at `entry`, then move its price to `now`."""
    market = _market(name)
    venue.list_market(market, yes_price=entry)
    venue.buy(market, "YES", usd)
    if now is not None:
        venue.set_price(market, "YES", now)


# ── Exits free cash, as the Venue reports it ─────────────────────────────────

def test_stop_loss_frees_balance():
    pm, account, venue = _make_components(cash=50.0)
    _hold(venue, "stop_tok", usd=10.0, entry=0.50, now=0.20)   # -60%: 20 shares worth $4
    before = account.refresh().cash

    pm.manage_portfolio(lambda *a, **kw: None)

    assert account.refresh().cash == pytest.approx(before + 4.0)


def test_take_profit_exit_increases_balance():
    pm, account, venue = _make_components(cash=50.0)
    _hold(venue, "tp_tok", usd=10.0, entry=0.40, now=0.50)     # +25%: 25 shares worth $12.50
    before = account.refresh().cash

    pm.manage_portfolio(lambda *a, **kw: None)

    assert account.refresh().cash == pytest.approx(before + 12.5)


# ── free_up_capital sells weakest first ──────────────────────────────────────

def _spend_all(venue):
    """Park the remaining cash in a third, flat holding so cash is $0."""
    _hold(venue, "parked", usd=venue.cash(), entry=0.50)


def test_free_up_capital_sells_weakest_first():
    pm, account, venue = _make_components(cash=40.0)
    _hold(venue, "strong", usd=10.0, entry=0.50, now=0.45)      # -10%
    _hold(venue, "weak", usd=10.0, entry=0.50, now=0.35)        # -30%
    _spend_all(venue)

    sold = []
    pm.free_up_capital(
        required_amount=4.0,
        log_func=lambda level, asset, token, *a, **kw: sold.append(token) if level == "OPPORTUNITY-SWAP" else None,
    )

    assert sold[0] == "weak"


def test_free_up_capital_stops_when_enough():
    pm, account, venue = _make_components(cash=40.0)
    _hold(venue, "pos_a", usd=10.0, entry=0.50, now=0.35)       # weakest: frees $7
    _hold(venue, "pos_b", usd=10.0, entry=0.50, now=0.40)
    _spend_all(venue)

    sold = []
    freed = pm.free_up_capital(
        required_amount=5.0,
        log_func=lambda level, asset, token, *a, **kw: sold.append(token) if level == "OPPORTUNITY-SWAP" else None,
    )

    assert freed is True
    assert sold == ["pos_a"]
    assert account.cash == pytest.approx(7.0)


# ── Account spend + PortfolioManager do not double-count ─────────────────────

def test_budget_spend_and_portfolio_manage_independent():
    pm, account, venue = _make_components(cash=100.0)

    account.record_fill("crypto", 5.0)
    assert account.snapshot().spent_today == 5.0
    assert pm.bridge.spent_today == 5.0

    # manage_portfolio with no positions → no change to spent
    pm.manage_portfolio(lambda *a, **kw: None)
    assert account.snapshot().spent_today == 5.0  # unchanged

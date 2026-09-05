import sys
from unittest.mock import MagicMock

# Mock streamlit before any project import.
_st_mock = MagicMock()
_st_mock.cache_resource = lambda fn: fn  # no-op decorator
sys.modules["streamlit"] = _st_mock

# ui/components.py does `import streamlit.components.v1 as components` for its
# ECharts embeds — register that submodule explicitly too, since Python's
# import machinery won't resolve "streamlit.components.v1" through a mocked
# parent package's (fake) __path__.
_st_components_mock = MagicMock()
_st_components_v1_mock = MagicMock()
_st_mock.components = _st_components_mock
_st_components_mock.v1 = _st_components_v1_mock
sys.modules["streamlit.components"] = _st_components_mock
sys.modules["streamlit.components.v1"] = _st_components_v1_mock

# Mock eth_utils (Ethereum dependency not installable in this test env).
_eth_mock = MagicMock()
_eth_mock.to_checksum_address = lambda addr: str(addr)
sys.modules["eth_utils"] = _eth_mock

import pytest

from core.bridge import DataBridge
from core.models import MarketData
from core.trading_config import TradingConfig
from core.wallet_context import WalletContext


@pytest.fixture
def bridge():
    return DataBridge()


@pytest.fixture
def dry_run_config():
    return TradingConfig(
        trading_mode="dry_run",
        min_ev=0.30,
        bankroll_usd=1000.0,
        daily_limit_usd=15.0,
        max_bet_size_usd=3.0,
        min_trading_balance=5.0,
        max_daily_trades=10,
    )


@pytest.fixture
def wallet_context(tmp_path, dry_run_config, bridge):
    """A fully self-contained WalletContext: its own config, bridge, and temp db path."""
    return WalletContext(
        wallet_id="test_wallet",
        config=dry_run_config,
        bridge=bridge,
        db_path=str(tmp_path / "trades.db"),
    )


@pytest.fixture
def sample_market():
    return MarketData(
        market_id="tok_btc",
        asset_type="Crypto::BTCUSDT",
        strike_price=100_000.0,
        question="Will BTC exceed $100,000 by end of month?",
        market_name="Bitcoin — Will BTC exceed $100,000?",
        initial_price=0.50,
        volume=500_000.0,
        expiry_date="2026-12-31",
        no_market_id="tok_btc_no",
    )


@pytest.fixture
def noop_log():
    return lambda *a, **kw: None


@pytest.fixture
def wired_wallet_context(tmp_path, dry_run_config, bridge):
    """WalletContext with all four runtime components wired up.

    For tests that need mocked components (e.g. a MagicMock() executor),
    build your own context with the mocks attached directly instead of
    using this fixture.
    """
    from trading.executor import TradeExecutor
    from polymarket import PolymarketScannerHunter
    from trading.risk_manager import PortfolioManager
    from trading.budget_manager import BudgetManager
    from trading.decision_pipeline import sync_live_account_state

    ctx = WalletContext(
        wallet_id="test",
        config=dry_run_config,
        bridge=bridge,
        db_path=str(tmp_path / "test.db"),
    )
    ctx.executor = TradeExecutor(config=ctx.config)
    ctx.scanner = PolymarketScannerHunter(
        bridge=ctx.bridge,
        executor=ctx.executor,
        config=ctx.config,
    )
    ctx.portfolio_manager = PortfolioManager(
        bridge=ctx.bridge,
        executor=ctx.executor,
        config=ctx.config,
        hunter=ctx.scanner,
        db_path=ctx.db_path,
    )
    # Sync live balance before building BudgetManager so its initial_balance
    # reflects ctx.bridge.current_balance rather than freezing at 0.0 — same
    # dependency order as ui/dashboard.py and core/wallet_manager.py.
    sync_live_account_state(ctx.bridge, ctx.executor, ctx.portfolio_manager, lambda *a, **kw: None)
    ctx.budget_manager = BudgetManager(
        bridge=ctx.bridge,
        config=ctx.config,
        initial_balance=float(ctx.bridge.current_balance),
    )
    return ctx

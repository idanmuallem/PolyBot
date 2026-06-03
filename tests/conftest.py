import sys
from unittest.mock import MagicMock

# Mock streamlit before any project import.
_st_mock = MagicMock()
_st_mock.cache_resource = lambda fn: fn  # no-op decorator
sys.modules["streamlit"] = _st_mock

# Mock eth_utils (Ethereum dependency not installable in this test env).
_eth_mock = MagicMock()
_eth_mock.to_checksum_address = lambda addr: str(addr)
sys.modules["eth_utils"] = _eth_mock

import pytest

from core.bridge import DataBridge
from core.models import MarketData
from core.trading_config import TradingConfig


@pytest.fixture
def bridge():
    return DataBridge()


@pytest.fixture
def dry_run_config():
    return TradingConfig(
        dry_run=True,
        min_ev=0.30,
        bankroll_usd=1000.0,
        daily_limit_usd=15.0,
        max_bet_size_usd=3.0,
        min_trading_balance=5.0,
        max_daily_trades=10,
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

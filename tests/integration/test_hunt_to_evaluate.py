"""Integration: Hunter → Brain pipeline with mocked external APIs."""
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

from freezegun import freeze_time

from brains.crypto import HybridCryptoBrain
from core.models import MarketData
from core.trading_config import TradingConfig
from hunters.clients.ccxt_client import CCXTDataClient
from hunters.crypto import CryptoHunter


def _gamma_event(title="Bitcoin Price Market", price=0.55, volume=500_000,
                 question="Will BTC exceed $95000?", market_id="tok1", no_id="tok2"):
    return {
        "title": title,
        "slug": "bitcoin-price-market",
        "markets": [
            {
                "closed": False,
                "clobTokenIds": json.dumps([market_id, no_id]),
                "lastTradePrice": price,
                "volume": volume,
                "question": question,
                "groupItemTitle": "Yes",
            }
        ],
    }


def _mock_resp(events, status=200):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = events
    return r


# ── Hunt returns valid MarketData ─────────────────────────────────────────────

def test_hunt_returns_marketdata_with_valid_fields():
    hunter = CryptoHunter(symbols=["BTCUSDT"])
    event = _gamma_event(question="Will BTC exceed $95000?", price=0.55, volume=500_000)

    with patch.object(CCXTDataClient, "get_latest_value", return_value=95_000.0), \
         patch("hunters.base.crequests.get") as mock_gamma:
        mock_gamma.side_effect = [_mock_resp([event]), _mock_resp([])]

        result = hunter.hunt(skip_ids=[])

    assert result is not None
    assert result.market_id == "tok1"
    assert result.strike_price == 95_000.0
    assert result.initial_price == 0.55
    assert result.volume == 500_000.0
    assert result.asset_type == "Crypto::BTCUSDT"


# ── Hunt result feeds brain evaluate ─────────────────────────────────────────

@freeze_time("2026-06-02T00:00:00+00:00")
def test_hunt_result_feeds_brain_evaluate():
    hunter = CryptoHunter(symbols=["BTCUSDT"])
    expiry = (datetime(2026, 6, 2, tzinfo=timezone.utc) + timedelta(days=10)).isoformat()
    event = _gamma_event(
        question="Will BTC exceed $95000?", price=0.55, volume=500_000,
    )
    # Inject expiry into the market after hunt
    with patch.object(CCXTDataClient, "get_latest_value", return_value=97_000.0), \
         patch("hunters.base.crequests.get") as mock_gamma:
        mock_gamma.side_effect = [_mock_resp([event]), _mock_resp([])]
        market = hunter.hunt(skip_ids=[])

    assert market is not None
    # Override expiry to a known date for deterministic brain output
    market.expiry_date = expiry

    brain = HybridCryptoBrain()
    signal = brain.evaluate(market, 97_000.0, min_ev=0.01)

    assert 0.0 <= signal.fair_value <= 1.0
    assert isinstance(signal.expected_value, float)
    assert isinstance(signal.kelly_size, float)


# ── Asset mismatch guard ──────────────────────────────────────────────────────

def test_asset_mismatch_guard_blocks_cross_contamination():
    # BTC-tagged hunter should reject an ETH market title
    hunter = CryptoHunter(symbols=["BTCUSDT"])
    event = _gamma_event(
        title="Ethereum Price Market",
        question="Will ETH exceed $5000?",
        price=0.50, volume=500_000,
    )
    with patch.object(CCXTDataClient, "get_latest_value", return_value=95_000.0), \
         patch("hunters.base.crequests.get") as mock_gamma:
        # Both alias scans ("Bitcoin" and "BTC") get the eth event
        mock_gamma.side_effect = [
            _mock_resp([event]), _mock_resp([]),  # "Bitcoin" alias pages
            _mock_resp([event]), _mock_resp([]),  # "BTC" alias pages
        ]
        result = hunter.hunt(skip_ids=[])

    assert result is None


# ── Scanner TTE filters ───────────────────────────────────────────────────────

@freeze_time("2026-06-02T00:00:00+00:00")
def test_scanner_rejects_market_with_tte_too_short():
    from polymarket import PolymarketScannerHunter
    from core.bridge import DataBridge

    bridge = DataBridge()
    bridge.current_balance = 100.0
    config = TradingConfig(dry_run=True, min_tte_minutes=60, max_tte_days=180)
    executor = MagicMock()

    # Market expires in 10 minutes
    expiry = (datetime(2026, 6, 2, tzinfo=timezone.utc) + timedelta(minutes=10)).isoformat()
    short_market = MarketData(
        market_id="tok_short", asset_type="Crypto::BTCUSDT",
        strike_price=95_000.0, question="Will BTC exceed $95000?",
        market_name="BTC Short", initial_price=0.50, volume=500_000.0,
        expiry_date=expiry,
    )

    mock_hunter = MagicMock()
    mock_hunter.hunt.return_value = short_market

    scanner = PolymarketScannerHunter(
        bridge=bridge, executor=executor, config=config, hunters=[mock_hunter]
    )
    log_calls = []
    market, hunter = scanner.get_active_markets(lambda level, *a, **kw: log_calls.append(level))

    assert market is None
    assert "FILTERED" in log_calls


@freeze_time("2026-06-02T00:00:00+00:00")
def test_scanner_rejects_market_with_tte_too_long():
    from polymarket import PolymarketScannerHunter
    from core.bridge import DataBridge

    bridge = DataBridge()
    config = TradingConfig(dry_run=True, min_tte_minutes=60, max_tte_days=180)
    executor = MagicMock()

    # Market expires in 365 days (> max_tte_days=180)
    expiry = (datetime(2026, 6, 2, tzinfo=timezone.utc) + timedelta(days=365)).isoformat()
    far_market = MarketData(
        market_id="tok_far", asset_type="Crypto::BTCUSDT",
        strike_price=95_000.0, question="Will BTC exceed $95000?",
        market_name="BTC Far", initial_price=0.50, volume=500_000.0,
        expiry_date=expiry,
    )

    mock_hunter = MagicMock()
    mock_hunter.hunt.return_value = far_market

    scanner = PolymarketScannerHunter(
        bridge=bridge, executor=executor, config=config, hunters=[mock_hunter]
    )
    log_calls = []
    market, hunter = scanner.get_active_markets(lambda level, *a, **kw: log_calls.append(level))

    assert market is None
    assert "FILTERED" in log_calls

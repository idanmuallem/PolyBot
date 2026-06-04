import json
import time
from unittest.mock import MagicMock, patch

import pytest

from core.models import MarketData
from hunters.parsers import extract_crypto_strike


# ── Minimal concrete hunter for testing _scan_polymarket ─────────────────────

from hunters.base import BasePolymarketHunter


class _TestHunter(BasePolymarketHunter):
    """Minimal subclass that exposes _scan_polymarket for direct testing."""

    def hunt(self, skip_ids=None, add_cooldown_func=None):
        return self._scan_polymarket(
            95_000.0, "bitcoin", skip_ids=skip_ids or [],
            add_cooldown_func=add_cooldown_func,
        )

    def get_anchor_value(self):
        return 95_000.0

    def get_live_truth(self, market):
        return 95_000.0

    def get_topic_type(self):
        return "Crypto"

    def extract_strike(self, text, anchor):
        return extract_crypto_strike(text, anchor)

    def get_search_aliases(self):
        return ["bitcoin", "btc"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _event(title="Bitcoin Price Market", price=0.55, volume=500_000, question="Will BTC exceed $95000?",
           market_id="tok1", no_id="tok2", closed=False):
    return {
        "title": title,
        "slug": "bitcoin-price-market",
        "markets": [
            {
                "closed": closed,
                "clobTokenIds": json.dumps([market_id, no_id]),
                "lastTradePrice": price,
                "volume": volume,
                "question": question,
                "groupItemTitle": "Yes",
            }
        ],
    }


def _resp(events, status=200):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = events
    return r


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_scan_returns_market_on_valid_event():
    hunter = _TestHunter()
    with patch("hunters.base.crequests.get", side_effect=[_resp([_event()]), _resp([])]):
        result = hunter.hunt()
    assert result is not None
    assert result.market_id == "tok1"
    assert result.strike_price == 95_000.0


def test_scan_skips_cooldown_ids():
    hunter = _TestHunter()
    with patch("hunters.base.crequests.get", side_effect=[_resp([_event(market_id="tok1")]), _resp([])]):
        result = hunter.hunt(skip_ids=["tok1"])
    assert result is None


def test_scan_rejects_price_below_floor():
    hunter = _TestHunter()
    cooldown_calls = []
    with patch("hunters.base.crequests.get", side_effect=[_resp([_event(price=0.20)]), _resp([])]):
        result = hunter.hunt(add_cooldown_func=lambda mid: cooldown_calls.append(mid))
    assert result is None
    assert "tok1" in cooldown_calls


def test_scan_rejects_price_above_ceiling():
    hunter = _TestHunter()
    cooldown_calls = []
    with patch("hunters.base.crequests.get", side_effect=[_resp([_event(price=0.90)]), _resp([])]):
        result = hunter.hunt(add_cooldown_func=lambda mid: cooldown_calls.append(mid))
    assert result is None
    assert "tok1" in cooldown_calls


def test_scan_rejects_no_strike_extracted():
    hunter = _TestHunter()
    cooldown_calls = []
    # No numbers in question → extract_crypto_strike returns None
    with patch("hunters.base.crequests.get", side_effect=[
        _resp([_event(question="Will Bitcoin hit a new all-time high?")]), _resp([])
    ]):
        result = hunter.hunt(add_cooldown_func=lambda mid: cooldown_calls.append(mid))
    assert result is None
    assert "tok1" in cooldown_calls


def test_scan_rejects_volume_below_minimum():
    hunter = _TestHunter()
    with patch("hunters.base.crequests.get", side_effect=[_resp([_event(volume=1_000)]), _resp([])]):
        result = hunter.hunt()
    assert result is None


def test_scan_picks_highest_volume_market():
    hunter = _TestHunter()
    event_low = _event(volume=100_000, market_id="tokLow", question="Will BTC exceed $95000?")
    event_high = _event(volume=300_000, market_id="tokHigh", question="Will BTC exceed $95000?")
    with patch("hunters.base.crequests.get", side_effect=[_resp([event_low, event_high]), _resp([])]):
        result = hunter.hunt()
    assert result is not None
    assert result.volume == 300_000.0


def test_scan_stops_on_empty_page():
    hunter = _TestHunter()
    mock_get = MagicMock(side_effect=[_resp([_event()]), _resp([])])
    with patch("hunters.base.crequests.get", mock_get):
        hunter.hunt()
    assert mock_get.call_count == 2  # stopped after the empty page


def test_required_keywords_filter():
    hunter = _TestHunter()
    event_match = {
        "title": "Bitcoin price prediction",
        "slug": "bitcoin-price",
        "markets": [_event()["markets"][0]],
    }
    event_nomatch = {
        "title": "Crypto general discussion",
        "slug": "crypto-general",
        "markets": [_event(market_id="tok_other", question="Will BTC exceed $95000?")["markets"][0]],
    }
    with patch("hunters.base.crequests.get", side_effect=[
        _resp([event_match, event_nomatch]), _resp([])
    ]):
        result = hunter._scan_polymarket(
            95_000.0, "bitcoin",
            required_keywords=["price"],
        )
    # Only the event with "price" in title matches
    assert result is not None
    assert result.market_id == "tok1"


def test_cooldown_cache_expires_after_600_seconds():
    from polymarket import PolymarketScannerHunter
    from core.trading_config import TradingConfig
    from core.bridge import DataBridge

    bridge = DataBridge()
    config = TradingConfig(dry_run=True)
    executor = MagicMock()

    scanner = PolymarketScannerHunter(bridge=bridge, executor=executor, config=config)
    scanner.seen_markets["tok_old"] = time.time() - 601  # already expired

    active = scanner._get_active_seen_ids()
    assert "tok_old" not in active


def test_add_to_cooldown():
    from polymarket import PolymarketScannerHunter
    from core.trading_config import TradingConfig
    from core.bridge import DataBridge

    bridge = DataBridge()
    config = TradingConfig(dry_run=True)
    executor = MagicMock()

    scanner = PolymarketScannerHunter(bridge=bridge, executor=executor, config=config)
    scanner.add_to_cooldown("new_tok")
    assert "new_tok" in scanner.seen_markets

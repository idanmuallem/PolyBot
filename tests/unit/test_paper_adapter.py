"""Unit tests for PaperAdapter — the paper trading bridge."""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from core.models import Position


# Skip all tests if polymarket-paper-trader is not installed
pm_trader = pytest.importorskip("pm_trader")

from trading.paper_adapter import PaperAdapter, PAPER_ENGINE_AVAILABLE


class TestPaperAdapterInit:
    def test_creates_engine_with_default_balance(self, tmp_path):
        adapter = PaperAdapter(data_dir=str(tmp_path), initial_balance=500.0)
        assert adapter.is_ready
        balance = adapter.get_cash_balance()
        assert balance == pytest.approx(500.0)

    def test_reuses_existing_account(self, tmp_path):
        adapter1 = PaperAdapter(data_dir=str(tmp_path), initial_balance=500.0)
        assert adapter1.is_ready
        # Second adapter should find the existing account
        adapter2 = PaperAdapter(data_dir=str(tmp_path), initial_balance=999.0)
        assert adapter2.is_ready
        # Balance should still be 500, not re-initialized to 999
        assert adapter2.get_cash_balance() == pytest.approx(500.0)

    def test_not_ready_when_engine_unavailable(self):
        with patch("trading.paper_adapter.PAPER_ENGINE_AVAILABLE", False):
            adapter = PaperAdapter.__new__(PaperAdapter)
            adapter.engine = None
            adapter._token_map = {}
            assert not adapter.is_ready


class TestPaperAdapterBuySell:
    def test_execute_buy_returns_false_without_slug(self, tmp_path):
        adapter = PaperAdapter(data_dir=str(tmp_path), initial_balance=1000.0)
        result = adapter.execute_buy(
            slug=None, condition_id=None,
            token_id="tok1", side="YES", amount_usd=10.0,
        )
        assert result is False

    def test_execute_sell_returns_false_without_mapping(self, tmp_path):
        adapter = PaperAdapter(data_dir=str(tmp_path), initial_balance=1000.0)
        result = adapter.execute_sell("unknown_token", 5.0)
        assert result is False

    def test_skip_warning_rate_limited_per_token(self, tmp_path, caplog):
        """A market perpetually missing slug/condition_id (e.g. re-evaluated
        every loop tick) must not flood the logs with one WARNING per call —
        rate-limited to once per _SKIP_WARNING_INTERVAL_SECONDS per token."""
        import logging
        adapter = PaperAdapter(data_dir=str(tmp_path), initial_balance=1000.0)

        with caplog.at_level(logging.WARNING, logger="trading.paper_adapter"):
            for _ in range(5):
                result = adapter.execute_buy(
                    slug=None, condition_id=None,
                    token_id="tok_missing", side="YES", amount_usd=10.0,
                )
                assert result is False

        skip_warnings = [r for r in caplog.records if "skipping paper fill" in r.message]
        assert len(skip_warnings) == 1

    def test_skip_warning_not_rate_limited_across_different_tokens(self, tmp_path, caplog):
        import logging
        adapter = PaperAdapter(data_dir=str(tmp_path), initial_balance=1000.0)

        with caplog.at_level(logging.WARNING, logger="trading.paper_adapter"):
            adapter.execute_buy(slug=None, condition_id=None, token_id="tokA", side="YES", amount_usd=10.0)
            adapter.execute_buy(slug=None, condition_id=None, token_id="tokB", side="YES", amount_usd=10.0)

        skip_warnings = [r for r in caplog.records if "skipping paper fill" in r.message]
        assert len(skip_warnings) == 2

    def test_token_map_populated_on_buy(self, tmp_path):
        adapter = PaperAdapter(data_dir=str(tmp_path), initial_balance=10000.0)
        # Mock the engine.buy to avoid real API calls
        mock_trade = MagicMock()
        mock_trade.trade.shares = 10.0
        mock_trade.trade.avg_price = 0.50
        mock_trade.trade.fee = 0.01
        mock_trade.trade.slippage = 5.0
        adapter.engine.buy = MagicMock(return_value=mock_trade)

        result = adapter.execute_buy(
            slug="test-market", condition_id=None,
            token_id="tok_yes", side="YES", amount_usd=50.0,
            no_token_id="tok_no",
        )
        assert result is True
        assert "tok_yes" in adapter._token_map
        assert adapter._token_map["tok_yes"] == ("test-market", "yes")

    def test_no_side_maps_no_token(self, tmp_path):
        adapter = PaperAdapter(data_dir=str(tmp_path), initial_balance=10000.0)
        mock_trade = MagicMock()
        mock_trade.trade.shares = 10.0
        mock_trade.trade.avg_price = 0.50
        mock_trade.trade.fee = 0.01
        mock_trade.trade.slippage = 5.0
        adapter.engine.buy = MagicMock(return_value=mock_trade)

        result = adapter.execute_buy(
            slug="test-market", condition_id=None,
            token_id="tok_yes", side="NO", amount_usd=50.0,
            no_token_id="tok_no",
        )
        assert result is True
        # NO side should map the no_token_id
        assert "tok_no" in adapter._token_map
        assert adapter._token_map["tok_no"] == ("test-market", "no")


class TestPaperAdapterPositions:
    def test_get_positions_returns_empty_when_no_engine(self):
        adapter = PaperAdapter.__new__(PaperAdapter)
        adapter.engine = None
        adapter._token_map = {}
        assert adapter.get_positions() == []

    def test_get_positions_maps_to_polybot_position(self, tmp_path):
        adapter = PaperAdapter(data_dir=str(tmp_path), initial_balance=10000.0)
        adapter._token_map["tok1"] = ("test-market", "yes")

        adapter.engine.get_portfolio = MagicMock(return_value=[{
            "market_slug": "test-market",
            "outcome": "yes",
            "shares": 10.0,
            "avg_entry_price": 0.40,
            "live_price": 0.55,
            "current_value": 5.50,
        }])

        positions = adapter.get_positions()
        assert len(positions) == 1

        pos = positions[0]
        assert isinstance(pos, Position)
        assert pos.initial_price == pytest.approx(0.40)
        assert pos.current_price == pytest.approx(0.55)
        assert pos.shares == pytest.approx(10.0)
        assert pos.side == "YES"
        assert pos.token_id == "tok1"


class TestPaperAdapterBalance:
    def test_get_cash_balance(self, tmp_path):
        adapter = PaperAdapter(data_dir=str(tmp_path), initial_balance=750.0)
        assert adapter.get_cash_balance() == pytest.approx(750.0)

    def test_get_total_value_equals_cash_with_no_positions(self, tmp_path):
        adapter = PaperAdapter(data_dir=str(tmp_path), initial_balance=750.0)
        assert adapter.get_total_value() == pytest.approx(750.0)

    def test_fallback_balance_when_not_ready(self):
        adapter = PaperAdapter.__new__(PaperAdapter)
        adapter.engine = None
        adapter._token_map = {}
        with patch.dict(os.environ, {"PAPER_BALANCE_USD": "2000.0"}):
            assert adapter.get_cash_balance() == pytest.approx(2000.0)

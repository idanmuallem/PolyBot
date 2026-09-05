"""Unit tests for PaperAdapter — the paper trading bridge."""
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
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

    def test_place_limit_buy_tags_token_as_arbitrage(self, tmp_path):
        """place_limit_buy() is arbitrage's sole paper-fill path -- every
        token it records must end up in self._arbitrage_tokens so
        resolve_closed_markets() can identify it later."""
        adapter = PaperAdapter(data_dir=str(tmp_path), initial_balance=10000.0)
        adapter.engine.place_limit_order = MagicMock(return_value={"id": 1})

        order = adapter.place_limit_buy(
            slug="arb-market", condition_id=None,
            token_id="arb_tok", side="YES", amount_usd=25.0, limit_price=0.40,
        )

        assert order == {"id": 1}
        assert adapter._token_map.get("arb_tok") == ("arb-market", "yes")
        assert "arb_tok" in adapter._arbitrage_tokens

    def test_execute_buy_does_not_tag_token_as_arbitrage(self, tmp_path):
        """The crypto/model path's entry point (execute_buy) must never mark
        its own tokens as arbitrage."""
        adapter = PaperAdapter(data_dir=str(tmp_path), initial_balance=10000.0)
        mock_trade = MagicMock()
        mock_trade.trade.shares = 10.0
        mock_trade.trade.avg_price = 0.50
        mock_trade.trade.fee = 0.01
        mock_trade.trade.slippage = 5.0
        adapter.engine.buy = MagicMock(return_value=mock_trade)

        adapter.execute_buy(
            slug="crypto-market", condition_id=None,
            token_id="crypto_tok", side="YES", amount_usd=50.0,
        )

        assert "crypto_tok" not in adapter._arbitrage_tokens

    def test_arbitrage_tag_persists_across_reload(self, tmp_path):
        adapter = PaperAdapter(data_dir=str(tmp_path), initial_balance=10000.0)
        adapter.engine.place_limit_order = MagicMock(return_value={"id": 1})
        adapter.place_limit_buy(
            slug="arb-market", condition_id=None,
            token_id="arb_tok", side="YES", amount_usd=25.0, limit_price=0.40,
        )

        reloaded = PaperAdapter(data_dir=str(tmp_path), initial_balance=10000.0)

        assert reloaded._token_map.get("arb_tok") == ("arb-market", "yes")
        assert "arb_tok" in reloaded._arbitrage_tokens


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

    def test_get_positions_does_not_use_slug_as_token_id_when_unmapped(self, tmp_path, caplog):
        """A real, genuinely-open engine position whose _token_map entry is
        missing (e.g. lost across a restart — see execute_buy's
        _save_token_map()) must not be given the market slug as a fake
        token_id: execute_sell()'s own _token_map lookup can never match a
        slug (only real token_ids are ever stored as keys), so that
        fallback made the position look normal everywhere downstream while
        actually being permanently unsellable. The reconstructed token_id
        must be a distinct, obviously-broken sentinel instead."""
        import logging
        adapter = PaperAdapter(data_dir=str(tmp_path), initial_balance=10000.0)
        # Deliberately no _token_map entry for "orphaned-market"/"yes".

        adapter.engine.get_portfolio = MagicMock(return_value=[{
            "market_slug": "orphaned-market",
            "outcome": "yes",
            "shares": 95.83,
            "avg_entry_price": 0.01,
            "live_price": 0.01,
            "current_value": 0.96,
        }])

        with caplog.at_level(logging.WARNING, logger="trading.paper_adapter"):
            positions = adapter.get_positions()

        assert len(positions) == 1
        pos = positions[0]
        assert pos.market_id == "orphaned-market"
        assert pos.token_id != "orphaned-market"
        assert "orphaned-market" in pos.token_id  # still identifiable, just not passable as a real id
        assert pos.shares == pytest.approx(95.83)

        # A sell attempt against the reconstructed id must fail cleanly
        # (not silently "succeed"), since _token_map has no entry keyed by
        # this sentinel or by the raw slug.
        assert adapter.execute_sell(pos.token_id, pos.shares) is False

        assert any("no _token_map entry" in r.message for r in caplog.records)

    def test_execute_sell_invalidates_positions_cache(self, tmp_path):
        """get_positions() caches for _positions_cache_ttl seconds (30s).
        Without invalidating that cache on a successful sell, a caller that
        reads positions again right after (e.g. PortfolioManager.
        manage_portfolio()'s trailing _refresh_portfolio()) still sees the
        pre-sell snapshot and re-attempts to sell the same, already-closed
        position on every tick until the cache naturally expires — the
        observed "phantom repeated sell" symptom. Immediately re-querying
        positions after a successful sell must reflect the sale, not the
        stale cache."""
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

        # Prime the cache with the pre-sell snapshot.
        positions = adapter.get_positions()
        assert len(positions) == 1

        # The engine's own sell succeeds and now reports the position gone —
        # but adapter.get_positions() would still return the cached,
        # pre-sell snapshot for up to 30s if the cache isn't invalidated.
        adapter.engine.sell = MagicMock(
            return_value=SimpleNamespace(trade=SimpleNamespace(amount_usd=5.50, avg_price=0.55))
        )
        adapter.engine.get_portfolio = MagicMock(return_value=[])

        assert adapter.execute_sell("tok1", 10.0) is True

        positions_after_sell = adapter.get_positions()
        assert positions_after_sell == []


class TestResolveClosedMarketsArbitrageGate:
    """resolve_closed_markets(resolve_arbitrage=...) -- the ENABLE_ARBITRAGE
    kill switch's coverage of the periodic resolve-check, not just new-trade
    entry. A position is "arbitrage" here purely by having been recorded via
    place_limit_buy() (arbitrage's sole paper-fill path) -- see
    self._arbitrage_tokens."""

    @staticmethod
    def _fake_position(condition_id, slug, outcome):
        return SimpleNamespace(market_condition_id=condition_id, market_slug=slug, outcome=outcome)

    @staticmethod
    def _fake_resolve_result(slug, outcome, payout=1.0):
        return SimpleNamespace(position=SimpleNamespace(market_slug=slug, outcome=outcome), payout=payout)

    def test_resolve_arbitrage_false_skips_arbitrage_market_entirely(self, tmp_path):
        adapter = PaperAdapter(data_dir=str(tmp_path), initial_balance=100.0)
        adapter._token_map = {
            "arb_tok": ("arb-slug", "yes"),
            "crypto_tok": ("crypto-slug", "yes"),
        }
        adapter._arbitrage_tokens = {"arb_tok"}

        mock_engine = MagicMock()
        mock_engine.db.get_open_positions.return_value = [
            self._fake_position("cond_arb", "arb-slug", "yes"),
            self._fake_position("cond_crypto", "crypto-slug", "yes"),
        ]
        mock_engine.api.get_market.return_value = SimpleNamespace(closed=True)
        mock_engine.resolve_market.return_value = [self._fake_resolve_result("crypto-slug", "yes")]
        adapter.engine = mock_engine

        count = adapter.resolve_closed_markets(resolve_arbitrage=False)

        assert count == 1
        mock_engine.resolve_all.assert_not_called()
        # The arbitrage market's slug never even reached get_market/resolve_market --
        # confirms the arbitrage-specific branch was never entered, not just that
        # its result was discarded.
        mock_engine.api.get_market.assert_called_once_with("crypto-slug")
        mock_engine.resolve_market.assert_called_once_with("crypto-slug")

    def test_resolve_arbitrage_true_uses_engine_resolve_all(self, tmp_path):
        """Sanity check: with the flag on, the original resolve_all() path
        (which makes no arbitrage/crypto distinction) still runs unchanged."""
        adapter = PaperAdapter(data_dir=str(tmp_path), initial_balance=100.0)
        mock_engine = MagicMock()
        mock_engine.resolve_all.return_value = [self._fake_resolve_result("crypto-slug", "yes")]
        adapter.engine = mock_engine

        count = adapter.resolve_closed_markets(resolve_arbitrage=True)

        assert count == 1
        mock_engine.resolve_all.assert_called_once()

    def test_resolve_arbitrage_false_still_resolves_crypto_when_no_arbitrage_open(self, tmp_path):
        """No arbitrage positions in the book at all -- the crypto-only case
        must resolve exactly as it always has."""
        adapter = PaperAdapter(data_dir=str(tmp_path), initial_balance=100.0)
        adapter._token_map = {"crypto_tok": ("crypto-slug", "yes")}
        adapter._arbitrage_tokens = set()

        mock_engine = MagicMock()
        mock_engine.db.get_open_positions.return_value = [
            self._fake_position("cond_crypto", "crypto-slug", "yes"),
        ]
        mock_engine.api.get_market.return_value = SimpleNamespace(closed=True)
        mock_engine.resolve_market.return_value = [self._fake_resolve_result("crypto-slug", "yes")]
        adapter.engine = mock_engine

        count = adapter.resolve_closed_markets(resolve_arbitrage=False)

        assert count == 1
        mock_engine.resolve_market.assert_called_once_with("crypto-slug")


class TestResolveClosedMarketsLogging:
    """resolve_closed_markets(log_func=...) — the EXPIRED hunt_history row
    that makes a market resolution visible on the dashboard (previously
    logger.info-only, invisible outside stdout)."""

    @staticmethod
    def _fake_resolve_result(slug, outcome, payout=7.5, shares=10.0, avg_entry_price=0.4,
                              question="Will it happen?", realized_pnl=3.5):
        position = SimpleNamespace(
            market_slug=slug, outcome=outcome, shares=shares,
            avg_entry_price=avg_entry_price, market_question=question,
            realized_pnl=realized_pnl,
        )
        return SimpleNamespace(position=position, payout=payout)

    def test_logs_expired_row_per_resolved_position(self, tmp_path):
        adapter = PaperAdapter(data_dir=str(tmp_path), initial_balance=100.0)
        adapter._token_map = {"tok1": ("crypto-slug", "yes")}
        mock_engine = MagicMock()
        mock_engine.resolve_all.return_value = [
            self._fake_resolve_result("crypto-slug", "yes", payout=10.0, shares=10.0, avg_entry_price=0.4)
        ]
        adapter.engine = mock_engine

        log_func = MagicMock()
        count = adapter.resolve_closed_markets(resolve_arbitrage=True, log_func=log_func)

        assert count == 1
        log_func.assert_called_once()
        level, asset_type, token_id, payload = log_func.call_args[0]
        assert level == "EXPIRED"
        assert token_id == "tok1"
        assert payload["sold"] is True
        assert payload["shares"] == 10.0
        assert payload["price"] == pytest.approx(1.0)  # payout/shares = 10.0/10.0
        assert payload["initial_price"] == pytest.approx(0.4)
        assert payload["side"] == "YES"
        assert payload["payout"] == pytest.approx(10.0)

    def test_unmapped_token_falls_back_to_explicit_marker(self, tmp_path):
        """No _token_map entry for the resolved slug/outcome — same
        '__unmapped__' convention get_positions() uses, so a broken mapping
        is obvious in the dashboard rather than silently dropped."""
        adapter = PaperAdapter(data_dir=str(tmp_path), initial_balance=100.0)
        adapter._token_map = {}
        mock_engine = MagicMock()
        mock_engine.resolve_all.return_value = [self._fake_resolve_result("mystery-slug", "no")]
        adapter.engine = mock_engine

        log_func = MagicMock()
        adapter.resolve_closed_markets(resolve_arbitrage=True, log_func=log_func)

        _, _, token_id, _ = log_func.call_args[0]
        assert token_id == "__unmapped__:mystery-slug:no"

    def test_no_log_func_still_works(self, tmp_path):
        """log_func is optional — omitting it must not break the existing
        count-returning behavior other callers rely on."""
        adapter = PaperAdapter(data_dir=str(tmp_path), initial_balance=100.0)
        mock_engine = MagicMock()
        mock_engine.resolve_all.return_value = [self._fake_resolve_result("crypto-slug", "yes")]
        adapter.engine = mock_engine

        assert adapter.resolve_closed_markets(resolve_arbitrage=True) == 1


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

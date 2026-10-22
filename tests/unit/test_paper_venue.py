"""PaperVenue behaviour beyond the shared Venue contract
(tests/unit/test_venue_contract.py): configuration, restart persistence,
and paper-specific refusals. Real pm_trader in tmp_path, stubbed market data."""

import json
import logging
from types import SimpleNamespace

import pytest

pytest.importorskip("pm_trader")

from core.trading_config import TradingConfig
from tests.venue_harness import make_paper_venue
from trading.paper_venue import PaperVenue
from trading.venue import MarketRef

BTC = MarketRef(yes_token_id="btc-yes", no_token_id="btc-no", condition_id="0xbtc", slug="btc-100k")
NO_IDS = MarketRef(yes_token_id="tok-missing", no_token_id="tok-missing-no")


def _restart(data_dir, stub, starting_balance=1.0):
    return PaperVenue(data_dir=data_dir, starting_balance=starting_balance, market_api=stub)


class TestConfiguration:
    def test_existing_account_keeps_its_balance(self, tmp_path):
        venue, stub = make_paper_venue(tmp_path, 500.0)
        assert venue.cash() == pytest.approx(500.0)

        assert _restart(tmp_path, stub, starting_balance=999.0).cash() == pytest.approx(500.0)

    def test_reads_no_environment_variables(self, tmp_path, monkeypatch):
        env_db_dir = tmp_path / "from_env"
        monkeypatch.setenv("PAPER_BALANCE_USD", "1.0")
        monkeypatch.setenv("TRADES_DB_PATH", str(env_db_dir / "trades.db"))

        venue = PaperVenue(data_dir=tmp_path / "configured", starting_balance=750.0)

        assert venue.cash() == pytest.approx(750.0)
        assert (tmp_path / "configured").is_dir()
        assert not env_db_dir.exists()

    def test_for_wallet_uses_wallet_db_path_and_config_balance(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PAPER_BALANCE_USD", "1.0")
        ctx = SimpleNamespace(
            db_path=str(tmp_path / "wallet_a" / "trades.db"),
            config=TradingConfig(trading_mode="dry_run", paper_balance_usd=321.0),
        )

        venue = PaperVenue.for_wallet(ctx)

        assert venue.cash() == pytest.approx(321.0)
        assert (tmp_path / "wallet_a" / "paper_trading" / "paper.db").exists()


class TestMarketIdentification:
    def test_buy_without_slug_or_condition_id_is_refused(self, tmp_path):
        venue, _ = make_paper_venue(tmp_path, 1000.0)
        assert venue.buy(NO_IDS, "YES", 10.0) is False
        assert venue.place_limit_buy(NO_IDS, "YES", 10.0, limit_price=0.4) is None
        assert venue.cash() == pytest.approx(1000.0)

    def test_missing_ids_warning_is_rate_limited_per_token(self, tmp_path, caplog):
        venue, _ = make_paper_venue(tmp_path, 1000.0)
        other = MarketRef(yes_token_id="tok-other")

        with caplog.at_level(logging.WARNING, logger="trading.paper_venue"):
            for _ in range(5):
                venue.buy(NO_IDS, "YES", 10.0)
            venue.buy(other, "YES", 10.0)

        skip_warnings = [r for r in caplog.records if "skipping paper fill" in r.message]
        assert len(skip_warnings) == 2  # one per token, not one per call

    def test_buy_by_condition_id_alone(self, tmp_path):
        venue, stub = make_paper_venue(tmp_path, 1000.0)
        stub.list_market(BTC, yes_price=0.40)
        by_condition = MarketRef(yes_token_id="btc-yes", no_token_id="btc-no", condition_id="0xbtc")

        assert venue.buy(by_condition, "NO", 30.0) is True
        assert venue.sell("btc-no", 50.0) is True


class TestRestartPersistence:
    def test_position_bought_before_restart_is_sellable_after(self, tmp_path):
        venue, stub = make_paper_venue(tmp_path, 1000.0)
        stub.list_market(BTC, yes_price=0.40)
        venue.buy(BTC, "NO", 30.0)

        restarted = _restart(tmp_path, stub)

        assert [p.token_id for p in restarted.open_positions()] == ["btc-no"]
        assert restarted.sell("btc-no", 50.0) is True

    def test_arbitrage_tag_survives_restart(self, tmp_path):
        venue, stub = make_paper_venue(tmp_path, 1000.0)
        stub.list_market(BTC, yes_price=0.40)
        order = venue.place_limit_buy(BTC, "YES", 20.0, limit_price=0.40)
        venue.poll_limit_orders([order])
        stub.close_market(BTC, "YES")

        restarted = _restart(tmp_path, stub)

        assert restarted.resolve_closed_markets(resolve_arbitrage=False) == 0
        assert restarted.resolve_closed_markets(resolve_arbitrage=True) == 1

    def test_legacy_token_map_format_still_loads(self, tmp_path):
        venue, stub = make_paper_venue(tmp_path, 1000.0)
        stub.list_market(BTC, yes_price=0.40)
        venue.buy(BTC, "YES", 20.0)
        (tmp_path / "token_map.json").write_text(json.dumps({"btc-yes": ["btc-100k", "yes"]}))

        assert _restart(tmp_path, stub).sell("btc-yes", 50.0) is True

    def test_position_with_lost_token_map_is_visibly_unsellable(self, tmp_path, caplog):
        """The engine's ledger outlived token_map.json. The position must
        not masquerade under its slug (which no sell can ever match); it
        gets an obviously broken id and selling it fails honestly."""
        venue, stub = make_paper_venue(tmp_path, 1000.0)
        stub.list_market(BTC, yes_price=0.40)
        venue.buy(BTC, "YES", 20.0)
        (tmp_path / "token_map.json").unlink()

        restarted = _restart(tmp_path, stub)
        with caplog.at_level(logging.WARNING, logger="trading.paper_venue"):
            positions = restarted.open_positions()

        assert len(positions) == 1
        assert positions[0].token_id == "__unmapped__:btc-100k:yes"
        assert restarted.sell(positions[0].token_id, positions[0].shares) is False
        assert restarted.sell("btc-100k", positions[0].shares) is False
        assert any("no _token_map entry" in r.message for r in caplog.records)

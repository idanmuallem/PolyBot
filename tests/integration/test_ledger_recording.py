"""Integration: the engine records every model (crypto) entry, sale,
resolution and expiry in the Ledger exactly once (ticket #13).

Real SQLite (tmp_path) for both hunt_history and the Ledger. Entries are
driven through the pipeline's evaluate -> risk -> execute stages, as the
other pipeline integration tests do, until a tick-level seam exists (#15).
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from freezegun import freeze_time

import ui.data_manager as dm
from core.bridge import DataBridge
from core.models import MarketData, Position
from core.trading_config import TradingConfig
from core.wallet_context import WalletContext
from polymarket import PolymarketScannerHunter
from trading.budget_manager import BudgetManager
from trading.executor import TradeExecutor
from trading.fake_venue import FakeVenue
from trading.venue import MarketRef
from trading.ledger import Ledger
from trading.risk_manager import PortfolioManager


def _make_pipeline(db_path, balance=100.0, venue=None):
    from trading.decision_pipeline import SequentialTradingPipeline, sync_live_account_state

    dm.init_db(db_path)
    bridge = DataBridge()
    bridge.current_balance = balance
    config = TradingConfig(
        trading_mode="dry_run", min_ev=0.30, bankroll_usd=1000.0,
        daily_limit_usd=15.0, max_bet_size_usd=3.0,
        max_daily_trades=10, min_trading_balance=1.0,
        pricing_mode="legacy",
    )
    ctx = WalletContext(wallet_id="test_wallet", config=config, bridge=bridge, db_path=db_path)
    ctx.ledger = Ledger(db_path)

    def log_func(level, asset_type, token_id, payload):
        dm.log_event(bridge, level, asset_type, token_id, payload, db_path=db_path)

    with patch("trading.executor.TradeExecutor.get_open_positions", return_value=[]), \
         patch("trading.executor.TradeExecutor.get_balance", return_value=balance):
        ctx.executor = TradeExecutor(config=ctx.config, venue=venue or FakeVenue(cash=balance))
        ctx.scanner = PolymarketScannerHunter(bridge=bridge, executor=ctx.executor, config=config)
        ctx.portfolio_manager = PortfolioManager(
            bridge=bridge, executor=ctx.executor, config=config,
            hunter=ctx.scanner, db_path=db_path, ledger=ctx.ledger,
        )
        sync_live_account_state(bridge, ctx.executor, ctx.portfolio_manager, log_func)
        ctx.budget_manager = BudgetManager(bridge=bridge, config=config, initial_balance=balance)
        pipeline = SequentialTradingPipeline(ctx=ctx, log_func=log_func, delay=0.01)
    return pipeline, ctx


def _market(price):
    expiry = (datetime(2026, 6, 2, tzinfo=timezone.utc) + timedelta(days=10)).isoformat()
    return MarketData(
        market_id="tok1", asset_type="Crypto::BTCUSDT", strike_price=95_000.0,
        question="Will BTC exceed $95,000?", market_name="Bitcoin — BTC price market",
        initial_price=price, volume=500_000.0, expiry_date=expiry,
        no_market_id="tok1_no", condition_id="cond1", slug="btc-95k",
    )


def _enter(pipeline, price, prob):
    market = _market(price)
    if isinstance(pipeline.executor.venue, FakeVenue):
        pipeline.executor.venue.list_market(MarketRef.from_market(market), price)
    hunter = MagicMock()
    hunter.get_live_truth.return_value = 97_000.0
    with patch("brains.crypto.HybridCryptoBrain._calculate_probability", return_value=prob):
        candidate = pipeline._stage_evaluate_ev(market, hunter)
    assert candidate is not None
    approved_bet, risk_ctx = pipeline._stage_risk_and_budget(candidate)
    assert approved_bet > 0.0
    pipeline._stage_execute(candidate, approved_bet, risk_ctx)
    return candidate, approved_bet


# ── entries ──────────────────────────────────────────────────────────────────

@freeze_time("2026-06-02T00:00:00+00:00")
def test_one_model_fill_counts_once_despite_auto_trade_track_and_dry_run_rows(tmp_path):
    """Regression (double-count): one fill logs AUTO-TRADE, DRY-RUN and
    TRACK to hunt_history; trade stats must count it exactly once."""
    db = str(tmp_path / "trades.db")
    pipeline, ctx = _make_pipeline(db)

    _enter(pipeline, price=0.40, prob=0.70)

    levels = dm.get_level_counts(db)
    assert levels.get("AUTO-TRADE") == 1
    assert levels.get("DRY-RUN") == 1
    assert levels.get("TRACK") == 1
    assert dm.get_trade_stats(db)["total_trades"] == 1
    assert len(dm.fetch_transactions(db)) == 1


@freeze_time("2026-06-02T00:00:00+00:00")
def test_model_entry_is_recorded_with_its_entry_details(tmp_path):
    db = str(tmp_path / "trades.db")
    pipeline, ctx = _make_pipeline(db)

    candidate, approved_bet = _enter(pipeline, price=0.40, prob=0.70)

    holding = ctx.ledger.holding("tok1")
    assert holding is not None
    assert holding.condition_id == "cond1"
    assert holding.side == "YES"
    assert holding.entry_price == pytest.approx(0.40)
    assert holding.shares == pytest.approx(approved_bet / 0.40, abs=0.01)
    assert holding.pre_prob == pytest.approx(candidate.pre_prob)
    assert holding.post_prob == pytest.approx(candidate.post_prob)
    assert holding.asset_type == "Crypto::BTCUSDT"
    assert holding.venue == "paper"
    assert holding.strategy == "crypto"


@freeze_time("2026-06-02T00:00:00+00:00")
def test_live_entry_is_tagged_with_the_live_venue(tmp_path):
    from tests.unit.test_clob_venue import StubClobClient
    from trading.clob_venue import ClobVenue

    db = str(tmp_path / "trades.db")
    pipeline, ctx = _make_pipeline(db, venue=ClobVenue(client=StubClobClient(), proxy_address="", http_get=None))

    _enter(pipeline, price=0.40, prob=0.70)

    assert ctx.ledger.holding("tok1").venue == "live"


@freeze_time("2026-06-02T00:00:00+00:00")
def test_no_side_entry_is_recorded_under_the_no_token(tmp_path):
    db = str(tmp_path / "trades.db")
    pipeline, ctx = _make_pipeline(db)

    _enter(pipeline, price=0.60, prob=0.10)

    holding = ctx.ledger.holding_for("cond1", "NO")
    assert holding is not None
    assert holding.token_id == "tok1_no"
    assert holding.entry_price == pytest.approx(0.40)
    assert ctx.ledger.holding("tok1") is None


# ── paper resolution ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_paper_resolution_is_recorded_once_by_the_engine_loop(tmp_path):
    import asyncio
    from unittest.mock import AsyncMock

    from trading.fake_venue import FakeVenue
    from trading.venue import MarketRef

    btc = MarketRef(yes_token_id="tok1", no_token_id="tok1_no", condition_id="cond1", slug="btc-95k")
    venue = FakeVenue(cash=100.0)
    venue.list_market(btc, yes_price=0.40)
    venue.buy(btc, "YES", 4.0)  # 10 shares
    venue.close_market(btc, "YES")

    db = str(tmp_path / "trades.db")
    pipeline, ctx = _make_pipeline(db, venue=venue)
    ctx.ledger.record_entry(token_id="tok1", condition_id="cond1", side="YES", shares=10.0,
                            price=0.40, venue="paper", strategy="crypto")
    pipeline._stage_hunt = AsyncMock(return_value=None)
    pipeline._stage_strategy_scan = AsyncMock(return_value=None)
    pipeline.portfolio_manager.manage_portfolio = MagicMock()

    try:
        await asyncio.wait_for(pipeline.run_forever(), timeout=0.3)
    except asyncio.TimeoutError:
        pass

    assert venue.open_positions() == []
    assert ctx.ledger.holding("tok1") is None
    assert ctx.ledger.realized_pnl() == pytest.approx(6.0)
    assert [t["kind"] for t in ctx.ledger.transactions()] == ["RESOLVED", "BUY"]


# ── sales / expiry ───────────────────────────────────────────────────────────

def _held(token_id="tok1", side="YES", shares=10.0, initial=0.40, current=0.80):
    return Position(
        market_id="cond1", token_id=token_id, initial_price=initial, current_price=current,
        shares=shares, value=shares * current, pnl_ratio=(current - initial) / initial, side=side,
    )


def _portfolio(tmp_path, positions, dry_run=True):
    db = str(tmp_path / "trades.db")
    dm.init_db(db)
    ledger = Ledger(db)
    config = TradingConfig(trading_mode="dry_run", take_profit_pct=0.5, stop_loss_pct=-0.5)
    executor = MagicMock()
    executor.dry_run = dry_run
    executor.get_open_positions.return_value = positions
    executor.get_balance.return_value = 100.0
    bridge = DataBridge()
    bridge.current_balance = 100.0
    pm = PortfolioManager(bridge=bridge, executor=executor, config=config, db_path=db, ledger=ledger)
    return pm, executor, ledger, db


def test_take_profit_sale_is_recorded_once(tmp_path):
    pm, executor, ledger, db = _portfolio(tmp_path, [_held(current=0.80)])
    ledger.record_entry(token_id="tok1", condition_id="cond1", side="YES", shares=10.0,
                        price=0.40, venue="paper", strategy="crypto")
    executor.sell_position.return_value = True

    pm.manage_portfolio(lambda *a: None)

    assert ledger.holding("tok1") is None
    sale = ledger.transactions()[0]
    assert (sale["kind"], sale["reason"]) == ("SELL", "TAKE-PROFIT")
    assert sale["price"] == pytest.approx(0.80)
    assert ledger.realized_pnl() == pytest.approx(4.0)
    assert len(ledger.transactions()) == 2


def test_sale_of_arbitrage_leg_keeps_its_ledger_strategy_tag(tmp_path):
    """The strategy tag on a sale comes from the Ledger holding, not from
    hunt_history payloads: an arbitrage leg with no asset_type on record is
    still sold as arbitrage, in its basket."""
    pm, executor, ledger, db = _portfolio(tmp_path, [_held(current=0.80)])
    ledger.record_entry(token_id="tok1", condition_id="cond1", side="YES", shares=10.0,
                        price=0.40, venue="paper", strategy="arbitrage", group_id="g1")
    executor.sell_position.return_value = True

    pm.manage_portfolio(lambda *a: None)

    sale = ledger.transactions()[0]
    assert (sale["kind"], sale["strategy"], sale["group_id"]) == ("SELL", "arbitrage", "g1")


def test_failed_sale_is_not_recorded(tmp_path):
    pm, executor, ledger, db = _portfolio(tmp_path, [_held(current=0.80)])
    ledger.record_entry(token_id="tok1", condition_id="cond1", side="YES", shares=10.0,
                        price=0.40, venue="paper", strategy="crypto")
    executor.sell_position.return_value = False

    pm.manage_portfolio(lambda *a: None)

    assert ledger.holding("tok1") is not None
    assert len(ledger.transactions()) == 1


def test_free_up_capital_sale_is_recorded(tmp_path):
    pm, executor, ledger, db = _portfolio(tmp_path, [_held(current=0.50)])
    ledger.record_entry(token_id="tok1", condition_id="cond1", side="YES", shares=10.0,
                        price=0.40, venue="paper", strategy="crypto")
    executor.sell_position.return_value = True

    pm.free_up_capital(1_000.0, lambda *a: None)

    sale = ledger.transactions()[0]
    assert (sale["kind"], sale["reason"]) == ("SELL", "OPPORTUNITY-SWAP")
    assert ledger.holding("tok1") is None


def test_portfolio_cull_sale_is_recorded(tmp_path):
    pm, executor, ledger, db = _portfolio(tmp_path, [_held(current=0.50)])
    ledger.record_entry(token_id="tok1", condition_id="cond1", side="YES", shares=10.0,
                        price=0.40, venue="paper", strategy="crypto",
                        post_prob=0.50)  # live_ev ~ 0 at 0.50: weak
    executor.sell_position.return_value = True

    pm.optimize_for_candidate(new_candidate_ev=1.0, log_func=lambda *a: None)

    sale = ledger.transactions()[0]
    assert (sale["kind"], sale["reason"]) == ("SELL", "PORTFOLIO-CULL")
    assert ledger.holding("tok1") is None


def test_live_position_given_up_after_max_attempts_is_recorded_as_one_expiry(tmp_path):
    pm, executor, ledger, db = _portfolio(tmp_path, [_held(current=0.01)], dry_run=False)
    ledger.record_entry(token_id="tok1", condition_id="cond1", side="YES", shares=10.0,
                        price=0.40, venue="live", strategy="crypto")
    executor.sell_position.return_value = False

    for _ in range(pm.MAX_EXIT_ATTEMPTS + 2):
        pm.manage_portfolio(lambda *a: None)

    expiries = [t for t in ledger.transactions() if t["kind"] == "EXPIRED"]
    assert len(expiries) == 1
    assert expiries[0]["confirmed"] is False
    assert ledger.holding("tok1") is None

"""Integration: the Account's budget (spend limits, one trade counter, the
daily reset) applies identically to model and arbitrage entries, driven one
engine tick at a time (tests/tick_harness.py) on a FakeVenue with a real
Ledger. The rules themselves are covered at the Account seam in
tests/unit/test_account_budget.py."""
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from core.models import MarketData
from core.trading_config import TradingConfig
from tests.tick_harness import FixedMarketScanner, FixedSignals, make_tick_pipeline
from trading.fake_venue import FakeVenue
from trading.strategies.base import StrategySignal
from trading.venue import MarketRef

GROUP = "event_sum:evt1"


def _config(**overrides):
    values = dict(
        trading_mode="dry_run", min_ev=0.30, bankroll_usd=1000.0,
        daily_limit_usd=15.0, crypto_daily_limit_usd=15.0, arbitrage_daily_limit_usd=15.0,
        max_bet_size_usd=3.0, max_daily_trades=10, min_trading_balance=1.0,
        loop_delay_seconds=0.01, pricing_mode="legacy", enable_arbitrage=False,
    )
    values.update(overrides)
    config = TradingConfig(**values)
    config.arbitrage_order_timeout_seconds = 0.05
    return config


def _btc_market(token="tok1", price=0.40):
    expiry = (datetime.now(timezone.utc) + timedelta(days=10)).isoformat()
    return MarketData(
        market_id=token, asset_type="Crypto::BTCUSDT", strike_price=95_000.0,
        question=f"Will BTC exceed $95,000? ({token})", market_name=f"BTC market {token}",
        initial_price=price, volume=500_000.0, expiry_date=expiry, no_market_id=f"{token}_no",
    )


def _leg(token_id, price=0.40):
    market = MarketData(
        market_id=token_id, asset_type="Arbitrage::EventSum", strike_price=0.0,
        question=f"Outcome {token_id}?", market_name=f"Test Event - {token_id}",
        initial_price=price, volume=100_000.0, no_market_id=f"{token_id}_no",
        condition_id=f"cond_{token_id}", slug=f"slug-{token_id}",
    )
    return StrategySignal(
        market=market, side="YES", price=price, bet_amount_usd=1.0,
        strategy_type="arbitrage", reasoning="test", group_id=GROUP, edge=0.10,
    )


def _pipeline(tmp_path, config, markets=(), legs=(), clock=None):
    venue = FakeVenue(cash=1000.0)
    for market in markets:
        venue.list_market(MarketRef.from_market(market), market.initial_price)
    for leg in legs:
        venue.list_market(MarketRef.from_market(leg.market), leg.price)
    scanner = FixedMarketScanner(markets[0], live_truth=97_000.0) if markets else None
    pipeline, ctx, rows = make_tick_pipeline(
        tmp_path, paper=venue, config=config, scanner=scanner, clock=clock,
    )
    if legs:
        pipeline.strategies = [FixedSignals(list(legs))]
    return pipeline, ctx, rows, venue


async def _tick(pipeline):
    # The brain sees a 0.70 fair value for a market priced 0.40: EV clears
    # min_ev and Kelly sizes far past the $3 max-bet ceiling.
    with patch("brains.crypto.HybridCryptoBrain._calculate_probability", return_value=0.70):
        await pipeline.tick()


def _rejections(rows):
    return [payload.get("reason") for level, _, payload in rows
            if level == "REJECTED" and isinstance(payload, dict)]


@pytest.mark.asyncio
async def test_a_model_fill_is_spent_against_the_account(tmp_path):
    pipeline, ctx, rows, venue = _pipeline(tmp_path, _config(), markets=[_btc_market()])

    await _tick(pipeline)

    assert "TRACK" in [level for level, _, _ in rows]
    assert pipeline.account.snapshot().spent_today == pytest.approx(3.0)   # capped at max bet
    assert ctx.bridge.spent_today == pytest.approx(3.0)
    assert ctx.bridge.current_balance == pytest.approx(venue.cash())


@pytest.mark.asyncio
async def test_arbitrage_fills_are_spent_and_refresh_cash(tmp_path):
    pipeline, ctx, rows, venue = _pipeline(
        tmp_path, _config(enable_arbitrage=True), legs=[_leg("tokA"), _leg("tokB")],
    )

    await _tick(pipeline)

    assert pipeline.account.snapshot().spent_today == pytest.approx(2.0)
    assert ctx.bridge.spent_today == pytest.approx(2.0)
    assert venue.cash() == pytest.approx(998.0)
    assert pipeline.account.cash == pytest.approx(998.0)
    assert ctx.bridge.current_balance == pytest.approx(998.0)


@pytest.mark.asyncio
async def test_arbitrage_and_model_trades_share_one_daily_trade_counter(tmp_path):
    """Two arbitrage legs fill; with max_daily_trades=2 the model entry in
    the same tick is refused before any order is placed."""
    pipeline, ctx, rows, venue = _pipeline(
        tmp_path, _config(enable_arbitrage=True, max_daily_trades=2),
        markets=[_btc_market()], legs=[_leg("tokA"), _leg("tokB")],
    )

    await _tick(pipeline)

    assert "global_daily_trade_limit_would_be_exceeded" in _rejections(rows)
    assert "TRACK" not in [level for level, _, _ in rows]
    assert {h.token_id for h in ctx.ledger.open_positions()} == {"tokA", "tokB"}


@pytest.mark.asyncio
async def test_model_trades_refused_at_the_strategy_trade_count(tmp_path):
    pipeline, ctx, rows, venue = _pipeline(
        tmp_path, _config(crypto_max_daily_trades=1), markets=[_btc_market("tok1")],
    )
    await _tick(pipeline)
    pipeline.hunter.market = _btc_market("tok2")
    venue.list_market(MarketRef.from_market(pipeline.hunter.market), 0.40)

    await _tick(pipeline)

    assert "daily_trade_limit_would_be_exceeded" in _rejections(rows)
    assert {h.token_id for h in ctx.ledger.open_positions()} == {"tok1"}


@pytest.mark.asyncio
async def test_the_first_tick_of_a_new_utc_day_resets_the_budget(tmp_path):
    now = {"t": datetime(2026, 10, 21, 23, 30, tzinfo=timezone.utc)}
    pipeline, ctx, rows, venue = _pipeline(
        tmp_path, _config(daily_limit_usd=3.0, crypto_daily_limit_usd=3.0),
        markets=[_btc_market("tok1")], clock=lambda: now["t"],
    )
    await _tick(pipeline)
    pipeline.hunter.market = _btc_market("tok2")
    venue.list_market(MarketRef.from_market(pipeline.hunter.market), 0.40)

    await _tick(pipeline)                       # same day: budget spent
    assert "daily_limit_reached" in _rejections(rows)

    now["t"] = datetime(2026, 10, 22, 0, 5, tzinfo=timezone.utc)
    await _tick(pipeline)                       # new day: budget back

    assert {h.token_id for h in ctx.ledger.open_positions()} == {"tok1", "tok2"}
    assert ctx.bridge.spent_today == pytest.approx(3.0)

"""Integration: strategy signals flow through the same executor as
model-driven trades, tagged with strategy_type for separate tracking."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.bridge import DataBridge
from core.trading_config import TradingConfig
from core.wallet_context import WalletContext
from polymarket import PolymarketClient, PolymarketScannerHunter
from trading.budget_manager import BudgetManager
from trading.executor import TradeExecutor
from trading.risk_manager import PortfolioManager


def _underpriced_event(event_id="evt1", n=3, price=0.30, expiry_days=None, title="Test Multi-Outcome Event"):
    """expiry_days: None (no expiry field), a single int (applied to every
    leg), or a list of per-leg day offsets (len must equal n)."""
    if expiry_days is not None and not isinstance(expiry_days, list):
        expiry_days = [expiry_days] * n

    def _end_date_iso(i):
        if expiry_days is None:
            return None
        return (datetime.now(timezone.utc) + timedelta(days=expiry_days[i])).isoformat()

    markets = []
    for i in range(n):
        market = {
            "closed": False,
            "clobTokenIds": [f"{event_id}_tok{i}", f"{event_id}_tok{i}_no"],
            "lastTradePrice": price,
            "question": f"Outcome {i}?",
            "groupItemTitle": f"Outcome {i}",
            "volume": 100_000.0,
        }
        end_date_iso = _end_date_iso(i)
        if end_date_iso is not None:
            market["endDateIso"] = end_date_iso
        markets.append(market)

    return {"id": event_id, "title": title, "markets": markets}


def _make_pipeline(balance=100.0, config=None):
    from trading.decision_pipeline import SequentialTradingPipeline, sync_live_account_state

    bridge = DataBridge()
    bridge.current_balance = balance
    bridge.current_portfolio = []
    bridge.open_position_value = 0.0
    bridge.open_positions_value = 0.0

    if config is None:
        config = TradingConfig(
            dry_run=True, min_ev=0.30, bankroll_usd=1000.0,
            daily_limit_usd=15.0, max_bet_size_usd=3.0,
            max_daily_trades=10, min_trading_balance=1.0,
        )

    ctx = WalletContext(wallet_id="test_wallet", config=config, bridge=bridge, db_path="test_wallet_trades.db")
    log_calls = []
    log_func = lambda level, asset_type, token_id, payload: log_calls.append(
        {"level": level, "asset_type": asset_type, "token_id": token_id, "payload": payload}
    )

    with patch("trading.executor.TradeExecutor.get_open_positions", return_value=[]), \
         patch("trading.executor.TradeExecutor.get_balance", return_value=balance):
        ctx.executor = TradeExecutor(config=ctx.config)
        ctx.scanner = PolymarketScannerHunter(bridge=ctx.bridge, executor=ctx.executor, config=ctx.config)
        ctx.portfolio_manager = PortfolioManager(
            bridge=ctx.bridge, executor=ctx.executor, config=ctx.config,
            hunter=ctx.scanner, db_path=ctx.db_path,
        )
        sync_live_account_state(ctx.bridge, ctx.executor, ctx.portfolio_manager, log_func)
        ctx.budget_manager = BudgetManager(
            bridge=ctx.bridge, config=ctx.config, initial_balance=float(ctx.bridge.current_balance),
        )

        pipeline = SequentialTradingPipeline(
            ctx=ctx,
            log_func=log_func,
            delay=0.01,
        )

    return pipeline, bridge, log_calls


# ── Signals execute through the executor, tagged strategy_type ───────────────

@pytest.mark.asyncio
async def test_underpriced_event_executes_every_leg_dry_run():
    pipeline, bridge, log_calls = _make_pipeline(balance=100.0)

    with patch.object(PolymarketClient, "get_multi_outcome_events", return_value=[_underpriced_event(n=3)]):
        await pipeline._stage_strategy_scan()

    dry_run_entries = [c for c in log_calls if c["level"] == "DRY-RUN"]
    leg_entries = [c for c in log_calls if c["level"] == "STRATEGY-LEG"]
    group_entries = [c for c in log_calls if c["level"] == "STRATEGY-GROUP"]

    assert len(dry_run_entries) == 3
    assert len(leg_entries) == 3
    assert len(group_entries) == 1
    assert group_entries[0]["payload"]["executed_legs"] == 3


@pytest.mark.asyncio
async def test_all_strategy_logs_tagged_arbitrage():
    pipeline, bridge, log_calls = _make_pipeline(balance=100.0)

    with patch.object(PolymarketClient, "get_multi_outcome_events", return_value=[_underpriced_event(n=2)]):
        await pipeline._stage_strategy_scan()

    strategy_logs = [c for c in log_calls if c["level"] in ("DRY-RUN", "STRATEGY-LEG", "STRATEGY-GROUP")]
    assert strategy_logs, "expected at least one strategy-tagged log entry"
    for entry in strategy_logs:
        assert entry["payload"].get("strategy_type") == "arbitrage", entry


@pytest.mark.asyncio
async def test_budget_manager_records_every_leg():
    pipeline, bridge, log_calls = _make_pipeline(balance=100.0)

    with patch.object(PolymarketClient, "get_multi_outcome_events", return_value=[_underpriced_event(n=3)]):
        await pipeline._stage_strategy_scan()

    # 3 legs at the strategy's default trade_size_usd (1.0) each.
    assert pipeline.budget_manager.total_spent_today == pytest.approx(3.0, abs=0.01)


# ── Fairly priced event: no signals, no trades ────────────────────────────────

@pytest.mark.asyncio
async def test_fairly_priced_event_executes_nothing():
    pipeline, bridge, log_calls = _make_pipeline(balance=100.0)
    fair_event = _underpriced_event(n=2, price=0.50)  # sums to 1.0, no edge

    with patch.object(PolymarketClient, "get_multi_outcome_events", return_value=[fair_event]):
        await pipeline._stage_strategy_scan()

    assert not [c for c in log_calls if c["level"] in ("STRATEGY-LEG", "DRY-RUN")]
    assert pipeline.budget_manager.total_spent_today == 0.0


# ── Guardrails: cash/budget and daily-trade-limit gating ─────────────────────

@pytest.mark.asyncio
async def test_insufficient_cash_rejects_whole_group():
    pipeline, bridge, log_calls = _make_pipeline(balance=1.0)  # can't afford 3 legs at $1 each... actually needs < total_cost
    bridge.current_balance = 0.50  # below the 3-leg $3.00 total cost

    with patch.object(PolymarketClient, "get_multi_outcome_events", return_value=[_underpriced_event(n=3)]):
        await pipeline._stage_strategy_scan()

    assert not [c for c in log_calls if c["level"] == "STRATEGY-LEG"]
    assert any(
        c["level"] == "REJECTED" and c["payload"].get("reason") == "insufficient_cash"
        for c in log_calls
    )
    assert pipeline.budget_manager.total_spent_today == 0.0


@pytest.mark.asyncio
async def test_already_held_group_is_skipped_not_rebought():
    """Restart-safety: _filled_arb_groups (in-memory) resets on every
    process restart, but real holdings — reflected in bridge.current_portfolio
    — persist across restarts. A group with a leg already in the portfolio
    must be skipped even on a "fresh" pipeline instance that has never seen
    this group_id before, or a redeploy re-buys everything it already
    holds (the confirmed root cause of a runaway-trade-rate incident)."""
    pipeline, bridge, log_calls = _make_pipeline(balance=100.0)
    event = _underpriced_event(n=3)

    # Simulate "already holding" one leg of this group, as if positions
    # persisted across a restart that wiped _filled_arb_groups.
    held = SimpleNamespace(asset_id="evt1_tok0")
    bridge.current_portfolio = [held]

    with patch.object(PolymarketClient, "get_multi_outcome_events", return_value=[event]):
        await pipeline._stage_strategy_scan()

    assert not [c for c in log_calls if c["level"] == "STRATEGY-LEG"]
    assert any(
        c["level"] == "SCAN-SKIP" and c["payload"].get("reason") == "already_owned_in_portfolio"
        for c in log_calls
    )
    assert pipeline.budget_manager.total_spent_today == 0.0
    assert "event_sum:evt1" in pipeline._filled_arb_groups


@pytest.mark.asyncio
async def test_global_daily_trade_limit_rejects_group_even_within_strategy_allowance():
    """Belt-and-suspenders: a generous per-strategy arbitrage_max_daily_trades
    must not let a single group blow past the account-wide max_daily_trades
    ceiling."""
    config = TradingConfig(
        dry_run=True, min_ev=0.30, bankroll_usd=1000.0,
        daily_limit_usd=1500.0, max_bet_size_usd=3.0,
        max_daily_trades=2,               # global ceiling: only 2 trades/day
        arbitrage_max_daily_trades=200,   # strategy's own bucket allows way more
        min_trading_balance=1.0,
    )
    pipeline, bridge, log_calls = _make_pipeline(balance=1000.0, config=config)

    with patch.object(PolymarketClient, "get_multi_outcome_events", return_value=[_underpriced_event(n=3)]):
        await pipeline._stage_strategy_scan()

    assert not [c for c in log_calls if c["level"] == "STRATEGY-LEG"]
    assert any(
        c["level"] == "REJECTED" and c["payload"].get("reason") == "global_daily_trade_limit_would_be_exceeded"
        for c in log_calls
    )


@pytest.mark.asyncio
async def test_daily_trade_limit_rejects_whole_group():
    config = TradingConfig(
        dry_run=True, min_ev=0.30, bankroll_usd=1000.0,
        daily_limit_usd=1500.0, max_bet_size_usd=3.0,
        max_daily_trades=2, arbitrage_max_daily_trades=2,
        min_trading_balance=1.0,  # fewer than the 3 legs
    )
    pipeline, bridge, log_calls = _make_pipeline(balance=1000.0, config=config)

    with patch.object(PolymarketClient, "get_multi_outcome_events", return_value=[_underpriced_event(n=3)]):
        await pipeline._stage_strategy_scan()

    assert not [c for c in log_calls if c["level"] == "STRATEGY-LEG"]
    assert any(
        c["level"] == "REJECTED" and c["payload"].get("reason") == "daily_trade_limit_would_be_exceeded"
        for c in log_calls
    )


# ── Phase 2: TTE filter — skip groups with an over-long leg ──────────────────

@pytest.mark.asyncio
async def test_tte_filter_skips_group_with_long_dated_leg():
    config = TradingConfig(
        dry_run=True, min_ev=0.30, bankroll_usd=1000.0,
        daily_limit_usd=15.0, max_bet_size_usd=3.0,
        max_daily_trades=10, min_trading_balance=1.0,
        max_tte_days=90,
    )
    pipeline, bridge, log_calls = _make_pipeline(balance=100.0, config=config)
    # 2 legs at 10 days out, 1 leg at 365 days out — one over-long leg voids
    # the whole group.
    event = _underpriced_event(n=3, expiry_days=[10, 10, 365])

    with patch.object(PolymarketClient, "get_multi_outcome_events", return_value=[event]):
        await pipeline._stage_strategy_scan()

    assert not [c for c in log_calls if c["level"] == "STRATEGY-LEG"]
    assert any(
        c["level"] == "FILTERED" and c["payload"].get("reason") == "tte_exceeds_max"
        for c in log_calls
    )
    assert pipeline.budget_manager.total_spent_today == 0.0


@pytest.mark.asyncio
async def test_tte_filter_allows_group_within_max_tte():
    config = TradingConfig(
        dry_run=True, min_ev=0.30, bankroll_usd=1000.0,
        daily_limit_usd=15.0, max_bet_size_usd=3.0,
        max_daily_trades=10, min_trading_balance=1.0,
        max_tte_days=90,
    )
    pipeline, bridge, log_calls = _make_pipeline(balance=100.0, config=config)
    # All 3 legs well within the 90-day ceiling.
    event = _underpriced_event(n=3, expiry_days=30)

    with patch.object(PolymarketClient, "get_multi_outcome_events", return_value=[event]):
        await pipeline._stage_strategy_scan()

    assert not [c for c in log_calls if c["level"] == "FILTERED" and c["payload"].get("reason") == "tte_exceeds_max"]
    leg_entries = [c for c in log_calls if c["level"] == "STRATEGY-LEG"]
    assert len(leg_entries) == 3


# ── Phase 4: crypto-first arbitrage priority ──────────────────────────────────

@pytest.mark.asyncio
async def test_crypto_events_execute_before_general_events():
    config = TradingConfig(
        dry_run=True, min_ev=0.30, bankroll_usd=1000.0,
        daily_limit_usd=15.0, max_bet_size_usd=3.0,
        max_daily_trades=10, min_trading_balance=1.0,
        arbitrage_crypto_first=True,
    )
    pipeline, bridge, log_calls = _make_pipeline(balance=100.0, config=config)

    general_event = _underpriced_event(event_id="general_evt", n=2, title="Who wins the election?")
    crypto_event = _underpriced_event(event_id="crypto_evt", n=2, title="Will Bitcoin hit $100k?")

    # Scanner returns the general event first — crypto-first ordering must
    # still execute the crypto group before it.
    with patch.object(PolymarketClient, "get_multi_outcome_events", return_value=[general_event, crypto_event]):
        await pipeline._stage_strategy_scan()

    group_entries = [c for c in log_calls if c["level"] == "STRATEGY-GROUP"]
    executed_order = [c["payload"]["group_id"] for c in group_entries]

    assert executed_order == ["event_sum:crypto_evt", "event_sum:general_evt"]


@pytest.mark.asyncio
async def test_crypto_first_disabled_preserves_scanner_order():
    config = TradingConfig(
        dry_run=True, min_ev=0.30, bankroll_usd=1000.0,
        daily_limit_usd=15.0, max_bet_size_usd=3.0,
        max_daily_trades=10, min_trading_balance=1.0,
        arbitrage_crypto_first=False,
    )
    pipeline, bridge, log_calls = _make_pipeline(balance=100.0, config=config)

    general_event = _underpriced_event(event_id="general_evt", n=2, title="Who wins the election?")
    crypto_event = _underpriced_event(event_id="crypto_evt", n=2, title="Will Bitcoin hit $100k?")

    with patch.object(PolymarketClient, "get_multi_outcome_events", return_value=[general_event, crypto_event]):
        await pipeline._stage_strategy_scan()

    group_entries = [c for c in log_calls if c["level"] == "STRATEGY-GROUP"]
    executed_order = [c["payload"]["group_id"] for c in group_entries]

    assert executed_order == ["event_sum:general_evt", "event_sum:crypto_evt"]


# ── Scan-interval throttling ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_scan_is_throttled_within_interval():
    pipeline, bridge, log_calls = _make_pipeline(balance=100.0)
    pipeline.strategy_scan_interval = 999.0  # effectively "don't rescan"

    with patch.object(PolymarketClient, "get_multi_outcome_events", return_value=[]) as mock_fetch:
        await pipeline._stage_strategy_scan()
        await pipeline._stage_strategy_scan()

    assert mock_fetch.call_count == 1


# ── Graceful degradation ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_event_fetch_failure_does_not_raise():
    pipeline, bridge, log_calls = _make_pipeline(balance=100.0)

    with patch.object(PolymarketClient, "get_multi_outcome_events", side_effect=Exception("network down")):
        await pipeline._stage_strategy_scan()  # must not raise

    assert any(c["level"] == "SYNC-WARN" for c in log_calls)


@pytest.mark.asyncio
async def test_strategy_scan_failure_does_not_raise():
    pipeline, bridge, log_calls = _make_pipeline(balance=100.0)

    with patch.object(PolymarketClient, "get_multi_outcome_events", return_value=[_underpriced_event(n=2)]), \
         patch.object(pipeline.strategies[0], "scan", side_effect=Exception("boom")):
        await pipeline._stage_strategy_scan()  # must not raise

    assert any(c["level"] == "SYNC-WARN" for c in log_calls)


@pytest.mark.asyncio
async def test_no_strategies_registered_is_a_noop():
    pipeline, bridge, log_calls = _make_pipeline(balance=100.0)
    pipeline.strategies = []

    with patch.object(PolymarketClient, "get_multi_outcome_events") as mock_fetch:
        await pipeline._stage_strategy_scan()

    mock_fetch.assert_not_called()

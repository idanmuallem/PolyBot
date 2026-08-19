"""Integration: strategy signals flow through the same executor as
model-driven trades, tagged with strategy_type for separate tracking."""
from unittest.mock import patch

import pytest

from core.bridge import DataBridge
from core.trading_config import TradingConfig
from core.wallet_context import WalletContext
from polymarket import PolymarketClient, PolymarketScannerHunter
from trading.budget_manager import BudgetManager
from trading.executor import TradeExecutor
from trading.risk_manager import PortfolioManager


def _underpriced_event(event_id="evt1", n=3, price=0.30):
    return {
        "id": event_id,
        "title": "Test Multi-Outcome Event",
        "markets": [
            {
                "closed": False,
                "clobTokenIds": [f"tok{i}", f"tok{i}_no"],
                "lastTradePrice": price,
                "question": f"Outcome {i}?",
                "groupItemTitle": f"Outcome {i}",
                "volume": 100_000.0,
            }
            for i in range(n)
        ],
    }


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
        c["level"] == "REJECTED" and c["payload"].get("reason") == "insufficient_cash_or_budget_for_all_legs"
        for c in log_calls
    )
    assert pipeline.budget_manager.total_spent_today == 0.0


@pytest.mark.asyncio
async def test_daily_trade_limit_rejects_whole_group():
    config = TradingConfig(
        dry_run=True, min_ev=0.30, bankroll_usd=1000.0,
        daily_limit_usd=1500.0, max_bet_size_usd=3.0,
        max_daily_trades=2, min_trading_balance=1.0,  # fewer than the 3 legs
    )
    pipeline, bridge, log_calls = _make_pipeline(balance=1000.0, config=config)

    with patch.object(PolymarketClient, "get_multi_outcome_events", return_value=[_underpriced_event(n=3)]):
        await pipeline._stage_strategy_scan()

    assert not [c for c in log_calls if c["level"] == "STRATEGY-LEG"]
    assert any(
        c["level"] == "REJECTED" and c["payload"].get("reason") == "daily_trade_limit_would_be_exceeded"
        for c in log_calls
    )


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

"""Integration: strategy signals flow through the same executor as
model-driven trades, tagged with strategy_type for separate tracking."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.bridge import DataBridge
from core.models import MarketData
from core.trading_config import TradingConfig
from core.wallet_context import WalletContext
from polymarket import PolymarketClient, PolymarketScannerHunter
from trading.budget_manager import BudgetManager
from trading.executor import TradeExecutor
from trading.risk_manager import PortfolioManager
from trading.strategies.base import StrategySignal


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
            trading_mode="dry_run", min_ev=0.30, bankroll_usd=1000.0,
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
        # These tests exercise arbitrage-group execution mechanics (budget
        # gating, TTE filtering, scan ordering, ...) with MarketData
        # fixtures that don't carry a slug/condition_id — not real
        # paper-engine fidelity. A real PaperAdapter (constructed
        # unconditionally by TradeExecutor.__init__ whenever pm_trader is
        # importable) would now honestly report those buys as failed (see
        # trading/executor.py's dry-run branches and
        # _simulate_full_fill_arbitrage_group's fill accounting), which
        # isn't what these tests are about. See tests/unit/test_executor.py's
        # _make_executor() for the same rationale.
        ctx.executor.paper = None
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
async def test_fully_held_group_is_skipped_not_rebought():
    """Restart-safety: _filled_arb_groups (in-memory) resets on every
    process restart, but real holdings — reflected in bridge.current_portfolio
    — persist across restarts. A group whose ENTIRE currently-valid basket
    is already held must be skipped even on a "fresh" pipeline instance
    that has never seen this group_id before, or a redeploy re-buys
    everything it already holds (the confirmed root cause of a
    runaway-trade-rate incident). See test_partially_held_group_* below
    for the complementary case: a basket only PARTLY held must NOT be
    skipped — see _group_already_held's docstring for why the check is
    ALL-legs, not ANY-leg."""
    pipeline, bridge, log_calls = _make_pipeline(balance=100.0)
    event = _underpriced_event(n=3)

    # Simulate holding every leg, as if positions persisted across a
    # restart that wiped _filled_arb_groups.
    bridge.current_portfolio = [
        SimpleNamespace(asset_id=f"evt1_tok{i}", shares=3.0, initial_price=0.30)
        for i in range(3)
    ]

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
async def test_partially_held_group_not_skipped_only_missing_leg_bought():
    """A basket where 2 of 3 currently-valid legs are already held (e.g.
    the event gained a new candidate/outcome after the original buy — see
    _group_already_held's docstring for the real Nobel Peace Prize case
    this is modeled on) must NOT be skipped, and the pipeline must only
    attempt to buy the still-missing leg — not re-buy the 2 it already
    holds."""
    pipeline, bridge, log_calls = _make_pipeline(balance=100.0)
    event = _underpriced_event(n=3)

    bridge.current_portfolio = [
        SimpleNamespace(asset_id="evt1_tok0", shares=10.0, initial_price=0.30),
        SimpleNamespace(asset_id="evt1_tok1", shares=10.0, initial_price=0.30),
    ]

    with patch.object(PolymarketClient, "get_multi_outcome_events", return_value=[event]):
        await pipeline._stage_strategy_scan()

    assert not [c for c in log_calls if c["level"] == "SCAN-SKIP"]
    dry_run_entries = [c for c in log_calls if c["level"] == "DRY-RUN"]
    leg_entries = [c for c in log_calls if c["level"] == "STRATEGY-LEG"]
    assert [c["token_id"] for c in dry_run_entries] == ["evt1_tok2"]
    assert [c["token_id"] for c in leg_entries] == ["evt1_tok2"]

    group_entries = [c for c in log_calls if c["level"] == "STRATEGY-GROUP"]
    assert len(group_entries) == 1
    assert group_entries[0]["payload"]["n_legs"] == 1
    assert group_entries[0]["payload"]["legs_already_held"] == 2
    assert group_entries[0]["payload"]["legs_total_in_basket"] == 3

    assert "event_sum:evt1" in pipeline._filled_arb_groups


@pytest.mark.asyncio
async def test_partial_hold_topup_trims_new_leg_to_held_share_count():
    """The missing leg is bought for a fixed dollar amount, which can land
    far more shares than the already-held legs hold — e.g. a cheap
    long-shot outcome added to the event after the original buy. The true
    guaranteed-set size is bounded by the SCARCEST leg in the whole
    basket, held or new, so the excess on the newly-bought leg must be
    trimmed back to the held legs' share count, not kept as naked
    exposure on a candidate that can't actually complete a full set."""
    pipeline, bridge, log_calls = _make_pipeline(balance=100.0)
    event = _underpriced_event(n=3)
    event["markets"][2]["lastTradePrice"] = 0.05  # cheap long-shot leg: $1 buys ~20 shares

    bridge.current_portfolio = [
        SimpleNamespace(asset_id="evt1_tok0", shares=2.0, initial_price=0.30),
        SimpleNamespace(asset_id="evt1_tok1", shares=2.0, initial_price=0.30),
    ]

    with patch.object(PolymarketClient, "get_multi_outcome_events", return_value=[event]):
        await pipeline._stage_strategy_scan()

    leg_entries = [c for c in log_calls if c["level"] == "STRATEGY-LEG"]
    assert len(leg_entries) == 1
    assert leg_entries[0]["token_id"] == "evt1_tok2"
    assert leg_entries[0]["payload"]["shares"] == pytest.approx(2.0)  # trimmed to the held legs' 2.0 shares

    assert any(c["level"] == "DRY-RUN-SELL" for c in log_calls)  # ~18 excess shares sold back
    assert "event_sum:evt1" in pipeline._filled_arb_groups


@pytest.mark.asyncio
async def test_global_daily_trade_limit_rejects_group_even_within_strategy_allowance():
    """Belt-and-suspenders: a generous per-strategy arbitrage_max_daily_trades
    must not let a single group blow past the account-wide max_daily_trades
    ceiling."""
    config = TradingConfig(
        trading_mode="dry_run", min_ev=0.30, bankroll_usd=1000.0,
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
        trading_mode="dry_run", min_ev=0.30, bankroll_usd=1000.0,
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
        trading_mode="dry_run", min_ev=0.30, bankroll_usd=1000.0,
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
        trading_mode="dry_run", min_ev=0.30, bankroll_usd=1000.0,
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
        trading_mode="dry_run", min_ev=0.30, bankroll_usd=1000.0,
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
        trading_mode="dry_run", min_ev=0.30, bankroll_usd=1000.0,
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


# ── ENABLE_ARBITRAGE kill switch: early exit, not a downstream cap ───────────

@pytest.mark.asyncio
async def test_enable_arbitrage_false_skips_scan_entirely_no_api_call():
    """config.enable_arbitrage=False must stop _stage_strategy_scan() before
    any Gamma API call and before strategy.scan() runs at all — not just
    reject every signal downstream (that's what ARBITRAGE_MAX_DAILY_TRADES=0
    already does, and is explicitly NOT what this flag is for)."""
    config = TradingConfig(
        trading_mode="dry_run", min_ev=0.30, bankroll_usd=1000.0,
        daily_limit_usd=15.0, max_bet_size_usd=3.0,
        max_daily_trades=10, min_trading_balance=1.0,
        enable_arbitrage=False,
    )
    pipeline, bridge, log_calls = _make_pipeline(balance=100.0, config=config)

    with patch.object(PolymarketClient, "get_multi_outcome_events") as mock_fetch, \
         patch.object(pipeline.strategies[0], "scan", new=AsyncMock()) as mock_scan:
        await pipeline._stage_strategy_scan()

    mock_fetch.assert_not_called()
    mock_scan.assert_not_called()
    assert not log_calls  # zero log entries of any kind — the stage never ran


@pytest.mark.asyncio
async def test_enable_arbitrage_true_default_preserves_current_behavior():
    """Default (unset/True) must behave exactly as before this flag existed."""
    pipeline, bridge, log_calls = _make_pipeline(balance=100.0)
    assert pipeline.config.enable_arbitrage is True

    with patch.object(PolymarketClient, "get_multi_outcome_events", return_value=[_underpriced_event(n=3)]):
        await pipeline._stage_strategy_scan()

    assert [c for c in log_calls if c["level"] == "STRATEGY-GROUP"]


# ── Group-level guaranteed-profit re-verification against real fills ─────────
#
# EventSumStrategy.scan() only checks profitability once, against a stale
# lastTradePrice snapshot with a flat fee estimate — never re-verified
# against what was actually filled. These tests exercise
# _execute_strategy_group() directly, with TradeExecutor.execute_arbitrage_group
# mocked to return a crafted (fills, arb_sets, unfilled) result, so the
# post-fill verification/unwind logic can be tested independently of
# whatever a real (or simulated) order book happens to return.

_FAKE_ARBITRAGE_STRATEGY = SimpleNamespace(strategy_type="arbitrage")


def _signal(token_id, price, bet_amount_usd=1.0, group_id="event_sum:evt1", edge=0.10):
    market = MarketData(
        market_id=token_id,
        asset_type="Arbitrage::EventSum",
        strike_price=0.0,
        question=f"Outcome {token_id}?",
        market_name=f"Test Event - {token_id}",
        initial_price=price,
        volume=100_000.0,
        no_market_id=f"{token_id}_no",
    )
    return StrategySignal(
        market=market, side="YES", price=price, bet_amount_usd=bet_amount_usd,
        strategy_type="arbitrage", reasoning="test", group_id=group_id, edge=edge,
    )


async def _run_group(pipeline, signals, arb_result):
    """Run _execute_strategy_group with execute_arbitrage_group's result
    controlled directly, and sell_position spied on (but still a real call
    through to the dry-run/no-paper-adapter path, so it returns True)."""
    group_id = signals[0].group_id
    with patch.object(pipeline.executor, "execute_arbitrage_group", new=AsyncMock(return_value=arb_result)), \
         patch.object(pipeline.executor, "sell_position", wraps=pipeline.executor.sell_position) as mock_sell:
        await pipeline._execute_strategy_group(_FAKE_ARBITRAGE_STRATEGY, group_id, signals)
    return mock_sell


@pytest.mark.asyncio
async def test_arbitrage_rejects_group_not_profitable_after_real_fills():
    """Theoretical scan-time edge (2 legs at $0.49 = $0.98, ~2% before
    fees) can still lose money once real fills are worse than requested —
    here each $1 leg only received 1.8 shares (real cost ~$0.556/share)
    instead of the ~2.04 shares $1/$0.49 implies, for a real -$0.20 net."""
    pipeline, bridge, log_calls = _make_pipeline(balance=100.0)
    signals = [_signal("tokA", 0.49), _signal("tokB", 0.49)]

    arb_result = {
        "success": True, "arb_sets": 1.8,
        "fills": {"tokA": 1.8, "tokB": 1.8},
        "unfilled": [], "surplus": {},
    }
    mock_sell = await _run_group(pipeline, signals, arb_result)

    assert mock_sell.call_count == 2
    sold_shares = {call.args[0]: call.args[1] for call in mock_sell.call_args_list}
    assert sold_shares == {"tokA": 1.8, "tokB": 1.8}

    assert not [c for c in log_calls if c["level"] == "STRATEGY-LEG"]
    assert any(
        c["level"] == "REJECTED" and c["payload"].get("reason") == "not_profitable_after_real_fills"
        for c in log_calls
    )
    assert "event_sum:evt1" not in pipeline._filled_arb_groups
    assert "event_sum:evt1" in pipeline._exhausted_arb_groups


@pytest.mark.asyncio
async def test_arbitrage_executes_when_genuinely_profitable_after_real_fills():
    """2 legs at $0.40 (healthy edge), real fills matching what was
    requested (2.5 shares/$1 each) — should execute, not unwind."""
    pipeline, bridge, log_calls = _make_pipeline(balance=100.0)
    signals = [_signal("tokA", 0.40), _signal("tokB", 0.40)]

    arb_result = {
        "success": True, "arb_sets": 2.5,
        "fills": {"tokA": 2.5, "tokB": 2.5},
        "unfilled": [], "surplus": {},
    }
    mock_sell = await _run_group(pipeline, signals, arb_result)

    mock_sell.assert_not_called()
    leg_entries = [c for c in log_calls if c["level"] == "STRATEGY-LEG"]
    assert len(leg_entries) == 2
    assert all(c["payload"]["executed"] for c in leg_entries)
    assert not [c for c in log_calls if c["level"] == "REJECTED"]
    assert "event_sum:evt1" in pipeline._filled_arb_groups


@pytest.mark.asyncio
async def test_arbitrage_trims_surplus_when_fills_are_unequal_but_profitable():
    """2 legs at $0.30/$0.35 (healthy edge), but real fills come back
    unequal (5.0 vs 2.5 shares for the same $1 spend, as thin books on
    illiquid long-shot outcomes would produce) — the matched 2.5-share set
    is genuinely profitable, so it executes, but the 2.5-share surplus on
    the overfilled leg is naked exposure and must be sold back, not kept."""
    pipeline, bridge, log_calls = _make_pipeline(balance=100.0)
    signals = [_signal("tokA", 0.30), _signal("tokB", 0.35)]

    arb_result = {
        "success": True, "arb_sets": 2.5,
        "fills": {"tokA": 5.0, "tokB": 2.5},
        "unfilled": [], "surplus": {"tokA": 2.5},
    }
    mock_sell = await _run_group(pipeline, signals, arb_result)

    mock_sell.assert_called_once()
    assert mock_sell.call_args.args[0] == "tokA"
    assert mock_sell.call_args.args[1] == 2.5
    leg_entries = {c["token_id"]: c for c in log_calls if c["level"] == "STRATEGY-LEG"}
    assert leg_entries["tokA"]["payload"]["shares"] == 2.5  # trimmed, not the raw 5.0 fill
    assert leg_entries["tokB"]["payload"]["shares"] == 2.5
    assert "event_sum:evt1" in pipeline._filled_arb_groups


@pytest.mark.asyncio
async def test_arbitrage_unwinds_when_group_incomplete():
    """2 legs, only one fills at all (the other unfilled) — arb_sets is 0
    (no complementary set exists without every leg), so the leg that DID
    fill must be sold back entirely, not held as a naked one-sided bet."""
    pipeline, bridge, log_calls = _make_pipeline(balance=100.0)
    signals = [_signal("tokA", 0.30), _signal("tokB", 0.35)]

    arb_result = {
        "success": False, "arb_sets": 0.0,
        "fills": {"tokA": 3.0},
        "unfilled": ["tokB"], "surplus": {},
    }
    mock_sell = await _run_group(pipeline, signals, arb_result)

    mock_sell.assert_called_once()
    assert mock_sell.call_args.args[0] == "tokA"
    assert mock_sell.call_args.args[1] == 3.0

    assert not [c for c in log_calls if c["level"] == "STRATEGY-LEG"]
    assert any(
        c["level"] == "REJECTED" and c["payload"].get("reason") == "group_incomplete_unwound_partial_fills"
        for c in log_calls
    )
    assert "event_sum:evt1" not in pipeline._filled_arb_groups
    assert "event_sum:evt1" in pipeline._exhausted_arb_groups


# ── End-to-end: real _execute_paper_limit_arbitrage_group, not a mocked
# execute_arbitrage_group result — confirms the whole chain (limit-order
# placement/polling in executor.py -> execute_arbitrage_group's dispatch ->
# _execute_strategy_group's post-fill verification/unwind) works together,
# not just each piece in isolation.

def _mock_paper_limit_orders(fills_by_token: dict):
    condition_to_token = {}

    def fake_place_limit_buy(slug, condition_id, token_id, side, amount_usd, limit_price, no_token_id=None):
        cond = f"cond_{token_id}"
        condition_to_token[cond] = token_id
        return {"id": f"order_{token_id}", "market_condition_id": cond, "outcome": side.lower()}

    def fake_get_position_shares(condition_id, outcome):
        token_id = condition_to_token.get(condition_id)
        return fills_by_token.get(token_id, 0.0)

    paper = MagicMock()
    paper.place_limit_buy.side_effect = fake_place_limit_buy
    paper.check_pending_limit_orders.return_value = []
    paper.get_position_shares.side_effect = fake_get_position_shares
    paper.cancel_limit_order.return_value = True
    return paper


@pytest.mark.asyncio
async def test_paper_limit_orders_end_to_end_some_legs_fill_others_dont():
    """A 2-leg group where the limit-order path genuinely fills one leg and
    never fills the other: the real _execute_paper_limit_arbitrage_group
    voids the group (arb_sets=0, per its own unfilled-leg rule), and that
    correctly flows into _execute_strategy_group's incomplete-group unwind —
    the leg that did fill gets sold back, not held as a naked position."""
    pipeline, bridge, log_calls = _make_pipeline(balance=100.0)
    pipeline.executor.paper = _mock_paper_limit_orders({"tokA": 3.33})  # tokB never fills
    pipeline.config.arbitrage_order_timeout_seconds = 0.05  # keep the poll loop's deadline short

    with patch.object(pipeline.executor, "sell_position", return_value=True) as mock_sell:
        await pipeline._execute_strategy_group(
            _FAKE_ARBITRAGE_STRATEGY, "event_sum:evt1",
            [_signal("tokA", 0.30), _signal("tokB", 0.35)],
        )

    pipeline.executor.paper.place_limit_buy.assert_called()
    assert pipeline.executor.paper.place_limit_buy.call_count == 2
    mock_sell.assert_called_once()
    assert mock_sell.call_args.args[0] == "tokA"
    assert mock_sell.call_args.args[1] == pytest.approx(3.33)

    assert not [c for c in log_calls if c["level"] == "STRATEGY-LEG"]
    assert any(
        c["level"] == "REJECTED" and c["payload"].get("reason") == "group_incomplete_unwound_partial_fills"
        for c in log_calls
    )
    assert "event_sum:evt1" not in pipeline._filled_arb_groups


@pytest.mark.asyncio
async def test_paper_limit_orders_end_to_end_all_legs_fill():
    pipeline, bridge, log_calls = _make_pipeline(balance=100.0)
    pipeline.executor.paper = _mock_paper_limit_orders({"tokA": 3.33, "tokB": 2.85})
    pipeline.config.arbitrage_order_timeout_seconds = 0.05

    await pipeline._execute_strategy_group(
        _FAKE_ARBITRAGE_STRATEGY, "event_sum:evt1",
        [_signal("tokA", 0.30), _signal("tokB", 0.35)],
    )

    leg_entries = [c for c in log_calls if c["level"] == "STRATEGY-LEG"]
    assert len(leg_entries) == 2
    assert all(c["payload"]["executed"] for c in leg_entries)
    assert "event_sum:evt1" in pipeline._filled_arb_groups

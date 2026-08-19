"""E2E: one complete iteration of SequentialTradingPipeline with all external APIs mocked.

NOTE: @freeze_time is intentionally NOT used here because freezegun patches
time.monotonic(), which breaks asyncio.wait_for() and asyncio.sleep().
Dynamic expiry dates are used instead.
"""
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import pytest

from core.bridge import DataBridge
from core.models import MarketData
from core.trading_config import TradingConfig
from core.wallet_context import WalletContext
from hunters.crypto import CryptoHunter
from polymarket import PolymarketClient
from trading.executor import TradeExecutor


_DRY_CONFIG = TradingConfig(
    dry_run=True,
    min_ev=0.30,
    bankroll_usd=1000.0,
    daily_limit_usd=15.0,
    max_bet_size_usd=3.0,
    max_daily_trades=10,
    loop_delay_seconds=0.01,
    min_trading_balance=1.0,
    min_tte_minutes=60,
    max_tte_days=180,
    private_key="",
    proxy_address="",
    # This test exercises generic pipeline plumbing (DRY-RUN/AUTO-TRADE/TRACK
    # firing, budget tracking, cooldown caching) with numbers tuned against
    # the brain's raw fair value — see test_full_pipeline_dryrun_wang_mode
    # below for the Wang-adjusted path.
    pricing_mode="legacy",
)


@pytest.mark.asyncio
async def test_single_pipeline_loop_dry_run(tmp_path):
    """
    Full pipeline cycle: Hunt → Evaluate → Risk → Execute in dry-run mode.
    All external APIs mocked. Verifies:
      - DRY-RUN log entry is produced (executor fires)
      - AUTO-TRADE log entry exists (Bug #1 regression: no NameError)
      - TRACK log entry has "fair", "ev", "bet_usd", "side" keys
      - total_spent_today increases
      - market is added to seen_markets cooldown cache
    """
    bridge = DataBridge()
    bridge.current_balance = 100.0
    bridge.current_portfolio = []
    bridge.open_position_value = 0.0
    bridge.open_positions_value = 0.0
    bridge.live_trading = False

    # 29 days from now: standard_bs model; price=0.35 gives EV > 0.30
    expiry = (datetime.now(timezone.utc) + timedelta(days=29)).isoformat()
    sample_market = MarketData(
        market_id="tok_btc",
        asset_type="Crypto::BTCUSDT",
        strike_price=95_000.0,
        question="Will BTC exceed $95,000 by next month?",
        market_name="Bitcoin — Will BTC exceed $95,000?",
        initial_price=0.35,
        volume=500_000.0,
        expiry_date=expiry,
        no_market_id="tok_btc_no",
    )

    log_calls = []

    def spy_log(level, asset_type, token_id, payload):
        log_calls.append({
            "level": level,
            "asset_type": asset_type,
            "token_id": token_id,
            "payload": payload,
        })

    from trading.decision_pipeline import SequentialTradingPipeline

    ctx = WalletContext(
        wallet_id="test_wallet",
        config=_DRY_CONFIG,
        bridge=bridge,
        db_path=str(tmp_path / "trades.db"),
    )

    with patch.object(TradeExecutor, "get_open_positions", return_value=[]), \
         patch.object(TradeExecutor, "get_balance", return_value=100.0), \
         patch.object(CryptoHunter, "hunt", return_value=sample_market), \
         patch.object(CryptoHunter, "get_live_truth", return_value=97_000.0), \
         patch.object(PolymarketClient, "get_multi_outcome_events", return_value=[]):

        pipeline = SequentialTradingPipeline(ctx=ctx, log_func=spy_log, delay=0.01)

        # Run the loop; TimeoutError is expected after 1 second
        try:
            await asyncio.wait_for(pipeline.run_forever(), timeout=1.0)
        except asyncio.TimeoutError:
            pass

    # 1. Executor fired → DRY-RUN log exists
    assert any(c["level"] == "DRY-RUN" for c in log_calls), (
        "No DRY-RUN entry. Log levels seen: " + str([c["level"] for c in log_calls])
    )

    # 2. AUTO-TRADE log exists → Bug #1 regression guard (no NameError before this line)
    assert any(c["level"] == "AUTO-TRADE" for c in log_calls), (
        "No AUTO-TRADE entry — NameError may have been raised in evaluate_and_execute"
    )

    # 3. Budget was spent
    assert pipeline.budget_manager.total_spent_today > 0, "total_spent_today should be > 0"

    # 4. Market was cached as seen
    assert "tok_btc" in pipeline.hunter.seen_markets

    # 5. TRACK entry has required payload keys
    track_entries = [c for c in log_calls if c["level"] == "TRACK"]
    assert track_entries, "No TRACK entry found"
    track_payload = track_entries[0]["payload"]
    for key in ("fair", "ev", "bet_usd", "side"):
        assert key in track_payload, f"TRACK payload missing key: {key!r}"
    # legacy mode: no Wang adjustment applied.
    assert track_payload["pricing_mode"] == "legacy"
    assert track_payload["wang_fair_value"] is None


_WANG_DRY_CONFIG = TradingConfig(
    dry_run=True,
    min_ev=0.30,
    bankroll_usd=1000.0,
    daily_limit_usd=15.0,
    max_bet_size_usd=3.0,
    max_daily_trades=10,
    loop_delay_seconds=0.01,
    min_trading_balance=1.0,
    min_tte_minutes=60,
    max_tte_days=180,
    private_key="",
    proxy_address="",
    pricing_mode="wang",
    wang_base_lambda=0.183,
    wang_min_edge=0.05,
)


@pytest.mark.asyncio
async def test_full_pipeline_loop_dry_run_wang_mode(tmp_path):
    """Same loop as test_single_pipeline_loop_dry_run, but in Wang pricing
    mode: verifies a trade still fires end-to-end and the TRACK payload
    carries the Wang fields (raw_probability, wang_lambda, wang_fair_value,
    wang_edge) needed to compare Wang-adjusted decisions against the legacy
    raw-EV path.
    """
    bridge = DataBridge()
    bridge.current_balance = 100.0
    bridge.current_portfolio = []
    bridge.open_position_value = 0.0
    bridge.open_positions_value = 0.0
    bridge.live_trading = False

    # Priced at PRICE_FLOOR so the (smaller, by design) Wang-adjusted edge
    # still clears both wang_min_edge and min_ev.
    expiry = (datetime.now(timezone.utc) + timedelta(days=29)).isoformat()
    sample_market = MarketData(
        market_id="tok_btc_wang",
        asset_type="Crypto::BTCUSDT",
        strike_price=95_000.0,
        question="Will BTC exceed $95,000 by next month?",
        market_name="Bitcoin — Will BTC exceed $95,000?",
        initial_price=0.30,
        volume=500_000.0,
        expiry_date=expiry,
        no_market_id="tok_btc_wang_no",
    )

    log_calls = []

    def spy_log(level, asset_type, token_id, payload):
        log_calls.append({
            "level": level,
            "asset_type": asset_type,
            "token_id": token_id,
            "payload": payload,
        })

    from trading.decision_pipeline import SequentialTradingPipeline

    ctx = WalletContext(
        wallet_id="test_wallet_wang",
        config=_WANG_DRY_CONFIG,
        bridge=bridge,
        db_path=str(tmp_path / "trades_wang.db"),
    )

    with patch.object(TradeExecutor, "get_open_positions", return_value=[]), \
         patch.object(TradeExecutor, "get_balance", return_value=100.0), \
         patch.object(CryptoHunter, "hunt", return_value=sample_market), \
         patch.object(CryptoHunter, "get_live_truth", return_value=97_000.0), \
         patch.object(PolymarketClient, "get_multi_outcome_events", return_value=[]):

        pipeline = SequentialTradingPipeline(ctx=ctx, log_func=spy_log, delay=0.01)
        assert pipeline.pricing_mode == "wang"

        try:
            await asyncio.wait_for(pipeline.run_forever(), timeout=1.0)
        except asyncio.TimeoutError:
            pass

    assert any(c["level"] == "DRY-RUN" for c in log_calls), (
        "No DRY-RUN entry. Log levels seen: " + str([c["level"] for c in log_calls])
    )

    track_entries = [c for c in log_calls if c["level"] == "TRACK"]
    assert track_entries, "No TRACK entry found"
    payload = track_entries[0]["payload"]

    assert payload["pricing_mode"] == "wang"
    for key in ("raw_probability", "wang_lambda", "wang_fair_value", "wang_edge"):
        assert key in payload and payload[key] is not None, f"TRACK payload missing/None: {key!r}"
    # The Wang adjustment should have moved fair value away from the brain's
    # raw probability — that's the whole point of routing through PricingEngine.
    assert payload["fair"] != pytest.approx(payload["raw_probability"])

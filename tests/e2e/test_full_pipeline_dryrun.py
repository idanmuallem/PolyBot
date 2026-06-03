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
from hunters.crypto import CryptoHunter
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
)


@pytest.mark.asyncio
async def test_single_pipeline_loop_dry_run():
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

    with patch("core.trading_config.TradingConfig.from_env", return_value=_DRY_CONFIG), \
         patch.object(TradeExecutor, "get_open_positions", return_value=[]), \
         patch.object(TradeExecutor, "get_balance", return_value=100.0), \
         patch.object(CryptoHunter, "hunt", return_value=sample_market), \
         patch.object(CryptoHunter, "get_live_truth", return_value=97_000.0):

        pipeline = SequentialTradingPipeline(bridge=bridge, log_func=spy_log, delay=0.01)

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

"""Integration: pipeline stages 2-4 (Evaluate → Risk → Execute) with mocked executor."""
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest
from freezegun import freeze_time

from core.bridge import DataBridge
from core.models import MarketData
from core.trading_config import TradingConfig


def _make_pipeline(config=None, balance=100.0):
    from trading.decision_pipeline import SequentialTradingPipeline

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

    log_calls = []

    with patch("trading.executor.TradingConfig.from_env", return_value=config), \
         patch("trading.decision_pipeline.TradingConfig.from_env", return_value=config), \
         patch("trading.executor.TradeExecutor.get_open_positions", return_value=[]), \
         patch("trading.executor.TradeExecutor.get_balance", return_value=balance):
        pipeline = SequentialTradingPipeline(
            bridge=bridge,
            log_func=lambda level, *a, **kw: log_calls.append(level),
            delay=0.01,
        )

    return pipeline, bridge, log_calls


def _market(price=0.40, strike=95_000.0, expiry_days=10, asset_type="Crypto::BTCUSDT"):
    from freezegun import freeze_time as _ft
    expiry = (datetime(2026, 6, 2, tzinfo=timezone.utc) + timedelta(days=expiry_days)).isoformat()
    return MarketData(
        market_id="tok1", asset_type=asset_type,
        strike_price=strike, question="Will BTC exceed $95,000?",
        market_name="Bitcoin — BTC price market",
        initial_price=price, volume=500_000.0,
        expiry_date=expiry, no_market_id="tok1_no",
    )


# ── Stage 2: _stage_evaluate_ev ──────────────────────────────────────────────

@freeze_time("2026-06-02T00:00:00+00:00")
def test_evaluate_picks_yes_side():
    """price=0.40, fair=0.70 → EV_YES=0.75 > EV_NO=-0.50 → side=YES."""
    pipeline, bridge, _ = _make_pipeline()
    market = _market(price=0.40)

    mock_hunter = MagicMock()
    mock_hunter.get_live_truth.return_value = 97_000.0

    with patch("brains.crypto.HybridCryptoBrain._calculate_probability", return_value=0.70):
        candidate = pipeline._stage_evaluate_ev(market, mock_hunter)

    assert candidate is not None
    assert candidate.side == "YES"
    assert candidate.final_ev == pytest.approx(0.75, rel=0.01)


@freeze_time("2026-06-02T00:00:00+00:00")
def test_evaluate_picks_no_side():
    """price=0.80, fair=0.30 → EV_NO=2.5 > EV_YES → side=NO."""
    pipeline, bridge, _ = _make_pipeline()
    market = _market(price=0.80)

    mock_hunter = MagicMock()
    mock_hunter.get_live_truth.return_value = 97_000.0

    with patch("brains.crypto.HybridCryptoBrain._calculate_probability", return_value=0.30):
        candidate = pipeline._stage_evaluate_ev(market, mock_hunter)

    assert candidate is not None
    assert candidate.side == "NO"
    # entry_price for NO side = 1 - 0.80 = 0.20
    assert candidate.entry_price == pytest.approx(0.20, abs=0.001)


@freeze_time("2026-06-02T00:00:00+00:00")
def test_evaluate_returns_none_on_missing_live_truth():
    pipeline, bridge, log_calls = _make_pipeline()
    market = _market(price=0.50)
    mock_hunter = MagicMock()
    mock_hunter.get_live_truth.return_value = None  # unavailable

    candidate = pipeline._stage_evaluate_ev(market, mock_hunter)
    assert candidate is None
    assert "SCAN-SKIP" in log_calls


# ── Stage 3: _stage_risk_and_budget ──────────────────────────────────────────

@freeze_time("2026-06-02T00:00:00+00:00")
def test_stage_risk_rejects_low_ev():
    pipeline, bridge, log_calls = _make_pipeline()
    market = _market(price=0.50)
    mock_hunter = MagicMock()
    mock_hunter.get_live_truth.return_value = 97_000.0

    # fair=0.55, price=0.50 → EV_YES=0.10 < 0.30 threshold
    with patch("brains.crypto.HybridCryptoBrain._calculate_probability", return_value=0.55):
        candidate = pipeline._stage_evaluate_ev(market, mock_hunter)

    # Either rejected in stage_evaluate (not tradable) or stage_risk
    if candidate is not None:
        approved_bet, risk_ctx = pipeline._stage_risk_and_budget(candidate)
        assert approved_bet == 0.0


@freeze_time("2026-06-02T00:00:00+00:00")
def test_stage_risk_rejects_entry_price_out_of_bounds():
    pipeline, bridge, log_calls = _make_pipeline()
    # price=0.90 → above PRICE_CEILING, filtered in _stage_evaluate_ev
    market = _market(price=0.90)
    mock_hunter = MagicMock()
    mock_hunter.get_live_truth.return_value = 97_000.0

    candidate = pipeline._stage_evaluate_ev(market, mock_hunter)
    # Should be filtered (price out of bounds) before even reaching stage_risk
    assert candidate is None
    assert "FILTERED" in log_calls


@freeze_time("2026-06-02T00:00:00+00:00")
def test_stage_risk_approves_valid_candidate():
    pipeline, bridge, _ = _make_pipeline(balance=100.0)
    market = _market(price=0.40)
    mock_hunter = MagicMock()
    mock_hunter.get_live_truth.return_value = 97_000.0

    with patch("brains.crypto.HybridCryptoBrain._calculate_probability", return_value=0.70):
        candidate = pipeline._stage_evaluate_ev(market, mock_hunter)

    assert candidate is not None
    approved_bet, risk_ctx = pipeline._stage_risk_and_budget(candidate)
    assert approved_bet > 0.0
    assert risk_ctx is not None


# ── Stage 4: _stage_execute ───────────────────────────────────────────────────

@freeze_time("2026-06-02T00:00:00+00:00")
def test_stage_execute_records_budget_on_success():
    pipeline, bridge, _ = _make_pipeline(balance=100.0)
    market = _market(price=0.40)
    mock_hunter = MagicMock()
    mock_hunter.get_live_truth.return_value = 97_000.0

    with patch("brains.crypto.HybridCryptoBrain._calculate_probability", return_value=0.70):
        candidate = pipeline._stage_evaluate_ev(market, mock_hunter)

    assert candidate is not None
    approved_bet, risk_ctx = pipeline._stage_risk_and_budget(candidate)
    assert approved_bet > 0.0

    pipeline._stage_execute(candidate, approved_bet, risk_ctx)
    assert pipeline.budget_manager.total_spent_today == pytest.approx(approved_bet, abs=0.01)

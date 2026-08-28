"""Integration: pipeline stages 2-4 (Evaluate → Risk → Execute) with mocked executor."""
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from freezegun import freeze_time

from brains.pricing_engine import wang_transform
from core.bridge import DataBridge
from core.models import MarketData
from core.trading_config import TradingConfig
from core.wallet_context import WalletContext
from polymarket import PolymarketScannerHunter
from trading.budget_manager import BudgetManager
from trading.executor import TradeExecutor
from trading.risk_manager import PortfolioManager


def _make_pipeline(config=None, balance=100.0):
    from trading.decision_pipeline import SequentialTradingPipeline, sync_live_account_state

    bridge = DataBridge()
    bridge.current_balance = balance
    bridge.current_portfolio = []
    bridge.open_position_value = 0.0
    bridge.open_positions_value = 0.0

    if config is None:
        # pricing_mode="legacy": these tests exercise generic pipeline
        # mechanics (side selection, risk gating, execution) using the
        # brain's raw fair value directly — see test_pipeline_wang_mode.py
        # for Wang-adjustment-specific coverage.
        config = TradingConfig(
            dry_run=True, min_ev=0.30, bankroll_usd=1000.0,
            daily_limit_usd=15.0, max_bet_size_usd=3.0,
            max_daily_trades=10, min_trading_balance=1.0,
            pricing_mode="legacy",
        )

    ctx = WalletContext(wallet_id="test_wallet", config=config, bridge=bridge, db_path="test_wallet_trades.db")

    log_calls = []
    log_func = lambda level, *a, **kw: log_calls.append(level)

    with patch("trading.executor.TradeExecutor.get_open_positions", return_value=[]), \
         patch("trading.executor.TradeExecutor.get_balance", return_value=balance):
        ctx.executor = TradeExecutor(config=ctx.config)
        # These tests exercise pipeline stage mechanics (side selection, risk
        # gating, execution) with MarketData fixtures that don't carry a
        # slug/condition_id — not real paper-engine fidelity. A real
        # PaperAdapter (constructed unconditionally by TradeExecutor.__init__
        # whenever pm_trader is importable) would now honestly report those
        # buys as failed (see trading/executor.py's dry-run branches), which
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


def _market(price=0.40, strike=95_000.0, expiry_days=10, asset_type="Crypto::BTCUSDT"):
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


# ── Stage 2: already-held guards (restart-proof real-portfolio check +
#    in-memory _simulated_positions backstop) ─────────────────────────────────

@freeze_time("2026-06-02T00:00:00+00:00")
def test_already_owned_check_catches_no_side_holding_after_restart():
    """Regression: a NO-side paper fill is recorded under the NO token's own
    id (no_market_id), not the market's YES-canonical token_id used here for
    hunting -- see PaperAdapter.execute_buy()'s `outcome == "no"` branch.
    Simulates a fresh process restart: _simulated_positions (the in-memory
    backstop) is empty, exactly as it is right after a container restart,
    so only the real-portfolio check (bridge.current_portfolio, restored
    from disk) can catch this."""
    pipeline, bridge, log_calls = _make_pipeline()
    market = _market(price=0.40)  # market_id="tok1", no_market_id="tok1_no"
    assert pipeline._simulated_positions == set()  # fresh-restart precondition

    bridge.current_portfolio = [SimpleNamespace(token_id="tok1_no")]

    candidate = pipeline._stage_evaluate_ev(market, MagicMock())

    assert candidate is None
    assert "SCAN-SKIP" in log_calls


@freeze_time("2026-06-02T00:00:00+00:00")
def test_already_owned_check_still_catches_yes_side_holding():
    """Regression: the original token_id match (YES-side holdings, where
    the market's own token_id IS what a paper/live fill is recorded under)
    must keep working unchanged."""
    pipeline, bridge, log_calls = _make_pipeline()
    market = _market(price=0.40)  # market_id="tok1"
    bridge.current_portfolio = [SimpleNamespace(token_id="tok1")]

    candidate = pipeline._stage_evaluate_ev(market, MagicMock())

    assert candidate is None
    assert "SCAN-SKIP" in log_calls


@freeze_time("2026-06-02T00:00:00+00:00")
def test_already_owned_check_ignores_unrelated_holding():
    """A held position for a different market must not block evaluation."""
    pipeline, bridge, _ = _make_pipeline()
    market = _market(price=0.40)  # market_id="tok1", no_market_id="tok1_no"
    bridge.current_portfolio = [SimpleNamespace(token_id="unrelated_tok")]

    mock_hunter = MagicMock()
    mock_hunter.get_live_truth.return_value = 97_000.0

    with patch("brains.crypto.HybridCryptoBrain._calculate_probability", return_value=0.70):
        candidate = pipeline._stage_evaluate_ev(market, mock_hunter)

    assert candidate is not None


@freeze_time("2026-06-02T00:00:00+00:00")
def test_already_held_simulated_skip_still_works_during_normal_operation():
    """The in-memory _simulated_positions backstop (distinct from the
    real-portfolio check above) must still fire unchanged -- this is the
    same-session guard that catches a just-bought token before the next
    portfolio refresh reflects it. Not restart-related; must survive this
    change untouched."""
    pipeline, bridge, log_calls = _make_pipeline()
    market = _market(price=0.40)  # market_id="tok1"
    pipeline._simulated_positions.add("tok1")

    candidate = pipeline._stage_evaluate_ev(market, MagicMock())

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


# ── Wang pricing mode (PricingEngine wiring) ─────────────────────────────────

def _wang_pipeline(balance=100.0, wang_base_lambda=0.183, wang_min_edge=0.0, min_ev=0.30):
    # wang_min_edge defaults to 0.0 here: market blending (BaseBrain.evaluate())
    # shrinks edges relative to the old hierarchical-only Wang math, so most of
    # these tests (which check metadata propagation, not the min-edge gate
    # itself) pass an edge floor low enough not to incidentally trip it.
    # test_wang_mode_skips_market_below_min_edge exercises the gate directly.
    config = TradingConfig(
        dry_run=True, min_ev=min_ev, bankroll_usd=1000.0,
        daily_limit_usd=15.0, max_bet_size_usd=3.0,
        max_daily_trades=10, min_trading_balance=1.0,
        pricing_mode="wang", wang_base_lambda=wang_base_lambda, wang_min_edge=wang_min_edge,
    )
    return _make_pipeline(config=config, balance=balance)


@freeze_time("2026-06-02T00:00:00+00:00")
def test_wang_mode_populates_pricing_fields():
    pipeline, bridge, _ = _wang_pipeline()
    market = _market(price=0.30, expiry_days=10)
    mock_hunter = MagicMock()
    mock_hunter.get_live_truth.return_value = 97_000.0

    with patch("brains.crypto.HybridCryptoBrain._calculate_probability", return_value=0.70):
        candidate = pipeline._stage_evaluate_ev(market, mock_hunter)

    assert candidate is not None
    assert candidate.pricing_mode == "wang"
    assert candidate.pre_prob == pytest.approx(0.70)
    # post_prob is Wang-adjusted then market-blended, not the raw probability.
    assert candidate.post_prob != pytest.approx(0.70)

    expected_wang_fair = wang_transform(0.70, pipeline.config.wang_lambda)
    assert candidate.wang_fair_value == pytest.approx(expected_wang_fair)
    # post_prob (post market-blend) is distinct from wang_fair_value
    # (Wang-only, pre-blend) now that blending sits between the two.
    expected_blended = (
        pipeline.config.model_weight * expected_wang_fair
        + (1.0 - pipeline.config.model_weight) * 0.30
    )
    assert candidate.post_prob == pytest.approx(expected_blended)

    assert candidate.wang_lambda is not None
    assert candidate.wang_edge == pytest.approx(candidate.post_prob - 0.30)
    # EV is computed off the final (Wang + blended) post_prob, not the raw probability.
    assert candidate.final_ev == pytest.approx(candidate.post_prob / 0.30 - 1.0, rel=0.01)


@freeze_time("2026-06-02T00:00:00+00:00")
def test_wang_mode_skips_market_below_min_edge():
    pipeline, bridge, log_calls = _wang_pipeline(wang_min_edge=0.05)
    # Market price set exactly to the Wang-only fair value: blending a value
    # with itself is a no-op, so the final post_prob == market_price and
    # wang_edge == 0.0 exactly.
    wang_fair = wang_transform(0.70, pipeline.config.wang_lambda)
    market = _market(price=wang_fair, expiry_days=10)
    mock_hunter = MagicMock()
    mock_hunter.get_live_truth.return_value = 97_000.0

    with patch("brains.crypto.HybridCryptoBrain._calculate_probability", return_value=0.70):
        candidate = pipeline._stage_evaluate_ev(market, mock_hunter)

    assert candidate is None
    assert "SCAN-SKIP" in log_calls


@freeze_time("2026-06-02T00:00:00+00:00")
def test_legacy_mode_leaves_wang_fields_none():
    pipeline, bridge, _ = _make_pipeline()  # default _make_pipeline config is pricing_mode="legacy"
    market = _market(price=0.40)
    mock_hunter = MagicMock()
    mock_hunter.get_live_truth.return_value = 97_000.0

    with patch("brains.crypto.HybridCryptoBrain._calculate_probability", return_value=0.70):
        candidate = pipeline._stage_evaluate_ev(market, mock_hunter)

    assert candidate is not None
    assert candidate.pricing_mode == "legacy"
    assert candidate.pre_prob == pytest.approx(0.70)
    assert candidate.post_prob == pytest.approx(0.70)  # unchanged by any Wang adjustment
    assert candidate.wang_lambda is None
    assert candidate.wang_fair_value is None
    assert candidate.wang_edge is None


@freeze_time("2026-06-02T00:00:00+00:00")
def test_wang_mode_track_log_includes_pricing_fields():
    pipeline, bridge, log_calls = _wang_pipeline()
    market = _market(price=0.30, expiry_days=10)
    mock_hunter = MagicMock()
    mock_hunter.get_live_truth.return_value = 97_000.0

    captured = []
    pipeline.log_func = lambda level, *a, **kw: captured.append((level, a, kw))

    # raw=0.85 (rather than a coinflip-adjacent value): market blending
    # (BaseBrain.evaluate()) dampens edges enough that this needs to clear
    # min_ev=0.30 at price=0.30, the most favorable price PRICE_FLOOR allows.
    with patch("brains.crypto.HybridCryptoBrain._calculate_probability", return_value=0.85):
        candidate = pipeline._stage_evaluate_ev(market, mock_hunter)
    assert candidate is not None

    approved_bet, risk_ctx = pipeline._stage_risk_and_budget(candidate)
    assert approved_bet > 0.0
    pipeline._stage_execute(candidate, approved_bet, risk_ctx)

    track_calls = [c for c in captured if c[0] == "TRACK"]
    assert track_calls, "No TRACK entry found"
    payload = track_calls[0][1][2]
    for key in ("pre_prob", "wang_lambda", "wang_fair_value", "wang_edge", "pricing_mode"):
        assert key in payload, f"TRACK payload missing key: {key!r}"
    assert payload["pricing_mode"] == "wang"
    assert payload["pre_prob"] == pytest.approx(0.85)


# ── Phase 7: strategy_type / kelly_fraction_used / correlation_exposure ──────

@freeze_time("2026-06-02T00:00:00+00:00")
def test_model_driven_candidate_tagged_strategy_type_model():
    pipeline, bridge, _ = _wang_pipeline()
    market = _market(price=0.30, expiry_days=10)
    mock_hunter = MagicMock()
    mock_hunter.get_live_truth.return_value = 97_000.0

    with patch("brains.crypto.HybridCryptoBrain._calculate_probability", return_value=0.70):
        candidate = pipeline._stage_evaluate_ev(market, mock_hunter)

    assert candidate is not None
    assert candidate.strategy_type == "model"


@freeze_time("2026-06-02T00:00:00+00:00")
def test_candidate_kelly_fraction_used_matches_config():
    config = TradingConfig(
        dry_run=True, min_ev=0.30, bankroll_usd=1000.0,
        daily_limit_usd=15.0, max_bet_size_usd=3.0,
        max_daily_trades=10, min_trading_balance=1.0,
        pricing_mode="wang", kelly_fraction=0.10, wang_min_edge=0.0,
    )
    pipeline, bridge, _ = _make_pipeline(config=config)
    market = _market(price=0.30, expiry_days=10)
    mock_hunter = MagicMock()
    mock_hunter.get_live_truth.return_value = 97_000.0

    with patch("brains.crypto.HybridCryptoBrain._calculate_probability", return_value=0.70):
        candidate = pipeline._stage_evaluate_ev(market, mock_hunter)

    assert candidate is not None
    assert candidate.kelly_fraction_used == pytest.approx(0.10)


@freeze_time("2026-06-02T00:00:00+00:00")
def test_correlation_exposure_reflects_open_book():
    pipeline, bridge, _ = _wang_pipeline()
    # SimpleNamespace, not MagicMock: a bare Position-like object whose
    # missing attributes are genuinely absent (MagicMock auto-vivifies any
    # attribute access, which would short-circuit the opportunity_map lookup
    # this test is actually exercising).
    bridge.current_portfolio = [SimpleNamespace(token_id="existing_tok")]
    bridge.opportunity_map = {"existing_tok": {"asset_type": "Crypto::BTCUSDT"}}
    market = _market(price=0.30, expiry_days=10, asset_type="Crypto::BTCUSDT")
    mock_hunter = MagicMock()
    mock_hunter.get_live_truth.return_value = 97_000.0

    with patch("brains.crypto.HybridCryptoBrain._calculate_probability", return_value=0.70):
        candidate = pipeline._stage_evaluate_ev(market, mock_hunter)

    assert candidate is not None
    # Same asset already open -> fully correlated with the book.
    assert candidate.correlation_exposure == pytest.approx(1.0)


@freeze_time("2026-06-02T00:00:00+00:00")
def test_correlation_exposure_zero_with_empty_book():
    pipeline, bridge, _ = _wang_pipeline()
    market = _market(price=0.30, expiry_days=10)
    mock_hunter = MagicMock()
    mock_hunter.get_live_truth.return_value = 97_000.0

    with patch("brains.crypto.HybridCryptoBrain._calculate_probability", return_value=0.70):
        candidate = pipeline._stage_evaluate_ev(market, mock_hunter)

    assert candidate is not None
    assert candidate.correlation_exposure == 0.0


@freeze_time("2026-06-02T00:00:00+00:00")
def test_analytics_fields_present_in_track_log():
    pipeline, bridge, log_calls = _wang_pipeline()
    market = _market(price=0.30, expiry_days=10)
    mock_hunter = MagicMock()
    mock_hunter.get_live_truth.return_value = 97_000.0

    captured = []
    pipeline.log_func = lambda level, *a, **kw: captured.append((level, a, kw))

    # raw=0.85: see test_wang_mode_track_log_includes_pricing_fields for why
    # 0.70 no longer clears min_ev at PRICE_FLOOR once market blending applies.
    with patch("brains.crypto.HybridCryptoBrain._calculate_probability", return_value=0.85):
        candidate = pipeline._stage_evaluate_ev(market, mock_hunter)
    assert candidate is not None

    approved_bet, risk_ctx = pipeline._stage_risk_and_budget(candidate)
    pipeline._stage_execute(candidate, approved_bet, risk_ctx)

    track_calls = [c for c in captured if c[0] == "TRACK"]
    assert track_calls, "No TRACK entry found"
    payload = track_calls[0][1][2]
    for key in ("strategy_type", "kelly_fraction_used", "correlation_exposure"):
        assert key in payload, f"TRACK payload missing key: {key!r}"
    assert payload["strategy_type"] == "model"

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from freezegun import freeze_time

from core.bridge import DataBridge
from core.models import Position
from core.trading_config import TradingConfig
from core.wallet_context import WalletContext
from trading.gate import Ok
from trading.ledger import Ledger
from trading.risk_manager import PortfolioManager


def _make_pm(take_profit=0.20, stop_loss=-0.50, min_hold_ev=-0.10, ledger=None):
    bridge = DataBridge()
    bridge.current_balance = 50.0
    bridge.current_portfolio = []
    bridge.open_position_value = 0.0
    bridge.unrealized_pnl = 0.0
    config = TradingConfig(
        take_profit_pct=take_profit,
        stop_loss_pct=stop_loss,
        min_hold_ev=min_hold_ev,
        trading_mode="dry_run",
    )
    ctx = WalletContext(wallet_id="test_wallet", config=config, bridge=bridge, db_path="test_wallet_trades.db")
    executor = MagicMock()
    executor.get_open_positions.return_value = []
    pm = PortfolioManager(bridge=ctx.bridge, executor=executor, config=ctx.config,
                          db_path=ctx.db_path, ledger=ledger)
    return pm, bridge, executor


@pytest.fixture
def ledger(tmp_path):
    return Ledger(str(tmp_path / "trades.db"))


def _enter(ledger, token_id, side="YES", strategy="crypto", **details):
    """Record an entry fill for token_id in the Ledger with the given
    entry details (pre_prob, post_prob, wang_edge, asset_type, ...)."""
    ledger.record_entry(token_id=token_id, condition_id=f"cond-{token_id}", side=side,
                        shares=10.0, price=0.40, venue="paper", strategy=strategy, **details)


def _pos(pnl_ratio=0.0, live_ev=0.0, token_id="tok1", price=0.50, shares=10.0, value=5.0, side="YES"):
    return Position(
        market_id="mkt1",
        token_id=token_id,
        initial_price=0.40,
        current_price=price,
        shares=shares,
        value=value,
        pnl_ratio=pnl_ratio,
        side=side,
        live_ev=live_ev,
    )


# ── manage_portfolio: exit logic ──────────────────────────────────────────────

def test_take_profit_triggered():
    pm, bridge, executor = _make_pm()
    pos = _pos(pnl_ratio=0.25, token_id="take_tok")
    executor.get_open_positions.side_effect = [[pos], []]
    executor.sell_position.return_value = True

    log_levels = []
    pm.manage_portfolio(lambda level, *a, **kw: log_levels.append(level))

    executor.sell_position.assert_called_once()
    assert "TAKE-PROFIT" in log_levels


def test_take_profit_logs_initial_price_for_per_share_pnl():
    """The dashboard's Avg $/share (Closed) stat needs both entry
    (initial_price) and exit (price) in the logged payload."""
    pm, bridge, executor = _make_pm()
    pos = _pos(pnl_ratio=0.25, token_id="take_tok", price=0.50)  # initial_price=0.40 (fixture default)
    executor.get_open_positions.side_effect = [[pos], []]
    executor.sell_position.return_value = True

    payloads = []
    pm.manage_portfolio(lambda level, asset_type, token_id, payload: payloads.append((level, payload)))

    tp_payload = next(p for level, p in payloads if level == "TAKE-PROFIT")
    assert tp_payload["initial_price"] == pytest.approx(0.40)
    assert tp_payload["price"] == pytest.approx(0.50)


def test_stop_loss_triggered():
    pm, bridge, executor = _make_pm()
    pos = _pos(pnl_ratio=-0.60, token_id="stop_tok")
    executor.get_open_positions.side_effect = [[pos], []]
    executor.sell_position.return_value = True

    log_levels = []
    pm.manage_portfolio(lambda level, *a, **kw: log_levels.append(level))

    executor.sell_position.assert_called_once()
    assert "STOP-LOSS" in log_levels


def test_exit_position_stops_retrying_after_max_attempts():
    """A position whose sell keeps failing (e.g. its market resolved and
    vanished) must stop being retried every cycle after MAX_EXIT_ATTEMPTS
    failures — not hammer sell_position()/the log forever (a real incident
    retried one dead position ~2,000 times over 15+ hours with no backoff)."""
    pm, bridge, executor = _make_pm()
    pos = _pos(pnl_ratio=-1.0, token_id="dead_tok")
    executor.sell_position.return_value = False

    log_calls = []
    log_func = lambda level, asset_type, token_id, payload: log_calls.append(payload)

    for _ in range(pm.gate.MAX_EXIT_ATTEMPTS + 5):
        pm._exit_position(pos, "STOP-LOSS", -0.5, {"pnl_ratio": -1.0}, log_func)

    assert executor.sell_position.call_count == pm.gate.MAX_EXIT_ATTEMPTS
    assert len(log_calls) == pm.gate.MAX_EXIT_ATTEMPTS
    assert log_calls[-1]["gave_up_after_max_attempts"] is True
    assert "gave_up_after_max_attempts" not in log_calls[0]


def test_exit_position_retry_count_resets_after_a_successful_sell():
    pm, bridge, executor = _make_pm()
    pos = _pos(pnl_ratio=-1.0, token_id="flaky_tok")
    log_func = lambda *a, **kw: None

    executor.sell_position.return_value = False
    for _ in range(pm.gate.MAX_EXIT_ATTEMPTS - 1):
        pm._exit_position(pos, "STOP-LOSS", -0.5, {}, log_func)
    executor.sell_position.return_value = True
    pm._exit_position(pos, "STOP-LOSS", -0.5, {}, log_func)

    executor.sell_position.return_value = False
    for _ in range(pm.gate.MAX_EXIT_ATTEMPTS - 1):
        pm._exit_position(pos, "STOP-LOSS", -0.5, {}, log_func)
    assert isinstance(pm.gate.can_exit(pos), Ok)


def test_gave_up_in_live_mode_logs_expired_flag():
    """Live mode's gave-up signal (unsellable — see __init__'s comment on
    what a resolved/closed market looks like via the positions API) should
    surface as a visible EXPIRED row, not just silently stop retrying."""
    pm, bridge, executor = _make_pm()
    executor.dry_run = False  # live mode
    pos = _pos(pnl_ratio=-1.0, token_id="dead_tok")
    executor.sell_position.return_value = False

    log_calls = []
    log_func = lambda level, asset_type, token_id, payload: log_calls.append((level, payload))

    for _ in range(pm.gate.MAX_EXIT_ATTEMPTS):
        pm._exit_position(pos, "STOP-LOSS", -0.5, {"pnl_ratio": -1.0}, log_func)

    expired_calls = [p for level, p in log_calls if level == "EXPIRED"]
    assert len(expired_calls) == 1  # fires exactly once, on the attempt that hits the cap
    assert expired_calls[0]["sold"] is False
    assert expired_calls[0]["shares"] == pytest.approx(10.0)
    assert expired_calls[0]["initial_price"] == pytest.approx(0.40)


def test_gave_up_in_dry_run_mode_does_not_log_expired():
    """Paper mode's real resolutions are logged separately by
    PaperVenue.resolve_closed_markets(); a paper position that's merely
    unsellable (e.g. a broken token mapping — not a market resolution)
    should not be mislabeled EXPIRED here."""
    pm, bridge, executor = _make_pm()
    executor.dry_run = True
    pos = _pos(pnl_ratio=-1.0, token_id="dead_tok")
    executor.sell_position.return_value = False

    log_levels = []
    log_func = lambda level, *a, **kw: log_levels.append(level)

    for _ in range(pm.gate.MAX_EXIT_ATTEMPTS):
        pm._exit_position(pos, "STOP-LOSS", -0.5, {"pnl_ratio": -1.0}, log_func)

    assert "EXPIRED" not in log_levels


def test_successful_live_exit_never_logs_expired():
    pm, bridge, executor = _make_pm()
    executor.dry_run = False
    pos = _pos(pnl_ratio=-0.60, token_id="stop_tok")
    executor.sell_position.return_value = True

    log_levels = []
    log_func = lambda level, *a, **kw: log_levels.append(level)
    pm._exit_position(pos, "STOP-LOSS", -0.5, {"pnl_ratio": -0.60}, log_func)

    assert log_levels == ["STOP-LOSS"]


# ── Post-exit cooldown (prevents same-market buy/exit/re-buy churn) ──────────

@freeze_time("2026-06-02T00:00:00+00:00")
@pytest.mark.parametrize("level", ["EV-CONVERGENCE", "WANG-EDGE-DECAY", "STOP-LOSS"])
def test_loss_type_exit_sets_cooldown(level):
    pm, bridge, executor = _make_pm()
    pos = _pos(pnl_ratio=-0.30, token_id="churny_tok")
    executor.sell_position.return_value = True

    assert pm.gate.exit_cooldown_until("churny_tok") is None
    pm._exit_position(pos, level, -0.5, {}, lambda *a, **kw: None)

    cooldown_until = pm.gate.exit_cooldown_until("churny_tok")
    # frozen "now" (2026-06-02T00:00:00Z) plus the default 24h cooldown window
    assert cooldown_until == datetime(2026, 6, 3, 0, 0, tzinfo=timezone.utc)


@freeze_time("2026-06-02T00:00:00+00:00")
def test_take_profit_exit_does_not_set_cooldown():
    """A profitable exit means the model was right about the market --
    re-entering it afterward is a legitimate trade, not churn."""
    pm, bridge, executor = _make_pm()
    pos = _pos(pnl_ratio=0.25, token_id="tp_tok")
    executor.sell_position.return_value = True

    pm._exit_position(pos, "TAKE-PROFIT", 0.20, {}, lambda *a, **kw: None)

    assert pm.gate.exit_cooldown_until("tp_tok") is None


@freeze_time("2026-06-02T00:00:00+00:00")
def test_failed_sell_does_not_set_cooldown():
    """Nothing actually closed -- the position is still open, so a cooldown
    (which is about blocking re-entry after a close) doesn't apply."""
    pm, bridge, executor = _make_pm()
    pos = _pos(pnl_ratio=-0.60, token_id="unsellable_tok")
    executor.sell_position.return_value = False

    pm._exit_position(pos, "STOP-LOSS", -0.5, {}, lambda *a, **kw: None)

    assert pm.gate.exit_cooldown_until("unsellable_tok") is None


def test_exit_cooldown_expires_and_is_pruned():
    pm, bridge, executor = _make_pm()
    pos = _pos(pnl_ratio=-0.30, token_id="churny_tok")
    executor.sell_position.return_value = True

    with freeze_time("2026-06-02T00:00:00+00:00"):
        pm._exit_position(pos, "EV-CONVERGENCE", -0.10, {}, lambda *a, **kw: None)
        assert pm.gate.exit_cooldown_until("churny_tok") is not None

    # Just under the 24h window: still blocked.
    with freeze_time("2026-06-02T23:59:00+00:00"):
        assert pm.gate.exit_cooldown_until("churny_tok") is not None

    # Past the 24h window: expired.
    with freeze_time("2026-06-03T01:00:00+00:00"):
        assert pm.gate.exit_cooldown_until("churny_tok") is None


def test_ev_convergence_triggered():
    pm, bridge, executor = _make_pm()
    # pnl within bounds but live_ev below min_hold_ev
    pos = _pos(pnl_ratio=0.05, live_ev=-0.15, token_id="ev_tok")
    executor.get_open_positions.side_effect = [[pos], []]
    executor.sell_position.return_value = True

    log_levels = []
    pm.manage_portfolio(lambda level, *a, **kw: log_levels.append(level))

    executor.sell_position.assert_called_once()
    assert "EV-CONVERGENCE" in log_levels


def test_no_exit_within_bounds():
    pm, bridge, executor = _make_pm()
    pos = _pos(pnl_ratio=0.10, live_ev=0.0, token_id="fine_tok")
    executor.get_open_positions.return_value = [pos]

    pm.manage_portfolio(lambda *a, **kw: None)

    executor.sell_position.assert_not_called()


# ── manage_portfolio: Wang edge decay exit ────────────────────────────────────

def test_wang_edge_decay_triggers_exit(ledger):
    pm, bridge, executor = _make_pm(ledger=ledger)
    # pre_prob=0.55 -> Wang fair_value (base_lambda=0.183) ≈ 0.6212;
    # pricing the position exactly at that fair value collapses the edge to 0.
    pos = _pos(pnl_ratio=0.05, live_ev=0.0, token_id="decayed_tok", price=0.6212104251426875)
    _enter(ledger, "decayed_tok", pre_prob=0.55)
    executor.get_open_positions.side_effect = [[pos], []]
    executor.sell_position.return_value = True

    log_levels = []
    pm.manage_portfolio(lambda level, *a, **kw: log_levels.append(level))

    executor.sell_position.assert_called_once()
    assert "WANG-EDGE-DECAY" in log_levels


def test_healthy_wang_edge_does_not_trigger_exit(ledger):
    pm, bridge, executor = _make_pm(ledger=ledger)
    # Same pre_prob, but priced well below fair value -> healthy edge.
    pos = _pos(pnl_ratio=0.05, live_ev=0.0, token_id="healthy_tok", price=0.35)
    _enter(ledger, "healthy_tok", pre_prob=0.55)
    executor.get_open_positions.return_value = [pos]

    log_levels = []
    pm.manage_portfolio(lambda level, *a, **kw: log_levels.append(level))

    executor.sell_position.assert_not_called()
    assert "WANG-EDGE-DECAY" not in log_levels


# ── manage_portfolio: ENABLE_ARBITRAGE gate ───────────────────────────────────

def test_arbitrage_position_skipped_entirely_when_disabled(ledger):
    """With ENABLE_ARBITRAGE=False, an arbitrage-tagged position must not be
    touched by manage_portfolio() at all -- not exited, not even stashed into
    position_analytics -- while a crypto position in the same book is still
    handled normally in the same loop."""
    pm, bridge, executor = _make_pm(ledger=ledger)
    pm.enable_arbitrage = False

    arb_pos = _pos(pnl_ratio=0.99, token_id="arb_tok")        # would hit TAKE-PROFIT if processed
    crypto_pos = _pos(pnl_ratio=0.25, token_id="crypto_tok")  # genuinely hits TAKE-PROFIT
    _enter(ledger, "arb_tok", strategy="arbitrage", asset_type="Arbitrage::EventSum")
    _enter(ledger, "crypto_tok", asset_type="Crypto::BTCUSDT")
    executor.get_open_positions.side_effect = [[arb_pos, crypto_pos], []]
    executor.sell_position.return_value = True

    log_calls = []
    pm.manage_portfolio(lambda level, asset_type, token_id, payload: log_calls.append((level, token_id)))

    # Zero calls into the exit path for the arbitrage leg -- only the crypto
    # position was ever handed to sell_position.
    executor.sell_position.assert_called_once()
    assert executor.sell_position.call_args[0][0] == "crypto_tok"

    assert not any(token_id == "arb_tok" for _level, token_id in log_calls)
    assert any(token_id == "crypto_tok" for _level, token_id in log_calls)
    assert "arb_tok" not in bridge.position_analytics
    assert "crypto_tok" in bridge.position_analytics


def test_arbitrage_position_processed_normally_when_enabled(ledger):
    """Sanity check: the gate only fires when the flag is off -- the same
    arbitrage-tagged position is exited normally when ENABLE_ARBITRAGE is on."""
    pm, bridge, executor = _make_pm(ledger=ledger)
    assert pm.enable_arbitrage is True

    arb_pos = _pos(pnl_ratio=0.99, token_id="arb_tok")
    _enter(ledger, "arb_tok", strategy="arbitrage", asset_type="Arbitrage::EventSum")
    executor.get_open_positions.side_effect = [[arb_pos], []]
    executor.sell_position.return_value = True

    log_levels = []
    pm.manage_portfolio(lambda level, *a, **kw: log_levels.append(level))

    executor.sell_position.assert_called_once()
    assert "TAKE-PROFIT" in log_levels


def test_wang_edge_decay_skipped_without_raw_probability():
    # No Ledger holding -> nothing to re-evaluate;
    # falls through to the existing P&L-based checks untouched.
    pm, bridge, executor = _make_pm()
    pos = _pos(pnl_ratio=0.10, live_ev=0.0, token_id="unknown_tok")
    executor.get_open_positions.return_value = [pos]

    log_levels = []
    pm.manage_portfolio(lambda level, *a, **kw: log_levels.append(level))

    executor.sell_position.assert_not_called()
    assert "WANG-EDGE-DECAY" not in log_levels


def test_take_profit_still_wins_over_decayed_edge(ledger):
    # Hard limits are checked first: a position hitting take-profit exits as
    # TAKE-PROFIT even though its Wang edge has also collapsed to zero.
    pm, bridge, executor = _make_pm(take_profit=0.20, ledger=ledger)
    pos = _pos(pnl_ratio=0.25, live_ev=0.0, token_id="tp_tok", price=0.6212104251426875)
    _enter(ledger, "tp_tok", pre_prob=0.55)
    executor.get_open_positions.side_effect = [[pos], []]
    executor.sell_position.return_value = True

    log_levels = []
    pm.manage_portfolio(lambda level, *a, **kw: log_levels.append(level))

    executor.sell_position.assert_called_once()
    assert "TAKE-PROFIT" in log_levels
    assert "WANG-EDGE-DECAY" not in log_levels


def test_wang_edge_decay_translates_no_side_probability(ledger):
    # pre_prob is always recorded in YES-space; for a NO position the
    # "probability my side wins" is 1 - pre_prob. p=0.30 (YES) ->
    # 0.70 (NO) -> Wang fair_value ≈ 0.7603 for that translated probability;
    # pricing the NO leg exactly there collapses its edge to 0.
    pm, bridge, executor = _make_pm(ledger=ledger)
    pos = _pos(pnl_ratio=0.05, live_ev=0.0, token_id="no_tok",
               price=0.7603411908017963, side="NO")
    _enter(ledger, "no_tok", side="NO", pre_prob=0.30)
    executor.get_open_positions.side_effect = [[pos], []]
    executor.sell_position.return_value = True

    log_levels = []
    pm.manage_portfolio(lambda level, *a, **kw: log_levels.append(level))

    executor.sell_position.assert_called_once()
    assert "WANG-EDGE-DECAY" in log_levels


def test_wang_decay_threshold_is_half_wang_min_edge():
    pm, _, _ = _make_pm()
    assert pm.wang_decay_threshold == pytest.approx(pm.wang_min_edge / 2.0)


# ── Correlation exposure (trading/correlation.py wiring) ─────────────────────

def test_get_book_asset_types_resolves_via_ledger(ledger):
    pm, bridge, executor = _make_pm(ledger=ledger)
    pos_a = _pos(token_id="tokA", price=0.5)
    pos_b = _pos(token_id="tokB", price=0.5)
    _enter(ledger, "tokA", asset_type="Crypto::BTCUSDT")
    _enter(ledger, "tokB", asset_type="Weather::Miami")
    executor.get_open_positions.return_value = [pos_a, pos_b]

    pm._refresh_portfolio()
    asset_types = pm.get_book_asset_types()

    assert set(asset_types) == {"Crypto::BTCUSDT", "Weather::Miami"}


def test_correlation_exposure_for_averages_against_book(ledger):
    pm, bridge, executor = _make_pm(ledger=ledger)
    pos = _pos(token_id="tokA", price=0.5)
    _enter(ledger, "tokA", asset_type="Crypto::BTCUSDT")
    executor.get_open_positions.return_value = [pos]
    pm._refresh_portfolio()

    # Same market already open -> exposure to another BTC candidate is 1.0.
    assert pm.correlation_exposure_for("Crypto::BTCUSDT") == 1.0
    # Unrelated category -> 0.0.
    assert pm.correlation_exposure_for("Weather::Miami") == 0.0


def test_correlation_exposure_zero_with_no_open_positions():
    pm, bridge, executor = _make_pm()
    executor.get_open_positions.return_value = []
    pm._refresh_portfolio()

    assert pm.correlation_exposure_for("Crypto::BTCUSDT") == 0.0


# ── Position analytics snapshot (dashboard edge-decay tracking) ──────────────

def test_manage_portfolio_stashes_position_analytics(ledger):
    pm, bridge, executor = _make_pm(ledger=ledger)
    pos = _pos(pnl_ratio=0.05, token_id="analytics_tok", price=0.40)
    _enter(ledger, "analytics_tok", pre_prob=0.50, wang_edge=0.10, asset_type="Crypto::BTCUSDT")
    executor.get_open_positions.return_value = [pos]

    pm.manage_portfolio(lambda *a, **kw: None)

    snapshot = bridge.position_analytics["analytics_tok"]
    assert snapshot["asset_type"] == "Crypto::BTCUSDT"
    assert snapshot["pre_prob"] == pytest.approx(0.50)
    assert snapshot["entry_wang_edge"] == pytest.approx(0.10)
    assert snapshot["current_wang_edge"] is not None
    assert snapshot["edge_delta"] == pytest.approx(snapshot["current_wang_edge"] - 0.10)


def test_position_analytics_none_without_raw_probability():
    pm, bridge, executor = _make_pm()
    pos = _pos(pnl_ratio=0.05, token_id="unknown_tok", price=0.40)
    executor.get_open_positions.return_value = [pos]

    pm.manage_portfolio(lambda *a, **kw: None)

    snapshot = bridge.position_analytics["unknown_tok"]
    assert snapshot["current_wang_edge"] is None
    assert snapshot["edge_delta"] is None


# ── optimize_for_candidate ────────────────────────────────────────────────────

def test_cull_weak_position(ledger):
    pm, bridge, executor = _make_pm(ledger=ledger)
    # Position: post_prob=0.10, current_price=0.50 → live_ev=-0.80
    # Candidate: ev=0.55; 0.55 >= -0.80 + 0.10 → cull ✓
    pos = _pos(pnl_ratio=0.0, token_id="weak_tok", price=0.50, shares=10.0, value=5.0)
    _enter(ledger, "weak_tok", post_prob=0.10)
    executor.get_open_positions.return_value = [pos]
    executor.sell_position.return_value = True

    freed = pm.optimize_for_candidate(
        new_candidate_ev=0.55, min_improvement=0.10, log_func=lambda *a, **kw: None
    )

    assert freed > 0
    executor.sell_position.assert_called_once()


def test_no_cull_competitive_position(ledger):
    pm, bridge, executor = _make_pm(ledger=ledger)
    # Position: post_prob=0.80, current_price=0.50 → live_ev=0.60
    # Candidate: ev=0.55; 0.55 < 0.60 + 0.10=0.70 → don't cull ✓
    pos = _pos(pnl_ratio=0.0, token_id="comp_tok", price=0.50, shares=10.0, value=5.0)
    _enter(ledger, "comp_tok", post_prob=0.80)
    executor.get_open_positions.return_value = [pos]

    freed = pm.optimize_for_candidate(
        new_candidate_ev=0.55, min_improvement=0.10, log_func=lambda *a, **kw: None
    )

    assert freed == 0.0
    executor.sell_position.assert_not_called()


# ── free_up_capital ───────────────────────────────────────────────────────────

def test_free_up_capital_sells_weakest_first():
    pm, bridge, executor = _make_pm()
    bridge.current_balance = 0.0  # force liquidation
    pos_strong = _pos(pnl_ratio=-0.10, token_id="strong", price=0.50, shares=10.0, value=5.0)
    pos_weak = _pos(pnl_ratio=-0.30, token_id="weak", price=0.50, shares=10.0, value=5.0)
    executor.get_open_positions.return_value = [pos_strong, pos_weak]
    executor.sell_position.return_value = True

    sold_tokens = []
    pm.free_up_capital(
        required_amount=4.0,
        log_func=lambda level, asset, token, *a, **kw: sold_tokens.append(token),
    )

    # Weakest position (pnl_ratio=-0.30) is sold first
    assert sold_tokens[0] == "weak"


# ── Every exit path goes through the Gate ────────────────────────────────────

def test_dead_position_retry_cap_spans_cull_swap_and_rule_exits(ledger):
    """Regression (98748ae), extended to the cull and swap paths: a position
    that can never sell is tried at most MAX_EXIT_ATTEMPTS times in total,
    whichever exit path makes the attempt. Cull and swap used to bypass the
    cap and retry it forever."""
    pm, bridge, executor = _make_pm(ledger=ledger)
    _enter(ledger, "dead_tok", post_prob=0.50)
    executor.get_open_positions.return_value = [_pos(pnl_ratio=-1.0, token_id="dead_tok")]
    executor.sell_position.return_value = False
    noop = lambda *a, **kw: None

    for _ in range(3):
        pm.optimize_for_candidate(new_candidate_ev=1.0, log_func=noop)  # portfolio cull
        pm.free_up_capital(1_000.0, noop)                                # opportunity swap
        pm.manage_portfolio(noop)                                        # stop-loss

    assert executor.sell_position.call_count == 5
    assert pm.gate.can_exit(_pos(token_id="dead_tok")).reason == "exit_attempts_exhausted"


@freeze_time("2026-06-02T00:00:00+00:00")
@pytest.mark.parametrize("path", ["cull", "swap"])
def test_cull_and_swap_sales_start_the_post_exit_cooldown(ledger, path):
    pm, bridge, executor = _make_pm(ledger=ledger)
    _enter(ledger, "weak_tok", post_prob=0.50)
    executor.get_open_positions.return_value = [_pos(pnl_ratio=-0.10, token_id="weak_tok")]
    executor.sell_position.return_value = True

    if path == "cull":
        pm.optimize_for_candidate(new_candidate_ev=1.0, log_func=lambda *a: None)
    else:
        pm.free_up_capital(1_000.0, lambda *a: None)

    assert pm.gate.exit_cooldown_until("weak_tok") == datetime(2026, 6, 3, tzinfo=timezone.utc)

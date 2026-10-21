"""Account: the engine's single writer of cash. It refreshes from the active
Venue (+ the Ledger), owns equity, peak equity and the drawdown circuit
breaker, and publishes one snapshot to the bridge."""
import pytest

from core.bridge import DataBridge
from trading.account import Account
from trading.fake_venue import FakeVenue
from trading.ledger import Ledger
from trading.venue import MarketRef

BTC = MarketRef(yes_token_id="btc-yes", no_token_id="btc-no", condition_id="0xbtc", slug="btc-100k")


@pytest.fixture
def ledger(tmp_path):
    return Ledger(str(tmp_path / "trades.db"))


def _account(venue, ledger=None, bridge=None, max_drawdown_pct=0.20, log=None):
    return Account(
        active_venue=lambda: venue,
        bridge=bridge if bridge is not None else DataBridge(),
        ledger=ledger,
        max_drawdown_pct=max_drawdown_pct,
        log_func=(lambda level, *a: log.append(level)) if log is not None else None,
    )


def test_refresh_publishes_venue_cash_and_positions_to_the_bridge(ledger):
    venue = FakeVenue(cash=100.0)
    venue.list_market(BTC, yes_price=0.40)
    venue.buy(BTC, "YES", 20.0)               # 50 shares at 0.40
    ledger.record_entry(token_id="btc-yes", condition_id="0xbtc", side="YES", shares=50.0,
                        price=0.40, venue="paper", strategy="crypto")
    venue.set_price(BTC, "YES", 0.50)         # marked up: 50 x 0.50 = 25
    bridge = DataBridge()

    snap = _account(venue, ledger, bridge).refresh()

    assert snap.cash == pytest.approx(80.0)
    assert snap.open_position_value == pytest.approx(25.0)
    assert snap.equity == pytest.approx(105.0)
    assert snap.unrealized_pnl == pytest.approx(5.0)   # Ledger entry 0.40 marked to 0.50
    assert bridge.current_balance == pytest.approx(80.0)
    assert bridge.open_position_value == pytest.approx(25.0)
    assert bridge.unrealized_pnl == pytest.approx(5.0)


# ── Drawdown circuit breaker (carried over from the pipeline) ────────────────

def _book_worth(venue, yes_price):
    """Mark the $50 / 100-share BTC YES holding: equity = 50 + 100 x price."""
    venue.set_price(BTC, "YES", yes_price)


def _holding_venue():
    """$100 account: $50 cash plus 100 BTC YES shares bought at 0.50."""
    venue = FakeVenue(cash=100.0)
    venue.list_market(BTC, yes_price=0.50)
    venue.buy(BTC, "YES", 50.0)
    return venue


def test_breaker_pauses_entries_past_the_drawdown_limit():
    venue, log = _holding_venue(), []
    account = _account(venue, max_drawdown_pct=0.20, log=log)
    account.refresh()

    _book_worth(venue, 0.25)                  # equity 75: 25% down from the $100 peak
    snap = account.refresh()

    assert account.drawdown_paused is True
    assert snap.peak_equity == pytest.approx(100.0)
    assert snap.drawdown_pct == pytest.approx(0.25)
    assert "CIRCUIT-BREAKER" in log


def test_no_pause_within_the_drawdown_limit():
    venue, log = _holding_venue(), []
    account = _account(venue, max_drawdown_pct=0.20, log=log)
    account.refresh()

    _book_worth(venue, 0.40)                  # equity 90: 10% down
    account.refresh()

    assert account.drawdown_paused is False
    assert "CIRCUIT-BREAKER" not in log


def test_pause_clears_once_equity_recovers():
    venue, log = _holding_venue(), []
    account = _account(venue, max_drawdown_pct=0.20, log=log)
    account.refresh()
    _book_worth(venue, 0.25)
    account.refresh()

    _book_worth(venue, 0.50)
    account.refresh()

    assert account.drawdown_paused is False
    assert "CIRCUIT-BREAKER-CLEAR" in log


def test_peak_equity_ratchets_up_with_new_highs():
    venue = _holding_venue()
    account = _account(venue, max_drawdown_pct=0.20)
    _book_worth(venue, 1.00)                  # equity 150: new peak
    account.refresh()

    _book_worth(venue, 0.75)                  # equity 125: only ~16.7% off the $150 peak
    snap = account.refresh()

    assert snap.peak_equity == pytest.approx(150.0)
    assert account.drawdown_paused is False


def test_breaker_logs_only_on_the_transition():
    venue, log = _holding_venue(), []
    account = _account(venue, max_drawdown_pct=0.20, log=log)
    account.refresh()
    _book_worth(venue, 0.25)

    account.refresh()
    account.refresh()

    assert log.count("CIRCUIT-BREAKER") == 1


def test_bridge_shows_the_breaker_state_the_account_uses():
    venue, bridge = _holding_venue(), DataBridge()
    account = _account(venue, bridge=bridge, max_drawdown_pct=0.20)
    account.refresh()
    _book_worth(venue, 0.30)                  # equity 80: exactly 20% down

    account.refresh()

    assert bridge.drawdown_paused is True
    assert bridge.peak_equity == pytest.approx(100.0)
    assert bridge.current_drawdown_pct == pytest.approx(0.20)


# ── Balance flicker / single cash writer ─────────────────────────────────────

def test_bridge_cash_is_the_venue_cash_after_every_refresh(ledger):
    """Regression (balance flicker): independent writers (budget sync off a
    stale startup base, optimistic post-trade/post-sale adjustments, the
    venue sync) each wrote cash and the last one won, so the displayed
    balance bounced between values. Now only the Account writes it, always
    from the Venue."""
    venue, bridge = FakeVenue(cash=100.0), DataBridge()
    venue.list_market(BTC, yes_price=0.50)
    account = _account(venue, ledger, bridge)
    account.refresh()

    venue.buy(BTC, "YES", 10.0)
    account.record_fill("crypto", 10.0)
    assert bridge.current_balance == pytest.approx(100.0)   # budget bookkeeping never touches cash

    account.refresh()
    assert bridge.current_balance == pytest.approx(90.0)
    venue.sell("btc-yes", 20.0)
    account.refresh()
    account.refresh()
    assert bridge.current_balance == pytest.approx(100.0)


def test_failed_venue_read_keeps_the_last_known_cash():
    class FlakyVenue(FakeVenue):
        down = False

        def cash(self):
            if self.down:
                raise ConnectionError("collateral endpoint unreachable")
            return super().cash()

    venue, log = FlakyVenue(cash=100.0), []
    account = _account(venue, log=log)
    account.refresh()

    venue.down = True
    snap = account.refresh()

    assert snap.cash == pytest.approx(100.0)
    assert account.drawdown_paused is False
    assert "SYNC-WARN" in log


def test_refresh_follows_the_active_venue():
    """Reads go to whichever Venue is active at refresh time (paper <-> live
    switching swaps it between ticks)."""
    paper, live = FakeVenue(cash=100.0), FakeVenue(cash=40.0)
    active = {"venue": paper}
    account = Account(active_venue=lambda: active["venue"], bridge=DataBridge())
    assert account.refresh().cash == pytest.approx(100.0)

    active["venue"] = live

    assert account.refresh().cash == pytest.approx(40.0)


def test_switching_venue_starts_the_drawdown_baseline_from_the_new_venues_equity():
    """$1000 on paper, $50 on live: the switch is not a 95% drawdown."""
    paper, live = FakeVenue(cash=1000.0), FakeVenue(cash=50.0)
    live.name = "live"
    active = {"venue": paper}
    log = []
    account = Account(active_venue=lambda: active["venue"], bridge=DataBridge(),
                      log_func=lambda level, *a: log.append(level))
    account.refresh()

    active["venue"] = live
    snap = account.refresh()

    assert snap.peak_equity == pytest.approx(50.0)
    assert snap.drawdown_pct == 0.0
    assert snap.drawdown_paused is False
    assert "CIRCUIT-BREAKER" not in log

    live._cash = 30.0                     # a real 40% drawdown on live still trips it
    assert account.refresh().drawdown_paused is True

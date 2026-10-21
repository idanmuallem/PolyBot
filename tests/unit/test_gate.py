"""Gate.can_enter: one entry-eligibility check for model markets and
arbitrage baskets, backed by Ledger holdings and the post-exit cooldown."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from freezegun import freeze_time

from core.models import MarketData
from trading.gate import Gate, Ok, Skip
from trading.ledger import Ledger


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "trades.db")


@pytest.fixture
def ledger(db_path):
    return Ledger(db_path)


def _market(yes="tok1", no="tok1_no", condition_id="0xc1"):
    return MarketData(
        market_id=yes, asset_type="Crypto::BTCUSDT", strike_price=95_000.0,
        question="Will BTC exceed $95,000?", market_name="BTC market",
        initial_price=0.40, volume=1.0, no_market_id=no, condition_id=condition_id,
    )


def _buy(ledger, token_id, side, condition_id="0xc1", shares=10.0, price=0.40, venue="paper", **kw):
    ledger.record_entry(
        token_id=token_id, condition_id=condition_id, side=side, shares=shares,
        price=price, venue=venue, strategy=kw.pop("strategy", "crypto"), **kw,
    )


# ── Model markets ────────────────────────────────────────────────────────────

def test_unheld_market_may_be_entered(ledger):
    assert isinstance(Gate(ledger).can_enter(_market()), Ok)


def test_refuses_market_whose_no_side_is_held(ledger):
    """Regression (16bb257): a NO-side fill is held under the NO token, not
    the market's YES-canonical id, and used to slip past the held check."""
    _buy(ledger, "tok1_no", "NO")

    decision = Gate(ledger).can_enter(_market())

    assert isinstance(decision, Skip)
    assert decision.reason == "already_owned_in_portfolio"


def test_refuses_market_whose_yes_side_is_held(ledger):
    _buy(ledger, "tok1", "YES")
    assert Gate(ledger).can_enter(_market()).reason == "already_owned_in_portfolio"


def test_unrelated_holding_does_not_block(ledger):
    _buy(ledger, "other_tok", "YES", condition_id="0xother")
    assert isinstance(Gate(ledger).can_enter(_market()), Ok)


def test_already_held_survives_restart(db_path):
    """Regression: the in-memory held set reset on every restart while the
    holdings persisted, so a redeploy re-bought everything it held. A fresh
    Ledger and Gate over the same database still refuse."""
    _buy(Ledger(db_path), "tok1_no", "NO")

    decision = Gate(Ledger(db_path)).can_enter(_market())

    assert decision.reason == "already_owned_in_portfolio"


def test_refuses_backfilled_holding_with_no_condition_id(ledger):
    """Positions backfilled from hunt_history have condition_id=None, so a
    (condition_id, side) lookup misses them; the token match must not."""
    _buy(ledger, "tok1_no", "NO", condition_id=None)

    assert Gate(ledger).can_enter(_market()).reason == "already_owned_in_portfolio"


def test_refuses_by_condition_id_when_the_token_is_not_known(ledger):
    _buy(ledger, "some_no_token", "NO", condition_id="0xc1")

    assert Gate(ledger).can_enter(_market(no=None)).reason == "already_owned_in_portfolio"


def test_closed_position_no_longer_blocks(ledger):
    _buy(ledger, "tok1", "YES")
    ledger.record_exit(token_id="tok1", shares=10.0, price=0.50, reason="TAKE-PROFIT", venue="paper")

    assert isinstance(Gate(ledger).can_enter(_market()), Ok)


def test_holdings_on_the_other_venue_do_not_block(ledger):
    _buy(ledger, "tok1", "YES", venue="paper")

    assert isinstance(Gate(ledger, venue_name_fn=lambda: "live").can_enter(_market()), Ok)
    assert isinstance(Gate(ledger, venue_name_fn=lambda: "paper").can_enter(_market()), Skip)


def test_no_ledger_holds_nothing():
    assert isinstance(Gate(None).can_enter(_market()), Ok)


# ── Post-exit cooldown ───────────────────────────────────────────────────────

def _sold(gate, token_id, reason="EV-CONVERGENCE"):
    gate.record_exit_attempt(SimpleNamespace(token_id=token_id), sold=True, reason=reason)


@freeze_time("2026-06-02T00:00:00+00:00")
def test_refuses_inside_the_post_exit_cooldown_on_either_token(ledger):
    for token in ("tok1", "tok1_no"):
        gate = Gate(ledger)
        _sold(gate, token)
        decision = gate.can_enter(_market())
        assert isinstance(decision, Skip)
        assert decision.reason == "exit_cooldown"
        assert decision.detail == {"until": str(datetime(2026, 6, 3, tzinfo=timezone.utc))}


@pytest.mark.parametrize("reason", [
    "STOP-LOSS", "WANG-EDGE-DECAY", "EV-CONVERGENCE", "PORTFOLIO-CULL", "OPPORTUNITY-SWAP",
])
def test_every_loss_type_exit_starts_a_24h_cooldown(ledger, reason):
    gate = Gate(ledger)
    with freeze_time("2026-06-02T00:00:00+00:00"):
        _sold(gate, "tok1", reason)
        assert gate.exit_cooldown_until("tok1") == datetime(2026, 6, 3, tzinfo=timezone.utc)
    with freeze_time("2026-06-02T23:59:00+00:00"):
        assert gate.can_enter(_market()).reason == "exit_cooldown"
    with freeze_time("2026-06-03T01:00:00+00:00"):
        assert gate.exit_cooldown_until("tok1") is None
        assert isinstance(gate.can_enter(_market()), Ok)


def test_take_profit_exit_starts_no_cooldown(ledger):
    """A profitable exit means the model was right about the market, so
    re-entering it is a legitimate trade, not churn."""
    gate = Gate(ledger)
    _sold(gate, "tok1", "TAKE-PROFIT")
    assert gate.exit_cooldown_until("tok1") is None


def test_failed_exit_starts_no_cooldown(ledger):
    gate = Gate(ledger)
    gate.record_exit_attempt(SimpleNamespace(token_id="tok1"), sold=False, reason="STOP-LOSS")
    assert gate.exit_cooldown_until("tok1") is None


def test_buy_exit_rebuy_churn_is_refused_until_the_cooldown_ends(ledger):
    """Regression (8e9b2d5): after a loss-type exit, an oscillating EV
    estimate re-bought the market it had just sold. The position is closed,
    so only the cooldown stands between exit and re-buy."""
    gate = Gate(ledger)

    with freeze_time("2026-06-02T00:00:00+00:00"):
        _buy(ledger, "tok1", "YES")
        assert gate.can_enter(_market()).reason == "already_owned_in_portfolio"

        ledger.record_exit(token_id="tok1", shares=10.0, price=0.30, reason="EV-CONVERGENCE", venue="paper")
        _sold(gate, "tok1")
        assert gate.can_enter(_market()).reason == "exit_cooldown"

    with freeze_time("2026-06-03T00:00:01+00:00"):  # window over
        assert isinstance(gate.can_enter(_market()), Ok)


# ── Arbitrage baskets ────────────────────────────────────────────────────────

def _leg(i, side="YES"):
    market = _market(yes=f"leg{i}", no=f"leg{i}_no", condition_id=f"0xleg{i}")
    return SimpleNamespace(market=market, side=side)


def test_unheld_basket_buys_every_leg(ledger):
    legs = [_leg(0), _leg(1), _leg(2)]

    decision = Gate(ledger).can_enter(legs)

    assert isinstance(decision, Ok)
    assert list(decision.legs) == legs
    assert decision.held == ()


def test_fully_held_basket_is_refused(ledger):
    """Regression: the in-memory filled-groups guard reset on restart and a
    redeploy re-bought every basket it held."""
    for i in range(3):
        _buy(ledger, f"leg{i}", "YES", condition_id=f"0xleg{i}", strategy="arbitrage")

    decision = Gate(ledger).can_enter([_leg(0), _leg(1), _leg(2)])

    assert decision.reason == "already_owned_in_portfolio"


def test_partly_held_basket_buys_only_the_missing_legs(ledger):
    legs = [_leg(0), _leg(1), _leg(2)]
    _buy(ledger, "leg0", "YES", condition_id=None, shares=2.0, price=0.30, strategy="arbitrage")

    decision = Gate(ledger).can_enter(legs)

    assert isinstance(decision, Ok)
    assert list(decision.legs) == legs[1:]
    [(held_leg, holding)] = decision.held
    assert held_leg is legs[0]
    assert (holding.shares, holding.entry_price) == (2.0, 0.30)


def test_basket_with_a_leg_held_on_the_opposite_side_is_refused(ledger):
    """Holding a leg market's other side means the basket can't be completed
    without holding both sides of one market."""
    _buy(ledger, "leg1_no", "NO", condition_id="0xleg1")

    decision = Gate(ledger).can_enter([_leg(0), _leg(1)])

    assert decision.reason == "already_owned_in_portfolio"


def test_basket_with_a_missing_leg_in_cooldown_is_refused(ledger):
    gate = Gate(ledger)
    _sold(gate, "leg1")

    decision = gate.can_enter([_leg(0), _leg(1)])

    assert decision.reason == "exit_cooldown"


# ── can_exit: exit attempt cap ───────────────────────────────────────────────

def _position(token_id="tok1"):
    return SimpleNamespace(token_id=token_id, shares=10.0, current_price=0.30, side="YES")


def test_dead_position_is_refused_after_five_failed_exits(ledger):
    """Regression (98748ae): a position whose market vanished can never
    sell, and was retried every tick (~2,000 times in one incident). After
    five failed attempts, whichever exit path made them, the Gate refuses."""
    gate = Gate(ledger)
    pos = _position("dead_tok")

    gave_up = [gate.record_exit_attempt(pos, sold=False, reason=reason)
               for reason in ("STOP-LOSS", "PORTFOLIO-CULL", "OPPORTUNITY-SWAP", "STOP-LOSS")]
    assert gave_up == [False] * 4
    assert isinstance(gate.can_exit(pos), Ok)

    assert gate.record_exit_attempt(pos, sold=False, reason="PORTFOLIO-CULL") is True
    decision = gate.can_exit(pos)
    assert isinstance(decision, Skip)
    assert decision.reason == "exit_attempts_exhausted"
    assert isinstance(gate.can_exit(_position("other_tok")), Ok)

"""The Gate: the single owner of "may I enter this market?".

`can_enter(market)` for a model entry and `can_enter(legs)` for an
arbitrage basket both answer Ok or Skip(reason), from the same two facts:

- holdings in the Ledger (so the answer survives restarts), on either side
  of the market, matched by YES/NO token and by (condition_id, side);
- the post-exit cooldown, read through `exit_cooldown_until(token)`.

`can_exit` (#22) belongs here too and should reuse the same cooldown source.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

from trading.venue import MarketRef, normalize_side


@dataclass(frozen=True)
class Ok:
    """Entry allowed. For a basket, `legs` are the legs still to buy and
    `held` pairs each already-held leg with its Ledger holding."""
    legs: tuple = ()
    held: tuple = ()

    def __bool__(self) -> bool:
        return True


@dataclass(frozen=True)
class Skip:
    reason: str
    detail: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return False


ALREADY_OWNED = "already_owned_in_portfolio"
EXIT_COOLDOWN = "exit_cooldown"


class Gate:
    def __init__(
        self,
        ledger,
        exit_cooldown_until: Optional[Callable[[str], Optional[datetime]]] = None,
        venue: Optional[Callable[[], str]] = None,
    ):
        """`ledger` may be None (nothing is held). `venue()` names the
        active venue ("paper" | "live"); when given, holdings recorded on the
        other venue don't count."""
        self._ledger = ledger
        self._cooldown_until = exit_cooldown_until or (lambda _token: None)
        self._venue = venue

    # ------------------------------------------------------------------

    def can_enter(self, target: Any) -> Ok | Skip:
        if isinstance(target, (list, tuple)):
            return self._can_enter_legs(list(target))
        return self._can_enter_market(target)

    def _can_enter_market(self, market) -> Ok | Skip:
        ref = MarketRef.from_market(market)
        holdings = self._Holdings(self._open_holdings())
        if holdings.on_market(ref) is not None:
            return Skip(ALREADY_OWNED)
        cooling = self._cooldown(ref)
        return cooling if cooling is not None else Ok()

    def _cooldown(self, ref: MarketRef) -> Optional[Skip]:
        """Skip if either of the market's tokens is inside its post-exit
        cooldown (a NO-side exit is recorded under the NO token)."""
        for token in (ref.yes_token_id, ref.no_token_id):
            until = self._cooldown_until(token) if token else None
            if until is not None:
                return Skip(EXIT_COOLDOWN, {"until": str(until)})
        return None

    def _can_enter_legs(self, legs: list) -> Ok | Skip:
        """A basket may be entered while at least one leg is missing. Legs
        already held on their own side are reported, not re-bought; a leg
        whose market is held on the other side, or is cooling down after an
        exit, refuses the whole basket (it could never be completed)."""
        holdings = self._Holdings(self._open_holdings())
        missing, held = [], []
        for leg in legs:
            ref = MarketRef.from_market(leg.market)
            side = normalize_side(leg.side)
            own = holdings.on_side(ref, side)
            if own is not None:
                held.append((leg, own))
                continue
            if holdings.on_side(ref, "NO" if side == "YES" else "YES") is not None:
                return Skip(ALREADY_OWNED)
            cooling = self._cooldown(ref)
            if cooling is not None:
                return cooling
            missing.append(leg)
        if legs and not missing:
            return Skip(ALREADY_OWNED)
        return Ok(legs=tuple(missing), held=tuple(held))

    # ------------------------------------------------------------------

    def _open_holdings(self) -> list:
        if self._ledger is None:
            return []
        venue = self._venue() if self._venue is not None else None
        return self._ledger.open_positions(venue=venue)

    class _Holdings:
        """Open Ledger holdings indexed by token and by (condition_id, side).
        Backfilled holdings have no condition_id, so token matching is what
        finds them."""

        def __init__(self, holdings: list):
            self.by_token = {h.token_id: h for h in holdings}
            self.by_condition = {
                (h.condition_id, normalize_side(h.side)): h for h in holdings if h.condition_id
            }

        def on_side(self, ref: MarketRef, side: str):
            token = ref.token_for(side)
            if side == "NO" and not ref.no_token_id:
                token = None  # no NO token known: only condition_id can match
            held = self.by_token.get(token) if token else None
            if held is None and ref.condition_id:
                held = self.by_condition.get((ref.condition_id, side))
            return held

        def on_market(self, ref: MarketRef):
            return self.on_side(ref, "YES") or self.on_side(ref, "NO")

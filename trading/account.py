"""Account: the engine's single writer of cash and account state.

It refreshes from the active Venue (authoritative for cash and share
quantities) and the Ledger (entry prices, for unrealized P&L), owns equity,
peak equity and the drawdown circuit breaker, and publishes one snapshot to
the bridge. Nothing else in the engine writes cash.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class AccountSnapshot:
    cash: float = 0.0
    open_position_value: float = 0.0
    unrealized_pnl: float = 0.0
    peak_equity: float = 0.0
    drawdown_pct: float = 0.0
    drawdown_paused: bool = False

    @property
    def equity(self) -> float:
        return self.cash + self.open_position_value


class Account:
    def __init__(
        self,
        venue: Callable[[], object],
        bridge,
        ledger=None,
        max_drawdown_pct: float = 0.20,
        log_func: Optional[Callable] = None,
    ):
        self._venue = venue
        self._bridge = bridge
        self._ledger = ledger
        self._max_drawdown_pct = float(max_drawdown_pct)
        self._log = log_func or (lambda *a, **kw: None)
        self._snapshot = AccountSnapshot()

    # -- reads ----------------------------------------------------------------

    def snapshot(self) -> AccountSnapshot:
        return self._snapshot

    @property
    def cash(self) -> float:
        return self._snapshot.cash

    @property
    def equity(self) -> float:
        return self._snapshot.equity

    @property
    def drawdown_paused(self) -> bool:
        """True pauses new entries; exits keep running."""
        return self._snapshot.drawdown_paused

    # -- refresh --------------------------------------------------------------

    def refresh(self) -> AccountSnapshot:
        """Re-read cash and positions from the active Venue, recompute
        equity and the drawdown breaker, and publish the snapshot. A Venue
        read that fails keeps the last known value for that field."""
        prev = self._snapshot
        venue = self._venue()

        cash = prev.cash
        try:
            cash = float(venue.cash())
        except Exception as exc:
            self._log("SYNC-WARN", "Account", "balance", {"reason": "balance_fetch_failed", "error": str(exc)})

        position_value, unrealized = prev.open_position_value, prev.unrealized_pnl
        try:
            positions = list(venue.open_positions())
            position_value = sum(float(getattr(p, "value", 0.0) or 0.0) for p in positions)
            unrealized = self._unrealized_pnl(positions, getattr(venue, "name", None))
        except Exception as exc:
            self._log("SYNC-WARN", "Account", "portfolio", {"reason": "positions_fetch_failed", "error": str(exc)})

        # Drawdown circuit breaker. The peak-tracking/threshold pattern
        # mirrors oracle3's StandardRiskManager._check_drawdown()
        # (oracle3/risk/risk_manager.py) -
        # https://github.com/YichengYang-Ethan/oracle3, licensed under the
        # Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
        equity = cash + position_value
        peak = max(prev.peak_equity, equity)
        drawdown_pct = (peak - equity) / peak if peak > 0 else 0.0
        paused = drawdown_pct >= self._max_drawdown_pct

        self._snapshot = AccountSnapshot(
            cash=cash,
            open_position_value=position_value,
            unrealized_pnl=unrealized,
            peak_equity=peak,
            drawdown_pct=drawdown_pct,
            drawdown_paused=paused,
        )
        self._publish()
        self._log_breaker_transition(prev.drawdown_paused, self._snapshot)
        return self._snapshot

    def _log_breaker_transition(self, was_paused: bool, snap: AccountSnapshot) -> None:
        fields = {
            "peak_equity": round(snap.peak_equity, 4),
            "current_equity": round(snap.equity, 4),
            "drawdown_pct": round(snap.drawdown_pct, 4),
        }
        if snap.drawdown_paused and not was_paused:
            self._log("CIRCUIT-BREAKER", "Pipeline", "drawdown", {
                "reason": "max_drawdown_pct breached — new trade entry paused; "
                          "position management/exits still run",
                **fields,
                "max_drawdown_pct": self._max_drawdown_pct,
            })
        elif was_paused and not snap.drawdown_paused:
            self._log("CIRCUIT-BREAKER-CLEAR", "Pipeline", "drawdown", {
                "reason": "equity recovered above max_drawdown_pct — new trade entry resumed",
                **fields,
            })

    def _unrealized_pnl(self, positions, venue_name: Optional[str]) -> float:
        """Ledger entry prices marked to the Venue's current prices."""
        if self._ledger is None:
            return 0.0
        marks = {str(p.token_id): float(p.current_price or 0.0) for p in positions}
        try:
            return float(self._ledger.unrealized_pnl(marks, venue=venue_name))
        except Exception as exc:
            print(f"[ACCOUNT] Unrealized P&L unavailable: {exc}")
            return 0.0

    def _publish(self) -> None:
        snap, bridge = self._snapshot, self._bridge
        bridge.current_balance = snap.cash
        bridge.open_position_value = snap.open_position_value
        bridge.unrealized_pnl = snap.unrealized_pnl
        bridge.peak_equity = snap.peak_equity
        bridge.current_drawdown_pct = snap.drawdown_pct
        bridge.drawdown_paused = snap.drawdown_paused

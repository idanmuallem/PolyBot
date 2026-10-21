"""Account: the engine's single writer of cash and account state.

It refreshes from the active Venue (authoritative for cash and share
quantities) and the Ledger (entry prices, for unrealized P&L), owns equity,
peak equity and the drawdown circuit breaker, and publishes one snapshot to
the bridge. Nothing else in the engine writes cash.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Callable, Optional

from core.trading_config import TradingConfig
from trading.venue import PositionsUnavailable

logger = logging.getLogger(__name__)


def _utc(now: datetime) -> datetime:
    """*now* in UTC; a naive datetime is taken to be UTC already."""
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


@dataclass(frozen=True)
class AccountSnapshot:
    cash: float = 0.0
    open_position_value: float = 0.0
    unrealized_pnl: float = 0.0
    peak_equity: float = 0.0
    drawdown_pct: float = 0.0
    drawdown_paused: bool = False
    spent_today: float = 0.0
    watch_only: bool = False

    @property
    def equity(self) -> float:
        return self.cash + self.open_position_value


@dataclass(frozen=True)
class Approved:
    """reserve() said yes: spend up to *usd*."""
    usd: float

    def __bool__(self) -> bool:
        return True


@dataclass(frozen=True)
class Refused:
    """reserve() said no, and why."""
    reason: str
    detail: dict

    def __bool__(self) -> bool:
        return False


class Account:
    def __init__(
        self,
        active_venue: Callable[[], object],
        bridge,
        ledger=None,
        max_drawdown_pct: float = 0.20,
        log_func: Optional[Callable] = None,
        config: Optional[TradingConfig] = None,
    ):
        self._config = config if config is not None else TradingConfig()
        self._active_venue = active_venue
        self._bridge = bridge
        self._ledger = ledger
        self._max_drawdown_pct = float(max_drawdown_pct)
        self._log = log_func or (lambda *a, **kw: None)
        self._snapshot = AccountSnapshot()
        self._spent: dict[str, float] = {}
        self._trades: dict[str, int] = {}
        self._day = None  # UTC date today's spend and trade counts belong to
        self._peak_venue = None  # name of the Venue peak equity was measured on
        self._positions_unavailable = False  # an outage is being logged (see refresh())

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

    # -- budget ---------------------------------------------------------------

    def reserve(self, strategy: str, usd: float, trades: int = 1) -> "Approved | Refused":
        """May *strategy* spend *usd* on *trades* trades now (an arbitrage
        basket is one trade per leg)?

        Refused if the trades don't all fit in today's per-strategy or
        account-wide trade count, or nothing is left of today's budget.
        Otherwise Approved with the amount allowed: *usd* capped at
        max_bet_size_usd per trade and at what is left of the strategy's
        and the account's daily budget. Reserving holds nothing:
        record_fill() is what spends.
        """
        trades = int(trades)
        strategy_trades = self._trades.get(strategy, 0)
        if strategy_trades + trades > self._strategy_max_trades(strategy):
            return Refused("daily_trade_limit_would_be_exceeded", {
                "strategy": strategy, "trades": trades,
                "strategy_trades_today": strategy_trades,
                "strategy_max_daily_trades": self._strategy_max_trades(strategy),
            })
        # Same reasoning as the global spend ceiling: per-strategy trade
        # caps may sum past max_daily_trades, which TradingConfig only warns
        # about, so the account-wide count is a ceiling of its own.
        trades_today = sum(self._trades.values())
        global_max = int(self._config.max_daily_trades)
        if trades_today + trades > global_max:
            return Refused("global_daily_trade_limit_would_be_exceeded", {
                "strategy": strategy, "trades": trades,
                "trades_today": trades_today, "global_max_daily_trades": global_max,
            })

        remaining = self._remaining_budget(strategy)
        approved = min(float(usd), float(self._config.max_bet_size_usd) * trades, remaining)
        if approved <= 0:
            return Refused("daily_limit_reached", self._budget_detail(strategy))
        return Approved(usd=approved)

    def record_fill(self, strategy: str, usd: float) -> None:
        """One *strategy* entry filled for *usd*: it counts as one trade
        and *usd* of spend against today's limits."""
        self._spent[strategy] = self._spent.get(strategy, 0.0) + float(usd)
        self._trades[strategy] = self._trades.get(strategy, 0) + 1
        self._snapshot = replace(self._snapshot, spent_today=sum(self._spent.values()))
        self._publish()

    def daily_reset(self, now: datetime) -> bool:
        """Start the budget day *now* falls on (UTC): today's spend and
        trade counts are whatever the Ledger recorded for that day, so a
        restart resumes them and a new day starts from zero. True if it
        moved to a later day; the first call only sets the day."""
        day = _utc(now).date()
        if day == self._day:
            return False
        first = self._day is None
        self._day = day
        self._spent, self._trades = self._recorded_entries(day.isoformat())
        self._snapshot = replace(self._snapshot, spent_today=sum(self._spent.values()))
        self._publish()
        return not first

    def _recorded_entries(self, day: str) -> tuple[dict, dict]:
        if self._ledger is None:
            return {}, {}
        try:
            entries = self._ledger.entries_on(day)
        except Exception as exc:
            logger.warning(f"[ACCOUNT] Today's spend unavailable from the Ledger: {exc}")
            return {}, {}
        return ({s: usd for s, (usd, _n) in entries.items()},
                {s: n for s, (_usd, n) in entries.items()})

    def _strategy_limit(self, strategy: str) -> float:
        cfg = self._config
        limits = {
            "arbitrage": cfg.arbitrage_daily_limit_usd,
            "crypto": cfg.crypto_daily_limit_usd,
        }
        return float(limits.get(strategy, cfg.daily_limit_usd))

    def _strategy_max_trades(self, strategy: str) -> int:
        cfg = self._config
        caps = {
            "arbitrage": cfg.arbitrage_max_daily_trades,
            "crypto": cfg.crypto_max_daily_trades,
        }
        return int(caps.get(strategy, cfg.max_daily_trades))

    def _remaining_budget(self, strategy: str) -> float:
        # Per-strategy limits may sum past daily_limit_usd (TradingConfig
        # only warns), so the global limit caps every strategy too.
        strategy_left = self._strategy_limit(strategy) - self._spent.get(strategy, 0.0)
        global_left = float(self._config.daily_limit_usd) - sum(self._spent.values())
        return min(strategy_left, global_left)

    def _budget_detail(self, strategy: str) -> dict:
        return {
            "strategy": strategy,
            "strategy_daily_limit_usd": round(self._strategy_limit(strategy), 4),
            "strategy_spent_today": round(self._spent.get(strategy, 0.0), 4),
            "daily_limit_usd": round(float(self._config.daily_limit_usd), 4),
            "spent_today": round(sum(self._spent.values()), 4),
        }

    # -- refresh --------------------------------------------------------------

    def refresh(self) -> AccountSnapshot:
        """Re-read cash and positions from the active Venue, recompute
        equity and the drawdown breaker, and publish the snapshot. A Venue
        read that fails keeps the last known value for that field."""
        prev = self._snapshot
        venue = self._active_venue()

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
            self._positions_unavailable = False
        except PositionsUnavailable as exc:
            # An outage: logged once, not on every refresh it lasts.
            if not self._positions_unavailable:
                self._log("SYNC-WARN", "Account", "portfolio", {"reason": "positions_unavailable", "error": str(exc)})
            self._positions_unavailable = True
        except Exception as exc:
            self._log("SYNC-WARN", "Account", "portfolio", {"reason": "positions_fetch_failed", "error": str(exc)})

        # Drawdown circuit breaker. The peak-tracking/threshold pattern
        # mirrors oracle3's StandardRiskManager._check_drawdown()
        # (oracle3/risk/risk_manager.py) -
        # https://github.com/YichengYang-Ethan/oracle3, licensed under the
        # Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
        # Peak equity is per Venue: on a switch (Dry run <-> Live run) the
        # baseline restarts from the new Venue's equity, so the difference
        # between two accounts never reads as a drawdown.
        equity = cash + position_value
        venue_name = getattr(venue, "name", None)
        if venue_name != self._peak_venue:
            self._peak_venue = venue_name
            peak = equity
        else:
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
            spent_today=prev.spent_today,
            watch_only=cash < float(self._config.min_trading_balance),
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
            logger.warning(f"[ACCOUNT] Unrealized P&L unavailable: {exc}")
            return 0.0

    def _publish(self) -> None:
        snap, bridge = self._snapshot, self._bridge
        bridge.current_balance = snap.cash
        bridge.open_position_value = snap.open_position_value
        bridge.unrealized_pnl = snap.unrealized_pnl
        bridge.peak_equity = snap.peak_equity
        bridge.current_drawdown_pct = snap.drawdown_pct
        bridge.drawdown_paused = snap.drawdown_paused
        bridge.spent_today = snap.spent_today
        bridge.watch_only = snap.watch_only

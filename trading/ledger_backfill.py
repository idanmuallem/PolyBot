"""One-time backfill of historical trades from hunt_history into the Ledger.

Before the Ledger existed, the only record of trades was the hunt_history
event log. This migration replays that log once, in id order, and records
every trade it can reconstruct as Ledger rows stamped with the original
event times. It runs from ui.data_manager.init_db, in both the engine and
the dashboard process.

Idempotent and crash-safe: the whole replay and its completion marker (a
row in ledger_migrations) are written in one IMMEDIATE transaction. A run
that is interrupted (crash, redeploy, the container stopped after its WAL
checkpoint) commits nothing and is simply re-run on the next init; a run
that committed is never repeated. IMMEDIATE also serialises the two
processes: the second one waits, then finds the marker.

Only rows logged before the Ledger's earliest own trade are replayed, so a
database that already recorded trades into the Ledger is not double-counted.

How the log maps onto Ledger rows:
  model entry  TRACK (logged only for executed trades) paired with the
               executor fill row just before it (DRY-RUN / PAPER-TRADE ->
               paper, LIVE-TRADE -> live), which has the bought token
               (the NO token on a NO trade), price and shares. AUTO-TRADE
               is the pre-fill intent and is never an entry on its own.
  arbitrage    STRATEGY-LEG rows with executed fills (per-leg token, price,
               shares, side, group_id). ARBITRAGE-FILL only sets the
               basket's venue: live for "all_legs_filled", else paper.
  sale         The executor sell row (DRY-RUN-SELL / PAPER-SELL -> paper,
               SELL -> live). The exit row that follows it for the same
               token (TAKE-PROFIT, STOP-LOSS, ...) gives the reason, and its
               sold=False marks a failed sell, which is skipped. With no
               exit row it was an arbitrage unwind or a portfolio cull.
  resolution   EXPIRED: sold=True is a paid-out (paper) resolution at
               "price"; sold=False is live's unconfirmed expiry.
A sale or resolution of a token whose replayed position is already closed
is a phantom (the pre-6a06f35 repeated-sell bug) and is skipped.
hunt_history carries no condition_id, so backfilled positions have none.
"""

from __future__ import annotations

import ast
import json
import math
import sqlite3
from typing import Optional

from trading.ledger import Ledger
from trading.venue import whole_cent_shares

MIGRATION = "hunt_history_backfill_v1"

_BUY_VENUE = {"DRY-RUN": "paper", "PAPER-TRADE": "paper", "LIVE-TRADE": "live"}
_SELL_VENUE = {"DRY-RUN-SELL": "paper", "PAPER-SELL": "paper", "SELL": "live"}
_EXIT_REASONS = {"TAKE-PROFIT", "STOP-LOSS", "WANG-EDGE-DECAY", "EV-CONVERGENCE", "OPPORTUNITY-SWAP"}
_LIVE_ARBITRAGE_FILLS = {"all_legs_filled"}


def backfill_hunt_history(db_path: str) -> int:
    """Run the backfill unless it already ran. Returns the number of Ledger
    trade rows recorded (0 when it already ran)."""
    conn = sqlite3.connect(str(db_path), timeout=30, isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            recorded = _backfill(conn)
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
        return recorded
    finally:
        conn.close()


def _backfill(conn) -> int:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS ledger_migrations (name TEXT PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    if conn.execute("SELECT 1 FROM ledger_migrations WHERE name = ?", (MIGRATION,)).fetchone():
        return 0

    ledger = Ledger.on_connection(conn)
    recorded = 0
    has_log = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'hunt_history'"
    ).fetchone()
    if has_log:
        cutoff = conn.execute("SELECT MIN(timestamp) FROM trades").fetchone()[0]
        sql = "SELECT id, timestamp, level, asset_type, token_id, payload FROM hunt_history"
        params: tuple = ()
        if cutoff is not None:
            sql += " WHERE timestamp < ?"
            params = (cutoff,)
        rows = [
            _Row(r["timestamp"], str(r["level"] or ""), r["asset_type"], str(r["token_id"] or ""),
                 parse_payload(r["payload"]))
            for r in conn.execute(sql + " ORDER BY id", params).fetchall()
        ]
        recorded = _Replay(ledger, rows).run()

    conn.execute(
        "INSERT INTO ledger_migrations (name, applied_at) VALUES (?, datetime('now'))", (MIGRATION,)
    )
    return recorded


class _Row:
    __slots__ = ("ts", "level", "asset_type", "token", "payload")

    def __init__(self, ts, level, asset_type, token, payload):
        self.ts, self.level, self.asset_type, self.token, self.payload = (
            ts, level, asset_type, token, payload)


class _Replay:
    def __init__(self, ledger: Ledger, rows: list[_Row]):
        self.ledger = ledger
        self.rows = rows
        self.by_token: dict[str, list[int]] = {}
        for i, row in enumerate(rows):
            self.by_token.setdefault(row.token, []).append(i)
        self.pending_buy: Optional[_Row] = None
        self.auto_shares: dict[str, float] = {}
        self.mode = "paper"  # venue of the latest executor row seen
        self.group_venue: dict[str, str] = {}
        self.ever_opened: set[str] = set()
        self.recorded = 0

    def run(self) -> int:
        for i, row in enumerate(self.rows):
            level = row.level
            if level == "AUTO-TRADE":
                shares = _num(row.payload.get("shares"))
                if shares:
                    self.auto_shares[row.token] = shares
            elif level in _BUY_VENUE:
                self.mode = _BUY_VENUE[level]
                if row.payload.get("success") is not False:
                    self.pending_buy = row
            elif level == "TRACK":
                self._model_entry(row)
            elif level == "ARBITRAGE-FILL":
                venue = "live" if row.payload.get("reason") in _LIVE_ARBITRAGE_FILLS else "paper"
                self.mode = venue
                if row.payload.get("group_id"):
                    self.group_venue[str(row.payload["group_id"])] = venue
            elif level == "STRATEGY-LEG":
                self._arbitrage_entry(row)
            elif level in _SELL_VENUE:
                self.mode = _SELL_VENUE[level]
                self._sale(i, row)
            elif level == "EXPIRED":
                self._resolution(row)
        return self.recorded

    def _model_entry(self, track: _Row) -> None:
        fill, self.pending_buy = self.pending_buy, None
        p = track.payload
        if not p.get("executed", True) or fill is None:
            return
        side = _side(p.get("side") or fill.payload.get("side"))
        if fill.payload.get("side") and _side(fill.payload["side"]) != side:
            return
        price = _num(fill.payload.get("price"))
        if not price or price <= 0:
            return
        shares = _num(fill.payload.get("shares")) or self.auto_shares.get(fill.token)
        if not shares:
            bet = _num(fill.payload.get("bet_amount_usd")) or _num(p.get("bet_usd")) or 0.0
            shares = whole_cent_shares(bet, price)
        if shares <= 0:
            return
        post_prob = next((_num(p.get(k)) for k in ("post_prob", "fair", "fair_value")
                          if _num(p.get(k)) is not None), None)
        self.ledger.record_entry(
            token_id=fill.token, condition_id=None, side=side, shares=shares, price=price,
            venue=_BUY_VENUE[fill.level], strategy="crypto", asset_type=track.asset_type,
            market_name=p.get("market_name"), pre_prob=_num(p.get("pre_prob")),
            post_prob=post_prob, wang_edge=_num(p.get("wang_edge")), at=track.ts,
        )
        self.ever_opened.add(fill.token)
        self.recorded += 1

    def _arbitrage_entry(self, row: _Row) -> None:
        p = row.payload
        shares = _num(p.get("shares"))
        price = _num(p.get("price"))
        if not p.get("executed") or not shares or shares <= 0 or not price or price <= 0:
            return
        group_id = str(p["group_id"]) if p.get("group_id") else None
        self.ledger.record_entry(
            token_id=row.token, condition_id=None, side=_side(p.get("side")), shares=shares,
            price=price, venue=self.group_venue.get(group_id, self.mode), strategy="arbitrage",
            asset_type=row.asset_type, market_name=p.get("market_name"), group_id=group_id,
            at=row.ts,
        )
        self.ever_opened.add(row.token)
        self.recorded += 1

    def _exit_row(self, i: int) -> Optional[_Row]:
        """The exit-path row logged right after the executor sell row i for
        the same token, if any."""
        for j in self.by_token.get(self.rows[i].token, []):
            if j <= i:
                continue
            level = self.rows[j].level
            if level in _SELL_VENUE or level in _BUY_VENUE:
                return None
            if level in _EXIT_REASONS:
                return self.rows[j]
        return None

    def _sale(self, i: int, row: _Row) -> None:
        shares = _num(row.payload.get("shares"))
        price = _num(row.payload.get("price"))
        if not shares or shares <= 0 or price is None:
            return
        exit_row = self._exit_row(i)
        if exit_row is not None and "sold" in exit_row.payload and not exit_row.payload["sold"]:
            return
        holding = self.ledger.holding(row.token)
        if holding is None and row.token in self.ever_opened:
            return
        if exit_row is not None:
            reason = exit_row.level
        elif holding is not None and holding.strategy == "arbitrage":
            reason = "ARBITRAGE-UNWIND"
        else:
            reason = "PORTFOLIO-CULL"
        entry_price = _num(exit_row.payload.get("initial_price")) if exit_row is not None else None
        self.ledger.record_exit(
            token_id=row.token, shares=shares, price=price, reason=reason,
            venue=_SELL_VENUE[row.level], entry_price=entry_price, at=row.ts,
        )
        self.recorded += 1

    def _resolution(self, row: _Row) -> None:
        p = row.payload
        confirmed = bool(p.get("sold"))
        payout = _num(p.get("price")) if confirmed else None
        if confirmed and payout is None:
            return
        holding = self.ledger.holding(row.token)
        if holding is None and row.token in self.ever_opened:
            return
        shares = _num(p.get("shares")) or 0.0
        if holding is not None and shares <= 0:
            shares = holding.shares  # pre-fix paper rows logged 0 shares
        venue = holding.venue if holding is not None else ("paper" if confirmed else "live")
        self.ledger.record_resolution(
            token_id=row.token, shares=shares, payout_price=payout, venue=venue,
            entry_price=_num(p.get("initial_price")), side=p.get("side") or None,
            market_name=p.get("market_name") or None, at=row.ts,
        )
        self.recorded += 1


def parse_payload(raw) -> dict:
    """A hunt_history payload: JSON, or the Python repr log_event writes.

    The one hunt_history payload decoder. Besides this backfill, only the
    dashboard's diagnostics views (event table, terminal feed, EV chart,
    equity curve) use it; trading code never reads log payloads."""
    text = str(raw or "").strip()
    if not text:
        return {}
    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(text)
        except Exception:
            continue
        return value if isinstance(value, dict) else {}
    return {}


def _num(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _side(value) -> str:
    return "NO" if str(value or "").strip().upper() == "NO" else "YES"

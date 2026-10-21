"""Ledger: the typed record of trades and positions.

One owner for "what do I hold, what did it cost, and what did I make". Lives
in the wallet's trades database alongside hunt_history, which goes back to
being a diagnostics-only event log (see spec #12).

Two tables:
  positions — one row per position opened, with its entry details. Open
              until sold down to zero shares, resolved or expired.
  trades    — one row per trade event: an entry fill (BUY), a sale (SELL),
              a market resolution (RESOLVED) or an unconfirmed expiry
              (EXPIRED), or a live entry order that never filled
              (UNFILLED). Every row is tagged with venue (paper | live) and
              strategy tag (crypto | arbitrage).

Authority split: the Venue is authoritative for share quantities and cash;
the Ledger is authoritative for entry details, realized P&L and history.

Single writer: the engine process records into the Ledger; the dashboard
process only calls the read methods.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS positions (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        venue         TEXT NOT NULL,
        strategy      TEXT NOT NULL,
        token_id      TEXT NOT NULL,
        condition_id  TEXT,
        side          TEXT,
        asset_type    TEXT,
        market_name   TEXT,
        shares        REAL NOT NULL,
        entry_price   REAL NOT NULL,
        pre_prob      REAL,
        post_prob     REAL,
        wang_edge     REAL,
        group_id      TEXT,
        opened_at     TEXT NOT NULL,
        closed_at     TEXT,
        status        TEXT NOT NULL DEFAULT 'open',
        realized_pnl  REAL NOT NULL DEFAULT 0.0
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_positions_token ON positions (token_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_positions_market ON positions (condition_id, side, status)",
    """
    CREATE TABLE IF NOT EXISTS trades (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp     TEXT NOT NULL,
        venue         TEXT NOT NULL,
        strategy      TEXT NOT NULL,
        kind          TEXT NOT NULL,
        reason        TEXT,
        position_id   INTEGER,
        token_id      TEXT NOT NULL,
        condition_id  TEXT,
        side          TEXT,
        market_name   TEXT,
        shares        REAL NOT NULL,
        price         REAL,
        amount_usd    REAL,
        entry_price   REAL,
        realized_pnl  REAL,
        confirmed     INTEGER NOT NULL DEFAULT 1,
        group_id      TEXT
    )
    """,
)

# Columns added after a table first shipped: (table, column, type). Added by
# init_schema to databases created before them.
_ADDED_COLUMNS = (
    ("positions", "group_id", "TEXT"),
    ("trades", "group_id", "TEXT"),
)


@dataclass(frozen=True)
class Holding:
    """An open position as the Ledger knows it."""
    token_id: str
    condition_id: Optional[str]
    side: Optional[str]
    shares: float
    entry_price: float
    venue: str
    strategy: str
    asset_type: Optional[str]
    market_name: Optional[str]
    pre_prob: Optional[float]
    post_prob: Optional[float]
    wang_edge: Optional[float]
    opened_at: str
    group_id: Optional[str] = None  # arbitrage basket the leg belongs to


def _now() -> str:
    """Recorded times are UTC, the same clock as the Account's budget day."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class _Borrowed:
    """A caller's open connection, lent to a Ledger: entering and leaving it
    neither commits nor closes, so the caller owns the transaction."""

    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, *exc):
        return False


class Ledger:
    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        self._conn = None
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with self._connect() as conn:
            init_schema(conn)

    @classmethod
    def on_connection(cls, conn) -> "Ledger":
        """A Ledger that reads and writes through `conn` inside the caller's
        transaction (the hunt_history backfill records a whole history
        atomically this way). The caller commits or rolls back."""
        conn.row_factory = sqlite3.Row
        ledger = cls.__new__(cls)
        ledger.db_path = None
        ledger._conn = conn
        init_schema(conn)
        return ledger

    def _connect(self):
        if self._conn is not None:
            return _Borrowed(self._conn)
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Writes (engine only)
    # ------------------------------------------------------------------

    def record_entry(
        self, *, token_id: str, condition_id: Optional[str], side: str, shares: float,
        price: float, venue: str, strategy: str, asset_type: Optional[str] = None,
        market_name: Optional[str] = None, pre_prob: Optional[float] = None,
        post_prob: Optional[float] = None, wang_edge: Optional[float] = None,
        group_id: Optional[str] = None, at: Optional[str] = None,
    ) -> int:
        """Record an entry fill. Returns the position id. group_id ties an
        arbitrage leg to its basket; the position's later sales and its
        resolution inherit it. `at` overrides the recorded time
        ("YYYY-MM-DD HH:MM:SS"; default now)."""
        ts = at or _now()
        side = str(side or "").upper() or None
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO positions (venue, strategy, token_id, condition_id, side, asset_type, "
                "market_name, shares, entry_price, pre_prob, post_prob, wang_edge, group_id, opened_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (venue, strategy, str(token_id), condition_id, side, asset_type, market_name,
                 float(shares), float(price), pre_prob, post_prob, wang_edge, group_id, ts),
            )
            position_id = int(cur.lastrowid)
            conn.execute(
                "INSERT INTO trades (timestamp, venue, strategy, kind, position_id, token_id, "
                "condition_id, side, market_name, shares, price, amount_usd, entry_price, group_id) "
                "VALUES (?, ?, ?, 'BUY', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ts, venue, strategy, position_id, str(token_id), condition_id, side, market_name,
                 float(shares), float(price), float(shares) * float(price), float(price), group_id),
            )
        return position_id

    def record_exit(
        self, *, token_id: str, shares: float, price: float, reason: str, venue: str,
        strategy: Optional[str] = None, entry_price: Optional[float] = None,
        side: Optional[str] = None, market_name: Optional[str] = None,
        at: Optional[str] = None,
    ) -> int:
        """Record a confirmed sale of `shares` at `price`. `reason` is the
        exit path (TAKE-PROFIT, STOP-LOSS, PORTFOLIO-CULL, ...).

        The open holding for token_id supplies entry price, side and
        strategy. For a position the Ledger never saw opened (e.g. one
        opened before the Ledger existed), the caller's entry_price / side /
        strategy are used instead. Returns the trade id."""
        return self._close(
            kind="SELL", token_id=token_id, shares=shares, price=price, reason=reason,
            venue=venue, strategy=strategy, entry_price=entry_price, side=side,
            market_name=market_name, confirmed=True, at=at,
        )

    def record_resolution(
        self, *, token_id: str, shares: float, payout_price: Optional[float], venue: str,
        strategy: Optional[str] = None, entry_price: Optional[float] = None,
        side: Optional[str] = None, market_name: Optional[str] = None,
        at: Optional[str] = None,
    ) -> int:
        """Record a market resolution or expiry, which ends the position.

        payout_price is the per-share payout (1.0 winning / 0.0 losing).
        None means an unconfirmed expiry: the market most likely resolved
        but no payout is confirmed (live mode's gave-up-selling case), so
        the position is closed as 'expired' with no realized P&L."""
        confirmed = payout_price is not None
        return self._close(
            kind="RESOLVED" if confirmed else "EXPIRED", token_id=token_id, shares=shares,
            price=payout_price, reason="EXPIRED", venue=venue, strategy=strategy,
            entry_price=entry_price, side=side, market_name=market_name, confirmed=confirmed,
            at=at,
        )

    def record_unfilled(self, *, token_id: str, venue: str, at: Optional[str] = None) -> Optional[int]:
        """Close the open holding for token_id as an entry order that never
        filled: its whole size, no exit price and no realized P&L. Returns
        the trade id, or None if nothing was open."""
        holding = self.holding(token_id)
        if holding is None:
            return None
        return self._close(
            kind="UNFILLED", token_id=token_id, shares=holding.shares, price=None,
            reason="UNFILLED", venue=venue, strategy=None, entry_price=None, side=None,
            market_name=None, confirmed=False, at=at, closed_status="unfilled",
        )

    def _close(
        self, *, kind: str, token_id: str, shares: float, price: Optional[float],
        reason: Optional[str], venue: str, strategy: Optional[str],
        entry_price: Optional[float], side: Optional[str], market_name: Optional[str],
        confirmed: bool, at: Optional[str] = None, closed_status: Optional[str] = None,
    ) -> int:
        ts = at or _now()
        shares = float(shares)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM positions WHERE token_id = ? AND status = 'open' "
                "ORDER BY id DESC LIMIT 1",
                (str(token_id),),
            ).fetchone()

            position_id = None
            condition_id = None
            group_id = None
            if row is not None:
                position_id = int(row["id"])
                condition_id = row["condition_id"]
                group_id = row["group_id"]
                entry_price = float(row["entry_price"])
                side = row["side"]
                strategy = row["strategy"]
                market_name = market_name or row["market_name"]

            realized = None
            if confirmed and price is not None and entry_price is not None:
                realized = (float(price) - float(entry_price)) * shares

            if row is not None:
                remaining = float(row["shares"]) - shares
                # Resolution / expiry ends the position whatever is left;
                # a sale only closes it once sold down to zero.
                fully_closed = kind != "SELL" or remaining <= 1e-9
                status = "open"
                if fully_closed:
                    status = closed_status or ("closed" if confirmed else "expired")
                conn.execute(
                    "UPDATE positions SET shares = ?, status = ?, closed_at = ?, "
                    "realized_pnl = realized_pnl + ? WHERE id = ?",
                    (0.0 if fully_closed else remaining, status, ts if fully_closed else None,
                     realized or 0.0, position_id),
                )

            cur = conn.execute(
                "INSERT INTO trades (timestamp, venue, strategy, kind, reason, position_id, token_id, "
                "condition_id, side, market_name, shares, price, amount_usd, entry_price, "
                "realized_pnl, confirmed, group_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ts, venue, strategy or "crypto", kind, reason, position_id, str(token_id),
                 condition_id, (str(side).upper() if side else None), market_name, shares,
                 (float(price) if price is not None else None),
                 (float(price) * shares if price is not None else None),
                 (float(entry_price) if entry_price is not None else None),
                 realized, int(bool(confirmed)), group_id),
            )
            return int(cur.lastrowid)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def holding(self, token_id: str) -> Optional[Holding]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM positions WHERE token_id = ? AND status = 'open' "
                "ORDER BY id DESC LIMIT 1",
                (str(token_id),),
            ).fetchone()
        return _holding(row) if row else None

    def holding_for(self, condition_id: str, side: str) -> Optional[Holding]:
        """The open holding on one side of a market, keyed canonically on
        (condition_id, side) so a NO-side holding is never missed."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM positions WHERE condition_id = ? AND side = ? AND status = 'open' "
                "ORDER BY id DESC LIMIT 1",
                (str(condition_id), str(side or "").upper()),
            ).fetchone()
        return _holding(row) if row else None

    def open_positions(self, venue: Optional[str] = None) -> list[Holding]:
        sql = "SELECT * FROM positions WHERE status = 'open'"
        params: tuple = ()
        if venue is not None:
            sql += " AND venue = ?"
            params = (venue,)
        with self._connect() as conn:
            rows = conn.execute(sql + " ORDER BY id", params).fetchall()
        return [_holding(r) for r in rows]

    def spent_on(self, day: str) -> float:
        """Total $ of entry fills recorded on `day` ("YYYY-MM-DD", the date
        part of the recorded times), across venues and strategies."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(amount_usd), 0.0) FROM trades "
                "WHERE kind = 'BUY' AND substr(timestamp, 1, 10) = ?",
                (str(day),),
            ).fetchone()
        return float(row[0] or 0.0)

    def entries_on(self, day: str) -> dict[str, tuple[float, int]]:
        """Entry fills recorded on `day` ("YYYY-MM-DD", UTC), per strategy:
        {strategy: ($ spent, number of fills)}, across venues."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT strategy, COALESCE(SUM(amount_usd), 0.0), COUNT(*) FROM trades "
                "WHERE kind = 'BUY' AND substr(timestamp, 1, 10) = ? GROUP BY strategy",
                (str(day),),
            ).fetchall()
        return {str(r[0]): (float(r[1] or 0.0), int(r[2])) for r in rows}

    def _closing_trades(self, venue: Optional[str] = None) -> list:
        """Confirmed closes (sales and paid-out resolutions) with a known
        exit price, oldest first."""
        sql = ("SELECT * FROM trades WHERE kind != 'BUY' AND confirmed = 1 "
               "AND price IS NOT NULL")
        params: tuple = ()
        if venue is not None:
            sql += " AND venue = ?"
            params = (venue,)
        with self._connect() as conn:
            return conn.execute(sql + " ORDER BY id", params).fetchall()

    def realized_pnl(self, venue: Optional[str] = None) -> float:
        return float(sum(r["realized_pnl"] or 0.0 for r in self._closing_trades(venue)))

    def unrealized_pnl(self, marks: dict, venue: Optional[str] = None) -> float:
        """(mark - entry price) x shares over open holdings, given current
        per-token prices. Holdings with no mark contribute nothing."""
        total = 0.0
        for h in self.open_positions(venue):
            mark = marks.get(h.token_id)
            if mark is None:
                continue
            total += (float(mark) - h.entry_price) * h.shares
        return total

    def trade_stats(self, venue: Optional[str] = None) -> dict:
        """Dashboard trade stats. total_trades counts each entry fill and
        each confirmed close with a decidable outcome; win/loss rates and
        avg win/loss (in gross exit proceeds) come from the closes.
        TAKE-PROFIT is a win and STOP-LOSS a loss by definition; every other
        close is decided by exit price vs entry price (unchanged closes and
        closes with unknown entry price are left out)."""
        sql = "SELECT side FROM trades WHERE kind = 'BUY'"
        params: tuple = ()
        if venue is not None:
            sql += " AND venue = ?"
            params = (venue,)
        with self._connect() as conn:
            entry_sides = [str(r["side"] or "").upper() for r in conn.execute(sql, params)]

        wins: list[float] = []
        losses: list[float] = []
        side_results = {"YES": [0, 0], "NO": [0, 0]}  # side -> [wins, losses]
        for r in self._closing_trades(venue):
            gross = float(r["price"]) * float(r["shares"])
            if gross <= 0:
                continue
            if r["reason"] == "TAKE-PROFIT":
                is_win = True
            elif r["reason"] == "STOP-LOSS":
                is_win = False
            else:
                if r["entry_price"] is None:
                    continue
                delta = float(r["price"]) - float(r["entry_price"])
                if delta == 0:
                    continue
                is_win = delta > 0
            (wins if is_win else losses).append(gross)
            side = str(r["side"] or "").upper()
            if side in side_results:
                side_results[side][0 if is_win else 1] += 1

        def _rate(won: int, lost: int) -> float:
            return round(won / (won + lost) * 100.0, 2) if (won + lost) else 0.0

        return {
            "win_rate": _rate(len(wins), len(losses)),
            "total_trades": len(entry_sides) + len(wins) + len(losses),
            "avg_win": round(sum(wins) / len(wins), 2) if wins else 0.0,
            "avg_loss": round(sum(losses) / len(losses), 2) if losses else 0.0,
            "total_yes_trades": entry_sides.count("YES"),
            "yes_win_rate": _rate(*side_results["YES"]),
            "total_no_trades": entry_sides.count("NO"),
            "no_win_rate": _rate(*side_results["NO"]),
        }

    def transactions(self, limit: int = 100, venue: Optional[str] = None) -> list[dict]:
        """Every trade row (buys, sales, resolutions, expiries), newest first."""
        sql = "SELECT * FROM trades"
        params: tuple = ()
        if venue is not None:
            sql += " WHERE venue = ?"
            params = (venue,)
        with self._connect() as conn:
            rows = conn.execute(sql + " ORDER BY id DESC LIMIT ?", params + (int(limit),)).fetchall()

        out = []
        for r in rows:
            pnl_per_share = None
            if (r["kind"] != "BUY" and r["confirmed"] and r["price"] is not None
                    and r["entry_price"] is not None):
                pnl_per_share = float(r["price"]) - float(r["entry_price"])
            out.append({
                "time": r["timestamp"], "kind": r["kind"], "reason": r["reason"],
                "venue": r["venue"], "strategy": r["strategy"], "token_id": r["token_id"],
                "side": r["side"], "market_name": r["market_name"], "price": r["price"],
                "shares": r["shares"], "amount_usd": r["amount_usd"],
                "pnl_per_share": pnl_per_share, "confirmed": bool(r["confirmed"]),
                "group_id": r["group_id"],
            })
        return out

    def closed_trade_deltas(self, venue: Optional[str] = None) -> list[float]:
        """Per-share $ P&L (exit price - entry price) for every confirmed
        close whose entry price is known."""
        return [
            float(r["price"]) - float(r["entry_price"])
            for r in self._closing_trades(venue)
            if r["entry_price"] is not None
        ]


def init_schema(conn) -> None:
    """Create the Ledger tables if missing. Idempotent; also called from
    ui.data_manager.init_db so both processes see the same schema."""
    for statement in _SCHEMA:
        conn.execute(statement)
    for table, column, col_type in _ADDED_COLUMNS:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")


def _holding(row) -> Holding:
    return Holding(
        token_id=row["token_id"], condition_id=row["condition_id"], side=row["side"],
        shares=float(row["shares"]), entry_price=float(row["entry_price"]),
        venue=row["venue"], strategy=row["strategy"], asset_type=row["asset_type"],
        market_name=row["market_name"], pre_prob=row["pre_prob"], post_prob=row["post_prob"],
        wang_edge=row["wang_edge"], opened_at=row["opened_at"], group_id=row["group_id"],
    )

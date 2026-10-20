import json
import os
import sqlite3
from datetime import datetime

import pandas as pd

from trading.ledger import Ledger, init_schema as init_ledger_schema
from trading.ledger_backfill import backfill_hunt_history


# ---------------------------------------------------------------------------
# Payload parsing
# ---------------------------------------------------------------------------

def _parse_payload_value(payload_value) -> dict:
    if isinstance(payload_value, dict):
        return payload_value
    text = str(payload_value or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        try:
            import ast
            return ast.literal_eval(text)
        except Exception:
            return {}


# ---------------------------------------------------------------------------
# DB connection
# ---------------------------------------------------------------------------

def _open_db(db_path: str, timeout: int = 10):
    return sqlite3.connect(db_path, timeout=timeout)


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

def init_db(db_path: str):
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with _open_db(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hunt_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                level TEXT,
                asset_type TEXT,
                token_id TEXT,
                payload TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_snapshots (
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                cash REAL,
                positions_value REAL,
                total_value REAL
            )
            """
        )

        # engine_status / engine_control / open_positions: the shared-state
        # tables that let the trading engine run as its own OS process,
        # separate from the Streamlit dashboard process, with no networking
        # between them — see run_engine.py and core/wallet_manager.py.
        # Single-writer-per-table by convention (not enforced by SQLite):
        # the engine process only ever writes engine_status/open_positions
        # and reads engine_control; the dashboard process is the reverse.
        # This avoids either side doing a read-modify-write that could
        # clobber a field the other process just wrote.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS engine_status (
                id                  INTEGER PRIMARY KEY CHECK (id = 1),
                updated_at          TEXT,
                current_balance     REAL,
                starting_balance    REAL,
                open_position_value REAL,
                unrealized_pnl      REAL,
                watch_only          INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS engine_control (
                id                     INTEGER PRIMARY KEY CHECK (id = 1),
                live_trading_requested INTEGER,
                updated_at             TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS open_positions (
                token_id         TEXT PRIMARY KEY,
                market_id        TEXT,
                side             TEXT,
                shares           REAL,
                initial_price    REAL,
                current_price    REAL,
                value            REAL,
                pnl_ratio        REAL,
                asset_type       TEXT,
                wang_edge_entry  REAL,
                wang_edge_now    REAL,
                wang_edge_delta  REAL
            )
            """
        )

        # WAL mode: readers (the dashboard process) never block the writer
        # (the engine process) and vice versa. Persists in the database
        # file's header, so this only needs to be set once — but init_db()
        # is called by both processes at startup, so issuing it here is
        # cheap and idempotent rather than a real per-connection cost.
        # Ledger tables (trades / positions) — see trading/ledger.py.
        init_ledger_schema(conn)

        conn.execute("PRAGMA journal_mode=WAL")

    # One-time copy of pre-Ledger trade history into the Ledger. Atomic and
    # marker-guarded, so a failed or interrupted run commits nothing and is
    # retried on the next init; it must never stop either process starting.
    try:
        backfill_hunt_history(db_path)
    except Exception as e:
        print(f"[LEDGER] hunt_history backfill failed (will retry on next start): {e}")


def insert_paper_snapshot(db_path: str, cash: float, positions_value: float):
    total_value = cash + positions_value
    try:
        with _open_db(db_path) as conn:
            conn.execute(
                "INSERT INTO paper_snapshots (cash, positions_value, total_value) VALUES (?, ?, ?)",
                (float(cash), float(positions_value), float(total_value)),
            )
    except Exception as e:
        print(f"[PAPER] DB error saving snapshot: {e}")


# ---------------------------------------------------------------------------
# Engine <-> dashboard shared state (separate-process split — see
# run_engine.py). engine_status/open_positions: engine writes, dashboard
# reads. engine_control: dashboard writes, engine reads (polls once per loop
# tick — see SequentialTradingPipeline.run_forever() in
# trading/decision_pipeline.py).
# ---------------------------------------------------------------------------

def write_engine_status(
    db_path: str, balance: float, starting_balance: float,
    position_value: float, unrealized_pnl: float, watch_only: bool,
) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with _open_db(db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO engine_status "
                "(id, updated_at, current_balance, starting_balance, open_position_value, "
                " unrealized_pnl, watch_only) VALUES (1, ?, ?, ?, ?, ?, ?)",
                (ts, float(balance), float(starting_balance), float(position_value),
                 float(unrealized_pnl), int(bool(watch_only))),
            )
    except Exception as e:
        print(f"[ENGINE-STATUS] DB error writing status: {e}")


def read_engine_status(db_path: str) -> dict | None:
    try:
        with _open_db(db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT updated_at, current_balance, starting_balance, open_position_value, "
                "unrealized_pnl, watch_only FROM engine_status WHERE id = 1"
            ).fetchone()
    except Exception:
        return None
    if row is None:
        return None
    d = dict(row)
    d["watch_only"] = bool(d["watch_only"])
    return d


def write_live_trading_requested(db_path: str, value: bool) -> None:
    """Dashboard-side write: the operator's requested trading mode, polled
    by the engine process once per loop tick (see run_forever())."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with _open_db(db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO engine_control (id, live_trading_requested, updated_at) "
                "VALUES (1, ?, ?)",
                (int(bool(value)), ts),
            )
    except Exception as e:
        print(f"[ENGINE-CONTROL] DB error writing control: {e}")


def seed_live_trading_requested_if_absent(db_path: str, default_value: bool) -> None:
    """Initialize engine_control on first-ever startup only — INSERT OR
    IGNORE so an operator's existing preference is never clobbered by a
    later engine restart picking a fresh config-derived default."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with _open_db(db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO engine_control (id, live_trading_requested, updated_at) "
                "VALUES (1, ?, ?)",
                (int(bool(default_value)), ts),
            )
    except Exception as e:
        print(f"[ENGINE-CONTROL] DB error seeding control: {e}")


def read_live_trading_requested(db_path: str) -> bool:
    """Defaults to False (dry-run) when no control row exists yet — the safe
    default for an unattended process that should never trade real money
    before an operator (or the config-derived seed) has explicitly opted in."""
    try:
        with _open_db(db_path) as conn:
            row = conn.execute(
                "SELECT live_trading_requested FROM engine_control WHERE id = 1"
            ).fetchone()
    except Exception:
        return False
    if row is None:
        return False
    return bool(row[0])


def write_open_positions(db_path: str, positions: list, position_analytics: dict | None = None) -> None:
    """Wholesale replace (DELETE + INSERT in one transaction) — same
    overwrite-snapshot pattern as insert_paper_snapshot's caller, just for
    the current set of open positions instead of one aggregate row.

    `positions` is bridge.current_portfolio (a list of core.models.Position);
    `position_analytics` is bridge.position_analytics (Wang edge-decay/
    asset_type snapshots keyed by token_id, populated once per management
    cycle by PortfolioManager) — merged here into one row per position so
    the dashboard needs only a single read per render.
    """
    analytics = position_analytics or {}
    rows = []
    for p in positions:
        token_id = str(getattr(p, "token_id", "") or "")
        snapshot = analytics.get(token_id, {}) or {}
        rows.append((
            token_id,
            str(getattr(p, "market_id", "") or ""),
            str(getattr(p, "side", "") or ""),
            float(getattr(p, "shares", 0.0) or 0.0),
            float(getattr(p, "initial_price", 0.0) or 0.0),
            float(getattr(p, "current_price", 0.0) or 0.0),
            float(getattr(p, "value", 0.0) or 0.0),
            float(getattr(p, "pnl_ratio", 0.0) or 0.0),
            snapshot.get("asset_type"),
            snapshot.get("entry_wang_edge"),
            snapshot.get("current_wang_edge"),
            snapshot.get("edge_delta"),
        ))
    try:
        with _open_db(db_path) as conn:
            conn.execute("DELETE FROM open_positions")
            conn.executemany(
                "INSERT INTO open_positions "
                "(token_id, market_id, side, shares, initial_price, current_price, value, pnl_ratio, "
                " asset_type, wang_edge_entry, wang_edge_now, wang_edge_delta) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
    except Exception as e:
        print(f"[OPEN-POSITIONS] DB error writing positions: {e}")


def read_open_positions(db_path: str) -> list[dict]:
    try:
        with _open_db(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT token_id, market_id, side, shares, initial_price, current_price, value, "
                "pnl_ratio, asset_type, wang_edge_entry, wang_edge_now, wang_edge_delta "
                "FROM open_positions"
            ).fetchall()
            return [dict(row) for row in rows]
    except Exception:
        return []


def get_level_counts(db_path: str) -> dict:
    """Replaces the old in-memory bridge.level_counts for the Activity
    Breakdown chart — fully derivable from hunt_history, which already
    carries one row per logged event."""
    try:
        with _open_db(db_path) as conn:
            rows = conn.execute("SELECT level, COUNT(*) FROM hunt_history GROUP BY level").fetchall()
        return {str(level): int(count) for level, count in rows}
    except Exception:
        return {}


def get_recent_opportunity_map(db_path: str, scan_limit: int = 500) -> dict:
    """token_id -> most-recent EV-carrying hunt_history payload (market_name,
    ev, post_prob, asset_type). Replaces the old in-memory
    bridge.opportunity_map, which was itself built from exactly these same
    payloads (see the old log_event()) — so this is a direct re-derivation
    from hunt_history, not a new source of truth. Used both for the
    EV-by-market chart (get_latest_ev_by_token) and to label open positions
    with a readable market name (render_positions — Position itself only
    carries token_id/market_id, not a question/name).

    `market_name` is left as-is from the payload (possibly missing/None) —
    callers that need a guaranteed display string apply their own fallback
    (get_latest_ev_by_token falls back to the token_id itself; ui/components.py's
    _readable_market_name falls back to a reformatted market_id/token_id slug).
    """
    try:
        with _open_db(db_path) as conn:
            rows = conn.execute(
                "SELECT token_id, asset_type, payload FROM hunt_history ORDER BY id DESC LIMIT ?",
                (scan_limit,),
            ).fetchall()
    except Exception:
        return {}

    seen: dict = {}
    for token_id, asset_type, payload_raw in rows:
        token_id = str(token_id)
        if token_id in seen:
            continue
        payload = _parse_payload_value(payload_raw)
        if not isinstance(payload, dict) or "ev" not in payload:
            continue
        try:
            ev = float(payload["ev"])
        except (TypeError, ValueError):
            continue
        seen[token_id] = {
            "token_id": token_id,
            "asset_type": str(asset_type),
            "ev": ev,
            "post_prob": float(payload.get("post_prob", 0.0) or 0.0),
            "market_name": payload.get("market_name"),
        }
    return seen


def get_latest_ev_by_token(db_path: str, limit: int = 15, scan_limit: int = 500) -> list[dict]:
    """Top-N by EV, for the EV-by-market chart — a view over
    get_recent_opportunity_map()."""
    opp_map = get_recent_opportunity_map(db_path, scan_limit=scan_limit)
    items = [
        {**entry, "market_name": entry.get("market_name") or entry["token_id"]}
        for entry in opp_map.values()
    ]
    items.sort(key=lambda x: x["ev"], reverse=True)
    return items[:limit]


def get_terminal_feed(db_path: str, limit: int = 20) -> list[str]:
    """Recent hunt_history events formatted as one-line strings, replacing
    the old in-memory bridge.terminal_logs deque — same source data
    (hunt_history), formatted the same way the dashboard's log_func used to
    format it on the fly (see the removed _make_log_event() in
    ui/dashboard.py)."""
    try:
        with _open_db(db_path) as conn:
            rows = conn.execute(
                "SELECT level, asset_type, payload FROM hunt_history ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
    except Exception:
        return []

    lines = []
    for level, asset_type, payload_raw in rows:
        payload = _parse_payload_value(payload_raw)
        reason = str(payload.get("reason", "")).strip() if isinstance(payload, dict) else ""
        market_name = str(payload.get("market_name", "")).strip() if isinstance(payload, dict) else ""
        ev_value = payload.get("ev") if isinstance(payload, dict) else None
        detail = reason or market_name or (str(payload_raw)[:140] if payload_raw else "")
        ev_suffix = f" | ev={ev_value}" if ev_value is not None else ""
        lines.append(f"[{level}] {asset_type} - {detail}{ev_suffix}")
    return lines


# ---------------------------------------------------------------------------
# Event logging
# ---------------------------------------------------------------------------

# bridge.opportunity_map keeps one entry per token_id ever seen with an "ev"
# in its payload (REJECTED/TRACK/AUTO-TRADE/...), and — unlike the hunter's
# own cooldown cache (PolymarketScannerHunter.seen_markets, pruned by age
# every scan) — nothing ever removed an entry. Over a long-running engine
# process that's an unbounded, ever-growing dict: the confirmed cause of the
# Aug 29 2026 OOM (run_engine.py's RSS reached ~416MB on a 916MB host after
# ~29h). Cap it the same way ev_samples below is capped, evicting the
# least-recently-touched token first. risk_manager's opportunity_map lookups
# already fall back to a DB query on a miss, so evicting a still-open
# position's entry only costs a slower lookup, not correctness.
OPPORTUNITY_MAP_CAP = 500


def log_event(bridge, level, asset_type, token_id, payload, db_path: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    bridge.event_count += 1
    bridge.level_counts[str(level)] += 1

    if isinstance(payload, dict) and "ev" in payload:
        try:
            bridge.ev_samples.append(float(payload["ev"]))
            bridge.ev_samples = bridge.ev_samples[-300:]
        except Exception:
            pass

    payload_pretty = json.dumps(payload, ensure_ascii=False, sort_keys=True) if isinstance(payload, dict) else str(payload)
    print(f"[{ts}] [{level}] [{asset_type}] [{token_id}] {payload_pretty}")

    try:
        with _open_db(db_path) as conn:
            conn.execute(
                "INSERT INTO hunt_history (timestamp, level, asset_type, token_id, payload) VALUES (?, ?, ?, ?, ?)",
                (ts, str(level), str(asset_type), str(token_id), str(payload)),
            )
    except Exception as e:
        print(f"[DASHBOARD] Log DB error: {e}")

    if isinstance(payload, dict) and "ev" in payload:
        try:
            entry = {
                "token_id": str(token_id),
                "asset_type": str(asset_type),
                "ev": float(payload.get("ev", 0.0)),
                "post_prob": float(payload.get("post_prob", 0.0)),
                "market_name": str(
                    payload.get("market_name")
                    or bridge.market_name_by_token.get(str(token_id), bridge.market_question)
                ),
            }
            # Phase 7: carry the Wang/strategy/risk analytics fields through
            # too, so PortfolioManager's in-memory lookups (pre_prob,
            # wang_edge, asset_type for correlation) hit this fast path
            # instead of always falling back to a DB query.
            for key in (
                "pre_prob", "wang_lambda", "wang_fair_value", "wang_edge",
                "strategy_type", "kelly_fraction_used", "correlation_exposure",
            ):
                if payload.get(key) is not None:
                    entry[key] = payload[key]
            token_key = str(token_id)
            # Pop-then-set moves this key to the end of the dict's insertion
            # order (Python dicts preserve it) so eviction below always drops
            # the least-recently-touched token, not an arbitrary one.
            bridge.opportunity_map.pop(token_key, None)
            bridge.opportunity_map[token_key] = entry
            if len(bridge.opportunity_map) > OPPORTUNITY_MAP_CAP:
                del bridge.opportunity_map[next(iter(bridge.opportunity_map))]
        except Exception:
            pass

    if bridge.event_count - bridge.last_summary_at >= 10:
        bridge.last_summary_at = bridge.event_count
        live = bridge.level_counts.get("LIVE-TRADE", 0)
        dry = bridge.level_counts.get("DRY-RUN", 0)
        paper = bridge.level_counts.get("PAPER-TRADE", 0)
        avg_ev = sum(bridge.ev_samples) / len(bridge.ev_samples) if bridge.ev_samples else 0.0
        max_ev = max(bridge.ev_samples) if bridge.ev_samples else 0.0
        # TODO(log-noise): this periodic stdout dump (with its own "=" * 90
        # separators) duplicates data already in bridge/opportunity_map and
        # isn't part of the DB-backed structured log flow — worth deciding
        # whether it earns its keep or should move behind a log-level flag.
        print("=" * 90)
        print(f"[PERF] events={bridge.event_count} | live={live} | dry={dry} | paper={paper}")
        print(f"[PERF] avg_ev={avg_ev:.4f} | max_ev={max_ev:.4f} | daily_spend=${bridge.daily_spend:.2f} | balance=${bridge.current_balance:.2f}")
        print(f"[PERF] level_mix={dict(bridge.level_counts)}")
        print("=" * 90)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _extract_payload_columns(parsed: pd.Series) -> dict:
    """Pull named fields out of a series of payload dicts."""
    def _get(*keys):
        def fn(p):
            if not isinstance(p, dict):
                return None
            for k in keys:
                if p.get(k) is not None:
                    return p[k]
            return None
        return parsed.apply(fn)

    return {
        "Market Name": _get("market_name"),
        "Price":       _get("price", "market_price"),
        # "post_prob" is the current key (see decision_pipeline.py's TRACK
        # payload); "fair_value"/"fair" are kept as fallbacks for rows logged
        # before the pre_prob/post_prob rename (and "fair_value" is still
        # what TradeExecutor's own AUTO-TRADE/DRY-RUN log entries use).
        "Fair Value":  _get("post_prob", "fair_value", "fair"),
        "EV":          _get("ev"),
        "Bet ($)":     _get("bet_usd", "bet_amount_usd"),
        "Shares":      _get("shares"),
        "Model Used":  _get("model_used"),
        "Reject Reason": _get("reason"),
        "Side":        _get("side"),
        # Phase 7: Wang pricing / strategy-source / risk-sizing analytics —
        # expanded into the existing payload JSON rather than new DB columns,
        # matching how Phases 3-5 already log these fields per trade.
        "Raw Prob":    _get("pre_prob", "raw_probability"),
        "Entry k":     _get("wang_lambda"),  # dict key kept for continuity; see
        # core/models.py TradeSignal.wang_lambda — now holds entry_k, not a
        # Wang lambda, hence the relabeled column.
        "Wang FV":     _get("wang_fair_value"),
        "Wang Edge":   _get("wang_edge"),
        "Strategy":    _get("strategy_type"),
        "Kelly Frac":  _get("kelly_fraction_used"),
        "Correlation": _get("correlation_exposure"),
    }


_NUMERIC_DISPLAY_COLS = [
    "Price", "Fair Value", "EV", "Bet ($)", "Shares",
    "Raw Prob", "Entry k", "Wang FV", "Wang Edge", "Kelly Frac", "Correlation",
]


def process_logs_for_display(df: pd.DataFrame) -> pd.DataFrame:
    _empty_cols = ["Time", "Action", "Asset", "Side", "Market Name", "Reject Reason", "Price", "Fair Value", "EV", "Bet ($)", "Shares", "Token"]
    if df is None or df.empty:
        return pd.DataFrame(columns=_empty_cols)

    out = df.copy()

    out["Token"] = (
        out["token_id"].astype(str).apply(lambda t: f"{t[:4]}...{t[-4:]}" if len(t) > 8 else t)
        if "token_id" in out.columns else "-"
    )

    parsed = out["payload"].apply(_parse_payload_value) if "payload" in out.columns else pd.Series([{}] * len(out))
    for col, series in _extract_payload_columns(parsed).items():
        out[col] = series

    for col in _NUMERIC_DISPLAY_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["_ts"] = pd.to_datetime(out.get("timestamp"), errors="coerce")
    out = out.sort_values("_ts", ascending=False)
    out = out.rename(columns={"timestamp": "Time", "level": "Action", "asset_type": "Asset"})
    out["Time"] = out["_ts"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("-")
    out = out.drop(columns=[c for c in ["payload", "token_id", "_ts"] if c in out.columns])

    desired = [
        "Time", "Action", "Asset", "Side", "Strategy", "Market Name", "Reject Reason", "Model Used",
        "Price", "Fair Value", "EV", "Raw Prob", "Entry k", "Wang FV", "Wang Edge",
        "Bet ($)", "Kelly Frac", "Correlation", "Shares", "Token",
    ]
    out = out[[c for c in desired if c in out.columns]]

    numeric_cols = set(_NUMERIC_DISPLAY_COLS)
    for col in out.columns:
        if col not in numeric_cols:
            out[col] = out[col].fillna("-")
    return out


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def fetch_latest_history(db_path: str, limit: int = 50) -> pd.DataFrame:
    try:
        with _open_db(db_path) as conn:
            df = pd.read_sql_query(
                "SELECT timestamp, level, asset_type, token_id, payload FROM hunt_history ORDER BY id DESC LIMIT ?",
                conn,
                params=(limit,),
            )
        return process_logs_for_display(df)
    except Exception:
        return pd.DataFrame()


def get_trade_stats(db_path: str) -> dict:
    """Dashboard trade stats, read from the Ledger (trading/ledger.py) —
    hunt_history is a diagnostics log, not a trade record."""
    return Ledger(db_path).trade_stats()


def get_closed_trade_deltas(db_path: str) -> list:
    """Per-share $ P&L (exit price - entry price) for every confirmed
    close (sale or paid-out resolution) in the Ledger."""
    return Ledger(db_path).closed_trade_deltas()


_SELL_REASON_LABELS = {
    "TAKE-PROFIT": "SELL (Take-Profit)", "STOP-LOSS": "SELL (Stop-Loss)",
    "WANG-EDGE-DECAY": "SELL (Edge Decay)", "EV-CONVERGENCE": "SELL (EV Converged)",
    "PORTFOLIO-CULL": "SELL (Cull)", "OPPORTUNITY-SWAP": "SELL (Swap)",
    "ARBITRAGE-TRIM": "SELL (Arb Trim)", "ARBITRAGE-UNWIND": "SELL (Arb Unwind)",
}

_TRANSACTIONS_COLUMNS = [
    "Time", "Type", "Side", "Market Name", "Price", "Shares",
    "Amount ($)", "P&L ($/share)", "Status",
]


def _transaction_type_label(trade: dict) -> str:
    if trade["kind"] == "BUY":
        return "BUY"
    if trade["kind"] == "SELL":
        return _SELL_REASON_LABELS.get(trade.get("reason") or "", "SELL")
    return "EXPIRED"  # RESOLVED (paid out) or EXPIRED (unconfirmed)


def fetch_transactions(db_path: str, limit: int = 100) -> pd.DataFrame:
    """Every buy, sale, resolution and expiry in the Ledger, newest first —
    backs the "Transactions" table folded into the Portfolio view."""
    try:
        trades = Ledger(db_path).transactions(limit=limit)
    except Exception:
        return pd.DataFrame(columns=_TRANSACTIONS_COLUMNS)

    records = []
    for t in trades:
        if t["kind"] == "BUY":
            status = "Confirmed"
        else:
            status = "Sold" if t["confirmed"] else "Unconfirmed"
        records.append({
            "Time": t["time"],
            "Type": _transaction_type_label(t),
            "Side": str(t.get("side") or "-").upper(),
            "Market Name": t.get("market_name") or str(t["token_id"]),
            "Price": t["price"],
            "Shares": t["shares"],
            "Amount ($)": t["amount_usd"],
            "P&L ($/share)": t["pnl_per_share"],
            "Status": status,
        })

    df = pd.DataFrame(records, columns=_TRANSACTIONS_COLUMNS)
    if df.empty:
        return df
    for col in ("Price", "Shares", "Amount ($)", "P&L ($/share)"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_equity_curve(db_path: str) -> pd.DataFrame:
    try:
        with _open_db(db_path) as conn:
            df = pd.read_sql_query(
                "SELECT timestamp, payload FROM hunt_history WHERE level = 'TRACK' ORDER BY id ASC",
                conn,
            )
    except Exception:
        return pd.DataFrame(columns=["timestamp", "total_equity"])

    if df.empty:
        return pd.DataFrame(columns=["timestamp", "total_equity"])

    payloads = df["payload"].apply(_parse_payload_value)
    df["total_equity"] = payloads.apply(
        lambda p: float(p["total_equity"]) if isinstance(p, dict) and p.get("total_equity") is not None else None
    )
    df = df.dropna(subset=["total_equity"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.dropna(subset=["timestamp"])[["timestamp", "total_equity"]]


def restore_runtime_state(db_path: str, fallback_starting_balance: float, now: datetime | None = None) -> dict:
    """Engine state to resume from after a restart or redeploy.

    spent_today comes from the Ledger (today's entry fills). Balances resume
    from the last engine_status the engine published before it stopped;
    they are refreshed from the Venue on the first sync after startup.
    hunt_history is not consulted: it is a diagnostics log, not state.
    """
    state = {
        "starting_balance": float(fallback_starting_balance),
        "current_balance": float(fallback_starting_balance),
        "start_of_day_equity": 0.0,
        "spent_today": 0.0,
        "source": "default",
    }

    today = (now or datetime.now()).strftime("%Y-%m-%d")
    try:
        state["spent_today"] = Ledger(db_path).spent_on(today)
    except Exception as e:
        print(f"[RESTORE] Could not read today's spend from the Ledger: {e}")

    status = read_engine_status(db_path)
    if status:
        balance = float(status.get("current_balance") or 0.0)
        starting = float(status.get("starting_balance") or 0.0)
        if balance > 0 or starting > 0:
            state["current_balance"] = balance
            state["starting_balance"] = starting if starting > 0 else balance
            state["source"] = "engine_status"

    return state


def get_paper_snapshots(db_path: str, limit: int = 100) -> list[dict]:
    try:
        with _open_db(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT timestamp, cash, positions_value, total_value FROM paper_snapshots ORDER BY timestamp ASC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(row) for row in rows]
    except Exception:
        return []

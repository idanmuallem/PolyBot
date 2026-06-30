import json
import os
import shutil
import sqlite3
from datetime import datetime

import pandas as pd


DEFAULT_DB_PATH = "/app/trades.db"
FALLBACK_DB_PATH = "trades.db"

_ACTIVE_DB_PATH: str | None = None


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

def _resolve_db(db_path: str) -> str:
    global _ACTIVE_DB_PATH
    if _ACTIVE_DB_PATH:
        return _ACTIVE_DB_PATH

    for candidate in [db_path, DEFAULT_DB_PATH, FALLBACK_DB_PATH]:
        normalized = str(candidate or "").strip()
        if not normalized:
            continue
        if os.path.isdir(normalized):
            normalized = os.path.join(normalized, "trades.db")
        parent = os.path.dirname(normalized)
        if parent:
            os.makedirs(parent, exist_ok=True)
        try:
            with sqlite3.connect(normalized, timeout=5) as conn:
                conn.execute("SELECT 1")
            _ACTIVE_DB_PATH = normalized
            return normalized
        except sqlite3.OperationalError:
            continue

    # Last resort: trust the caller's path even without verification
    _ACTIVE_DB_PATH = db_path or DEFAULT_DB_PATH
    return _ACTIVE_DB_PATH


def _open_db(db_path: str, timeout: int = 10):
    return sqlite3.connect(_resolve_db(db_path), timeout=timeout)


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

def init_db(db_path: str = DEFAULT_DB_PATH):
    resolved = _resolve_db(db_path)

    # Migrate legacy misnamed DB (trades.db/trades.db) if present
    legacy = os.path.normpath(os.path.join("trades.db", "trades.db"))
    if os.path.exists(legacy) and not os.path.exists(resolved):
        parent = os.path.dirname(resolved)
        if parent:
            os.makedirs(parent, exist_ok=True)
        shutil.copy2(legacy, resolved)

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


# ---------------------------------------------------------------------------
# Event logging
# ---------------------------------------------------------------------------

def log_event(bridge, level, asset_type, token_id, payload, db_path: str = DEFAULT_DB_PATH):
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
            bridge.opportunity_map[str(token_id)] = {
                "token_id": str(token_id),
                "asset_type": str(asset_type),
                "ev": float(payload.get("ev", 0.0)),
                "fair": float(payload.get("fair", 0.0)),
                "market_name": str(
                    payload.get("market_name")
                    or bridge.market_name_by_token.get(str(token_id), bridge.market_question)
                ),
            }
        except Exception:
            pass

    if bridge.event_count - bridge.last_summary_at >= 10:
        bridge.last_summary_at = bridge.event_count
        live = bridge.level_counts.get("LIVE-TRADE", 0)
        dry = bridge.level_counts.get("DRY-RUN", 0)
        paper = bridge.level_counts.get("PAPER-TRADE", 0)
        avg_ev = sum(bridge.ev_samples) / len(bridge.ev_samples) if bridge.ev_samples else 0.0
        max_ev = max(bridge.ev_samples) if bridge.ev_samples else 0.0
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
        "Fair Value":  _get("fair_value", "fair"),
        "EV":          _get("ev"),
        "Bet ($)":     _get("bet_usd", "bet_amount_usd"),
        "Shares":      _get("shares"),
        "Model Used":  _get("model_used"),
        "Reject Reason": _get("reason"),
        "Side":        _get("side"),
    }


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

    for col in ["Price", "Fair Value", "EV", "Bet ($)", "Shares"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["_ts"] = pd.to_datetime(out.get("timestamp"), errors="coerce")
    out = out.sort_values("_ts", ascending=False)
    out = out.rename(columns={"timestamp": "Time", "level": "Action", "asset_type": "Asset"})
    out["Time"] = out["_ts"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("-")
    out = out.drop(columns=[c for c in ["payload", "token_id", "_ts"] if c in out.columns])

    desired = ["Time", "Action", "Asset", "Side", "Market Name", "Reject Reason", "Model Used",
               "Price", "Fair Value", "EV", "Bet ($)", "Shares", "Token"]
    out = out[[c for c in desired if c in out.columns]]

    numeric_cols = {"Price", "Fair Value", "EV", "Bet ($)", "Shares"}
    for col in out.columns:
        if col not in numeric_cols:
            out[col] = out[col].fillna("-")
    return out


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def fetch_latest_history(limit: int = 50, db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
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


def get_trade_stats(db_path: str = DEFAULT_DB_PATH) -> dict:
    _empty = {
        "win_rate": 0.0, "total_trades": 0, "avg_win": 0.0, "avg_loss": 0.0,
        "total_yes_trades": 0, "yes_win_rate": 0.0, "total_no_trades": 0, "no_win_rate": 0.0,
    }
    try:
        with _open_db(db_path) as conn:
            df = pd.read_sql_query(
                "SELECT level, payload FROM hunt_history WHERE level IN ('LIVE-TRADE','DRY-RUN','PAPER-TRADE','AUTO-TRADE','TAKE-PROFIT','STOP-LOSS')",
                conn,
            )
    except Exception:
        return _empty

    if df.empty:
        return _empty

    EXEC_LEVELS = {"LIVE-TRADE", "DRY-RUN", "PAPER-TRADE", "AUTO-TRADE"}
    total_trades = yes_trades = no_trades = 0
    yes_wins = no_wins = yes_losses = no_losses = 0
    realized_win: list = []
    realized_loss: list = []

    for level, payload_raw in zip(df["level"], df["payload"]):
        payload = _parse_payload_value(payload_raw)
        if not isinstance(payload, dict):
            continue

        side = str(payload.get("side", "")).upper()
        gross = float(payload.get("price", 0.0) or 0.0) * float(payload.get("shares", 0.0) or 0.0)

        if level in EXEC_LEVELS:
            total_trades += 1
            if side == "YES":
                yes_trades += 1
            elif side == "NO":
                no_trades += 1

        if level == "TAKE-PROFIT" and gross > 0:
            realized_win.append(gross)
            if side == "YES":
                yes_wins += 1
            elif side == "NO":
                no_wins += 1
        elif level == "STOP-LOSS" and gross > 0:
            realized_loss.append(gross)
            if side == "YES":
                yes_losses += 1
            elif side == "NO":
                no_losses += 1

    denom = len(realized_win) + len(realized_loss)
    yes_denom = yes_wins + yes_losses
    no_denom = no_wins + no_losses

    return {
        "win_rate":        round(len(realized_win) / denom * 100.0, 2) if denom > 0 else 0.0,
        "total_trades":    total_trades,
        "avg_win":         round(sum(realized_win) / len(realized_win), 2) if realized_win else 0.0,
        "avg_loss":        round(sum(realized_loss) / len(realized_loss), 2) if realized_loss else 0.0,
        "total_yes_trades": yes_trades,
        "yes_win_rate":    round(yes_wins / yes_denom * 100.0, 2) if yes_denom > 0 else 0.0,
        "total_no_trades": no_trades,
        "no_win_rate":     round(no_wins / no_denom * 100.0, 2) if no_denom > 0 else 0.0,
    }


def get_equity_curve(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
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


def restore_runtime_state(db_path: str, fallback_starting_balance: float) -> dict:
    state = {
        "starting_balance": float(fallback_starting_balance),
        "current_balance": float(fallback_starting_balance),
        "start_of_day_equity": 0.0,
        "spent_today": 0.0,
        "source": "default",
    }

    try:
        with _open_db(db_path) as conn:
            rows = conn.execute(
                "SELECT level, payload FROM hunt_history WHERE level IN ('LOOP-SUMMARY','TRACK') ORDER BY id DESC LIMIT 250"
            ).fetchall()
    except Exception:
        return state

    for level, payload_raw in rows:
        payload = _parse_payload_value(payload_raw)
        if not isinstance(payload, dict):
            continue

        if state["source"] == "default":
            for key in ("cash", "current_balance", "available_cash", "total_equity"):
                val = payload.get(key)
                if val is not None:
                    try:
                        fval = float(val)
                        if fval > 0:
                            state["current_balance"] = fval
                            state["starting_balance"] = fval
                            state["source"] = f"db:{level}:{key}"
                            break
                    except Exception:
                        continue

        for key in ("start_of_day_equity", "spent_today"):
            val = payload.get(key)
            if val is not None:
                try:
                    state[key] = float(val)
                except Exception:
                    pass

        if state["source"] != "default" and state["start_of_day_equity"] > 0:
            break

    return state

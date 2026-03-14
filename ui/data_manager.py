import ast
import json
import os
import shutil
import sqlite3
from datetime import datetime

import pandas as pd


DEFAULT_DB_PATH = "/app/trades.db"
FALLBACK_DB_PATH = os.path.join("trading", "trades.db")
LEGACY_DB_PATH = os.path.join("trades.db", "trades.db")
SECONDARY_FALLBACK_DB_PATH = "trades.db"

_ACTIVE_DB_PATH = None


def _parse_payload_value(payload_value):
    if isinstance(payload_value, dict):
        return payload_value
    if payload_value is None:
        return {}
    payload_text = str(payload_value).strip()
    if not payload_text:
        return {}
    try:
        return ast.literal_eval(payload_text)
    except Exception:
        try:
            return json.loads(payload_text)
        except Exception:
            return {}


def _normalize_db_path(db_path: str) -> str:
    normalized = str(db_path or DEFAULT_DB_PATH)

    if os.path.normpath(normalized) == os.path.normpath("trades.db"):
        normalized = DEFAULT_DB_PATH

    if os.path.isdir(normalized):
        normalized = os.path.join(normalized, "trades.db")

    parent = os.path.dirname(normalized)
    if parent:
        try:
            os.makedirs(parent, exist_ok=True)
        except Exception:
            normalized = FALLBACK_DB_PATH
            fallback_parent = os.path.dirname(normalized)
            if fallback_parent:
                os.makedirs(fallback_parent, exist_ok=True)

    return normalized


def _migrate_legacy_db_if_needed(target_db_path: str):
    legacy_db = os.path.normpath(LEGACY_DB_PATH)
    target_db = os.path.normpath(target_db_path)

    if target_db == legacy_db:
        return

    if os.path.exists(target_db):
        return

    if os.path.exists(legacy_db):
        parent = os.path.dirname(target_db)
        if parent:
            os.makedirs(parent, exist_ok=True)
        shutil.copy2(legacy_db, target_db)


def _candidate_db_paths(db_path: str):
    global _ACTIVE_DB_PATH

    candidates = []
    if _ACTIVE_DB_PATH:
        candidates.append(_ACTIVE_DB_PATH)
    candidates.extend(
        [
            db_path,
            DEFAULT_DB_PATH,
            FALLBACK_DB_PATH,
            SECONDARY_FALLBACK_DB_PATH,
        ]
    )

    normalized = []
    seen = set()
    for candidate in candidates:
        path = _normalize_db_path(candidate)
        if path not in seen:
            seen.add(path)
            normalized.append(path)
    return normalized


def _open_connection_with_fallback(db_path: str, timeout: int = 10):
    global _ACTIVE_DB_PATH

    last_error = None
    for candidate in _candidate_db_paths(db_path):
        try:
            with sqlite3.connect(candidate, timeout=timeout) as conn:
                conn.execute("SELECT 1")
            _ACTIVE_DB_PATH = candidate
            return sqlite3.connect(candidate, timeout=timeout)
        except sqlite3.OperationalError as exc:
            last_error = exc

    raise sqlite3.OperationalError(
        f"unable to open database file for all candidates: {_candidate_db_paths(db_path)} | last_error={last_error}"
    )


def init_db(db_path: str = DEFAULT_DB_PATH):
    for candidate in _candidate_db_paths(db_path):
        _migrate_legacy_db_if_needed(candidate)
        try:
            with sqlite3.connect(candidate, timeout=10) as conn:
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
            global _ACTIVE_DB_PATH
            _ACTIVE_DB_PATH = candidate
            return
        except sqlite3.OperationalError:
            continue

    raise sqlite3.OperationalError(
        f"unable to open database file for all candidates: {_candidate_db_paths(db_path)}"
    )


def log_event(bridge, level, asset_type, token_id, payload, db_path: str = DEFAULT_DB_PATH):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload_str = str(payload)

    bridge.event_count += 1
    bridge.level_counts[str(level)] += 1

    if isinstance(payload, dict) and "ev" in payload:
        try:
            bridge.ev_samples.append(float(payload.get("ev", 0.0)))
            bridge.ev_samples = bridge.ev_samples[-300:]
        except Exception:
            pass

    payload_pretty = json.dumps(payload, ensure_ascii=False, sort_keys=True) if isinstance(payload, dict) else str(payload)
    print(f"[{ts}] [{level}] [{asset_type}] [{token_id}] {payload_pretty}")

    try:
        with _open_connection_with_fallback(db_path, timeout=10) as conn:
            conn.execute(
                """
                INSERT INTO hunt_history (timestamp, level, asset_type, token_id, payload)
                VALUES (?, ?, ?, ?, ?)
                """,
                (ts, str(level), str(asset_type), str(token_id), payload_str),
            )
    except Exception as e:
        print(f"[DASHBOARD] Log DB error: {e}")

    if isinstance(payload, dict) and "ev" in payload:
        try:
            payload_market_name = payload.get("market_name")
            bridge.opportunity_map[str(token_id)] = {
                "token_id": str(token_id),
                "asset_type": str(asset_type),
                "ev": float(payload.get("ev", 0.0)),
                "fair": float(payload.get("fair", 0.0)),
                "market_name": str(payload_market_name) if payload_market_name else bridge.market_name_by_token.get(str(token_id), bridge.market_question),
            }
        except Exception:
            pass

    if bridge.event_count - bridge.last_summary_at >= 10:
        bridge.last_summary_at = bridge.event_count
        live_trades = bridge.level_counts.get("LIVE-TRADE", 0)
        dry_run_trades = bridge.level_counts.get("DRY-RUN", 0)
        paper_trades = bridge.level_counts.get("PAPER-TRADE", 0)
        watch_only_skips = bridge.level_counts.get("WATCH-ONLY", 0)
        avg_ev = sum(bridge.ev_samples) / len(bridge.ev_samples) if bridge.ev_samples else 0.0
        max_ev = max(bridge.ev_samples) if bridge.ev_samples else 0.0

        print("=" * 90)
        print(
            f"[PERF] events={bridge.event_count} | live={live_trades} | dry_run={dry_run_trades} | "
            f"paper={paper_trades} | watch_only={watch_only_skips}"
        )
        print(
            f"[PERF] avg_ev={avg_ev:.4f} | max_ev={max_ev:.4f} | "
            f"daily_spend=${bridge.daily_spend:.2f} | est_balance=${bridge.current_balance:.2f}"
        )
        print(f"[PERF] level_mix={dict(bridge.level_counts)}")
        print("=" * 90)


def process_logs_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Time", "Action", "Side", "Market Name", "Price", "Fair Value", "EV", "Bet ($)", "Shares", "Token"])

    transformed = df.copy()

    if "token_id" in transformed.columns:
        transformed["Token"] = transformed["token_id"].astype(str).apply(
            lambda token: f"{token[:4]}...{token[-4:]}" if len(token) > 8 else token
        )
    else:
        transformed["Token"] = "-"

    parsed_payload = transformed["payload"].apply(_parse_payload_value) if "payload" in transformed.columns else pd.Series([{}] * len(transformed))

    transformed["Market Name"] = parsed_payload.apply(lambda p: p.get("market_name") if isinstance(p, dict) else None)
    transformed["Price"] = parsed_payload.apply(lambda p: p.get("price", p.get("market_price")) if isinstance(p, dict) else None)
    transformed["Fair Value"] = parsed_payload.apply(lambda p: p.get("fair_value", p.get("fair")) if isinstance(p, dict) else None)
    transformed["EV"] = parsed_payload.apply(lambda p: p.get("ev") if isinstance(p, dict) else None)
    transformed["Bet ($)"] = parsed_payload.apply(lambda p: p.get("bet_usd", p.get("bet_amount_usd")) if isinstance(p, dict) else None)
    transformed["Shares"] = parsed_payload.apply(lambda p: p.get("shares") if isinstance(p, dict) else None)
    transformed["Model Used"] = parsed_payload.apply(lambda p: p.get("model_used") if isinstance(p, dict) else None)
    transformed["Reject Reason"] = parsed_payload.apply(lambda p: p.get("reason") if isinstance(p, dict) else None)
    transformed["Side"] = parsed_payload.apply(lambda p: p.get("side") if isinstance(p, dict) else None)

    def _reject_metrics(payload):
        if not isinstance(payload, dict):
            return None

        numeric_keys = [
            "ev",
            "ev_yes",
            "ev_no",
            "threshold",
            "required_amount",
            "current_balance",
            "market_price",
            "fair_value",
            "position_size",
            "bet_amount_usd",
            "shares",
            "volume",
        ]

        parts = []
        for key in numeric_keys:
            if key in payload and payload.get(key) is not None:
                try:
                    value = float(payload.get(key))
                    parts.append(f"{key}={value:.4f}")
                except Exception:
                    parts.append(f"{key}={payload.get(key)}")

        return " | ".join(parts) if parts else None

    transformed["Reject Metrics"] = parsed_payload.apply(_reject_metrics)

    transformed["Price"] = pd.to_numeric(transformed["Price"], errors="coerce")
    transformed["Fair Value"] = pd.to_numeric(transformed["Fair Value"], errors="coerce")
    transformed["EV"] = pd.to_numeric(transformed["EV"], errors="coerce")
    transformed["Bet ($)"] = pd.to_numeric(transformed["Bet ($)"], errors="coerce")
    transformed["Shares"] = pd.to_numeric(transformed["Shares"], errors="coerce")

    transformed["timestamp_dt"] = pd.to_datetime(transformed.get("timestamp"), errors="coerce")
    transformed = transformed.sort_values("timestamp_dt", ascending=False)

    transformed = transformed.rename(columns={"timestamp": "Time", "level": "Action", "asset_type": "Asset"})
    transformed["Time"] = transformed["timestamp_dt"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("-")

    transformed = transformed.drop(columns=[c for c in ["payload", "token_id", "timestamp_dt"] if c in transformed.columns])

    desired_order = [
        "Time",
        "Action",
        "Asset",
        "Side",
        "Market Name",
        "Reject Reason",
        "Reject Metrics",
        "Model Used",
        "Price",
        "Fair Value",
        "EV",
        "Bet ($)",
        "Shares",
        "Token",
    ]
    transformed = transformed[[c for c in desired_order if c in transformed.columns]]

    numeric_cols = ["Price", "Fair Value", "EV", "Bet ($)", "Shares"]
    text_cols = [col for col in transformed.columns if col not in numeric_cols]
    for col in text_cols:
        transformed[col] = transformed[col].fillna("-")

    return transformed


def fetch_latest_history(limit: int = 50, db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Return display-ready latest history rows."""
    with _open_connection_with_fallback(db_path, timeout=10) as conn:
        history_df = pd.read_sql_query(
            "SELECT timestamp, level, asset_type, token_id, payload FROM hunt_history ORDER BY id DESC LIMIT ?",
            conn,
            params=(limit,),
        )
    return process_logs_for_display(history_df)


def get_trade_stats(db_path: str = DEFAULT_DB_PATH) -> dict:
    with _open_connection_with_fallback(db_path, timeout=10) as conn:
        trade_df = pd.read_sql_query(
            """
            SELECT level, payload
            FROM hunt_history
            WHERE level IN ('LIVE-TRADE', 'DRY-RUN', 'PAPER-TRADE', 'AUTO-TRADE', 'TAKE-PROFIT', 'STOP-LOSS')
            """,
            conn,
        )

    if trade_df.empty:
        return {
            "win_rate": 0.0,
            "total_trades": 0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }

    payloads = trade_df["payload"].apply(_parse_payload_value)

    realized_win = []
    realized_loss = []
    for level, payload in zip(trade_df["level"], payloads):
        if not isinstance(payload, dict):
            continue

        price = float(payload.get("price", 0.0) or 0.0)
        shares = float(payload.get("shares", 0.0) or 0.0)
        gross_value = price * shares

        if level == "TAKE-PROFIT" and gross_value > 0:
            realized_win.append(gross_value)
        elif level == "STOP-LOSS" and gross_value > 0:
            realized_loss.append(gross_value)

    trade_execution_levels = {"LIVE-TRADE", "DRY-RUN", "PAPER-TRADE", "AUTO-TRADE"}
    total_trades = int(trade_df["level"].isin(trade_execution_levels).sum())
    win_events = len(realized_win)
    loss_events = len(realized_loss)
    denom = win_events + loss_events
    win_rate = (float(win_events) / float(denom) * 100.0) if denom > 0 else 0.0

    yes_trades = 0
    no_trades = 0
    yes_wins = 0
    no_wins = 0
    yes_losses = 0
    no_losses = 0

    for idx, (level, payload) in enumerate(zip(trade_df["level"], payloads)):
        if not isinstance(payload, dict):
            continue
        if level not in trade_execution_levels:
            continue
        
        side = str(payload.get("side", "")).upper()
        if side == "YES":
            yes_trades += 1
        elif side == "NO":
            no_trades += 1

    for level, payload in zip(trade_df["level"], payloads):
        if not isinstance(payload, dict):
            continue
        side = str(payload.get("side", "")).upper()
        
        price = float(payload.get("price", 0.0) or 0.0)
        shares = float(payload.get("shares", 0.0) or 0.0)
        gross_value = price * shares
        
        if level == "TAKE-PROFIT" and gross_value > 0:
            if side == "YES":
                yes_wins += 1
            elif side == "NO":
                no_wins += 1
        elif level == "STOP-LOSS" and gross_value > 0:
            if side == "YES":
                yes_losses += 1
            elif side == "NO":
                no_losses += 1

    yes_denom = yes_wins + yes_losses
    no_denom = no_wins + no_losses
    yes_win_rate = (float(yes_wins) / float(yes_denom) * 100.0) if yes_denom > 0 else 0.0
    no_win_rate = (float(no_wins) / float(no_denom) * 100.0) if no_denom > 0 else 0.0

    return {
        "win_rate": round(win_rate, 2),
        "total_trades": total_trades,
        "avg_win": round(sum(realized_win) / len(realized_win), 2) if realized_win else 0.0,
        "avg_loss": round(sum(realized_loss) / len(realized_loss), 2) if realized_loss else 0.0,
        "total_yes_trades": yes_trades,
        "yes_win_rate": round(yes_win_rate, 2),
        "total_no_trades": no_trades,
        "no_win_rate": round(no_win_rate, 2),
    }


def get_equity_curve(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    with _open_connection_with_fallback(db_path, timeout=10) as conn:
        track_df = pd.read_sql_query(
            """
            SELECT id, timestamp, payload
            FROM hunt_history
            WHERE level = 'TRACK'
            ORDER BY id ASC
            """,
            conn,
        )

    if track_df.empty:
        return pd.DataFrame(columns=["timestamp", "total_equity"])

    payloads = track_df["payload"].apply(_parse_payload_value)
    track_df["total_equity"] = payloads.apply(
        lambda payload: float(payload.get("total_equity")) if isinstance(payload, dict) and payload.get("total_equity") is not None else None
    )
    track_df = track_df.dropna(subset=["total_equity"])
    if track_df.empty:
        return pd.DataFrame(columns=["timestamp", "total_equity"])

    track_df["timestamp"] = pd.to_datetime(track_df["timestamp"], errors="coerce")
    track_df = track_df.dropna(subset=["timestamp"])
    return track_df[["timestamp", "total_equity"]]


def get_system_throughput(db_path: str = DEFAULT_DB_PATH) -> dict:
    with _open_connection_with_fallback(db_path, timeout=10) as conn:
        throughput_df = pd.read_sql_query(
            """
            SELECT level, payload
            FROM hunt_history
            WHERE DATE(timestamp) = DATE('now', 'localtime')
            """,
            conn,
        )

    if throughput_df.empty:
        return {
            "total_scanned": 0,
            "total_rejected": 0,
            "timeouts_errors": 0,
        }

    level_series = throughput_df["level"].astype(str)
    payload_series = throughput_df["payload"].astype(str).str.lower()

    total_scanned = int(level_series.isin(["TRACK", "FILTERED", "SCAN-SKIP", "REJECTED"]).sum())
    total_rejected = int((level_series == "REJECTED").sum())
    timeouts_errors = int(
        level_series.str.contains("ERROR", na=False).sum()
        + payload_series.str.contains("timeout", na=False).sum()
    )

    return {
        "total_scanned": total_scanned,
        "total_rejected": total_rejected,
        "timeouts_errors": timeouts_errors,
    }

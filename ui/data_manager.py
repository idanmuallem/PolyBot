import ast
import json
import sqlite3
from datetime import datetime

import pandas as pd


def init_db(db_path: str = "trades.db"):
    with sqlite3.connect(db_path) as conn:
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


def log_event(bridge, level, asset_type, token_id, payload, db_path: str = "trades.db"):
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
        with sqlite3.connect(db_path, timeout=10) as conn:
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
        return pd.DataFrame(columns=["Time", "Action", "Market Name", "Price", "Fair Value", "EV", "Bet ($)", "Shares", "Token"])

    transformed = df.copy()

    if "token_id" in transformed.columns:
        transformed["Token"] = transformed["token_id"].astype(str).apply(
            lambda token: f"{token[:4]}...{token[-4:]}" if len(token) > 8 else token
        )
    else:
        transformed["Token"] = "-"

    def _parse_payload(payload_value):
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

    parsed_payload = transformed["payload"].apply(_parse_payload) if "payload" in transformed.columns else pd.Series([{}] * len(transformed))

    transformed["Market Name"] = parsed_payload.apply(lambda p: p.get("market_name") if isinstance(p, dict) else None)
    transformed["Price"] = parsed_payload.apply(lambda p: p.get("price", p.get("market_price")) if isinstance(p, dict) else None)
    transformed["Fair Value"] = parsed_payload.apply(lambda p: p.get("fair_value", p.get("fair")) if isinstance(p, dict) else None)
    transformed["EV"] = parsed_payload.apply(lambda p: p.get("ev") if isinstance(p, dict) else None)
    transformed["Bet ($)"] = parsed_payload.apply(lambda p: p.get("bet_usd", p.get("bet_amount_usd")) if isinstance(p, dict) else None)
    transformed["Shares"] = parsed_payload.apply(lambda p: p.get("shares") if isinstance(p, dict) else None)
    transformed["Model Used"] = parsed_payload.apply(lambda p: p.get("model_used") if isinstance(p, dict) else None)

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

    desired_order = ["Time", "Action", "Asset", "Market Name", "Model Used", "Price", "Fair Value", "EV", "Bet ($)", "Shares", "Token"]
    transformed = transformed[[c for c in desired_order if c in transformed.columns]]
    return transformed.fillna("-")


def fetch_latest_history(limit: int = 50, db_path: str = "trades.db") -> pd.DataFrame:
    """Return display-ready latest history rows."""
    with sqlite3.connect(db_path, timeout=10) as conn:
        history_df = pd.read_sql_query(
            "SELECT timestamp, level, asset_type, token_id, payload FROM hunt_history ORDER BY id DESC LIMIT ?",
            conn,
            params=(limit,),
        )
    return process_logs_for_display(history_df)

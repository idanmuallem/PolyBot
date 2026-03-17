import asyncio
import ast
import json
import os
import sqlite3
import threading

import streamlit as st

from trading.engine import run_market_monitor
import ui.data_manager as data_manager
from core.bridge import get_bridge
from ui.components import (
    render_equity_curve,
    render_ev_chart,
    render_history_table,
    render_kpis,
    render_positions,
    render_risk_gauge,
    render_system_throughput,
    render_terminal_feed,
    render_trade_stats,
)

st.set_page_config(page_title="PolyBot Quant Pro", page_icon="🛰️", layout="wide")
bridge = get_bridge()


def _as_bool(raw_value: str, default: bool) -> bool:
    if raw_value is None:
        return default
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def _validate_runtime_env() -> dict:
    required = ["POLYGON_PRIVATE_KEY", "POLY_ADDRESS"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise ValueError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Pass them at runtime with --env-file."
        )

    return {
        "dry_run": _as_bool(os.getenv("DRY_RUN", "true"), True),
        "paper_trade_mode": _as_bool(os.getenv("PAPER_TRADE_MODE", "false"), False),
        "daily_limit_usd": float(os.getenv("DAILY_LIMIT_USD", "100.0")),
        "paper_balance_usd": float(os.getenv("PAPER_BALANCE_USD", "1000.0")),
        "trades_db_path": os.getenv("TRADES_DB_PATH", "/app/trades.db"),
    }


def _parse_payload(payload_text: str) -> dict:
    if payload_text is None:
        return {}
    text = str(payload_text).strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return {}


def _restore_runtime_state(db_path: str, fallback_starting_balance: float) -> dict:
    state = {
        "starting_balance": float(fallback_starting_balance),
        "current_balance": float(fallback_starting_balance),
        "start_of_day_equity": 0.0,
        "spent_today": 0.0,
        "source": "default",
    }

    if not os.path.exists(db_path):
        return state

    try:
        with sqlite3.connect(db_path, timeout=10) as conn:
            rows = conn.execute(
                """
                SELECT level, payload
                FROM hunt_history
                WHERE level IN ('LOOP-SUMMARY', 'TRACK')
                ORDER BY id DESC
                LIMIT 250
                """
            ).fetchall()
    except Exception:
        return state

    for level, payload in rows:
        payload_obj = _parse_payload(payload)
        if not isinstance(payload_obj, dict):
            continue

        if state["source"] == "default":
            for key in ("cash", "current_balance", "available_cash", "total_equity"):
                if payload_obj.get(key) is not None:
                    try:
                        value = float(payload_obj.get(key))
                        if value > 0:
                            state["current_balance"] = value
                            state["starting_balance"] = value
                            state["source"] = f"db:{level}:{key}"
                            break
                    except Exception:
                        continue

        if payload_obj.get("start_of_day_equity") is not None:
            try:
                state["start_of_day_equity"] = float(payload_obj.get("start_of_day_equity"))
            except Exception:
                pass

        if payload_obj.get("spent_today") is not None:
            try:
                state["spent_today"] = float(payload_obj.get("spent_today"))
            except Exception:
                pass

        if state["source"] != "default" and state["start_of_day_equity"] > 0:
            break

    return state


runtime_env = _validate_runtime_env()
data_manager.init_db(runtime_env["trades_db_path"])
restored_state = _restore_runtime_state(
    db_path=runtime_env["trades_db_path"],
    fallback_starting_balance=runtime_env["paper_balance_usd"],
)

bridge.starting_balance = float(restored_state["starting_balance"])
bridge.current_balance = float(restored_state["current_balance"])
bridge.start_of_day_equity = float(restored_state["start_of_day_equity"])
bridge.spent_today = float(restored_state["spent_today"])
bridge.daily_spend = float(restored_state["spent_today"])
bridge.state_bootstrap_source = str(restored_state["source"])
bridge.live_trading = not (bool(runtime_env["dry_run"]) or bool(runtime_env["paper_trade_mode"]))


def dashboard_log_event(level, asset_type, token_id, payload):
    payload_dict = payload if isinstance(payload, dict) else {}
    reason = str(payload_dict.get("reason", "")).strip()
    market_name = str(payload_dict.get("market_name", "")).strip()
    ev_value = payload_dict.get("ev")
    detail = reason or market_name or str(payload)[:140]
    ev_suffix = f" | ev={ev_value}" if ev_value is not None else ""
    formatted_log = f"[{level}] {asset_type} - {detail}{ev_suffix}"
    bridge.terminal_logs.appendleft(formatted_log)

    if str(level) in {"REJECTED", "FILTERED", "SCAN-SKIP"} and token_id:
        bridge.seen_markets[str(token_id)] = str(payload_dict.get("market_name", ""))
    if len(bridge.seen_markets) > 500:
        keys = list(bridge.seen_markets.keys())
        for key in keys[:100]:
            bridge.seen_markets.pop(key, None)

    data_manager.log_event(
        bridge,
        level,
        asset_type,
        token_id,
        payload,
        db_path=runtime_env["trades_db_path"],
    )


if "engine_started" not in st.session_state:
    loop = asyncio.new_event_loop()
    threading.Thread(
        target=lambda: loop.run_until_complete(run_market_monitor(bridge, dashboard_log_event)),
        daemon=True,
    ).start()
    st.session_state.engine_started = True

st.title("🛰️ PolyBot: Quantitative Arbitrage Terminal")
st.caption("Real-time probability discovery and execution monitoring")

with st.sidebar:
    st.header("⚙️ Trading Mode")
    mode = st.toggle("Live Trading", value=bridge.live_trading)
    bridge.live_trading = bool(mode)
    st.write("Mode:", "Live Trading" if bridge.live_trading else "Dry Run")

    dry_run_enabled = not bridge.live_trading
    dot_color = "#16a34a" if dry_run_enabled else "#dc2626"
    dot_label = "DRY_RUN ENABLED" if dry_run_enabled else "DRY_RUN DISABLED"
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:8px;'>"
        f"<span style='height:10px;width:10px;border-radius:50%;background:{dot_color};display:inline-block;'></span>"
        f"<span>{dot_label}</span></div>",
        unsafe_allow_html=True,
    )

    if bridge.watch_only:
        st.warning("Watch-Only mode enabled by Balance Guard")

# PERFORMANCE: placeholders for partial redraws
row0_placeholder = st.empty()
row1_placeholder = st.empty()
row2_placeholder = st.empty()
row3_placeholder = st.empty()
row4_placeholder = st.empty()


def _render_dashboard_snapshot():
    current_token = str(getattr(bridge, "current_token_id", ""))
    if current_token:
        bridge.market_name_by_token[current_token] = bridge.market_question

    with row0_placeholder.container():
        render_kpis(bridge)

    with row1_placeholder.container():
        c1, c2 = st.columns([1, 1])
        with c1:
            render_system_throughput(data_manager, bridge)
        with c2:
            render_trade_stats(data_manager)

    with row2_placeholder.container():
        col1, col2 = st.columns([2, 1])
        with col1:
            render_equity_curve(data_manager)
        with col2:
            render_risk_gauge(bridge)

    with row3_placeholder.container():
        col1, col2 = st.columns([1, 1])
        with col1:
            render_ev_chart(bridge)
        with col2:
            render_positions(bridge)

    with row4_placeholder.container():
        st.divider()
        col1, col2 = st.columns([1, 2])
        with col1:
            render_terminal_feed(bridge)
        with col2:
            render_history_table(data_manager)


if hasattr(st, "fragment"):
    @st.fragment(run_every="2s")
    def live_fragment():
        _render_dashboard_snapshot()

    live_fragment()
else:
    _render_dashboard_snapshot()
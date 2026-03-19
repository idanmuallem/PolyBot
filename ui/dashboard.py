import asyncio
import ast
import json
import os
import sqlite3
import threading
import time
import pandas as pd
import streamlit as st

from trading.engine import run_market_monitor
import ui.data_manager as data_manager
from clients.polymarket import PolymarketClient
from core.bridge import get_bridge
from ui.components import render_equity_curve, render_ev_chart, render_positions

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


def _fetch_live_balance() -> tuple[float, bool]:
    proxy_address = str(os.getenv("POLY_ADDRESS", "")).strip()
    private_key = str(os.getenv("POLYGON_PRIVATE_KEY", "")).strip()
    try:
        balance = float(
            PolymarketClient().get_proxy_balance(
                proxy_address=proxy_address,
                private_key=private_key,
            )
        )
        return max(0.0, balance), True
    except Exception as exc:
        bridge.terminal_logs.appendleft(f"[BALANCE-ERROR] {exc}")
        return 0.0, False


def _render_global_kpis() -> None:
    c1, c2 = st.columns([1, 1])
    current_balance_label = (
        "$0.00 (Connection Error)"
        if bool(getattr(bridge, "balance_connection_error", False))
        else f"${float(getattr(bridge, 'current_balance', 0.0)):,.2f}"
    )
    c1.metric("Current Balance", current_balance_label)
    c2.metric("Total PnL", f"${float(getattr(bridge, 'total_pnl', 0.0)):,.2f}")


def _render_hunter_history_table() -> None:
    history_df = data_manager.fetch_latest_history(limit=80)
    if history_df.empty:
        st.info("No hunt history yet. Engine is scanning markets...")
        return

    keep_cols = ["Time", "Asset", "Side", "EV", "Action"]
    compact_df = history_df[[col for col in keep_cols if col in history_df.columns]].copy()
    if compact_df.empty:
        st.info("No display-ready events yet.")
        return

    if "EV" in compact_df.columns:
        compact_df["EV"] = pd.to_numeric(compact_df["EV"], errors="coerce")

    def _ev_color(value):
        if pd.isna(value):
            return ""
        numeric = float(value)
        if numeric > 0.5:
            return "color: #22c55e; font-weight: 700;"
        if numeric < 0.0:
            return "color: #ef4444; font-weight: 700;"
        return ""

    styled = compact_df.style
    if "EV" in compact_df.columns:
        styled = styled.format({"EV": "{:.3f}"}).map(_ev_color, subset=["EV"])
    st.dataframe(styled, hide_index=True, use_container_width=True)


def _render_compact_terminal_feed() -> None:
    logs = list(getattr(bridge, "terminal_logs", []))[:20]
    if not logs:
        st.info("No terminal logs yet.")
        return
    st.code("\n".join(logs), language="text")


def _render_hunter_view() -> None:
    st.markdown("### Hunter")
    col1, col2 = st.columns([2, 1])
    with col1:
        _render_hunter_history_table()
    with col2:
        _render_compact_terminal_feed()


def _render_portfolio_view() -> None:
    st.markdown("### Portfolio")
    render_positions(bridge)
    st.markdown("#### EV by Market")
    render_ev_chart(bridge)


def _render_balance_stats_row() -> None:
    stats = data_manager.get_trade_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Win Rate", f"{float(stats.get('win_rate', 0.0)):.2f}%")
    c2.metric("Total Trades", f"{int(stats.get('total_trades', 0))}")
    c3.metric("Avg Win", f"${float(stats.get('avg_win', 0.0)):,.2f}")
    c4.metric("Avg Loss", f"${float(stats.get('avg_loss', 0.0)):,.2f}")


def _render_balance_view() -> None:
    st.markdown("### Balance")
    _render_balance_stats_row()
    render_equity_curve(data_manager)


def _render_active_view(view_name: str) -> None:
    if view_name == "Hunter":
        _render_hunter_view()
        return
    if view_name == "Portfolio":
        _render_portfolio_view()
        return
    _render_balance_view()


def _ensure_engine_started_once() -> None:
    # The loop/thread references are stored in session_state so Streamlit reruns
    # (including navigation changes) do not spawn duplicate monitor workers.
    if st.session_state.get("engine_started"):
        return

    loop = asyncio.new_event_loop()

    def _runner() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_market_monitor(bridge, dashboard_log_event))

    thread = threading.Thread(target=_runner, daemon=True, name="polybot-engine")
    thread.start()

    st.session_state.engine_loop = loop
    st.session_state.engine_thread = thread
    st.session_state.engine_started = True


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


runtime_env = _validate_runtime_env()
data_manager.init_db(runtime_env["trades_db_path"])
restored_state = _restore_runtime_state(
    db_path=runtime_env["trades_db_path"],
    fallback_starting_balance=0.0,
)

live_balance, live_balance_ok = _fetch_live_balance()

bridge.starting_balance = float(restored_state["starting_balance"])
bridge.current_balance = float(live_balance)
bridge.balance_connection_error = not bool(live_balance_ok)
bridge.start_of_day_equity = float(restored_state["start_of_day_equity"])
bridge.spent_today = float(restored_state["spent_today"])
bridge.daily_spend = float(restored_state["spent_today"])
bridge.state_bootstrap_source = str(restored_state["source"])
bridge.live_trading = not (bool(runtime_env["dry_run"]) or bool(runtime_env["paper_trade_mode"]))
bridge.last_balance_sync_at = time.time()

if bridge.starting_balance <= 0.0:
    bridge.starting_balance = float(live_balance)
_ensure_engine_started_once()

st.title("🛰️ PolyBot: Quantitative Arbitrage Terminal")
st.caption("Minimal live terminal for scan, exposure, and balance decisions")

with st.sidebar:
    st.header("Navigation")
    if hasattr(st, "segmented_control"):
        active_view = st.segmented_control(
            "View",
            options=["Hunter", "Portfolio", "Balance"],
            default="Hunter",
        )
    else:
        active_view = st.radio("View", ["Hunter", "Portfolio", "Balance"], index=0)

    st.divider()
    st.header("Trading Mode")
    mode = st.toggle("Live Trading", value=bridge.live_trading)
    bridge.live_trading = bool(mode)
    st.caption("Live Trading" if bridge.live_trading else "Dry Run")

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


def _render_dashboard_snapshot(view_name: str):
    now_ts = time.time()
    last_sync = float(getattr(bridge, "last_balance_sync_at", 0.0) or 0.0)
    if (now_ts - last_sync) >= 15.0:
        live_balance, live_balance_ok = _fetch_live_balance()
        bridge.current_balance = float(live_balance)
        bridge.balance_connection_error = not bool(live_balance_ok)
        bridge.last_balance_sync_at = now_ts

    current_token = str(getattr(bridge, "current_token_id", ""))
    if current_token:
        bridge.market_name_by_token[current_token] = bridge.market_question

    _render_global_kpis()
    st.divider()
    _render_active_view(view_name)


if hasattr(st, "fragment"):
    @st.fragment(run_every="2s")
    def live_fragment():
        _render_dashboard_snapshot(active_view)

    live_fragment()
else:
    _render_dashboard_snapshot(active_view)
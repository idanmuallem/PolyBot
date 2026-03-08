import asyncio
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

data_manager.init_db()


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

    data_manager.log_event(bridge, level, asset_type, token_id, payload)


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
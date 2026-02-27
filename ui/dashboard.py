import asyncio
import threading

import streamlit as st

from orchestration.engine import run_market_monitor
import ui.data_manager as data_manager
from core.bridge import get_bridge
from ui.components import (
    render_ev_chart,
    render_history_table,
    render_kpis,
    render_positions,
)

st.set_page_config(page_title="PolyBot Quant Pro", page_icon="🛰️", layout="wide")
bridge = get_bridge()

data_manager.init_db()


def dashboard_log_event(level, asset_type, token_id, payload):
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
kpi_placeholder = st.empty()
chart_placeholder = st.empty()
positions_placeholder = st.empty()
history_placeholder = st.empty()


def _render_dashboard_snapshot():
    current_token = str(getattr(bridge, "current_token_id", ""))
    if current_token:
        bridge.market_name_by_token[current_token] = bridge.market_question

    with kpi_placeholder.container():
        with st.container():
            render_kpis(bridge)

    with chart_placeholder.container():
        with st.container():
            render_ev_chart(bridge)

    with positions_placeholder.container():
        with st.container():
            render_positions(bridge)

    with history_placeholder.container():
        with st.container():
            render_history_table(data_manager)


if hasattr(st, "fragment"):
    @st.fragment(run_every="2s")
    def live_fragment():
        _render_dashboard_snapshot()

    live_fragment()
else:
    _render_dashboard_snapshot()
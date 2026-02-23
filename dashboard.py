# dashboard.py
import streamlit as st
import threading
import asyncio
import sqlite3
import pandas as pd
import time
from datetime import datetime
from engine import run_market_monitor

# --- DATA BRIDGE ---
class DataBridge:
    def __init__(self):
        self.market_actual = 0.0
        self.market_poly = 0.0
        self.forecast = 0.0
        self.ev = 0.0
        self.status = "📡 Discovery Phase..."
        self.last_update = "N/A"
        self.automation_enabled = False

@st.cache_resource
def get_bridge(): return DataBridge()
bridge = get_bridge()

def log_trade(m, s, p, sz, stts):
    try:
        with sqlite3.connect("trades.db", timeout=10) as conn:
            conn.execute("INSERT INTO trades (timestamp, market, side, price, size, status) VALUES (?,?,?,?,?,?)", 
                         (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), m, s, p, sz, stts))
    except Exception as e: print(f"Log Error: {e}")

# --- UI CONFIG ---
st.set_page_config(page_title="PolyBot Quant Pro", page_icon="🛰️", layout="wide")

# Starts the engine loop in the background
if "engine_started" not in st.session_state:
    loop = asyncio.new_event_loop()
    threading.Thread(target=lambda: loop.run_until_complete(run_market_monitor(bridge, log_trade)), daemon=True).start()
    st.session_state.engine_started = True

# --- HEADER SECTION ---
st.title("🛰️ PolyBot: Quantitative Arbitrage Terminal")
st.markdown("##### Real-time latency exploitation between Binance Spot and Polymarket CLOB")

with st.container(border=True):
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("System Health", "🟢 ONLINE" if bridge.market_actual > 0 else "🟠 CONNECTING")
    h2.metric("Engine Status", bridge.status)
    h3.metric("Last Data Ping", bridge.last_update)
    h4.metric("Autopilot", "ACTIVE" if bridge.automation_enabled else "OFFLINE")

st.divider()

# --- CORE MARKET DATA ---
st.subheader("📊 Live Market Analysis")
with st.container(border=True):
    m1, m2, m3, m4 = st.columns(4)
    
    m1.metric("Binance BTC", f"${bridge.market_actual:,.2f}")
    m2.metric("Polymarket Price", f"${bridge.market_poly:.3f}")
    m3.metric("Math-Fair Value", f"{bridge.forecast:.1%}")
    
    # Delta coloring automatically switches based on positive/negative
    edge_color = "normal" if bridge.ev < 0.15 else "inverse"
    m4.metric("Trading Edge (EV)", f"{bridge.ev:.2%}", delta=f"{bridge.ev:.2%}", delta_color=edge_color)

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Bot Controls")
    bridge.automation_enabled = st.toggle("Enable Probability Trading", value=bridge.automation_enabled)
    
    st.divider()
    st.subheader("Manual Execution")
    trade_size = st.number_input("Shares to Buy", min_value=1, value=10)
    if st.button("Execute Manual Trade", type="primary", use_container_width=True):
        log_trade("AUTO-DETECT", "BUY", bridge.market_poly, trade_size, "MANUAL-LIVE")
        st.toast("Trade Executed!")

# --- RECENT ACTIVITY ---
st.subheader("📜 Recent Activity & Execution Ledger")
with st.container(border=True):
    try:
        with sqlite3.connect("trades.db", timeout=10) as conn:
            df = pd.read_sql_query("SELECT * FROM trades ORDER BY id DESC LIMIT 15", conn)
        
        if not df.empty:
            st.dataframe(df, width='stretch', hide_index=True)
        else:
            st.info("No trades logged yet. The engine is scanning for arbitrage opportunities...")
    except Exception:
        st.warning("Ledger is busy updating...")

# UI Heartbeat
time.sleep(2)
st.rerun()
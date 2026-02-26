# dashboard.py
import streamlit as st
import threading
import asyncio
import sqlite3
import pandas as pd
import time
from datetime import datetime
from engine import run_market_monitor

# --- UI CONFIG ---
# Must be the first Streamlit command
st.set_page_config(page_title="PolyBot Quant Pro", page_icon="🛰️", layout="wide")

# --- DB INIT ---
def init_db():
    with sqlite3.connect("trades.db") as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                market TEXT,
                side TEXT,
                price TEXT,
                size TEXT,
                status TEXT
            )
        """)
init_db()

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
        self.market_question = "Waiting for market..."
        self.market_asset_type = "N/A"

@st.cache_resource
def get_bridge(): return DataBridge()
bridge = get_bridge()

def log_trade(m, s, p, sz, stts="N/A", market_name="Unknown"):
    try:
        with sqlite3.connect("trades.db", timeout=10) as conn:
            conn.execute("INSERT INTO trades (timestamp, market, side, price, size, status) VALUES (?,?,?,?,?,?)", 
                         (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), str(market_name), str(s), str(p), str(sz), str(stts)))
    except Exception as e: print(f"Log Error: {e}")

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
st.caption(f"**Market:** {bridge.market_question}")

def format_live_value(asset_type, value):
    if asset_type.startswith("Crypto"):
        return f"${value:,.2f}"
    elif asset_type.startswith("Weather"):
        return f"{value:.1f}°C"
    elif asset_type.startswith("Economy"):
        return f"{value:.2f}%"
    else:
        return f"{value:,.2f}"

with st.container(border=True):
    m1, m2, m3, m4 = st.columns(4)
    
    asset_name = bridge.market_asset_type.split("::")[-1]
    m1.metric(f"Live: {asset_name}", format_live_value(bridge.market_asset_type, bridge.market_actual))
    m2.metric("Polymarket Price", f"${bridge.market_poly:.3f}")
    m3.metric("Math-Fair Value", f"{bridge.forecast:.1%}")
    
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
            # Convert all columns to string to avoid pyarrow type conversion errors
            df = df.astype(str)
            st.dataframe(df, width='stretch', hide_index=True)
        else:
            st.info("No trades logged yet. The engine is scanning for arbitrage opportunities...")
    except Exception as e:
        st.warning(f"Ledger is busy updating... ({str(e)[:50]})")

# UI Heartbeat
time.sleep(2)
st.rerun()
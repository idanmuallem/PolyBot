from collections import defaultdict
import collections

import streamlit as st


class DataBridge:
    def __init__(self):
        self.market_actual = 0.0
        self.market_poly = 0.0
        self.forecast = 0.0
        self.ev = 0.0
        self.status = "📡 Discovery Phase..."
        self.last_update = "N/A"
        self.market_question = "Waiting for market..."
        self.market_asset_type = "N/A"
        self.current_balance = 0.0
        self.daily_spend = 0.0
        self.watch_only = False
        self.live_trading = False
        self.opportunity_map = {}
        self.market_name_by_token = {}
        self.current_portfolio = []
        self.open_position_value = 0.0
        self.total_pnl = 0.0
        self.event_count = 0
        self.level_counts = defaultdict(int)
        self.ev_samples = []
        self.last_summary_at = 0
        self.terminal_logs = collections.deque(maxlen=20)
        self.seen_markets = {}


@st.cache_resource
def get_bridge() -> DataBridge:
    """Shared singleton bridge for dashboard reruns and engine thread."""
    return DataBridge()

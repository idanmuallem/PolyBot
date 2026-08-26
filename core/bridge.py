from collections import defaultdict, deque
from typing import Optional

import streamlit as st


class DataBridge:
    def __init__(self, wallet_id: Optional[str] = None):
        self.wallet_id = wallet_id       # which wallet this instance belongs to (None for the legacy singleton)
        self._engine_thread = None  # persists across Streamlit reruns via cache_resource
        self.market_actual = 0.0
        self.market_poly = 0.0
        self.forecast = 0.0
        self.ev = 0.0
        self.status = "📡 Discovery Phase..."
        self.last_update = "N/A"
        self.market_question = "Waiting for market..."
        self.market_asset_type = "N/A"
        self.starting_balance = 0.0
        self.current_balance = 0.0
        self.cash = 0.0
        self.balance_connection_error = False
        self.daily_spend = 0.0
        self.spent_today = 0.0
        self.start_of_day_equity = 0.0
        self.state_bootstrap_source = "uninitialized"
        self.watch_only = False
        self.live_trading = False
        self.drawdown_paused = False  # mirrors WalletContext.drawdown_paused, for dashboard display
        self.peak_equity = 0.0
        self.current_drawdown_pct = 0.0
        self.position_analytics = {}  # token_id -> Wang edge-decay/asset_type snapshot (see PortfolioManager)
        self.opportunity_map = {}
        self.market_name_by_token = {}
        self.current_portfolio = []
        self.open_position_value = 0.0
        self.open_positions_value = 0.0
        self.unrealized_pnl = 0.0
        self.event_count = 0
        self.level_counts = defaultdict(int)
        self.ev_samples = []
        self.last_summary_at = 0
        self.terminal_logs = deque(maxlen=20)
        self.seen_markets = {}


@st.cache_resource
def get_bridge() -> DataBridge:
    """Backward-compatible singleton bridge for the legacy single-wallet dashboard.

    This is not the only way to get a DataBridge — DataBridge is a plain class
    and multi-wallet callers should instantiate it directly (one per wallet,
    e.g. DataBridge(wallet_id="wallet_alpha")) rather than sharing this
    process-wide instance.
    """
    return DataBridge()

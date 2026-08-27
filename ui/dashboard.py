import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Ensure stdout/stderr handle Unicode (e.g. market names with ↓ ↑ symbols).
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import threading
import time

import pandas as pd
import streamlit as st

import ui.data_manager as data_manager
from polymarket import PolymarketClient
from core.bridge import get_bridge, DataBridge
from core.runtime_env import get_env as _get_env, restore_wallet_state, validate_runtime_env
from core.wallet_context import WalletContext
from ui.components import (
    fmt_dollars, render_activity_chart, render_equity_curve,
    render_ev_chart, render_positions,
)

st.set_page_config(page_title="PolyBot Quant Pro", page_icon="🛰️", layout="wide")


_balance_lock = threading.Lock()
_balance_value: float = 0.0
_balance_ok: bool = False
_balance_fetching: bool = False


def _do_balance_fetch() -> None:
    global _balance_value, _balance_ok, _balance_fetching
    try:
        proxy_address = _get_env("POLYMARKET_PROXY_ADDRESS", "POLY_ADDRESS")
        private_key = _get_env("POLYMARKET_PRIVATE_KEY", "POLYGON_PRIVATE_KEY")
        balance = float(
            PolymarketClient().get_proxy_balance(proxy_address=proxy_address, private_key=private_key)
        )
        with _balance_lock:
            _balance_value = max(0.0, balance)
            _balance_ok = True
    except Exception as exc:
        bridge.terminal_logs.appendleft(f"[BALANCE-ERROR] {exc}")
        with _balance_lock:
            _balance_ok = False
    finally:
        with _balance_lock:
            _balance_fetching = False


def _fetch_live_balance() -> tuple[float, bool]:
    """Return last-known balance immediately; refresh in background if idle."""
    global _balance_fetching
    with _balance_lock:
        already = _balance_fetching
        val, ok = _balance_value, _balance_ok
    if not already:
        with _balance_lock:
            _balance_fetching = True
        threading.Thread(target=_do_balance_fetch, daemon=True, name="balance-fetch").start()
    return val, ok


# ---------------------------------------------------------------------------
# Bootstrap (runs once per session; guarded against re-init)
# ---------------------------------------------------------------------------
#
# The dashboard no longer starts the trading engine itself — it used to
# (see git history for _ensure_engine_started_once / _make_log_event,
# removed here), spawning a background thread the first time a browser
# loaded this page. That tied the whole bot's operation to a browser being
# open, which defeated the point of running it unattended on a server: no
# browser connected meant no scanning, no trading, silently. The engine now
# runs as its own OS process (run_engine.py), started at container boot
# regardless of whether anyone ever opens this dashboard — see
# config/Docker/entrypoint.sh. This module is purely a reader: everything
# below comes from trades.db (see ui/data_manager.py's engine_status/
# engine_control/open_positions tables), never from an in-process engine
# thread.

def _bootstrap() -> None:
    global bridge, wallet_ctx
    # bridge is the cached @st.cache_resource singleton — same instance across
    # reruns. wallet_ctx itself is rebuilt each rerun (cheap; picks up env
    # changes) but always wraps that same bridge instance.
    bridge = get_bridge()
    wallet_ctx = validate_runtime_env(bridge)
    restore_wallet_state(wallet_ctx)

    # Kick off balance fetch in background — don't block startup.
    _fetch_live_balance()

    bridge.balance_connection_error = False
    bridge.last_balance_sync_at = 0.0


bridge: DataBridge = None
wallet_ctx: WalletContext = None
_bootstrap()


# ---------------------------------------------------------------------------
# KPI / status bar
# ---------------------------------------------------------------------------

def _render_global_kpis() -> None:
    c1, c2 = st.columns(2)
    # cash comes from engine_status (DB) via _compute_balance_snapshot() —
    # accurate in both modes, since the engine process syncs its own live
    # balance every loop tick regardless of dry-run/live. balance_connection_error
    # stays a dashboard-local diagnostic: it's only ever set True by this
    # process's own live-mode Polymarket balance poll (_fetch_live_balance),
    # unrelated to the split.
    cash, _, balance, total_deposits = _compute_balance_snapshot()
    balance_label = (
        "$0.00 (Connection Error)"
        if bool(getattr(bridge, "balance_connection_error", False))
        else fmt_dollars(cash)
    )
    c1.metric("Current Balance", balance_label)

    # Total PnL = Balance - Total Deposits: the true all-in figure (realized
    # + unrealized + fees + slippage), not just the open-positions-only
    # unrealized P&L (still computed engine-side by
    # PortfolioManager._refresh_portfolio() for anything that wants the
    # narrower figure). Shares _compute_balance_snapshot() with the Balance
    # view's own stats row so this KPI is numerically guaranteed to equal
    # "Balance - Total Deposits" as shown there.
    c2.metric("Total PnL", fmt_dollars(balance - total_deposits))


# ---------------------------------------------------------------------------
# Hunter view
# ---------------------------------------------------------------------------

def _render_hunter_history_table() -> None:
    history_df = data_manager.fetch_latest_history(wallet_ctx.db_path, limit=80)
    if history_df.empty:
        st.info("No hunt history yet. Engine is scanning markets...")
        return

    keep_cols = [
        "Time", "Action", "Asset", "Side", "Strategy", "EV",
        "Raw Prob", "Wang λ", "Wang Edge", "Market Name", "Reject Reason",
    ]
    compact_df = history_df[[col for col in keep_cols if col in history_df.columns]].copy()

    if "Market Name" in compact_df.columns:
        compact_df["Market Name"] = compact_df["Market Name"].apply(
            lambda v: v[:55] + "…" if isinstance(v, str) and len(v) > 55 else v
        )

    if compact_df.empty:
        st.info("No display-ready events yet.")
        return

    if "EV" in compact_df.columns:
        compact_df["EV"] = pd.to_numeric(compact_df["EV"], errors="coerce")

    def _ev_color(value):
        if pd.isna(value):
            return ""
        if float(value) > 0.5:
            return "color: #22c55e; font-weight: 700;"
        if float(value) < 0.0:
            return "color: #ef4444; font-weight: 700;"
        return ""

    styled = compact_df.style
    if "EV" in compact_df.columns:
        styled = styled.format({"EV": "{:.3f}"}).map(_ev_color, subset=["EV"])
    st.dataframe(styled, hide_index=True, use_container_width=True)


def _render_compact_terminal_feed() -> None:
    # Was bridge.terminal_logs (an in-memory rolling deque, filled live by
    # the engine's log_func) — now derived from hunt_history, since the
    # engine that fills it runs in a separate process (see run_engine.py).
    logs = data_manager.get_terminal_feed(wallet_ctx.db_path, limit=20)
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
    st.markdown("#### Activity Breakdown")
    render_activity_chart(data_manager.get_level_counts(wallet_ctx.db_path))


# ---------------------------------------------------------------------------
# Portfolio view
# ---------------------------------------------------------------------------

def _render_portfolio_view() -> None:
    st.markdown("### Portfolio")
    positions = data_manager.read_open_positions(wallet_ctx.db_path)
    opportunity_map = data_manager.get_recent_opportunity_map(wallet_ctx.db_path)
    render_positions(positions, opportunity_map)
    st.markdown("#### EV by Market")
    render_ev_chart(data_manager.get_latest_ev_by_token(wallet_ctx.db_path, limit=15))


# ---------------------------------------------------------------------------
# Balance view
# ---------------------------------------------------------------------------

def _avg(values: list) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def _compute_balance_snapshot() -> tuple[float, float, float, float]:
    """Returns (cash, holdings, balance, total_deposits).

    Reads cash/holdings/starting_balance from engine_status (DB), written
    once per loop tick by the engine process (see run_forever() in
    trading/decision_pipeline.py) — not from an in-process bridge, since the
    engine now runs as its own OS process (see run_engine.py) with no shared
    memory with this one. Previously this read straight off bridge state
    shared in-process; that's no longer possible post-split, and reading the
    real engine (rather than a live wallet API call) here also sidesteps the
    paper engine's thread-affine sqlite3 connection (see
    trading/paper_adapter.py) the same way the in-process version did.

    Shared by both the top-level "Total PnL" KPI and the Balance view's own
    stats row so the two stay numerically consistent by construction
    (Total PnL = Balance - Total Deposits) rather than by two separately
    maintained copies of the same formula drifting apart.
    """
    status = data_manager.read_engine_status(wallet_ctx.db_path) or {}
    is_paper_mode = not getattr(bridge, "live_trading", False)
    cash = float(status.get("current_balance", 0.0) or 0.0)
    holdings = float(status.get("open_position_value", 0.0) or 0.0)
    balance = cash + holdings

    # Total Deposits: paper mode has a stable, known constant (the account
    # is never topped up mid-run) — use it directly rather than
    # starting_balance, which is derived from the *most recent* balance
    # snapshot on every process restart (see restore_runtime_state() in
    # ui/data_manager.py), not the true original deposit. Live mode has no
    # such constant to fall back on, so starting_balance is the best
    # available approximation there — but it carries that same
    # restart-drift caveat, AND doesn't account for multiple real-world
    # deposits/withdrawals over an account's lifetime (no deposit ledger
    # exists to track those). Both are known limitations, not fixed here.
    if is_paper_mode:
        total_deposits = float(wallet_ctx.config.paper_balance_usd)
    else:
        total_deposits = float(status.get("starting_balance", 0.0) or 0.0)

    return cash, holdings, balance, total_deposits


def _render_balance_stats_row() -> None:
    cash, holdings, balance, total_deposits = _compute_balance_snapshot()

    stats = data_manager.get_trade_stats(wallet_ctx.db_path)
    closed_deltas = data_manager.get_closed_trade_deltas(wallet_ctx.db_path)
    open_deltas = [
        float(p.get("current_price", 0.0) or 0.0) - float(p.get("initial_price", 0.0) or 0.0)
        for p in data_manager.read_open_positions(wallet_ctx.db_path)
    ]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cash",           fmt_dollars(cash))
    c2.metric("Total Deposits", fmt_dollars(total_deposits))
    c3.metric("Holdings",       fmt_dollars(holdings))
    c4.metric("Balance",        fmt_dollars(balance))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Total Trades",             f"{int(stats.get('total_trades', 0))}")
    c6.metric("Avg $/share (Closed)",     fmt_dollars(_avg(closed_deltas)))
    c7.metric("Avg $/share (Open)",       fmt_dollars(_avg(open_deltas)))
    c8.metric("Avg $/share (Combined)",   fmt_dollars(_avg(closed_deltas + open_deltas)))


def _render_equity_chart() -> None:
    if not getattr(bridge, "live_trading", False):
        from ui.components import render_paper_equity_curve
        snapshots = data_manager.get_paper_snapshots(wallet_ctx.db_path)
        render_paper_equity_curve(snapshots)
    else:
        render_equity_curve(data_manager, wallet_ctx.db_path)


# The equity chart is an iframe (_echarts() -> components.html()): every
# render tears it down and rebuilds it from scratch, producing a visible
# white/grey flash. Riding the same 2s live_fragment as the rest of the
# dashboard meant that flash fired every 2s. Its underlying data can't
# change that fast anyway — paper snapshots are only written every ~3
# minutes (see PortfolioManager._refresh_portfolio()'s 180s throttle) — so
# it gets its own slower-refreshing fragment instead. Nested fragments
# rerun independently of their parent in Streamlit >= 1.37, so this cuts
# the flicker rate ~30x without restructuring the rest of the live loop.
if hasattr(st, "fragment"):
    @st.fragment(run_every="60s")
    def _render_equity_chart_fragment() -> None:
        _render_equity_chart()
else:
    _render_equity_chart_fragment = _render_equity_chart


def _render_balance_view() -> None:
    st.markdown("### Balance")
    _render_balance_stats_row()
    _render_equity_chart_fragment()


# ---------------------------------------------------------------------------
# View router
# ---------------------------------------------------------------------------

def _render_active_view(view_name: str) -> None:
    if view_name == "Hunter":
        _render_hunter_view()
    elif view_name == "Portfolio":
        _render_portfolio_view()
    else:
        _render_balance_view()


# ---------------------------------------------------------------------------
# Main render loop
# ---------------------------------------------------------------------------

def _render_dashboard_snapshot(view_name: str):
    now_ts = time.time()
    last_sync = float(getattr(bridge, "last_balance_sync_at", 0.0) or 0.0)
    if (now_ts - last_sync) >= 15.0:
        if bridge.live_trading:
            # Non-blocking: returns cached value and kicks off background refresh.
            live_balance, live_balance_ok = _fetch_live_balance()
            if live_balance > 0.0:
                bridge.current_balance = float(live_balance)
                bridge.balance_connection_error = not live_balance_ok
        # else: dry-run/paper mode. Don't touch bridge.current_balance here —
        # the paper pipeline keeps it in sync with the paper engine's actual
        # cash via _set_cash() in decision_pipeline.py. Overwriting it with
        # the real wallet balance was causing the Cash figure to flicker
        # between the two. Also skip the live-balance fetch entirely: it's
        # never used while in dry-run, so there's no reason to keep hitting
        # the real Polymarket API for it every 15s.
        bridge.last_balance_sync_at = now_ts

    _render_global_kpis()
    st.divider()
    _render_active_view(view_name)


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.title("🛰️ PolyBot: Quantitative Arbitrage Terminal")
st.caption("Minimal live terminal for scan, exposure, and balance decisions")

with st.sidebar:
    st.header("Navigation")
    if hasattr(st, "segmented_control"):
        active_view = st.segmented_control("View", options=["Hunter", "Portfolio", "Balance"], default="Hunter")
    else:
        active_view = st.radio("View", ["Hunter", "Portfolio", "Balance"], index=0)

    st.divider()
    st.header("Trading Mode")
    mode = st.toggle("Live Trading", value=bridge.live_trading)
    bridge.live_trading = bool(mode)
    # Also write to engine_control (DB): the engine that reads this now runs
    # as a separate process (see run_engine.py) and polls it once per loop
    # tick instead of reading bridge.live_trading directly (see
    # run_forever() in trading/decision_pipeline.py). The in-memory write
    # above is kept too — other renders on this process (the DRY_RUN
    # indicator dot right below, _compute_balance_snapshot's paper-mode
    # branch) still read it directly.
    data_manager.write_live_trading_requested(wallet_ctx.db_path, bool(mode))
    st.caption("Live Trading" if bridge.live_trading else "Dry Run")

    dot_color = "#16a34a" if not bridge.live_trading else "#dc2626"
    dot_label = "DRY_RUN ENABLED" if not bridge.live_trading else "DRY_RUN DISABLED"
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:8px;'>"
        f"<span style='height:10px;width:10px;border-radius:50%;background:{dot_color};display:inline-block;'></span>"
        f"<span>{dot_label}</span></div>",
        unsafe_allow_html=True,
    )

    # Was bridge.watch_only (engine-written in-process) — now read from
    # engine_status (DB), since the engine that sets it runs separately.
    _engine_status = data_manager.read_engine_status(wallet_ctx.db_path)
    if _engine_status and _engine_status.get("watch_only"):
        st.warning("Watch-Only mode enabled by Balance Guard")


if hasattr(st, "fragment"):
    @st.fragment(run_every="2s")
    def live_fragment():
        _render_dashboard_snapshot(active_view)

    live_fragment()
else:
    _render_dashboard_snapshot(active_view)

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

import asyncio
import os
import threading
import time

import pandas as pd
import streamlit as st

from trading.decision_pipeline import run_market_monitor
import ui.data_manager as data_manager
from polymarket import PolymarketClient
from core.bridge import get_bridge
from ui.components import render_equity_curve, render_ev_chart, render_positions

st.set_page_config(page_title="PolyBot Quant Pro", page_icon="🛰️", layout="wide")
bridge = get_bridge()


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _as_bool(raw: str, default: bool) -> bool:
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _get_env(*names: str) -> str:
    for name in names:
        value = str(os.getenv(name, "")).strip()
        if value:
            return value
    return ""


def _validate_runtime_env() -> dict:
    private_key = _get_env("POLYMARKET_PRIVATE_KEY", "POLYGON_PRIVATE_KEY")
    proxy_address = _get_env("POLYMARKET_PROXY_ADDRESS", "POLY_ADDRESS")
    if not private_key or not proxy_address:
        raise ValueError(
            "Missing required environment variables: "
            "POLYMARKET_PRIVATE_KEY/POLYGON_PRIVATE_KEY and "
            "POLYMARKET_PROXY_ADDRESS/POLY_ADDRESS. "
            "Pass them at runtime with --env-file."
        )
    return {
        "dry_run": _as_bool(os.getenv("DRY_RUN", "true"), True),
        "paper_trade_mode": _as_bool(os.getenv("PAPER_TRADE_MODE", "false"), False),
        "daily_limit_usd": float(os.getenv("DAILY_LIMIT_USD", "100.0")),
        "paper_balance_usd": float(os.getenv("PAPER_BALANCE_USD", "1000.0")),
        "trades_db_path": os.getenv("TRADES_DB_PATH", "/app/trades.db"),
    }


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
# Engine startup
# ---------------------------------------------------------------------------

def _ensure_engine_started_once() -> None:
    # Thread is stored on bridge (a @st.cache_resource singleton) so it
    # survives Streamlit reruns that would otherwise reset a module-level var.
    if bridge._engine_thread is not None and bridge._engine_thread.is_alive():
        return

    loop = asyncio.new_event_loop()

    def _runner() -> None:
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_market_monitor(bridge, _log_event))
        except Exception as exc:
            import traceback
            import tempfile
            bridge.terminal_logs.appendleft(f"[ENGINE-CRASH] {exc}")
            log_path = Path(tempfile.gettempdir()) / "polybot_crash.log"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[ENGINE-CRASH] {exc}\n{traceback.format_exc()}\n")

    bridge._engine_thread = threading.Thread(target=_runner, daemon=True, name="polybot-engine")
    bridge._engine_thread.start()


def _log_event(level, asset_type, token_id, payload):
    payload_dict = payload if isinstance(payload, dict) else {}
    reason = str(payload_dict.get("reason", "")).strip()
    market_name = str(payload_dict.get("market_name", "")).strip()
    ev_value = payload_dict.get("ev")
    detail = reason or market_name or str(payload)[:140]
    ev_suffix = f" | ev={ev_value}" if ev_value is not None else ""
    bridge.terminal_logs.appendleft(f"[{level}] {asset_type} - {detail}{ev_suffix}")

    if str(level) in {"REJECTED", "FILTERED", "SCAN-SKIP"} and token_id:
        bridge.seen_markets[str(token_id)] = str(payload_dict.get("market_name", ""))
        if len(bridge.seen_markets) > 500:
            keys = list(bridge.seen_markets.keys())
            for key in keys[:100]:
                bridge.seen_markets.pop(key, None)

    db_path = runtime_env.get("trades_db_path", "/app/trades.db")
    data_manager.log_event(bridge, level, asset_type, token_id, payload, db_path=db_path)


# ---------------------------------------------------------------------------
# Bootstrap (runs once per session; guarded against re-init)
# ---------------------------------------------------------------------------

def _bootstrap() -> None:
    global runtime_env
    # runtime_env is reset to {} on each Streamlit script rerun — re-populate it.
    runtime_env = _validate_runtime_env()
    data_manager.init_db(runtime_env["trades_db_path"])

    restored = data_manager.restore_runtime_state(
        db_path=runtime_env["trades_db_path"],
        fallback_starting_balance=0.0,
    )

    # Kick off balance fetch in background — don't block startup.
    _fetch_live_balance()

    bridge.starting_balance = float(restored["starting_balance"])
    bridge.current_balance = float(restored["current_balance"])
    bridge.balance_connection_error = False
    bridge.start_of_day_equity = float(restored["start_of_day_equity"])
    bridge.spent_today = float(restored["spent_today"])
    bridge.daily_spend = float(restored["spent_today"])
    bridge.state_bootstrap_source = str(restored["source"])
    bridge.live_trading = not (bool(runtime_env["dry_run"]) or bool(runtime_env["paper_trade_mode"]))
    bridge.last_balance_sync_at = 0.0

    if bridge.starting_balance <= 0.0:
        bridge.starting_balance = float(restored["current_balance"])

    _ensure_engine_started_once()


runtime_env: dict = {}
_bootstrap()


# ---------------------------------------------------------------------------
# KPI / status bar
# ---------------------------------------------------------------------------

def _render_global_kpis() -> None:
    c1, c2 = st.columns(2)
    balance_label = (
        "$0.00 (Connection Error)"
        if bool(getattr(bridge, "balance_connection_error", False))
        else f"${float(getattr(bridge, 'current_balance', 0.0)):,.2f}"
    )
    c1.metric("Current Balance", balance_label)
    c2.metric("Total PnL", f"${float(getattr(bridge, 'total_pnl', 0.0)):,.2f}")


# ---------------------------------------------------------------------------
# Hunter view
# ---------------------------------------------------------------------------

def _render_hunter_history_table() -> None:
    history_df = data_manager.fetch_latest_history(limit=80)
    if history_df.empty:
        st.info("No hunt history yet. Engine is scanning markets...")
        return

    keep_cols = ["Time", "Action", "Asset", "Side", "EV", "Market Name", "Reject Reason"]
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


# ---------------------------------------------------------------------------
# Portfolio view
# ---------------------------------------------------------------------------

def _render_portfolio_view() -> None:
    st.markdown("### Portfolio")
    render_positions(bridge)
    st.markdown("#### EV by Market")
    render_ev_chart(bridge)


# ---------------------------------------------------------------------------
# Balance view
# ---------------------------------------------------------------------------

def _render_balance_stats_row() -> None:
    stats = data_manager.get_trade_stats()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Win Rate",     f"{float(stats.get('win_rate', 0.0)):.2f}%")
    c2.metric("Total Trades", f"{int(stats.get('total_trades', 0))}")
    c3.metric("Avg Win",      f"${float(stats.get('avg_win', 0.0)):,.2f}")
    c4.metric("Avg Loss",     f"${float(stats.get('avg_loss', 0.0)):,.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("YES Trades",    f"{int(stats.get('total_yes_trades', 0))}")
    c6.metric("YES Win Rate",  f"{float(stats.get('yes_win_rate', 0.0)):.2f}%")
    c7.metric("NO Trades",     f"{int(stats.get('total_no_trades', 0))}")
    c8.metric("NO Win Rate",   f"{float(stats.get('no_win_rate', 0.0)):.2f}%")


def _render_balance_view() -> None:
    st.markdown("### Balance")
    _render_balance_stats_row()
    render_equity_curve(data_manager)


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
        # Non-blocking: returns cached value and kicks off background refresh.
        live_balance, live_balance_ok = _fetch_live_balance()
        if live_balance > 0.0:
            bridge.current_balance = float(live_balance)
            bridge.balance_connection_error = not live_balance_ok
        bridge.last_balance_sync_at = now_ts

    current_token = str(getattr(bridge, "current_token_id", ""))
    if current_token:
        bridge.market_name_by_token[current_token] = bridge.market_question

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
    st.caption("Live Trading" if bridge.live_trading else "Dry Run")

    dot_color = "#16a34a" if not bridge.live_trading else "#dc2626"
    dot_label = "DRY_RUN ENABLED" if not bridge.live_trading else "DRY_RUN DISABLED"
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:8px;'>"
        f"<span style='height:10px;width:10px;border-radius:50%;background:{dot_color};display:inline-block;'></span>"
        f"<span>{dot_label}</span></div>",
        unsafe_allow_html=True,
    )

    if bridge.watch_only:
        st.warning("Watch-Only mode enabled by Balance Guard")


if hasattr(st, "fragment"):
    @st.fragment(run_every="2s")
    def live_fragment():
        _render_dashboard_snapshot(active_view)

    live_fragment()
else:
    _render_dashboard_snapshot(active_view)

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

from dotenv import load_dotenv
import pandas as pd
import streamlit as st

from trading.decision_pipeline import run_market_monitor, sync_live_account_state
from trading.executor import TradeExecutor
from trading.risk_manager import PortfolioManager
from trading.budget_manager import BudgetManager
import ui.data_manager as data_manager
from polymarket import PolymarketClient, PolymarketScannerHunter
from core.bridge import get_bridge, DataBridge
from core.trading_config import TradingConfig
from core.wallet_context import WalletContext
from ui.components import (
    render_activity_chart, render_correlation_matrix, render_equity_curve,
    render_ev_chart, render_positions,
)

st.set_page_config(page_title="PolyBot Quant Pro", page_icon="🛰️", layout="wide")


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _get_env(*names: str) -> str:
    for name in names:
        value = str(os.getenv(name, "")).strip()
        if value:
            return value
    return ""


def _validate_runtime_env(bridge: DataBridge) -> WalletContext:
    """Build the single WalletContext this dashboard drives.

    Prefers a wallet config file (WALLET_CONFIG_PATH) if one is set; falls back
    to TradingConfig.from_env() for backward compatibility with deployments
    that only set process env vars and have no wallet config file yet.
    """
    # Load config/.env into the process environment before checking for
    # required vars below — otherwise a fresh `streamlit run` (with nothing
    # already exported into the shell) always fails this check even when
    # config/.env has real values, since TradingConfig.from_env() (which also
    # calls load_dotenv) only runs after this validation passes.
    load_dotenv("config/.env")

    wallet_config_path = _get_env("WALLET_CONFIG_PATH")

    if wallet_config_path:
        config = TradingConfig.from_file(wallet_config_path)
    else:
        private_key = _get_env("POLYMARKET_PRIVATE_KEY", "POLYGON_PRIVATE_KEY")
        proxy_address = _get_env("POLYMARKET_PROXY_ADDRESS", "POLY_ADDRESS")
        if not private_key or not proxy_address:
            raise ValueError(
                "Missing required environment variables: "
                "POLYMARKET_PRIVATE_KEY/POLYGON_PRIVATE_KEY and "
                "POLYMARKET_PROXY_ADDRESS/POLY_ADDRESS. "
                "Pass them at runtime with --env-file, or set WALLET_CONFIG_PATH "
                "to a wallet config.json."
            )
        config = TradingConfig.from_env()

    db_path = os.getenv("TRADES_DB_PATH", "/app/trades.db")
    return WalletContext(wallet_id="default", config=config, bridge=bridge, db_path=db_path)


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

def _make_log_event(ctx: WalletContext):
    """Build this wallet's log_func — captures ctx (and its db_path) via closure
    instead of reading db_path back out of a module-level dict each call."""

    def _log_event(level, asset_type, token_id, payload):
        payload_dict = payload if isinstance(payload, dict) else {}
        reason = str(payload_dict.get("reason", "")).strip()
        market_name = str(payload_dict.get("market_name", "")).strip()
        ev_value = payload_dict.get("ev")
        detail = reason or market_name or str(payload)[:140]
        ev_suffix = f" | ev={ev_value}" if ev_value is not None else ""
        ctx.bridge.terminal_logs.appendleft(f"[{level}] {asset_type} - {detail}{ev_suffix}")

        if str(level) in {"REJECTED", "FILTERED", "SCAN-SKIP"} and token_id:
            ctx.bridge.seen_markets[str(token_id)] = str(payload_dict.get("market_name", ""))
            if len(ctx.bridge.seen_markets) > 500:
                keys = list(ctx.bridge.seen_markets.keys())
                for key in keys[:100]:
                    ctx.bridge.seen_markets.pop(key, None)

        data_manager.log_event(ctx.bridge, level, asset_type, token_id, payload, db_path=ctx.db_path)

    return _log_event


def _ensure_engine_started_once(ctx: WalletContext) -> None:
    # Thread is stored on bridge (a @st.cache_resource singleton) so it
    # survives Streamlit reruns that would otherwise reset a module-level var.
    if ctx.bridge._engine_thread is not None and ctx.bridge._engine_thread.is_alive():
        return

    loop = asyncio.new_event_loop()

    def _runner() -> None:
        asyncio.set_event_loop(loop)
        log_func = _make_log_event(ctx)
        try:
            # Build this wallet's runtime components in dependency order and
            # attach them to ctx. This runs here (in the background thread,
            # only once — guarded by the early-return above) rather than in
            # _ensure_engine_started_once itself, so nothing network-touching
            # (CLOB auth in TradeExecutor.__init__, the live balance fetch
            # below) blocks Streamlit's main thread. WalletContext stays a
            # plain container — this is the one place that builds and wires
            # what it holds.
            ctx.executor = TradeExecutor(config=ctx.config)
            ctx.scanner = PolymarketScannerHunter(
                bridge=ctx.bridge,
                executor=ctx.executor,
                config=ctx.config,
            )
            ctx.portfolio_manager = PortfolioManager(
                bridge=ctx.bridge,
                executor=ctx.executor,
                config=ctx.config,
                hunter=ctx.scanner,
                db_path=ctx.db_path,
            )

            # Sync live balance before building BudgetManager so its
            # initial_balance reflects the account's real collateral, not
            # the DB-restored snapshot from _bootstrap().
            sync_live_account_state(ctx.bridge, ctx.executor, ctx.portfolio_manager, log_func)

            ctx.budget_manager = BudgetManager(
                bridge=ctx.bridge,
                config=ctx.config,
                initial_balance=float(ctx.bridge.current_balance),
            )

            loop.run_until_complete(run_market_monitor(ctx, log_func))
        except Exception as exc:
            import traceback
            import tempfile
            ctx.bridge.terminal_logs.appendleft(f"[ENGINE-CRASH] {exc}")
            log_path = Path(tempfile.gettempdir()) / "polybot_crash.log"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[ENGINE-CRASH] {exc}\n{traceback.format_exc()}\n")

    ctx.bridge._engine_thread = threading.Thread(target=_runner, daemon=True, name="polybot-engine")
    ctx.bridge._engine_thread.start()


# ---------------------------------------------------------------------------
# Bootstrap (runs once per session; guarded against re-init)
# ---------------------------------------------------------------------------

def _bootstrap() -> None:
    global bridge, wallet_ctx
    # bridge is the cached @st.cache_resource singleton — same instance across
    # reruns. wallet_ctx itself is rebuilt each rerun (cheap; picks up env
    # changes) but always wraps that same bridge instance.
    bridge = get_bridge()
    wallet_ctx = _validate_runtime_env(bridge)
    data_manager.init_db(wallet_ctx.db_path)

    restored = data_manager.restore_runtime_state(
        db_path=wallet_ctx.db_path,
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
    bridge.live_trading = not (bool(wallet_ctx.config.dry_run) or bool(wallet_ctx.config.paper_trade_mode))
    bridge.last_balance_sync_at = 0.0

    if bridge.starting_balance <= 0.0:
        bridge.starting_balance = float(restored["current_balance"])

    _ensure_engine_started_once(wallet_ctx)


bridge: DataBridge = None
wallet_ctx: WalletContext = None
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
    st.markdown("#### Activity Breakdown")
    render_activity_chart(bridge)


# ---------------------------------------------------------------------------
# Portfolio view
# ---------------------------------------------------------------------------

def _render_portfolio_view() -> None:
    st.markdown("### Portfolio")
    render_positions(bridge)
    st.markdown("#### EV by Market")
    render_ev_chart(bridge)
    st.markdown("#### Correlation")
    render_correlation_matrix(bridge, wallet_ctx.config)


# ---------------------------------------------------------------------------
# Balance view
# ---------------------------------------------------------------------------

def _avg(values: list) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def _render_balance_stats_row() -> None:
    # Deliberately read cash/holdings from bridge state (refreshed every
    # pipeline tick by sync_live_account_state()/_refresh_portfolio()) rather
    # than calling executor.get_balance()/get_open_positions() directly here.
    # Those route into PaperAdapter -> pm_trader.Engine, whose sqlite3
    # connection is thread-affine to the polybot-engine background thread —
    # calling it from Streamlit's own thread would hit the exact cross-thread
    # crash fixed for resolve_closed_markets() (see trading/decision_pipeline.py).
    is_paper_mode = not getattr(bridge, "live_trading", False)
    cash = float(getattr(bridge, "current_balance", 0.0) or 0.0)
    holdings = float(getattr(bridge, "open_position_value", 0.0) or 0.0)
    balance = cash + holdings

    # Total Deposits: paper mode has a stable, known constant (the account
    # is never topped up mid-run) — use it directly rather than
    # bridge.starting_balance, which is derived from the *most recent*
    # balance snapshot on every process restart (see restore_runtime_state()
    # in ui/data_manager.py), not the true original deposit. Live mode has
    # no such constant to fall back on, so bridge.starting_balance is the
    # best available approximation there — but it carries that same
    # restart-drift caveat, AND doesn't account for multiple real-world
    # deposits/withdrawals over an account's lifetime (no deposit ledger
    # exists to track those). Both are known limitations, not fixed here.
    if is_paper_mode:
        total_deposits = float(wallet_ctx.config.paper_balance_usd)
    else:
        total_deposits = float(getattr(bridge, "starting_balance", 0.0) or 0.0)

    stats = data_manager.get_trade_stats(wallet_ctx.db_path)
    closed_deltas = data_manager.get_closed_trade_deltas(wallet_ctx.db_path)
    open_deltas = [
        float(getattr(p, "current_price", 0.0) or 0.0) - float(getattr(p, "initial_price", 0.0) or 0.0)
        for p in (getattr(bridge, "current_portfolio", None) or [])
    ]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cash",           f"${cash:,.2f}")
    c2.metric("Total Deposits", f"${total_deposits:,.2f}")
    c3.metric("Holdings",       f"${holdings:,.2f}")
    c4.metric("Balance",        f"${balance:,.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Total Trades",             f"{int(stats.get('total_trades', 0))}")
    c6.metric("Avg $/share (Closed)",     f"${_avg(closed_deltas):,.2f}")
    c7.metric("Avg $/share (Open)",       f"${_avg(open_deltas):,.2f}")
    c8.metric("Avg $/share (Combined)",   f"${_avg(closed_deltas + open_deltas):,.2f}")


def _render_balance_view() -> None:
    st.markdown("### Balance")
    _render_balance_stats_row()
    
    if not getattr(bridge, "live_trading", False):
        from ui.components import render_paper_equity_curve
        snapshots = data_manager.get_paper_snapshots(wallet_ctx.db_path)
        render_paper_equity_curve(snapshots)
    else:
        render_equity_curve(data_manager, wallet_ctx.db_path)


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

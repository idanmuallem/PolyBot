from dataclasses import asdict
import inspect

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def _stretch_kwargs(api_fn):
    params = inspect.signature(api_fn).parameters
    if "width" in params:
        return {"width": "stretch"}
    return {}


def render_kpis(bridge):
    st.subheader("📌 Portfolio KPIs")
    k1, k2, k3 = st.columns([2, 1, 1])
    current_balance_label = (
        "$0.00 (Connection Error)"
        if bool(getattr(bridge, "balance_connection_error", False))
        else f"${float(getattr(bridge, 'current_balance', 0.0)):,.2f}"
    )
    k1.metric("Current Balance", current_balance_label)
    k2.metric("Open Position Value", f"${bridge.open_position_value:,.2f}")
    k3.metric("Total PnL", f"${bridge.total_pnl:,.2f}")


def render_ev_chart(bridge):
    st.subheader("📈 EV by Market")
    if not bridge.opportunity_map:
        st.info("No EV market data captured yet.")
        return

    ev_df = pd.DataFrame(list(bridge.opportunity_map.values()))
    if "market_name" not in ev_df.columns:
        ev_df["market_name"] = ev_df["token_id"]
    ev_df = ev_df.sort_values("ev", ascending=False).head(15)

    fig = px.bar(
        ev_df,
        x="market_name",
        y="ev",
        color="asset_type",
        title="EV by Market",
        labels={"market_name": "Market", "ev": "Expected Value"},
        template="plotly_dark",
    )
    fig.update_layout(
        xaxis_tickangle=-20,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"showgrid": False},
        yaxis={"showgrid": False},
    )
    st.plotly_chart(fig, **_stretch_kwargs(st.plotly_chart))


def render_positions(bridge):
    st.subheader("🧾 Open Positions")
    rows = []
    for pos in bridge.current_portfolio:
        try:
            rows.append(asdict(pos))
        except Exception:
            rows.append(pos.__dict__ if hasattr(pos, "__dict__") else {})

    if not rows:
        st.info("No open positions currently.")
        return

    pos_df = pd.DataFrame(rows)
    desired_cols = ["market_id", "token_id", "side", "shares", "initial_price", "current_price", "value", "pnl_percent"]
    pos_df = pos_df[[col for col in desired_cols if col in pos_df.columns]]

    styled = pos_df.style.format(
        {
            "initial_price": "{:.4f}",
            "current_price": "{:.4f}",
            "value": "${:,.2f}",
            "pnl_percent": "{:.2f}%",
        }
    ).map(
        lambda v: "color: #16a34a" if v > 0 else ("color: #dc2626" if v < 0 else ""),
        subset=["pnl_percent"],
    )

    if "side" in pos_df.columns:
        styled = styled.map(
            lambda v: "color: #16a34a; font-weight: 700;" if str(v).upper() == "YES" else (
                "color: #f59e0b; font-weight: 700;" if str(v).upper() == "NO" else ""
            ),
            subset=["side"],
        )

    st.dataframe(styled, hide_index=True, **_stretch_kwargs(st.dataframe))


def render_history_table(data_manager):
    st.subheader("🗂️ Hunt History")
    try:
        display_df = data_manager.fetch_latest_history(limit=50)
        if display_df.empty:
            st.info("No hunt history yet. Engine is scanning markets...")
            return

        def _style_action(action_value: str) -> str:
            color_map = {
                "TRACK": "#2563eb",
                "DRY-RUN": "#f59e0b",
                "AUTO-TRADE": "#16a34a",
            }
            color = color_map.get(str(action_value), "#e5e7eb")
            return f"color: {color}; font-weight: 600;"

        def _style_ev(value) -> str:
            try:
                ev = float(value)
            except Exception:
                return ""
            if ev >= 0.50:
                return "color: #22c55e; font-weight: 700;"
            if ev <= 0:
                return "color: #ef4444;"
            return ""

        def _style_side(value) -> str:
            side = str(value).upper()
            if side == "YES":
                return "color: #16a34a; font-weight: 700;"
            if side == "NO":
                return "color: #f59e0b; font-weight: 700;"
            return ""

        styled_df = display_df.style
        if "Action" in display_df.columns:
            styled_df = styled_df.map(_style_action, subset=["Action"])
        if "EV" in display_df.columns:
            styled_df = styled_df.map(_style_ev, subset=["EV"])
        if "Side" in display_df.columns:
            styled_df = styled_df.map(_style_side, subset=["Side"])

        st.dataframe(
            styled_df,
            hide_index=True,
            **_stretch_kwargs(st.dataframe),
            column_config={
                "Time": st.column_config.TextColumn("Time"),
                "Action": st.column_config.TextColumn("Action"),
                "Asset": st.column_config.TextColumn("Asset"),
                "Side": st.column_config.TextColumn("Side"),
                "Market Name": st.column_config.TextColumn("Market Name"),
                "Reject Reason": st.column_config.TextColumn("Reject Reason"),
                "Reject Metrics": st.column_config.TextColumn("Reject Metrics"),
                "Model Used": st.column_config.TextColumn("Model Used"),
                "Price": st.column_config.NumberColumn("Price", format="%.3f"),
                "Fair Value": st.column_config.NumberColumn("Fair Value", format="%.3f"),
                "EV": st.column_config.NumberColumn("EV", format="%.1f%%"),
                "Bet ($)": st.column_config.NumberColumn("Bet ($)", format="$%.2f"),
                "Shares": st.column_config.NumberColumn("Shares", format="%.2f"),
                "Token": st.column_config.TextColumn("Token"),
            },
        )
    except Exception as e:
        st.warning(f"Unable to load hunt history right now: {e}")


def render_trade_stats(data_manager):
    st.subheader("💹 Trade Stats")
    stats = data_manager.get_trade_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Win Rate", f"{float(stats.get('win_rate', 0.0)):.2f}%")
    c2.metric("Total Trades", f"{int(stats.get('total_trades', 0))}")
    c3.metric("Avg Win ($)", f"${float(stats.get('avg_win', 0.0)):,.2f}")
    c4.metric("Avg Loss ($)", f"${float(stats.get('avg_loss', 0.0)):,.2f}")
    
    st.markdown("##### Side Attribution")
    y1, y2, n1, n2 = st.columns(4)
    y1.metric("✅ YES Trades", f"{int(stats.get('total_yes_trades', 0))}")
    y2.metric("YES Win Rate", f"{float(stats.get('yes_win_rate', 0.0)):.2f}%")
    n1.metric("🔶 NO Trades", f"{int(stats.get('total_no_trades', 0))}")
    n2.metric("NO Win Rate", f"{float(stats.get('no_win_rate', 0.0)):.2f}%")


def render_system_throughput(data_manager, bridge):
    st.subheader("⚙️ System Throughput")
    throughput = data_manager.get_system_throughput()
    cooldown_count = len(getattr(bridge, "seen_markets", {}) or {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scanned Today", f"{int(throughput.get('total_scanned', 0))}")
    c2.metric("Rejected Today", f"{int(throughput.get('total_rejected', 0))}")
    c3.metric("Timeouts/Errors", f"{int(throughput.get('timeouts_errors', 0))}")
    c4.metric("Active Cooldowns", f"{cooldown_count}")


def render_equity_curve(data_manager):
    st.subheader("📉 Equity Curve")
    curve_df = data_manager.get_equity_curve()
    if curve_df.empty:
        st.info("No equity history available yet.")
        return

    fig = px.line(
        curve_df,
        x="timestamp",
        y="total_equity",
        title="Account Equity Over Time",
        labels={"timestamp": "Time", "total_equity": "Total Equity ($)"},
        template="plotly_dark",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"showgrid": False},
        yaxis={"showgrid": False},
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
    )
    st.plotly_chart(fig, **_stretch_kwargs(st.plotly_chart))


def render_risk_gauge(bridge):
    st.subheader("🛡️ Risk / Margin Gauge")
    total_equity = float(getattr(bridge, "current_balance", 0.0) or 0.0) + float(getattr(bridge, "open_position_value", 0.0) or 0.0)
    open_value = float(getattr(bridge, "open_position_value", 0.0) or 0.0)

    utilization = 0.0
    if total_equity > 0:
        utilization = max(0.0, min(100.0, (open_value / total_equity) * 100.0))

    bar_color = "#ef4444" if utilization > 50.0 else "#22c55e"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=utilization,
            number={"suffix": "%", "valueformat": ".1f"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": bar_color},
                "steps": [
                    {"range": [0, 50], "color": "rgba(34,197,94,0.25)"},
                    {"range": [50, 100], "color": "rgba(239,68,68,0.25)"},
                ],
                "threshold": {
                    "line": {"color": "#ef4444", "width": 3},
                    "thickness": 0.8,
                    "value": 50,
                },
            },
            title={"text": "Utilization"},
        )
    )
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", margin={"l": 10, "r": 10, "t": 50, "b": 10})
    st.plotly_chart(fig, **_stretch_kwargs(st.plotly_chart))


def render_terminal_feed(bridge):
    st.subheader("🖥️ Live Terminal Feed")
    logs = list(getattr(bridge, "terminal_logs", []))
    if not logs:
        st.info("No terminal logs yet.")
        return

    with st.container(border=True):
        st.code("\n".join(logs), language="text")

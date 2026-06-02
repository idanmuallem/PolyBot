from dataclasses import asdict

import pandas as pd
import plotly.express as px
import streamlit as st


def _apply_dark_layout(fig) -> None:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"showgrid": False},
        yaxis={"showgrid": False},
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
    )


def render_ev_chart(bridge):
    st.subheader("EV by Market")
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
    _apply_dark_layout(fig)
    fig.update_layout(xaxis_tickangle=-20)
    st.plotly_chart(fig, use_container_width=True)


def render_positions(bridge):
    positions = bridge.current_portfolio
    if not positions:
        st.info("No open positions currently.")
        return

    total_value = sum(float(getattr(p, "value", 0.0) or 0.0) for p in positions)
    total_pnl = sum(
        (float(getattr(p, "current_price", 0.0)) - float(getattr(p, "initial_price", 0.0)))
        * float(getattr(p, "shares", 0.0))
        for p in positions
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Open Positions", len(positions))
    c2.metric("Total Value", f"${total_value:,.2f}")
    c3.metric("Unrealized PnL", f"${total_pnl:,.2f}")

    rows = []
    for pos in positions:
        try:
            rows.append(asdict(pos))
        except Exception:
            rows.append(pos.__dict__ if hasattr(pos, "__dict__") else {})

    pos_df = pd.DataFrame(rows)
    desired_cols = ["market_id", "token_id", "side", "shares", "initial_price", "current_price", "value", "pnl_percent"]
    pos_df = pos_df[[c for c in desired_cols if c in pos_df.columns]]

    styled = pos_df.style.format({
        "initial_price": "{:.4f}",
        "current_price": "{:.4f}",
        "value": "${:,.2f}",
        "pnl_percent": "{:.2f}%",
    }).map(
        lambda v: "color: #16a34a" if v > 0 else ("color: #dc2626" if v < 0 else ""),
        subset=["pnl_percent"],
    )

    if "side" in pos_df.columns:
        styled = styled.map(
            lambda v: "color: #16a34a; font-weight: 700;" if str(v).upper() == "YES"
            else ("color: #f59e0b; font-weight: 700;" if str(v).upper() == "NO" else ""),
            subset=["side"],
        )

    st.dataframe(styled, hide_index=True, use_container_width=True)


def render_equity_curve(data_manager):
    st.subheader("Equity Curve")
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
    _apply_dark_layout(fig)
    st.plotly_chart(fig, use_container_width=True)

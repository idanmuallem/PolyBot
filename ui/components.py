from dataclasses import asdict
import inspect

import pandas as pd
import plotly.express as px
import streamlit as st


def _stretch_kwargs(api_fn):
    params = inspect.signature(api_fn).parameters
    if "width" in params:
        return {"width": "stretch"}
    return {}


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

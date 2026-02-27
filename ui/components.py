from dataclasses import asdict

import pandas as pd
import plotly.express as px
import streamlit as st


def render_kpis(bridge):
    st.subheader("📌 Portfolio KPIs")
    k1, k2, k3 = st.columns(3)
    k1.metric("Current Balance", f"${bridge.current_balance:,.2f}")
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
    fig.update_layout(xaxis_tickangle=-20)
    st.plotly_chart(fig, use_container_width=True)


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
    pos_df = pos_df[["market_id", "token_id", "shares", "initial_price", "current_price", "value", "pnl_percent"]]

    styled = pos_df.style.format(
        {
            "initial_price": "{:.4f}",
            "current_price": "{:.4f}",
            "value": "${:,.2f}",
            "pnl_percent": "{:.2f}%",
        }
    ).applymap(
        lambda v: "color: #16a34a" if v > 0 else ("color: #dc2626" if v < 0 else ""),
        subset=["pnl_percent"],
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


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

        styled_df = display_df.style
        if "EV" in display_df.columns:
            styled_df = styled_df.background_gradient(subset=["EV"], cmap="RdYlGn")
        if "Action" in display_df.columns:
            styled_df = styled_df.applymap(_style_action, subset=["Action"])

        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Time": st.column_config.TextColumn("Time"),
                "Action": st.column_config.TextColumn("Action"),
                "Asset": st.column_config.TextColumn("Asset"),
                "Market Name": st.column_config.TextColumn("Market Name"),
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

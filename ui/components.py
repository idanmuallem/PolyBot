from dataclasses import asdict

import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts


def render_ev_chart(bridge):
    st.subheader("EV by Market")
    if not bridge.opportunity_map:
        st.info("No EV market data captured yet.")
        return

    items = sorted(bridge.opportunity_map.values(), key=lambda x: x["ev"], reverse=True)[:15]
    names = [
        x["market_name"][:35] + ("…" if len(x["market_name"]) > 35 else "")
        for x in items
    ]
    evs = [round(x["ev"], 4) for x in items]

    bar_data = [
        {"value": ev, "itemStyle": {"color": "#22c55e" if ev > 0 else "#ef4444"}}
        for ev in evs
    ]

    options = {
        "backgroundColor": "transparent",
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": "3%", "right": "6%", "bottom": "3%", "containLabel": True},
        "xAxis": {
            "type": "value",
            "axisLabel": {"color": "#94a3b8"},
            "splitLine": {"lineStyle": {"color": "#1e293b"}},
        },
        "yAxis": {
            "type": "category",
            "data": names[::-1],
            "axisLabel": {"color": "#94a3b8", "fontSize": 11},
        },
        "series": [{
            "type": "bar",
            "data": bar_data[::-1],
            "markLine": {
                "silent": True,
                "data": [{"xAxis": 0}],
                "lineStyle": {"color": "#64748b", "type": "dashed"},
                "label": {"show": False},
            },
        }],
    }
    st_echarts(options=options, height="380px")


def render_equity_curve(data_manager):
    st.subheader("Equity Curve")
    df = data_manager.get_equity_curve()
    if df.empty:
        st.info("No equity history available yet.")
        return

    times = df["timestamp"].dt.strftime("%m-%d %H:%M").tolist()
    values = df["total_equity"].round(2).tolist()

    options = {
        "backgroundColor": "transparent",
        "tooltip": {
            "trigger": "axis",
            "formatter": "{b}<br/>Equity: ${c}",
            "axisPointer": {"type": "cross"},
        },
        "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
        "xAxis": {
            "type": "category",
            "data": times,
            "boundaryGap": False,
            "axisLabel": {"color": "#94a3b8", "rotate": 30},
            "axisLine": {"lineStyle": {"color": "#334155"}},
        },
        "yAxis": {
            "type": "value",
            "axisLabel": {"color": "#94a3b8", "formatter": "${value}"},
            "splitLine": {"lineStyle": {"color": "#1e293b"}},
        },
        "series": [{
            "type": "line",
            "data": values,
            "smooth": True,
            "symbol": "none",
            "lineStyle": {"color": "#22c55e", "width": 2},
            "itemStyle": {"color": "#22c55e"},
            "areaStyle": {
                "color": {
                    "type": "linear",
                    "x": 0, "y": 0, "x2": 0, "y2": 1,
                    "colorStops": [
                        {"offset": 0, "color": "rgba(34,197,94,0.30)"},
                        {"offset": 1, "color": "rgba(34,197,94,0.00)"},
                    ],
                }
            },
        }],
    }
    st_echarts(options=options, height="300px")


def render_activity_chart(bridge):
    counts = {k: v for k, v in bridge.level_counts.items() if v > 0}
    if not counts:
        return

    COLOR_MAP = {
        "DRY-RUN":        "#22c55e",
        "LIVE-TRADE":     "#16a34a",
        "PAPER-TRADE":    "#4ade80",
        "TRACK":          "#38bdf8",
        "TAKE-PROFIT":    "#3b82f6",
        "STOP-LOSS":      "#ef4444",
        "EV-CONVERGENCE": "#8b5cf6",
        "REJECTED":       "#f59e0b",
        "FILTERED":       "#fb923c",
        "SCAN-SKIP":      "#94a3b8",
        "PORTFOLIO-CULL": "#e879f9",
    }

    labels = list(counts.keys())
    bar_data = [
        {"value": counts[lbl], "itemStyle": {"color": COLOR_MAP.get(lbl, "#64748b")}}
        for lbl in labels
    ]

    options = {
        "backgroundColor": "transparent",
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
        "xAxis": {
            "type": "category",
            "data": labels,
            "axisLabel": {"color": "#94a3b8", "rotate": 25, "fontSize": 11},
            "axisLine": {"lineStyle": {"color": "#334155"}},
        },
        "yAxis": {
            "type": "value",
            "axisLabel": {"color": "#94a3b8"},
            "splitLine": {"lineStyle": {"color": "#1e293b"}},
        },
        "series": [{"type": "bar", "data": bar_data, "barMaxWidth": 50}],
    }
    st_echarts(options=options, height="260px")


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
    desired_cols = ["market_id", "token_id", "side", "shares", "initial_price", "current_price", "value", "pnl_ratio"]
    pos_df = pos_df[[c for c in desired_cols if c in pos_df.columns]]

    styled = pos_df.style.format({
        "initial_price": "{:.4f}",
        "current_price": "{:.4f}",
        "value": "${:,.2f}",
        "pnl_ratio": "{:.2%}",
    }).map(
        lambda v: "color: #16a34a" if v > 0 else ("color: #dc2626" if v < 0 else ""),
        subset=["pnl_ratio"],
    )

    if "side" in pos_df.columns:
        styled = styled.map(
            lambda v: "color: #16a34a; font-weight: 700;" if str(v).upper() == "YES"
            else ("color: #f59e0b; font-weight: 700;" if str(v).upper() == "NO" else ""),
            subset=["side"],
        )

    st.dataframe(styled, hide_index=True, use_container_width=True)

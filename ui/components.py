import json

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

_ECHARTS_CDN = "https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"


def fmt_dollars(value: float) -> str:
    """Format a dollar figure, killing the "-0.00" artifact: Python's
    string formatting sign-extends small negative floats that round to
    zero at 2 decimals (f"{-0.0003:.2f}" == "-0.00"), which reads as a
    nonsensical "negative zero" to anyone looking at the dashboard. Shared
    by every signed dollar display (Total PnL, Avg $/share, Cash, ...) —
    not just the ones that happened to show it first.
    """
    if abs(value) < 0.005:  # rounds to 0.00 at 2 decimals — kill the sign
        value = 0.0
    return f"${value:,.2f}"


def _echarts(options: dict, height: int = 350) -> None:
    opts_json = json.dumps(options, ensure_ascii=False)
    html = f"""
    <script src="{_ECHARTS_CDN}"></script>
    <div id="c" style="width:100%;height:{height}px;"></div>
    <script>
      var chart = echarts.init(document.getElementById('c'), null, {{renderer:'canvas'}});
      chart.setOption({opts_json});
      window.addEventListener('resize', function(){{ chart.resize(); }});
    </script>
    """
    components.html(html, height=height + 20)


def render_ev_chart(items: list[dict]):
    """`items` is data_manager.get_latest_ev_by_token()'s result — already
    sorted by EV descending and capped at the desired count."""
    st.subheader("EV by Market")
    if not items:
        st.info("No EV market data captured yet.")
        return

    names = [
        x["market_name"][:35] + ("…" if len(x["market_name"]) > 35 else "")
        for x in items
    ]
    evs = [round(x["ev"], 4) for x in items]
    bar_data = [
        {"value": ev, "itemStyle": {"color": "#22c55e" if ev > 0 else "#ef4444"}}
        for ev in evs
    ]

    _echarts({
        "backgroundColor": "transparent",
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": "2%", "right": "6%", "top": "4%", "bottom": "4%", "containLabel": True},
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
                "lineStyle": {"color": "#475569", "type": "dashed"},
                "label": {"show": False},
            },
        }],
    }, height=380)


def render_equity_curve(data_manager, db_path: str):
    st.subheader("Equity Curve")
    df = data_manager.get_equity_curve(db_path)
    if df.empty:
        st.info("No equity history available yet.")
        return

    times = df["timestamp"].dt.strftime("%m-%d %H:%M").tolist()
    values = df["total_equity"].round(2).tolist()

    _echarts({
        "backgroundColor": "transparent",
        "tooltip": {"trigger": "axis", "formatter": "{b}<br/>Equity: ${c}", "axisPointer": {"type": "cross"}},
        "grid": {"left": "2%", "right": "4%", "top": "4%", "bottom": "8%", "containLabel": True},
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
                    "type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
                    "colorStops": [
                        {"offset": 0, "color": "rgba(34,197,94,0.30)"},
                        {"offset": 1, "color": "rgba(34,197,94,0.00)"},
                    ],
                }
            },
        }],
    }, height=300)


def render_activity_chart(level_counts: dict):
    """`level_counts` is data_manager.get_level_counts()'s result (level ->
    count), replacing the old in-memory bridge.level_counts."""
    counts = {k: v for k, v in level_counts.items() if v > 0}
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

    _echarts({
        "backgroundColor": "transparent",
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": "2%", "right": "4%", "top": "4%", "bottom": "10%", "containLabel": True},
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
    }, height=260)


def _readable_market_name(pos: dict, opportunity_map: dict) -> str:
    """Human-readable market name for one position. A position row only
    carries market_id (the slug in paper mode) and token_id — the readable
    name/question lives in the EV opportunity map (populated by
    ui/data_manager.log_event for any EV-bearing log row, same source
    render_ev_chart() already reads market_name from), keyed by token_id.
    Falls back to reformatting the slug/id when no name is on record.
    """
    token_id = str(pos.get("token_id", "") or "")
    market_id = str(pos.get("market_id", "") or "")
    for key in (token_id, market_id):
        if not key:
            continue
        snapshot = opportunity_map.get(key)
        name = snapshot.get("market_name") if isinstance(snapshot, dict) else None
        if name:
            return str(name)

    slug = market_id or token_id
    return slug.replace("-", " ").replace("_", " ").title() if slug else "Unknown Market"


def render_positions(positions: list[dict], opportunity_map: dict | None = None):
    """`positions` is data_manager.read_open_positions()'s result;
    `opportunity_map` is data_manager.get_recent_opportunity_map()'s result
    — both DB-derived now that the engine runs as a separate process (see
    run_engine.py), replacing the old in-memory bridge.current_portfolio/
    bridge.opportunity_map reads."""
    if not positions:
        st.info("No open positions currently.")
        return

    opportunity_map = opportunity_map or {}

    total_value = sum(float(p.get("value", 0.0) or 0.0) for p in positions)
    total_pnl = sum(
        (float(p.get("current_price", 0.0) or 0.0) - float(p.get("initial_price", 0.0) or 0.0))
        * float(p.get("shares", 0.0) or 0.0)
        for p in positions
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Open Positions", len(positions))
    c2.metric("Total Value", fmt_dollars(total_value))
    c3.metric("Unrealized PnL", fmt_dollars(total_pnl))

    rows = []
    for pos in positions:
        shares = float(pos.get("shares", 0.0) or 0.0)
        initial_price = float(pos.get("initial_price", 0.0) or 0.0)
        current_price = float(pos.get("current_price", 0.0) or 0.0)
        rows.append({
            "Market": _readable_market_name(pos, opportunity_map),
            "Side": str(pos.get("side", "") or "").upper(),
            "Invested": shares * initial_price,
            "Entry Price": initial_price,
            "Current Price": current_price,
            "P&L": float(pos.get("pnl_ratio", 0.0) or 0.0),
        })

    pos_df = pd.DataFrame(rows, columns=["Market", "Side", "Invested", "Entry Price", "Current Price", "P&L"])

    styled = pos_df.style.format({
        "Invested": "${:,.2f}",
        "Entry Price": "${:,.4f}",
        "Current Price": "${:,.4f}",
        "P&L": "{:+.2%}",
    }).map(
        lambda v: "color: #16a34a" if v > 0 else ("color: #dc2626" if v < 0 else ""),
        subset=["P&L"],
    ).map(
        lambda v: "color: #16a34a; font-weight: 700;" if v == "YES"
        else ("color: #f59e0b; font-weight: 700;" if v == "NO" else ""),
        subset=["Side"],
    )

    st.dataframe(styled, hide_index=True, use_container_width=True)


def render_correlation_matrix(bridge, config):
    """Correlation matrix heatmap for currently open positions.

    The dashboard runs in a separate thread/process from the live pipeline,
    so this reads bridge.position_analytics (stashed every management cycle
    by PortfolioManager, same pattern as bridge.opportunity_map/current_portfolio)
    rather than holding a reference to a live PortfolioManager.

    Uses trading/correlation.py's CorrelationTracker — a static category/symbol
    estimate, not a live price-history correlation (that's future work; see
    that module's docstring).
    """
    st.subheader("Correlation Matrix")

    from trading.correlation import CorrelationTracker

    analytics = getattr(bridge, "position_analytics", {}) or {}
    open_tokens = {str(getattr(p, "token_id", "")) for p in (bridge.current_portfolio or [])}
    asset_types = sorted({
        snapshot["asset_type"]
        for token_id, snapshot in analytics.items()
        if token_id in open_tokens and snapshot.get("asset_type")
    })

    if len(asset_types) < 2:
        st.info("Need at least 2 open positions with known asset types to show correlation.")
        return

    tracker = CorrelationTracker(config=config)
    matrix = tracker.correlation_matrix(asset_types)
    grid = [[round(matrix[(a, b)], 2) for b in asset_types] for a in asset_types]

    heat_data = [
        [j, i, grid[i][j]]
        for i in range(len(asset_types))
        for j in range(len(asset_types))
    ]

    _echarts({
        "backgroundColor": "transparent",
        "tooltip": {"position": "top"},
        "grid": {"left": "18%", "right": "4%", "top": "4%", "bottom": "20%", "containLabel": True},
        "xAxis": {
            "type": "category", "data": asset_types,
            "axisLabel": {"color": "#94a3b8", "rotate": 30, "fontSize": 10},
            "splitArea": {"show": True},
        },
        "yAxis": {
            "type": "category", "data": asset_types,
            "axisLabel": {"color": "#94a3b8", "fontSize": 10},
            "splitArea": {"show": True},
        },
        "visualMap": {
            "min": 0, "max": 1, "calculable": True, "orient": "horizontal",
            "left": "center", "bottom": "0%",
            "inRange": {"color": ["#0f172a", "#38bdf8", "#ef4444"]},
            "textStyle": {"color": "#94a3b8"},
        },
        "series": [{
            "type": "heatmap",
            "data": heat_data,
            "label": {"show": True, "color": "#e2e8f0"},
        }],
    }, height=max(280, 60 * len(asset_types)))


def render_paper_equity_curve(snapshots: list[dict]):
    st.subheader("Paper Equity Curve")
    if not snapshots:
        st.info("Waiting for equity data...")
        return

    # Parse and format timestamps for the x-axis
    times = []
    for s in snapshots:
        ts = s.get("timestamp")
        if isinstance(ts, str):
            try:
                # Truncate string to simple MM-DD HH:MM if standard sqlite format
                parts = ts.split()
                if len(parts) == 2:
                    date_part, time_part = parts
                    times.append(f"{date_part[5:]} {time_part[:5]}")
                else:
                    times.append(ts)
            except Exception:
                times.append(ts)
        else:
            times.append(str(ts))

    cash_values = [round(s.get("cash", 0) or 0, 2) for s in snapshots]
    pos_values = [round(s.get("positions_value", 0) or 0, 2) for s in snapshots]
    total_values = [round(s.get("total_value", 0) or 0, 2) for s in snapshots]

    _echarts({
        "backgroundColor": "transparent",
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {"type": "cross"},
            # Series array order below is Cash(0), Positions(1), Total(2) —
            # {cN} tokens index into that same order.
            "formatter": "{b}<br/>Total: ${c2}<br/>Cash: ${c0}<br/>Positions: ${c1}",
        },
        "legend": {
            "data": ["Cash", "Positions", "Total"],
            "textStyle": {"color": "#94a3b8"},
            "top": "0%"
        },
        "grid": {"left": "2%", "right": "4%", "top": "12%", "bottom": "8%", "containLabel": True},
        "xAxis": {
            "type": "category",
            "data": times,
            "boundaryGap": False,
            "axisLabel": {"color": "#94a3b8", "rotate": 30},
            "axisLine": {"lineStyle": {"color": "#334155"}},
        },
        "yAxis": {
            "type": "value",
            # Zoom to where the data actually moves instead of always
            # anchoring at $0 — a $1000 paper balance fluctuating by a few
            # percent is invisible on a 0-1000+ axis. Note this means the
            # Cash/Positions stacked area no longer sits flush on the axis
            # line (expected trade-off for a readable range).
            "scale": True,
            "axisLabel": {"color": "#94a3b8", "formatter": "${value}"},
            "splitLine": {"lineStyle": {"color": "#1e293b"}},
        },
        "series": [
            {
                "name": "Cash",
                "type": "line",
                "stack": "Composition",
                "areaStyle": {},
                "smooth": True,
                "symbol": "none",
                "itemStyle": {"color": "#38bdf8"},
                "data": cash_values
            },
            {
                "name": "Positions",
                "type": "line",
                "stack": "Composition",
                "areaStyle": {},
                "smooth": True,
                "symbol": "none",
                "itemStyle": {"color": "#4ade80"},
                "data": pos_values
            },
            # Total — the primary line (drawn last so it renders on top of
            # the cash/positions composition), styled the same way
            # render_equity_curve() styles its total_equity line.
            {
                "name": "Total",
                "type": "line",
                "smooth": True,
                "symbol": "none",
                "z": 3,
                "lineStyle": {"color": "#22c55e", "width": 3},
                "itemStyle": {"color": "#22c55e"},
                "areaStyle": {
                    "color": {
                        "type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
                        "colorStops": [
                            {"offset": 0, "color": "rgba(34,197,94,0.30)"},
                            {"offset": 1, "color": "rgba(34,197,94,0.00)"},
                        ],
                    }
                },
                "data": total_values
            }
        ],
    }, height=300)

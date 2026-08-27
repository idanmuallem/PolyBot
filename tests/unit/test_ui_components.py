"""Tests for ui/components.py's data-shaping logic (Phase 7 additions).

streamlit itself is mocked globally in conftest.py, so these tests exercise
the real Python logic (matrix construction, column merging) rather than
actual rendering — st.* calls just no-op against the mock.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from core.bridge import DataBridge
from core.trading_config import TradingConfig
import ui.components as components


def _position(token_id="tok1", side="YES", shares=10.0, initial_price=0.40,
              current_price=0.50, value=5.0, pnl_ratio=0.25):
    return SimpleNamespace(
        market_id="mkt1", token_id=token_id, side=side, shares=shares,
        initial_price=initial_price, current_price=current_price,
        value=value, pnl_ratio=pnl_ratio,
    )


def _position_row(token_id="tok1", side="YES", shares=10.0, initial_price=0.40,
                   current_price=0.50, value=5.0, pnl_ratio=0.25):
    """render_positions() now takes data_manager.read_open_positions()'s
    result — plain dicts, one row per open position — rather than a bridge
    with live Position objects (see the process split in run_engine.py)."""
    return {
        "market_id": "mkt1", "token_id": token_id, "side": side, "shares": shares,
        "initial_price": initial_price, "current_price": current_price,
        "value": value, "pnl_ratio": pnl_ratio,
    }


# ── fmt_dollars ────────────────────────────────────────────────────────────

def test_fmt_dollars_kills_negative_zero_artifact():
    """f"{-0.001:.2f}" == "-0.00" — a small negative that rounds to zero at
    2 decimals still sign-extends, reading as a nonsensical "negative
    zero" to anyone looking at the dashboard."""
    assert components.fmt_dollars(-0.001) == "$0.00"
    assert components.fmt_dollars(-0.0049) == "$0.00"  # right at the rounding boundary
    assert components.fmt_dollars(0.0) == "$0.00"


def test_fmt_dollars_preserves_real_negative_values():
    assert components.fmt_dollars(-0.15) == "$-0.15"
    assert components.fmt_dollars(-21.51) == "$-21.51"


def test_fmt_dollars_preserves_positive_values():
    assert components.fmt_dollars(0.15) == "$0.15"
    assert components.fmt_dollars(1234.5) == "$1,234.50"


# ── render_correlation_matrix ─────────────────────────────────────────────────

def test_correlation_matrix_shows_info_with_fewer_than_two_assets():
    bridge = DataBridge()
    bridge.current_portfolio = [_position(token_id="tok1")]
    bridge.position_analytics = {"tok1": {"asset_type": "Crypto::BTCUSDT"}}

    with patch.object(components.st, "info") as mock_info, \
         patch.object(components, "_echarts") as mock_echarts:
        components.render_correlation_matrix(bridge, TradingConfig())

    mock_info.assert_called_once()
    mock_echarts.assert_not_called()


def test_correlation_matrix_renders_heatmap_for_two_or_more_assets():
    bridge = DataBridge()
    bridge.current_portfolio = [_position(token_id="tok1"), _position(token_id="tok2")]
    bridge.position_analytics = {
        "tok1": {"asset_type": "Crypto::BTCUSDT"},
        "tok2": {"asset_type": "Crypto::ETHUSDT"},
    }

    with patch.object(components, "_echarts") as mock_echarts:
        components.render_correlation_matrix(bridge, TradingConfig())

    mock_echarts.assert_called_once()
    options = mock_echarts.call_args[0][0]
    assert options["xAxis"]["data"] == ["Crypto::BTCUSDT", "Crypto::ETHUSDT"]
    heat_data = options["series"][0]["data"]
    # 2x2 grid -> 4 cells, including the (i, i) == 1.0 diagonal.
    assert len(heat_data) == 4
    assert any(cell[2] == 1.0 for cell in heat_data)


def test_correlation_matrix_ignores_analytics_for_closed_positions():
    bridge = DataBridge()
    # Only tok1 is actually open; tok2's analytics are stale from a closed position.
    bridge.current_portfolio = [_position(token_id="tok1")]
    bridge.position_analytics = {
        "tok1": {"asset_type": "Crypto::BTCUSDT"},
        "tok2": {"asset_type": "Crypto::ETHUSDT"},
    }

    with patch.object(components.st, "info") as mock_info, \
         patch.object(components, "_echarts") as mock_echarts:
        components.render_correlation_matrix(bridge, TradingConfig())

    mock_info.assert_called_once()
    mock_echarts.assert_not_called()


# ── render_positions: Market/Side/Invested/Entry/Current/P&L columns ─────────

def _render_positions_captured(positions, opportunity_map=None):
    captured = {}

    def _capture_dataframe(styled, **kwargs):
        captured["styled"] = styled

    with patch.object(components.st, "columns", return_value=(MagicMock(), MagicMock(), MagicMock())), \
         patch.object(components.st, "dataframe", side_effect=_capture_dataframe):
        components.render_positions(positions, opportunity_map)

    assert "styled" in captured
    return captured["styled"].data


def test_render_positions_has_exactly_the_redesigned_columns():
    positions = [_position_row(token_id="tok1")]
    opportunity_map = {"tok1": {"market_name": "Will Bitcoin Reach $85,000?"}}

    df = _render_positions_captured(positions, opportunity_map)

    assert list(df.columns) == ["Market", "Side", "Invested", "Entry Price", "Current Price", "P&L"]
    # No token_id/market_id/wang-edge leakage into the display.
    assert "token_id" not in df.columns
    assert "market_id" not in df.columns
    assert "Wang Edge (entry)" not in df.columns


def test_render_positions_uses_market_name_from_opportunity_map():
    positions = [_position_row(token_id="tok1", shares=10.0, initial_price=0.40,
                                current_price=0.50, pnl_ratio=0.25)]
    opportunity_map = {"tok1": {"market_name": "Will Bitcoin Reach $85,000?"}}

    df = _render_positions_captured(positions, opportunity_map)

    assert df.loc[0, "Market"] == "Will Bitcoin Reach $85,000?"
    assert df.loc[0, "Side"] == "YES"
    assert df.loc[0, "Invested"] == pytest.approx(10.0 * 0.40)
    assert df.loc[0, "Entry Price"] == pytest.approx(0.40)
    assert df.loc[0, "Current Price"] == pytest.approx(0.50)
    assert df.loc[0, "P&L"] == pytest.approx(0.25)


def test_render_positions_falls_back_to_formatted_slug_when_no_market_name():
    """No opportunity_map entry on record for this token — fall back to
    reformatting the slug (market_id in paper mode) into something readable
    rather than showing the raw hyphenated id."""
    positions = [_position_row(token_id="unknown_tok")]

    df = _render_positions_captured(positions, {})

    assert df.loc[0, "Market"] == "Mkt1"  # market_id="mkt1" from the _position_row fixture


def test_render_positions_handles_missing_analytics_gracefully():
    positions = [_position_row(token_id="unknown_tok")]

    with patch.object(components.st, "columns", return_value=(MagicMock(), MagicMock(), MagicMock())), \
         patch.object(components.st, "dataframe") as mock_dataframe:
        components.render_positions(positions, {})  # must not raise

    mock_dataframe.assert_called_once()

# ── render_paper_equity_curve ─────────────────────────────────────────────────

def test_render_paper_equity_curve_empty():
    with patch.object(components.st, "info") as mock_info, \
         patch.object(components, "_echarts") as mock_echarts:
        components.render_paper_equity_curve([])
    mock_info.assert_called_once_with("Waiting for equity data...")
    mock_echarts.assert_not_called()

def test_render_paper_equity_curve_with_data():
    snapshots = [
        {"timestamp": "2026-08-25 15:00:00", "cash": 500.0, "positions_value": 200.0, "total_value": 700.0},
        {"timestamp": "2026-08-25 15:03:00", "cash": 450.0, "positions_value": 260.0, "total_value": 710.0},
    ]
    with patch.object(components.st, "info") as mock_info, \
         patch.object(components, "_echarts") as mock_echarts:
        components.render_paper_equity_curve(snapshots)
    
    mock_info.assert_not_called()
    mock_echarts.assert_called_once()
    
    options = mock_echarts.call_args[0][0]
    # Verify times on X-axis (shortened as MM-DD HH:MM if standard)
    assert len(options["xAxis"]["data"]) == 2
    assert "08-25 15:00" in options["xAxis"]["data"][0]

    # Verify values on series — Cash/Positions stacked (composition), plus
    # an explicit, unstacked Total series carrying total_value (the primary
    # "is my paper account up or down" line).
    assert len(options["series"]) == 3
    by_name = {s["name"]: s for s in options["series"]}
    assert by_name["Cash"]["data"] == [500.0, 450.0]
    assert by_name["Positions"]["data"] == [200.0, 260.0]
    assert by_name["Total"]["data"] == [700.0, 710.0]

    # Total must not be stacked with cash/positions, and must be visually
    # distinguished (bolder line, drawn on top via z-order).
    assert "stack" not in by_name["Total"]
    assert by_name["Cash"].get("stack") == by_name["Positions"].get("stack")
    assert by_name["Total"]["lineStyle"]["width"] >= 3
    assert by_name["Total"].get("z", 0) > by_name["Cash"].get("z", 0)

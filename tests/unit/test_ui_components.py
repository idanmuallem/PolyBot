"""Tests for ui/components.py's data-shaping logic (Phase 7 additions).

streamlit itself is mocked globally in conftest.py, so these tests exercise
the real Python logic (matrix construction, column merging) rather than
actual rendering — st.* calls just no-op against the mock.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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


# ── render_positions: edge-decay columns ──────────────────────────────────────

def test_render_positions_merges_edge_decay_analytics():
    bridge = DataBridge()
    bridge.current_portfolio = [_position(token_id="tok1")]
    bridge.position_analytics = {
        "tok1": {"entry_wang_edge": 0.10, "current_wang_edge": 0.03, "edge_delta": -0.07},
    }

    captured = {}

    def _capture_dataframe(styled, **kwargs):
        captured["styled"] = styled

    with patch.object(components.st, "columns", return_value=(MagicMock(), MagicMock(), MagicMock())), \
         patch.object(components.st, "dataframe", side_effect=_capture_dataframe):
        components.render_positions(bridge)

    assert "styled" in captured
    df = captured["styled"].data
    assert df.loc[0, "Wang Edge (entry)"] == 0.10
    assert df.loc[0, "Wang Edge (now)"] == 0.03
    assert df.loc[0, "Edge Δ"] == -0.07


def test_render_positions_handles_missing_analytics_gracefully():
    bridge = DataBridge()
    bridge.current_portfolio = [_position(token_id="unknown_tok")]
    bridge.position_analytics = {}

    with patch.object(components.st, "columns", return_value=(MagicMock(), MagicMock(), MagicMock())), \
         patch.object(components.st, "dataframe") as mock_dataframe:
        components.render_positions(bridge)  # must not raise

    mock_dataframe.assert_called_once()

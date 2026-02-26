"""
Executor: Trade execution and risk management.

Separates execution logic from the orchestration layer.
Handles position sizing, risk checks, and trade firing.
"""

from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from models import MarketData


@dataclass
class RiskConfig:
    """Risk management configuration."""
    ev_threshold: float = 0.15  # Minimum EV to execute trade
    max_position_size: float = 1.0  # Max position as fraction of capital
    max_daily_trades: int = 10  # Max trades per day
    stop_loss_pct: float = 0.05  # Stop loss percentage


class TradeExecutor:
    """Handles trade execution with risk management.

    Responsibilities:
    - Evaluate expected value (EV) against thresholds
    - Check risk constraints
    - Fire execution callbacks
    - Position sizing (future enhancement)
    """

    def __init__(self, risk_config: Optional[RiskConfig] = None):
        """Initialize TradeExecutor.

        Args:
            risk_config: Risk configuration (uses defaults if not provided)
        """
        self.risk_config = risk_config or RiskConfig()
        self.trade_count_today = 0

    def evaluate_and_execute(
        self,
        market: MarketData,
        fair_value: float,
        ev: float,
        current_poly_price: float,
        log_func: Callable,
    ) -> bool:
        """Evaluate market conditions and execute trade if criteria are met.

        Args:
            market: MarketData object with market_id, asset_type, question, etc.
            fair_value: Fair value probability (our calculated price)
            ev: Expected value (fair_value - market_price) / market_price
            current_poly_price: Current Polymarket mid price
            log_func: Logging callback function

        Returns:
            True if trade was executed, False otherwise
        """
        asset_type = market.asset_type
        token_id = market.market_id

        # ========================================
        # 1. Check EV threshold
        # ========================================
        if ev <= self.risk_config.ev_threshold:
            return False

        # ========================================
        # 2. Check daily trade limit
        # ========================================
        if self.trade_count_today >= self.risk_config.max_daily_trades:
            log_func(
                "RISK",
                asset_type,
                token_id,
                f"Daily trade limit ({self.risk_config.max_daily_trades}) reached",
            )
            return False

        # ========================================
        # 3. Check market validity
        # ========================================
        if not self._validate_market(market, log_func):
            return False

        # ========================================
        # 4. Calculate position size (future enhancement)
        # ========================================
        position_size = self._calculate_position_size(ev, fair_value)

        # ========================================
        # 5. Execute trade
        # ========================================
        log_func(
            "AUTO-TRADE",
            asset_type,
            token_id,
            {
                "market_price": current_poly_price,
                "fair_value": fair_value,
                "ev": round(ev, 4),
                "position_size": position_size,
            },
        )

        self.trade_count_today += 1
        return True

    def _validate_market(self, market: MarketData, log_func: Callable) -> bool:
        """Validate market conditions before execution.

        Args:
            market: MarketData object
            log_func: Logging callback

        Returns:
            True if market is valid for trading
        """
        # Check required fields
        required_fields = [
            ("market_id", market.market_id),
            ("asset_type", market.asset_type),
            ("strike_price", market.strike_price),
            ("question", market.question),
        ]
        for field_name, field_value in required_fields:
            if field_value is None:
                log_func("VALIDATE", "Market", "Unknown", f"Missing {field_name}")
                return False

        return True

    def _calculate_position_size(self, ev: float, fair_value: float) -> float:
        """Calculate position size based on EV and fair value.

        Simple Kelly-like approach (future enhancement).

        Args:
            ev: Expected value
            fair_value: Fair value probability

        Returns:
            Position size as fraction of max (0.0 to 1.0)
        """
        # Simple: size proportional to EV, capped at max
        # Higher EV = larger position
        position = min(ev * 2.0, self.risk_config.max_position_size)
        return max(0.01, position)  # Minimum 1% if trading

    def reset_daily_count(self):
        """Reset daily trade counter (call at start of each trading day)."""
        self.trade_count_today = 0

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get current execution statistics.

        Returns:
            Dict with execution stats (trades today, etc.)
        """
        return {
            "trades_today": self.trade_count_today,
            "daily_limit": self.risk_config.max_daily_trades,
            "av_ev_threshold": self.risk_config.ev_threshold,
        }

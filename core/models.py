from dataclasses import dataclass
from typing import Optional

@dataclass
class MarketData:
    market_id: str
    asset_type: str
    strike_price: float
    question: str
    market_name: str
    initial_price: float
    volume: float
    expiry_date: Optional[str] = None


@dataclass
class TradeSignal:
    fair_value: float      # probability in [0.0, 1.0]
    expected_value: float  # (fair - price) / price
    kelly_size: float      # fraction of bankroll to risk
    is_tradable: bool      # meets EV threshold and positive kelly
    realized_pnl: float = 0.0


@dataclass
class Position:
    market_id: str
    token_id: str
    initial_price: float
    current_price: float
    shares: float
    value: float
    pnl_percent: float

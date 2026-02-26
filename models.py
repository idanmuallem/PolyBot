from dataclasses import dataclass

@dataclass
class MarketData:
    market_id: str
    asset_type: str
    strike_price: float
    question: str
    market_name: str
    initial_price: float
    volume: float


@dataclass
class TradeSignal:
    fair_value: float      # probability in [0.0, 1.0]
    expected_value: float  # (fair - price) / price
    kelly_size: float      # fraction of bankroll to risk
    is_tradable: bool      # meets EV threshold and positive kelly

from dataclasses import dataclass
from typing import Optional

# Market entry price bounds — shared by hunters, executor, and pipeline.
PRICE_FLOOR = 0.30
PRICE_CEILING = 0.85

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
    no_market_id: Optional[str] = None
    condition_id: Optional[str] = None   # Polymarket conditionId (0x...)
    slug: Optional[str] = None           # Polymarket market slug


@dataclass
class TradeSignal:
    post_prob: float        # final risk-adjusted P(YES): raw -> Wang -> market-blend
    expected_value: float   # EV of the chosen side, computed off post_prob
    kelly_size: float       # raw Kelly fraction (edge/odds) for the chosen side, unclamped
    is_tradable: bool       # meets EV threshold and positive kelly

    # Pricing diagnostics (see BaseBrain.evaluate() in brains/base.py) -
    # populated even when is_tradable is False, so callers can log why.
    pre_prob: float = 0.0            # the brain's unadjusted model output
    wang_fair_value: float = 0.0     # after Wang Transform only, before market blending
    wang_lambda: float = 0.0         # distortion parameter actually applied
    wang_edge: float = 0.0           # post_prob - market_price (post blend)
    confidence: float = 1.0          # [0, 1] scale applied to Kelly sizing downstream
    side: str = "YES"                # YES or NO - whichever side has the larger EV
    ev_yes: float = 0.0
    ev_no: float = 0.0
    entry_price: float = 0.0         # price of the chosen side (price_yes or 1 - price_yes)


@dataclass
class Position:
    market_id: str
    token_id: str
    initial_price: float
    current_price: float
    shares: float
    value: float
    pnl_ratio: float          # fraction: 0.25 = +25%, -0.50 = -50%
    side: str = "UNKNOWN"
    live_ev: float = 0.0

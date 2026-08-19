"""CorrelationTracker: category/symbol-based correlation estimate between
open positions.

This is the "cosmetic and logging" pass (Phase 7) — a full price-history
correlation engine (EWMA over realized returns, the way oracle3's
CorrelationAwareRiskManager tracks it in oracle3/risk/correlation_risk_manager.py)
is future work. For now this uses a static lookup: same market -> perfectly
correlated, same asset -> perfectly correlated, related crypto assets -> a
fixed approximate correlation, same category otherwise -> a flat moderate
correlation, different category -> uncorrelated. Good enough to flag "you
already have two BTC markets open" without needing a live price-series store.
"""

from typing import Dict, List, Tuple

from core.trading_config import TradingConfig

# Approximate pairwise crypto correlations (static, not fit from data).
_CRYPTO_CORRELATION = {
    frozenset({"BTC", "ETH"}): 0.85,
    frozenset({"BTC", "SOL"}): 0.75,
    frozenset({"ETH", "SOL"}): 0.80,
}
_SAME_CATEGORY_DEFAULT = 0.30  # e.g. two different Weather markets
_CROSS_CATEGORY_DEFAULT = 0.0  # e.g. a Crypto market vs a Weather market


class CorrelationTracker:
    """Estimates correlation exposure between open positions by asset_type.

    Config is injected through the constructor (WalletContext pattern), even
    though this first version doesn't need much from it yet — keeps the
    constructor shape consistent with PricingEngine/CCXTDataClient/strategies
    for whenever this grows into a real price-history tracker.
    """

    def __init__(self, config: TradingConfig):
        self.config = config

    def pairwise_correlation(self, asset_type_a: str, asset_type_b: str) -> float:
        """Estimated correlation between two "Category::Symbol" asset types."""
        if not asset_type_a or not asset_type_b:
            return 0.0
        if asset_type_a == asset_type_b:
            return 1.0

        cat_a, symbol_a = self._split(asset_type_a)
        cat_b, symbol_b = self._split(asset_type_b)
        if cat_a != cat_b:
            return _CROSS_CATEGORY_DEFAULT

        if cat_a == "crypto":
            base_a, base_b = self._crypto_base(symbol_a), self._crypto_base(symbol_b)
            if base_a == base_b:
                return 1.0
            return _CRYPTO_CORRELATION.get(frozenset({base_a, base_b}), _SAME_CATEGORY_DEFAULT)

        return _SAME_CATEGORY_DEFAULT

    def exposure_for_new_position(self, new_asset_type: str, open_asset_types: List[str]) -> float:
        """Average correlation between a candidate position and the current book.

        0.0 means the new position is uncorrelated with everything already
        open (or nothing is open yet); 1.0 means it's a duplicate of the book.
        """
        if not open_asset_types:
            return 0.0
        correlations = [self.pairwise_correlation(new_asset_type, a) for a in open_asset_types]
        return sum(correlations) / len(correlations)

    def correlation_matrix(self, asset_types: List[str]) -> Dict[Tuple[str, str], float]:
        """Full pairwise matrix (including the diagonal) for a set of asset types.

        Returns {(a, b): correlation} for every ordered pair, including (a, a).
        """
        matrix: Dict[Tuple[str, str], float] = {}
        for a in asset_types:
            for b in asset_types:
                matrix[(a, b)] = self.pairwise_correlation(a, b)
        return matrix

    @staticmethod
    def _split(asset_type: str) -> Tuple[str, str]:
        category, _, symbol = str(asset_type).partition("::")
        return category.strip().lower(), symbol.strip().upper()

    @staticmethod
    def _crypto_base(symbol: str) -> str:
        for base in ("BTC", "ETH", "SOL"):
            if symbol.startswith(base):
                return base
        return symbol

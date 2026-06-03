"""
CryptoHunter: Hunts Polymarket for cryptocurrency price prediction markets.
"""

from typing import Optional

from core.models import MarketData
from .parsers import extract_crypto_strike
from .clients.binance import BinanceClient

from .base import BasePolymarketHunter


class CryptoHunter(BasePolymarketHunter):
    """Hunt markets related to cryptocurrency prices (BTC, ETH, SOL).

    Anchor: Binance spot price via BinanceClient.
    """

    DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    _ALIAS_MAP = {
        "BTC": ["Bitcoin", "BTC"],
        "ETH": ["Ethereum", "ETH"],
        "SOL": ["Solana", "SOL"],
    }

    # Maps every search alias to its canonical Binance symbol so _resolve_keyword
    # can fetch the correct anchor regardless of which scan discovered the market.
    _ALIAS_TO_SYMBOL = {
        "bitcoin": "BTCUSDT", "btc": "BTCUSDT",
        "ethereum": "ETHUSDT", "eth": "ETHUSDT",
        "solana": "SOLUSDT", "sol": "SOLUSDT",
    }

    def __init__(self, symbols: Optional[list] = None, **kwargs):
        super().__init__(**kwargs)
        self.symbols = symbols or list(self.DEFAULT_SYMBOLS)
        self.binance_client = BinanceClient()

    def get_topic_type(self) -> str:
        return "Crypto"

    def get_anchor_value(self) -> Optional[float]:
        for symbol in self.symbols:
            price = self.binance_client.get_latest_value(symbol)
            if price and price > 0:
                return price
        return None

    def extract_strike(self, text: str, anchor: float) -> Optional[float]:
        return extract_crypto_strike(text, anchor)

    def get_search_aliases(self) -> list:
        return ["bitcoin", "btc", "ethereum", "eth", "solana", "sol"]

    def _resolve_keyword(self, anchor, event, market, current_keyword, matched_alias):
        # Look up the asset-specific Binance price so strike validation always
        # uses the correct anchor (e.g. ETH ~$2k, not BTC ~$67k).
        symbol = self._ALIAS_TO_SYMBOL.get(matched_alias.lower())
        if symbol:
            price = self.binance_client.get_latest_value(symbol)
            if price and price > 0:
                return price, symbol
        return anchor, current_keyword

    def hunt(self, skip_ids: list = None, add_cooldown_func=None) -> Optional[MarketData]:
        if skip_ids is None:
            skip_ids = []

        print(f"[CryptoHunter] Starting hunt: {len(self.symbols)} symbols, {len(skip_ids)} skipped")

        best_market = None
        highest_volume = 0.0

        for symbol in self.symbols:
            anchor_price = self.binance_client.get_latest_value(symbol)
            if not anchor_price or anchor_price <= 0:
                print(f"[CryptoHunter] No anchor for {symbol}, skipping")
                continue

            key = symbol.replace("USDT", "").replace("BUSD", "").upper()
            aliases = self._ALIAS_MAP.get(key, [key])

            for alias in aliases:
                found = self._scan_polymarket(
                    anchor_price,
                    alias,
                    skip_ids=skip_ids,
                    add_cooldown_func=add_cooldown_func,
                )
                # asset_type is already set correctly by _resolve_keyword inside
                # _scan_polymarket — do not override it here.
                if found and found.volume > highest_volume:
                    highest_volume = found.volume
                    best_market = found

        if best_market:
            asset_lower = best_market.asset_type.lower()
            q_lower = (best_market.market_name or "").lower()
            mismatch = (
                ("btcusdt" in asset_lower)
                and any(kw in q_lower for kw in ("eth", "ethereum", "sol", "solana"))
            ) or (
                ("ethusdt" in asset_lower)
                and any(kw in q_lower for kw in ("btc", "bitcoin", "sol", "solana"))
            ) or (
                ("solusdt" in asset_lower)
                and any(kw in q_lower for kw in ("btc", "bitcoin", "eth", "ethereum"))
            )
            if mismatch:
                print(f"[CryptoHunter] Skipping asset mismatch: {best_market.asset_type} / {best_market.market_name[:60]}")
                return None

            print(f"[CryptoHunter] Best market: {best_market.market_name.encode('ascii', errors='replace').decode()} | vol={best_market.volume:,.0f}")
        else:
            print("[CryptoHunter] No markets found")
        return best_market

    def get_live_truth(self, market: MarketData) -> Optional[float]:
        if not market or not market.asset_type.startswith("Crypto::"):
            return None
        try:
            symbol = market.asset_type.split("::", 1)[1]
            return self.binance_client.get_latest_value(symbol)
        except Exception as e:
            print(f"[CryptoHunter] get_live_truth error: {e}")
            return None

"""
CryptoHunter: Hunts Polymarket for cryptocurrency price prediction markets.
"""

from typing import Optional

from core.models import MarketData
from parsers import extract_crypto_strike
from clients.binance import BinanceClient

from .base import BasePolymarketHunter


class CryptoHunter(BasePolymarketHunter):
    """Hunt markets related to cryptocurrency prices (BTC, ETH, etc.).

    Anchor: Binance spot price via BinanceClient.
    """

    DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]

    _ALIAS_MAP = {
        "BTC": ["Bitcoin", "BTC"],
        "ETH": ["Ethereum", "ETH"],
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
                if found and found.volume > highest_volume:
                    highest_volume = found.volume
                    found.asset_type = f"{self.get_topic_type()}::{symbol}"
                    best_market = found

        if best_market:
            print(f"[CryptoHunter] Best market: {best_market.market_name} | vol={best_market.volume:,.0f}")
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

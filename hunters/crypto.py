"""
CryptoHunter: Hunts Polymarket for cryptocurrency price prediction markets.

Uses PolymarketClient for API access and BinanceClient for price data.
"""

from datetime import datetime
from typing import Optional

from core.models import MarketData
from parsers.crypto import extract_crypto_strike
from clients.polymarket import PolymarketClient
from clients.binance import BinanceClient

from .base import BasePolymarketHunter


class CryptoHunter(BasePolymarketHunter):
    """Hunt markets related to cryptocurrency prices (BTC, ETH, etc.).

    Anchor: Binance API (e.g., BTC/USDT latest price)
    Uses PolymarketClient for market discovery and BinanceClient for price data.
    """

    DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]  # Bitcoin prioritized first

    def __init__(self, symbols: Optional[list] = None, **kwargs):
        """Initialize CryptoHunter with clients.

        Args:
            symbols: List of trading symbols to hunt for (default: BTC, ETH).
            **kwargs: Passed to parent (e.g., polymarket_base)
        """
        super().__init__(**kwargs)
        self.symbols = symbols or list(self.DEFAULT_SYMBOLS)
        self.polymarket_client = PolymarketClient()
        self.binance_client = BinanceClient()

    def get_topic_type(self) -> str:
        return "Crypto"

    def get_anchor_value(self) -> Optional[float]:
        """Fetch the latest BTC or ETH price from Binance."""
        for symbol in self.symbols:
            price = self.binance_client.get_latest_value(symbol)
            if price > 0:
                return price
        return None

    def extract_strike(self, text: str, anchor: float) -> Optional[float]:
        """Extract cryptocurrency strike price using parser."""
        return extract_crypto_strike(text, anchor)

    def get_search_aliases(self) -> list:
        """Return all crypto-related aliases used for Polymarket event matching."""
        return ["bitcoin", "btc", "ethereum", "eth", "solana", "sol"]

    def hunt(self, skip_ids: list = None, add_cooldown_func=None, log_func=None) -> Optional[MarketData]:
        """Hunt for a crypto market.

        Tries each symbol in order. Returns the highest-volume market found.
        Respects skip_ids list to avoid markets in 10-minute cooldown.

        Args:
            skip_ids: List of market_ids to skip (in cooldown). Defaults to [].

        Returns:
            MarketData object or None.
        """
        if skip_ids is None:
            skip_ids = []
        
        print(f"[CryptoHunter] {datetime.now().isoformat()} - Starting hunt for {len(self.symbols)} symbols (skipping {len(skip_ids)} cooldown markets)")

        best_market = None
        highest_volume = 0.0

        # Try each symbol and search using multiple alias terms (Bitcoin/BTC, Ethereum/ETH)
        alias_map = {
            "BTC": ["Bitcoin", "BTC"],
            "ETH": ["Ethereum", "ETH"],
        }

        for symbol in self.symbols:
            print(f"[CryptoHunter] Trying symbol: {symbol}")

            # Get anchor price
            anchor_price = self.binance_client.get_latest_value(symbol)
            if not anchor_price or anchor_price <= 0:
                print(f"[CryptoHunter] Failed to fetch anchor for {symbol}, skipping")
                continue

            # Prepare alias list for searching Polymarket
            key = symbol.replace("USDT", "").replace("BUSD", "").upper()
            aliases = alias_map.get(key, [key, key.lower()])

            for alias in aliases:
                print(f"[CryptoHunter] Searching Polymarket for alias: {alias}")
                found = self._scan_polymarket(
                    anchor_price,
                    alias,
                    skip_ids=skip_ids,
                    add_cooldown_func=add_cooldown_func,
                    log_func=log_func,
                )
                if found and found.volume > highest_volume:
                    highest_volume = found.volume
                    # Ensure asset_type has the correct symbol
                    found.asset_type = f"{self.get_topic_type()}::{symbol}"
                    best_market = found

        if best_market:
            print(f"[CryptoHunter] Found best market: {best_market.market_id} with volume {best_market.volume}")
        else:
            print("[CryptoHunter] No crypto markets found after trying all symbols")
        
        return best_market

    def get_live_truth(self, market: MarketData) -> Optional[float]:
        """Fetch live BTC/ETH/crypto price from Binance for the given market.

        Args:
            market: MarketData object with asset_type like "Crypto::BTCUSDT"

        Returns:
            Current spot price or None on error.
        """
        if not market:
            return None

        try:
            asset_type = market.asset_type
            if not asset_type.startswith("Crypto::"):
                return None

            # Extract symbol (e.g., "BTCUSDT" from "Crypto::BTCUSDT")
            symbol = asset_type.split("::", 1)[1]
            return self.binance_client.get_latest_value(symbol)
        except Exception as e:
            print(f"[CryptoHunter] get_live_truth error: {e}")
            return None

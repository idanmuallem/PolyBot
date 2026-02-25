"""
CryptoHunter: Hunts Polymarket for cryptocurrency price prediction markets.

Uses Binance as the anchor source for live prices.
"""

import json
import re
from datetime import datetime
from typing import Optional, Dict, Any

import requests
from curl_cffi import requests as crequests

from .base import BaseHunter


class CryptoHunter(BaseHunter):
    """Hunt markets related to cryptocurrency prices (BTC, ETH, etc.).

    Anchor: Binance API (e.g., BTC/USDT latest price)
    """

    DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]

    def __init__(self, symbols: Optional[list] = None, **kwargs):
        """Initialize CryptoHunter.

        Args:
            symbols: List of trading symbols to hunt for (default: BTC, ETH).
                    Example: ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
            **kwargs: Passed to parent (e.g., polymarket_base)
        """
        super().__init__(**kwargs)
        self.symbols = symbols or list(self.DEFAULT_SYMBOLS)
        self._anchor_value = None

    def get_topic_type(self) -> str:
        return "Crypto"

    def get_anchor_value(self) -> Optional[float]:
        """Fetch the latest BTC or ETH price from Binance.

        Returns the anchor for the primary symbol.
        """
        if self.symbols:
            return self._get_binance_price(self.symbols[0])
        return None

    @staticmethod
    def _get_binance_price(symbol: str = "BTCUSDT") -> Optional[float]:
        """Fetch a single symbol price from Binance.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")

        Returns:
            Price as float, or None on failure.
        """
        try:
            res = requests.get(
                f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}",
                timeout=5
            )
            return float(res.json().get("price", 0.0))
        except Exception as e:
            print(f"[CryptoHunter] Failed to fetch {symbol}: {e}")
            return None

    @staticmethod
    def _extract_valid_strike(question: str, anchor_price: float, ratio_min: float = 0.2, ratio_max: float = 5.0) -> Optional[float]:
        """Extract a valid strike price from the market question.

        Looks for price patterns like "$1500", "$42.5k", "$1.2m" and validates
        the strike is within a reasonable ratio to the current anchor price.

        Args:
            question: Market question text
            anchor_price: Current market price for validation
            ratio_min: Minimum acceptable strike/anchor ratio (default: 0.2)
            ratio_max: Maximum acceptable strike/anchor ratio (default: 5.0)

        Returns:
            Valid strike price or None.
        """
        matches = re.finditer(r"\$(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*([kKmMbB])?", question)
        for match in matches:
            base_num = float(match.group(1).replace(",", ""))
            suffix = match.group(2).lower() if match.group(2) else ""

            if suffix == "k":
                base_num *= 1_000
            elif suffix == "m":
                base_num *= 1_000_000
            elif suffix == "b":
                base_num *= 1_000_000_000

            ratio = base_num / anchor_price if anchor_price else 0
            # Strict strike validation: only accept within ratio bounds
            if ratio_min < ratio < ratio_max:
                return base_num

        return None

    def _scan_polymarket(self, anchor: float, topic_type: str, max_pages: int = 5) -> Optional[Dict[str, Any]]:
        """Scan Polymarket for markets matching this crypto anchor.

        Selection criteria:
        - Price floor: Market price must be >= $0.18 (no longshots)
        - Strike validation: 0.2 < (strike/anchor) < 5.0
        - Volume optimization: Among valid markets, select highest volume

        Args:
            anchor: The anchor price (e.g., BTC/USDT last price)
            topic_type: Short string to match in event titles (e.g., "BTC", "ETH")
            max_pages: Max pages to scan

        Returns:
            Market dict with highest volume, or None if no suitable market found.
        """
        best_market = None
        highest_volume = 0.0

        for page in range(max_pages):
            params = {
                "active": "true",
                "closed": "false",
                "limit": 100,
                "offset": page * 100
            }
            try:
                resp = crequests.get(
                    self.polymarket_base,
                    params=params,
                    impersonate="chrome120",
                    timeout=15
                )
                if resp.status_code != 200:
                    break
                events = resp.json()
                if not events:
                    break

                for event in events:
                    title = event.get("title", "").lower()
                    slug = event.get("slug", "").lower()

                    # Match topic in title or slug
                    if topic_type.lower() not in title and topic_type.lower() not in slug:
                        continue

                    for market in event.get("markets", []):
                        if market.get("closed"):
                            continue

                        question = market.get("question", "")
                        current_price = float(market.get("lastTradePrice", 0) or 0)

                        # Rule 1: Price floor filter - reject markets below $0.18
                        if current_price < self.PRICE_FLOOR:
                            continue

                        # Rule 2: Price ceiling - reject markets at or above $1.00
                        if current_price >= 1.0:
                            continue

                        # Rule 3: Extract and validate strike (ratio 0.2 < strike/anchor < 5.0)
                        valid_strike = self._extract_valid_strike(
                            question, anchor,
                            ratio_min=self.STRIKE_RATIO_MIN,
                            ratio_max=self.STRIKE_RATIO_MAX
                        )
                        if not valid_strike:
                            continue

                        # Rule 4: Get token ID
                        tokens = market.get("clobTokenIds")
                        if isinstance(tokens, str):
                            try:
                                tokens = json.loads(tokens)
                            except Exception:
                                tokens = None

                        if not (isinstance(tokens, list) and tokens):
                            continue

                        # Rule 5: Extract volume (liquidity metric)
                        # Try different volume field names depending on API response
                        volume = float(market.get("volume", 0) or 0)
                        if volume == 0:
                            volume = float(market.get("liquidity", 0) or 0)
                        if volume == 0:
                            # Fallback: estimate liquidity from lastTradePrice and volume data
                            volume = float(market.get("tradingVolume", 0) or 0)

                        # Rule 6: Select market with highest volume
                        if volume > highest_volume:
                            highest_volume = volume
                            best_market = {
                                "market_id": str(tokens[0]).strip(),
                                "asset_type": f"Crypto::{topic_type.upper()}",
                                "strike_price": valid_strike,
                                "question": question,
                                "anchor_url": None,
                                "initial_price": current_price,
                                "volume": volume,
                            }

            except Exception as e:
                print(f"[CryptoHunter] Scan error on page {page}: {e}")
                break

        return best_market

    def hunt(self) -> Optional[Dict[str, Any]]:
        """Hunt for a crypto market.

        Tries each symbol in order. Returns the first valid market found.

        Returns:
            Market dict or None.
        """
        print(f"[CryptoHunter] {datetime.now().isoformat()} - Starting hunt for {len(self.symbols)} symbols")

        for symbol in self.symbols:
            print(f"[CryptoHunter] Trying symbol: {symbol}")

            # Get anchor price
            anchor_price = self._get_binance_price(symbol)
            if not anchor_price:
                print(f"[CryptoHunter] Failed to fetch anchor for {symbol}, skipping")
                continue

            # Scan Polymarket
            # Extract just the main part (e.g., "BTC" from "BTCUSDT")
            topic_match = symbol.replace("USDT", "").replace("BUSD", "")
            found = self._scan_polymarket(anchor_price, topic_match)

            if found:
                found["anchor_url"] = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
                print(f"[CryptoHunter] Found market for {symbol}: {found['market_id']}")
                self._anchor_value = anchor_price
                return found

        print("[CryptoHunter] No crypto markets found after trying all symbols")
        return None

    def get_live_truth(self, market: Dict[str, Any]) -> Optional[float]:
        """Fetch live BTC/ETH/crypto price from Binance for the given market.

        Args:
            market: Market dict with asset_type like "Crypto::BTCUSDT"

        Returns:
            Current spot price or None on error.
        """
        if not market:
            return None

        try:
            asset_type = market.get("asset_type", "")
            if not asset_type.startswith("Crypto::"):
                return None

            # Extract symbol (e.g., "BTCUSDT" from "Crypto::BTCUSDT")
            symbol = asset_type.split("::", 1)[1]
            return self._get_binance_price(symbol)
        except Exception as e:
            print(f"[CryptoHunter] get_live_truth error: {e}")
            return None

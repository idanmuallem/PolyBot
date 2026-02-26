"""
CryptoHunter: Hunts Polymarket for cryptocurrency price prediction markets.

Uses Binance as the anchor source for live prices.
"""

import json
import re
from datetime import datetime
from typing import Optional, Dict, Any

from curl_cffi import requests as crequests

from .base import BaseHunter


class CryptoHunter(BaseHunter):
    """Hunt markets related to cryptocurrency prices (BTC, ETH, etc.).

    Anchor: Binance API (e.g., BTC/USDT latest price)
    """

    DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]  # Bitcoin prioritized first

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
            res = crequests.get(
                f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}",
                timeout=5,
                impersonate="chrome120"
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
        # Relaxed regex: Optional '$' prefix to catch "Bitcoin 100k", "68,000", "Above 65500", or "ETH 3500"
        # Collect all numeric matches first (handles dates like 2026 then later '68,000')
        pattern = re.compile(r"(?:\$)?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*([kKmMbB])?")
        matches = list(pattern.finditer(question))
        candidates = []
        for match in matches:
            try:
                base_num = float(match.group(1).replace(",", ""))
            except Exception:
                continue
            suffix = (match.group(2) or "").lower()
            if suffix == "k":
                base_num *= 1_000
            elif suffix == "m":
                base_num *= 1_000_000
            elif suffix == "b":
                base_num *= 1_000_000_000
            candidates.append(base_num)

        # Evaluate candidates in order found; return the first that meets ratio checks
        for base_num in candidates:
            try:
                ratio = base_num / anchor_price if anchor_price else 0
            except Exception:
                continue
            if ratio_min < ratio < ratio_max:
                return base_num

        return None

    def _scan_polymarket(self, anchor: float, symbol: str, topic_type: str, max_pages: int = 5) -> Optional[Dict[str, Any]]:
        """Scan Polymarket for markets matching this crypto anchor.

        Selection criteria:
        - Price floor: Market price must be >= $0.18 (no longshots)
        - Strike validation: 0.2 < (strike/anchor) < 5.0
        - Volume optimization: Among valid markets, select highest volume

        Args:
            anchor: The anchor price (e.g., BTC/USDT last price)
            symbol: Full trading symbol (e.g., "BTCUSDT") for asset_type
            topic_type: Short string to match in event titles (e.g., "BTC", "ETH")
            max_pages: Max pages to scan

        Returns:
            Market dict with highest volume, or None if no suitable market found.
        """
        best_market = None
        highest_volume = 0.0
        lowest_distance = float("inf")

        # Optimization: Define aliases once outside the loop
        aliases = [topic_type.lower()]
        if topic_type.upper() == "BTC":
            aliases.append("bitcoin")
        elif topic_type.upper() == "ETH":
            aliases.append("ethereum")
        elif topic_type.upper() == "SOL":
            aliases.append("solana")

        for page in range(max_pages):
            params = {
                "active": "true",
                "closed": "false",
                "limit": 100,
                "offset": page * 100,
                "query": topic_type,
                "order": "volume",
                "ascending": "false",
            }
            try:
                resp = crequests.get(
                    self.polymarket_base,
                    params=params,
                    impersonate="chrome120",
                    timeout=15
                )
                if resp.status_code != 200:
                    print(f"[CryptoHunter] API Error {resp.status_code}: {resp.text}")
                    break
                events = resp.json()
                if not events:
                    break

                for event in events:
                    title = event.get("title", "").lower()
                    slug = event.get("slug", "").lower()

                    # Match topic in title or slug
                    # Expand search to include full names (e.g. BTC -> Bitcoin)
                    if not any(alias in title or alias in slug for alias in aliases):
                        continue

                    for market in event.get("markets", []):
                        if market.get("closed"):
                            continue

                        question = market.get("question", "")
                        # Deep concatenation: Combine event title, group title, market title, and outcomes
                        # This captures 'Bitcoin' from event title and "Above 68,000" from group bin
                        scannable_text = f"{event.get('title', '')} {market.get('groupItemTitle', '')} {market.get('title', '')} {question}".strip()
                        current_price = float(market.get("lastTradePrice", 0) or 0)
                        volume = float(market.get("volume", 0) or 0)
                        if volume == 0:
                            volume = float(market.get("liquidity", 0) or 0)
                        if volume == 0:
                            volume = float(market.get("tradingVolume", 0) or 0)

                        # Diagnostic logging BEFORE filters (if relevant to BTC/ETH)
                        if any(a in title or a in slug for a in aliases):
                            print(f"[DEBUG] Evaluating: {scannable_text} | Price: {current_price} | Vol: {volume}")

                        # Rule 1: Price floor filter - reject markets below PRICE_FLOOR
                        if current_price < self.PRICE_FLOOR:
                            if any(a in title or a in slug for a in aliases):
                                print(f"[DEBUG] Rejected: Price {current_price} below floor {self.PRICE_FLOOR}")
                            continue

                        # Rule 2: Price ceiling - reject markets priced above PRICE_CEILING
                        if current_price > self.PRICE_CEILING:
                            if any(a in title or a in slug for a in aliases):
                                print(f"[DEBUG] Rejected: Price {current_price} above ceiling {self.PRICE_CEILING}")
                            continue

                        # Rule 3: Extract and validate strike from scannable_text or outcomes
                        valid_strike = None
                        matched_outcome = None

                        # First try the combined scannable_text (includes event, group, outcome names)
                        valid_strike = self._extract_valid_strike(
                            scannable_text, anchor,
                            ratio_min=self.STRIKE_RATIO_MIN,
                            ratio_max=self.STRIKE_RATIO_MAX
                        )
                        if valid_strike is not None:
                            matched_outcome = scannable_text

                        # If still not found, iterate through outcomes and create sub_market_text for each
                        if valid_strike is None:
                            outcomes = market.get("outcomes") or []
                            for o in outcomes:
                                # Extract outcome name
                                outcome_name = None
                                if isinstance(o, dict):
                                    outcome_name = o.get("name") or o.get("label") or o.get("title")
                                else:
                                    outcome_name = str(o) if o else None

                                if not outcome_name:
                                    continue

                                # Deep concatenation: append outcome to scannable_text
                                sub_market_text = f"{scannable_text} {outcome_name}"
                                candidate = self._extract_valid_strike(
                                    sub_market_text, anchor,
                                    ratio_min=self.STRIKE_RATIO_MIN,
                                    ratio_max=self.STRIKE_RATIO_MAX
                                )
                                if candidate:
                                    valid_strike = candidate
                                    matched_outcome = outcome_name
                                    break

                        if not valid_strike:
                            if any(a in title or a in slug for a in aliases):
                                print(f"[DEBUG] Rejected: No valid strike extracted from '{scannable_text}'")
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

                        # Rule 5: Enforce minimum volume threshold
                        if volume < self.MIN_VOLUME:
                            if any(a in title or a in slug for a in aliases):
                                print(f"[DEBUG] Rejected: Volume {volume} below minimum {self.MIN_VOLUME}")
                            continue

                        # Compute distance if possible
                        try:
                            distance = abs(float(valid_strike) - float(anchor)) if valid_strike is not None else float("inf")
                        except Exception:
                            distance = float("inf")

                        # Determine if strike is within 10% of anchor
                        try:
                            within_10pct = (valid_strike is not None) and (abs(valid_strike - anchor) <= 0.10 * float(anchor))
                        except Exception:
                            within_10pct = False

                        # Liquidity-first: require higher liquidity to prioritize proximity; use MIN 5000
                        liquidity_threshold = max(self.MIN_VOLUME, 5000)

                        # Maintain two candidate slots: best_within (highest vol among within-10pct), best_fallback (smallest distance)
                        if not hasattr(self, "_best_within"):
                            self._best_within = None
                            self._best_within_vol = 0
                            self._best_fallback = None
                            self._best_fallback_dist = float("inf")
                            self._best_fallback_vol = 0

                        # Candidate qualifies for within-10pct consideration if liquidity threshold met
                        if volume >= liquidity_threshold and valid_strike is not None and within_10pct:
                            if volume > self._best_within_vol:
                                market_name = f"{event.get('title', '')} - {market.get('groupItemTitle', '')}".strip()
                                if not market_name.strip(" -"):
                                    market_name = matched_outcome or question
                                self._best_within = {
                                    "market_id": str(tokens[0]).strip(),
                                    "asset_type": f"Crypto::{symbol}",
                                    "strike_price": valid_strike,
                                    "question": question,
                                    "market_name": market_name,
                                    "anchor_url": None,
                                    "initial_price": current_price,
                                    "volume": volume,
                                }
                                self._best_within_vol = volume

                        # Fallback: consider for smallest distance among valid strikes
                        if valid_strike is not None:
                            if (distance < self._best_fallback_dist) or (distance == self._best_fallback_dist and volume > self._best_fallback_vol):
                                market_name = f"{event.get('title', '')} - {market.get('groupItemTitle', '')}".strip()
                                if not market_name.strip(" -"):
                                    market_name = matched_outcome or question
                                self._best_fallback = {
                                    "market_id": str(tokens[0]).strip(),
                                    "asset_type": f"Crypto::{symbol}",
                                    "strike_price": valid_strike,
                                    "question": question,
                                    "market_name": market_name,
                                    "anchor_url": None,
                                    "initial_price": current_price,
                                    "volume": volume,
                                }
                                self._best_fallback_dist = distance
                                self._best_fallback_vol = volume

            except Exception as e:
                print(f"[CryptoHunter] Scan error on page {page}: {e}")
                break

        # Prefer the within-10%-of-anchor candidate (high-liquidity) if present, else fallback by distance
        best = None
        if hasattr(self, "_best_within") and self._best_within is not None:
            best = self._best_within
        elif hasattr(self, "_best_fallback") and self._best_fallback is not None:
            best = self._best_fallback

        # Clean up temporary selection state
        for attr in ("_best_within", "_best_within_vol", "_best_fallback", "_best_fallback_dist", "_best_fallback_vol"):
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except Exception:
                    try:
                        delattr(self, attr)
                    except Exception:
                        pass

        return best

    def hunt(self) -> Optional[Dict[str, Any]]:
        """Hunt for a crypto market.

        Tries each symbol in order. Returns the first valid market found.

        Returns:
            Market dict or None.
        """
        print(f"[CryptoHunter] {datetime.now().isoformat()} - Starting hunt for {len(self.symbols)} symbols")

        # Try each symbol and search using multiple alias terms (Bitcoin/BTC, Ethereum/ETH)
        alias_map = {
            "BTC": ["Bitcoin", "BTC"],
            "ETH": ["Ethereum", "ETH"],
        }

        for symbol in self.symbols:
            print(f"[CryptoHunter] Trying symbol: {symbol}")

            # Get anchor price
            anchor_price = self._get_binance_price(symbol)
            if not anchor_price:
                print(f"[CryptoHunter] Failed to fetch anchor for {symbol}, skipping")
                continue

            # Prepare alias list for searching Polymarket
            key = symbol.replace("USDT", "").replace("BUSD", "").upper()
            aliases = alias_map.get(key, [key, key.lower()])

            for alias in aliases:
                print(f"[CryptoHunter] Searching Polymarket for alias: {alias}")
                found = self._scan_polymarket(anchor_price, symbol, alias)
                if found:
                    found["anchor_url"] = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
                    print(f"[CryptoHunter] Found market for {symbol} (alias={alias}): {found['market_id']}")
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

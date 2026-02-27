import time
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from brains import get_brain_for_asset_type
from brains.base import calculate_tte
from core.models import MarketData
from trading.decision_pipeline import DecisionContext, build_entry_pipeline
from .crypto import CryptoHunter
from .weather import WeatherHunter
from .economy import EconomyHunter


class PolymarketScannerHunter:
    """Coordinator hunter that runs one complete discovery+pricing pass."""

    def __init__(self, bridge, executor, budget_manager, config, hunters: Optional[list] = None):
        self.bridge = bridge
        self.executor = executor
        self.budget_manager = budget_manager
        self.config = config
        self.hunters = hunters or [CryptoHunter(), WeatherHunter(), EconomyHunter()]
        self.min_ev = float(config.min_ev)

        self.seen_markets = {}
        self.pipeline = build_entry_pipeline(executor=self.executor, budget_manager=self.budget_manager)

    def _get_active_seen_ids(self) -> List[str]:
        current_time = time.time()
        expired_ids = [mid for mid, ts in self.seen_markets.items() if current_time - ts >= 600]
        for mid in expired_ids:
            del self.seen_markets[mid]
        return list(self.seen_markets.keys())

    def get_active_markets(self, log_func) -> Tuple[Optional[MarketData], Optional[object]]:
        """Discovery pass across all hunters with TTE safety filters."""
        skip_ids = self._get_active_seen_ids()
        min_tte_days = float(self.config.min_tte_minutes) / (24.0 * 60.0)

        for hunter in self.hunters:
            hunter_name = hunter.__class__.__name__
            print(f"[SCANNER] Trying {hunter_name}... (skipping {len(skip_ids)} cooldown markets)")
            candidate_market = hunter.hunt(skip_ids=skip_ids)

            if not candidate_market:
                continue

            expiry_hint = (
                getattr(candidate_market, "expiry_date", None)
                or getattr(candidate_market, "question", None)
                or getattr(candidate_market, "market_name", None)
            )
            tte_days = calculate_tte(expiry_hint)

            if tte_days < min_tte_days or tte_days > float(self.config.max_tte_days):
                log_func(
                    "FILTERED",
                    candidate_market.asset_type,
                    candidate_market.market_id,
                    {
                        "market_name": candidate_market.market_name,
                        "reason": "TTE out of bounds",
                        "tte_days": round(tte_days, 4),
                        "min_tte_minutes": self.config.min_tte_minutes,
                        "max_tte_days": self.config.max_tte_days,
                    },
                )
                continue

            return candidate_market, hunter

        return None, None

    def fetch_order_book(self, market: MarketData) -> dict:
        """Fetch market orderbook snapshot (midpoint proxy for now)."""
        return {
            "token_id": market.market_id,
            "mid_price": float(market.initial_price),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def process_market(self, market: MarketData, hunter, log_func) -> bool:
        """Price, evaluate, and execute one market."""
        token_id = market.market_id
        asset_type = market.asset_type
        question = market.market_name

        self.bridge.status = f"🎯 {asset_type}: {question[:60]}..."
        self.bridge.market_question = question
        self.bridge.market_asset_type = asset_type
        self.bridge.current_token_id = token_id

        order_book = self.fetch_order_book(market)
        poly_price = float(order_book.get("mid_price", market.initial_price))
        self.bridge.market_poly = poly_price

        live_truth = hunter.get_live_truth(market)
        if live_truth is None:
            log_func("SCAN-SKIP", asset_type, token_id, {"reason": "live_truth unavailable"})
            return False

        self.bridge.market_actual = live_truth
        brain = get_brain_for_asset_type(asset_type)
        signal = brain.evaluate(market, live_truth, min_ev=self.min_ev)
        model_used = getattr(brain, "last_model_used", "unknown")

        self.bridge.forecast = signal.fair_value
        self.bridge.ev = signal.expected_value

        context = DecisionContext(
            market=market,
            asset_type=asset_type,
            token_id=token_id,
            question=question,
            signal=signal,
            model_used=model_used,
            poly_price=poly_price,
        )

        result = self.pipeline.handle(context, log_func)
        if result is None:
            result = context

        if result.status in {"not_tradable", "daily_limit"}:
            self.seen_markets[token_id] = time.time()

        if result.status == "watch_only":
            self.bridge.status = (
                f"[SYSTEM] Low Balance (${self.budget_manager.bridge.current_balance:.2f}). "
                "Operating in Watch-Only mode."
            )

        now = datetime.now(timezone.utc)
        self.bridge.last_update = now.strftime("%H:%M:%S")
        self.bridge.status = f"🎯 {asset_type}: {question}"
        log_func(
            "TRACK",
            asset_type,
            token_id,
            {
                "market_name": question,
                "model_used": model_used,
                "fair": round(signal.fair_value, 4),
                "ev": round(signal.expected_value, 4),
                "kelly": round(signal.kelly_size, 4),
                "bet_usd": round(result.actual_bet_usd, 2),
                "executed": result.executed,
                "status": result.status,
            },
        )
        return result.executed

    def scan_markets(self, log_func):
        """Execute one full scan-discover-process pass."""
        market, hunter = self.get_active_markets(log_func)
        if not market or not hunter:
            self.bridge.status = "❌ No markets found. Waiting..."
            return False
        return self.process_market(market, hunter, log_func)

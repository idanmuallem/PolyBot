# engine.py
"""
Engine: Lightweight orchestrator for the quantitative trading bot.

Responsibilities:
- Instantiate hunters, brains, and executor
- Run the main event loop (hunt -> get_live_truth -> brain.evaluate -> execute -> update UI)
- Coordinate between components

Data fetching:   delegated to Hunters (via get_live_truth)
Trade execution: delegated to TradeExecutor
Fair value calc: delegated to Brains (via evaluate)
"""

import asyncio
import time
from datetime import datetime, timezone
import os
from typing import Tuple

# Load .env automatically when present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from models import MarketData, TradeSignal
from hunters import get_default_hunters
from brains import get_brain_for_asset_type
from executor import TradeExecutor, RiskConfig

ROTATION_INTERVAL = 900  # Re-scan for the best market every 15 minutes


class MarketMonitor:
    """State keeper for the market monitoring engine.
    
    Maintains cooldown cache and execution state across hunt cycles.
    """
    
    def __init__(self):
        self.seen_markets = {}  # market_id: timestamp dictionary for 10-min cooldown
    
    def _get_active_seen_ids(self) -> list:
        """Return list of market_ids currently in cooldown (< 10 minutes old).
        
        Removes expired entries (>= 600 seconds old) from the cache.
        
        Returns:
            List of market_ids that should be skipped in current hunt.
        """
        current_time = time.time()
        expired_ids = [
            mid for mid, ts in self.seen_markets.items()
            if current_time - ts >= 600  # 10 minutes = 600 seconds
        ]
        for mid in expired_ids:
            del self.seen_markets[mid]
        
        return list(self.seen_markets.keys())


class BudgetTracker:
    """Tracks daily spending and enforces global daily risk limits."""
    
    def __init__(self, daily_limit_usd: float = 100.0, bankroll_usd: float = 1000.0):
        """Initialize budget tracker.
        
        Args:
            daily_limit_usd: Maximum total to spend in a single day
            bankroll_usd: Total bankroll (used for Kelly fraction calculations)
        """
        self.daily_limit_usd = daily_limit_usd
        self.bankroll_usd = bankroll_usd
        self.total_spent_today = 0.0
        self.day_start_time = time.time()
    
    def _reset_daily_stats(self):
        """Reset daily spending counter (call every 24 hours)."""
        self.total_spent_today = 0.0
        self.day_start_time = time.time()
        print(f"[BUDGET] Daily stats reset. Budget: ${self.daily_limit_usd}")
    
    def get_remaining_budget(self) -> float:
        """Get remaining budget for today."""
        return self.daily_limit_usd - self.total_spent_today
    
    def check_and_cap_bet(self, kelly_fraction: float) -> Tuple[float, bool]:
        """Check if trade fits in daily budget and cap if needed.
        
        Args:
            kelly_fraction: Kelly sizing fraction (0-1)
        
        Returns:
            Tuple of (actual_bet_usd, should_execute)
            - actual_bet_usd: Amount to execute (capped by budget)
            - should_execute: True if bet > 0, False if daily limit reached
        """
        # Check for 24-hour reset
        if time.time() - self.day_start_time >= 86400:
            self._reset_daily_stats()
        
        remaining = self.get_remaining_budget()
        desired_bet = kelly_fraction * self.bankroll_usd
        actual_bet = min(desired_bet, remaining)
        
        if actual_bet <= 0:
            print(f"[BUDGET] Daily limit reached (${self.daily_limit_usd} spent). Skipping trade.")
            return 0.0, False
        
        if actual_bet < desired_bet:
            print(f"[BUDGET] Kelly suggested ${desired_bet:.2f}, capping to ${actual_bet:.2f} (${remaining:.2f} remaining today)")
        
        return actual_bet, True
    
    def record_trade(self, amount_usd: float):
        """Record a completed trade against daily budget."""
        self.total_spent_today += amount_usd
        print(f"[BUDGET] Trade recorded: ${amount_usd:.2f}. Today's total: ${self.total_spent_today:.2f}/${self.daily_limit_usd:.2f}")


async def run_market_monitor(bridge, log_func):
    """Main orchestration loop.

    Delegates responsibilities:
    - Hunters: market discovery & live data fetching (respecting cooldown cache)
    - Brains:  fair value calculation via evaluate()
    - Executor: trade execution & risk management

    Args:
        bridge: State container with UI properties (status, market_poly, forecast, etc.)
        log_func: Logging callback function
    """
    # ========================================
    # SETUP: Initialize components
    # ========================================
    hunters = get_default_hunters()
    executor = TradeExecutor(risk_config=RiskConfig(ev_threshold=0.15))
    monitor = MarketMonitor()  # Initialize cooldown cache
    budget = BudgetTracker(daily_limit_usd=100.0, bankroll_usd=1000.0)  # Initialize budget tracker

    print(f"[ENGINE] Initialized {len(hunters)} hunters and TradeExecutor")
    print(f"[ENGINE] EV threshold: {executor.risk_config.ev_threshold}")
    print(f"[ENGINE] Daily limit: ${budget.daily_limit_usd} | Bankroll: ${budget.bankroll_usd}")

    while True:
        # ========================================
        # PHASE 1: HUNT (with cooldown awareness)
        # ========================================
        market = None
        hunter = None
        skip_ids = monitor._get_active_seen_ids()  # Get list of markets in cooldown

        for h in hunters:
            hunter_name = h.__class__.__name__
            print(f"[ENGINE] Trying {hunter_name}... (skipping {len(skip_ids)} cooldown markets)")
            market = h.hunt(skip_ids=skip_ids)  # Pass cooldown list to hunter

            if market:
                print(f"[ENGINE] {hunter_name} found: {market.market_id}")
                hunter = h
                break

        if not market:
            bridge.status = "❌ No markets found. Waiting 60s..."
            await asyncio.sleep(60)
            continue

        # ========================================
        # PHASE 2: EXTRACT MARKET METADATA
        # ========================================
        TOKEN_ID = market.market_id
        STRIKE = market.strike_price
        ASSET_TYPE = market.asset_type
        QUESTION = market.market_name

        bridge.status = f"🎯 {ASSET_TYPE}: {QUESTION[:60]}..."
        bridge.market_question = QUESTION
        bridge.market_asset_type = ASSET_TYPE
        print(f"[ENGINE] Tracking {ASSET_TYPE} => {TOKEN_ID} (strike {STRIKE})")

        # ========================================
        # PHASE 3: SELECT BRAIN
        # ========================================
        try:
            brain = get_brain_for_asset_type(ASSET_TYPE)
            brain_name = brain.__class__.__name__
            print(f"[ENGINE] Using {brain_name}")
        except ValueError as e:
            print(f"[ENGINE] Error: {e}")
            continue

        start_time = time.time()

        # ========================================
        # PHASE 4: LIVE TRACKING LOOP
        # ========================================
        while True:
            elapsed = time.time() - start_time
            if elapsed > ROTATION_INTERVAL:
                print("[ENGINE] Rotation interval reached, re-hunting...")
                break

            try:
                # --- A. Fetch Polymarket mid price ---
                from clients.polymarket import PolymarketClient
                poly_client = PolymarketClient()
                # Note: We'd need to add a method to PolymarketClient to get mid price
                # For now, use market's initial price as estimate
                poly_price = market.initial_price
                print(f"[ENGINE] Polymarket price for {TOKEN_ID}: {poly_price}")
                bridge.market_poly = poly_price  # Update bridge with Polymarket price

                # --- B. Fetch live value (delegated to hunter) ---
                live_truth = hunter.get_live_truth(market)
                if live_truth is None:
                    print(f"[ENGINE] Failed to get live truth for {ASSET_TYPE}, skipping this cycle")
                    await asyncio.sleep(5)
                    continue
                
                bridge.market_actual = live_truth
                print(f"[ENGINE] Live truth for {ASSET_TYPE}: {live_truth}")

                # --- C. Evaluate market using brain's template method ---
                signal: TradeSignal = brain.evaluate(market, live_truth, min_ev=executor.risk_config.ev_threshold)
                bridge.forecast = signal.fair_value
                bridge.ev = signal.expected_value
                print(f"[ENGINE] Brain evaluation: fair={signal.fair_value:.4f}, ev={signal.expected_value:.4f}, kelly={signal.kelly_size:.4f}")

                # --- D. Check tradability ---
                if not signal.is_tradable:
                    # Low EV or negative kelly: Add to cooldown cache and re-hunt
                    monitor.seen_markets[TOKEN_ID] = time.time()
                    print(f"[ENGINE] Market {TOKEN_ID} not tradable (EV={signal.expected_value:.4f}, Kelly={signal.kelly_size:.4f}). Entering 10m cooldown.")
                    bridge.status = f"⏸️ Low EV on {ASSET_TYPE}. Entering 10m cooldown..."
                    break  # Exit tracking loop, go back to hunt phase

                # --- E. Check daily budget and cap bet size ---
                actual_bet_usd, should_execute = budget.check_and_cap_bet(signal.kelly_size)
                
                if not should_execute:
                    # Daily limit exhausted: Enter cooldown and re-hunt
                    monitor.seen_markets[TOKEN_ID] = time.time()
                    bridge.status = f"💰 Daily limit reached. Re-hunting..."
                    break  # Exit tracking loop, go back to hunt phase
                
                # --- F. Delegate execution decision to executor ---
                executor.evaluate_and_execute(
                    market=market,
                    fair_value=signal.fair_value,
                    ev=signal.expected_value,
                    current_poly_price=poly_price,
                    log_func=log_func,
                )
                
                # Record the trade against daily budget
                budget.record_trade(actual_bet_usd)

                # --- G. Update UI ---
                now = datetime.now(timezone.utc)
                bridge.last_update = now.strftime("%H:%M:%S")
                bridge.status = f"🎯 {ASSET_TYPE}: {QUESTION}"
                log_func("TRACK", ASSET_TYPE, TOKEN_ID, 
                        {"fair": round(signal.fair_value, 4), "ev": round(signal.expected_value, 4), "kelly": round(signal.kelly_size, 4), "bet_usd": round(actual_bet_usd, 2)})

                await asyncio.sleep(2)

            except Exception as e:
                print(f"[ENGINE] Live update error: {e}")
                await asyncio.sleep(5)


def _get_live_truth_via_hunter(
    hunter,
    market: MarketData,
    asset_type: str,
    log_func,
) -> float:
    """Fetch live truth via the hunter's get_live_truth method.

    This encapsulates API calling logic away from the engine.

    Args:
        hunter: The active hunter instance
        market: MarketData object from hunt()
        asset_type: Asset type string
        log_func: Logging callback

    Returns:
        Live value or None on error
    """
    try:
        live_truth = hunter.get_live_truth(market)
        if live_truth is not None:
            print(f"[ENGINE] Live truth for {asset_type}: {live_truth}")
        return live_truth
    except Exception as e:
        print(f"[ENGINE] Error getting live truth: {e}")
        log_func("ERROR", asset_type, "N/A", str(e))
        return None
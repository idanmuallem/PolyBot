# engine.py
"""
Engine: Lightweight orchestrator for the quantitative trading bot.

Responsibilities:
- Instantiate hunters, brains, and executor
- Run the main event loop (hunt -> get_live_truth -> price -> execute -> update UI)
- Coordinate between components

Data fetching:   delegated to Hunters (via get_live_truth)
Trade execution: delegated to TradeExecutor
Fair value calc: delegated to Brains
"""

import asyncio
import time
from datetime import datetime, timezone
import os

# Load .env automatically when present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from curl_cffi import requests as crequests

from hunters import get_default_hunters
from brains import get_brain_for_asset_type
from executor import TradeExecutor, RiskConfig

ROTATION_INTERVAL = 900  # Re-scan for the best market every 15 minutes


async def run_market_monitor(bridge, log_func):
    """Main orchestration loop.

    Delegates responsibilities:
    - Hunters: market discovery & live data fetching
    - Brains:  fair value calculation
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

    print(f"[ENGINE] Initialized {len(hunters)} hunters and TradeExecutor")
    print(f"[ENGINE] EV threshold: {executor.risk_config.ev_threshold}")

    while True:
        # ========================================
        # PHASE 1: HUNT
        # ========================================
        market = None

        for hunter in hunters:
            hunter_name = hunter.__class__.__name__
            print(f"[ENGINE] Trying {hunter_name}...")
            market = hunter.hunt()

            if market:
                print(f"[ENGINE] {hunter_name} found: {market.get('market_id')}")
                break

        if not market:
            bridge.status = "❌ No markets found. Waiting 60s..."
            await asyncio.sleep(60)
            continue

        # ========================================
        # PHASE 2: EXTRACT MARKET METADATA
        # ========================================
        TOKEN_ID = market.get("market_id")
        STRIKE = market.get("strike_price")
        ASSET_TYPE = market.get("asset_type")
        QUESTION = market.get("question")

        bridge.status = f"🎯 Tracking: {ASSET_TYPE} - {QUESTION[:40]}..."
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
        EXPIRY_DATE = None

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
                mid_url = f"https://clob.polymarket.com/midpoint?token_id={TOKEN_ID}"
                mid_res = crequests.get(mid_url, impersonate="chrome120", timeout=5)
                poly_price = float(mid_res.json().get("mid", 0.50))
                bridge.market_poly = poly_price

                # --- B. Fetch live value (delegated to hunter) ---
                live_truth = _get_live_truth_via_hunter(hunter, market, ASSET_TYPE, log_func)
                if live_truth is None:
                    live_truth = 0.0

                # --- C. Compute time to expiry ---
                now = datetime.now(timezone.utc)
                if EXPIRY_DATE:
                    days_left = max(0.01, (EXPIRY_DATE - now).total_seconds() / 86400)
                else:
                    days_left = 7.0  # default

                # --- D. Calculate fair value (delegated to brain) ---
                fair_value = _get_fair_value_from_brain(brain, ASSET_TYPE, live_truth, STRIKE, days_left)
                bridge.forecast = fair_value

                # --- E. Calculate EV ---
                ev = calculate_ev(poly_price, fair_value)
                bridge.ev = ev

                # --- F. Delegate execution decision to executor ---
                executor.evaluate_and_execute(
                    market=market,
                    fair_value=fair_value,
                    ev=ev,
                    current_poly_price=poly_price,
                    log_func=log_func,
                )

                # --- G. Update UI ---
                bridge.last_update = now.strftime("%H:%M:%S")
                bridge.status = f"Tracking {ASSET_TYPE}: live={live_truth:.2f} forecast={fair_value:.3f} EV={ev:.3f}"
                log_func("TRACK", ASSET_TYPE, TOKEN_ID, {"fair": round(fair_value, 4), "ev": round(ev, 4)})

                await asyncio.sleep(2)

            except Exception as e:
                print(f"[ENGINE] Live update error: {e}")
                await asyncio.sleep(5)


def _get_live_truth_via_hunter(
    hunter,
    market: dict,
    asset_type: str,
    log_func,
) -> float:
    """Fetch live truth via the hunter's get_live_truth method.

    This encapsulates API calling logic away from the engine.

    Args:
        hunter: The active hunter instance
        market: Market dict from hunt()
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


def _get_fair_value_from_brain(
    brain,
    asset_type: str,
    live_truth: float,
    strike: float,
    days_left: float,
) -> float:
    """Calculate fair value via the brain's get_fair_value method.

    Extracts symbol/location/indicator and passes to brain.

    Args:
        brain: The active brain instance
        asset_type: Asset type string (e.g., "Crypto::BTCUSDT")
        live_truth: Current market value
        strike: Strike price
        days_left: Days to expiry

    Returns:
        Fair value probability
    """
    try:
        if asset_type.startswith("Crypto::"):
            symbol = asset_type.split("::", 1)[1]
            return brain.get_fair_value(live_truth, strike, days_left, symbol=symbol)
        elif asset_type.startswith("Weather::"):
            return brain.get_fair_value(live_truth, strike, days_left)
        elif asset_type.startswith("Economy::"):
            indicator = asset_type.split("::", 1)[1]
            return brain.get_fair_value(live_truth, strike, days_left, indicator=indicator)
        else:
            return 0.5
    except Exception as e:
        print(f"[ENGINE] Error calculating fair value: {e}")
        return 0.5


def calculate_ev(market_price: float, fair_value: float) -> float:
    """Calculate expected value.

    Args:
        market_price: Current Polymarket mid price
        fair_value: Our calculated fair value

    Returns:
        Expected value as a fraction
    """
    if market_price <= 0:
        return 0.0
    return (fair_value - market_price) / market_price
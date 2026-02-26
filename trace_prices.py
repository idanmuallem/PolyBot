#!/usr/bin/env python3
"""Trace price values through the engine."""

from models import MarketData
from hunters import CryptoHunter
from brains import get_brain_for_asset_type

# Hunt for a market
print("[TEST] Hunting for Bitcoin market...")
hunter = CryptoHunter()
market = hunter.hunt()

if market:
    print(f"\n[MARKET DATA]")
    print(f"  initial_price (Polymarket prob): {market.initial_price}")
    print(f"  strike_price (target price): {market.strike_price}")
    
    # Get brain
    brain = get_brain_for_asset_type(market.asset_type)
    
    # Get live truth
    live_truth = hunter.get_live_truth(market)
    print(f"\n[LIVE DATA]")
    print(f"  live_truth (spot price): {live_truth}")
    
    # Evaluate
    signal = brain.evaluate(market, live_truth, min_ev=0.02)
    print(f"\n[SIGNAL]")
    print(f"  fair_value: {signal.fair_value}")
    print(f"  expected_value: {signal.expected_value}")
    print(f"  kelly_size: {signal.kelly_size}")
    print(f"  is_tradable: {signal.is_tradable}")
    
    # What would dashboard show?
    print(f"\n[DASHBOARD DISPLAY]")
    print(f"  Live: ${live_truth:,.2f}")
    print(f"  Polymarket Price: ${market.initial_price:.3f}")  
    print(f"  Math-Fair Value: {signal.fair_value:.1%}")
    print(f"  Trading Edge (EV): {signal.expected_value:.2%}")
else:
    print("No market found!")

#!/usr/bin/env python3
"""Diagnostic script to test price extraction from Polymarket API."""

import json
from clients.polymarket import PolymarketClient

# Test direct API call
print("[DIAG] Testing PolymarketClient direct API call...")
client = PolymarketClient()
events = client.search_events("Bitcoin", limit=5, offset=0)

if events:
    print(f"[DIAG] Found {len(events)} events")
    for i, event in enumerate(events[:2]):  # Show first 2 events
        print(f"\n[EVENT {i}] {event.get('title', 'No title')}")
        markets = event.get('markets', [])
        print(f"  Markets: {len(markets)}")
        if markets:
            market = markets[0]
            print(f"  Market keys: {list(market.keys())}")
            print(f"  lastTradePrice: {market.get('lastTradePrice', 'NOT FOUND')}")
            print(f"  last_price: {market.get('last_price', 'NOT FOUND')}")
            print(f"  mid_price: {market.get('mid_price', 'NOT FOUND')}")
            print(f"  volume: {market.get('volume', 'NOT FOUND')}")
            print(f"  Full market object (first 500 chars):")
            print(f"    {str(market)[:500]}")
else:
    print("[DIAG] No events found")

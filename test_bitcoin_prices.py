#!/usr/bin/env python3
"""Test price extraction for Bitcoin markets."""

from clients.polymarket import PolymarketClient

client = PolymarketClient()

# Search for Bitcoin markets
print("[DIAG] Searching for Bitcoin markets...")
events = client.search_events("Bitcoin", limit=10, offset=0)

bitcoin_prices = []
for event in events:
    for market in event.get('markets', []):
        price = market.get('lastTradePrice')
        volume_str = market.get('volume', '0')
        try:
            volume = float(volume_str) if isinstance(volume_str, str) else volume_str
        except:
            volume = 0
        title = event.get('title', '')[:50]
        q = market.get('question', '')[:50]
        if price and volume > 0:
            bitcoin_prices.append({
                'price': price,
                'volume': volume,
                'title': title,
                'question': q
            })
            print(f"  [FOUND] Price: {price}, Volume: {volume:.0f} | Q: {q}")

if bitcoin_prices:
    print(f"\n[RESULT] Found {len(bitcoin_prices)} Bitcoin markets with prices")
    best = max(bitcoin_prices, key=lambda x: x['volume'])
    print(f"  Highest volume: price={best['price']}, vol={best['volume']:.0f}")
else:
    print("[RESULT] No Bitcoin markets with prices found")

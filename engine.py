# engine.py
import asyncio
import requests
import re
from datetime import datetime, timezone
from py_clob_client.client import ClobClient
from brain import calculate_fair_value, calculate_ev

def find_best_bitcoin_market(bad_ids=None):
    """Hunts the CLOB API for an ACTIVE Bitcoin market, ignoring bad/closed ones."""
    if bad_ids is None:
        bad_ids = []
        
    cursor = ""
    
    for _ in range(15):  # Scan up to 15 pages deep
        url = "https://clob.polymarket.com/markets"
        params = {"next_cursor": cursor} if cursor else {}
        
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                break
                
            data = resp.json()
            market_list = data.get('data', [])
            
            for m in market_list:
                # 🛑 THE FIX: Ignore markets that are closed, archived, or halted
                if not m.get('active') or m.get('closed') or not m.get('accepting_orders'):
                    continue
                    
                question = m.get('question', m.get('description', ''))
                
                if 'Bitcoin' in question or 'BTC' in question:
                    tokens = m.get('tokens', [])
                    
                    if isinstance(tokens, list) and len(tokens) >= 2:
                        yes_token = None
                        
                        for t in tokens:
                            if str(t.get('outcome', '')).lower() == 'yes':
                                yes_token = str(t.get('token_id'))
                                break
                        
                        if not yes_token:
                            yes_token = str(tokens[0].get('token_id'))
                            
                        # 🛑 THE FIX: Ignore IDs that previously caused a 404 error
                        if yes_token in bad_ids:
                            continue
                            
                        strike = 100000.0
                        price_match = re.search(r'\$(\d{1,3}(?:,\d{3})*)', question)
                        if price_match:
                            strike = float(price_match.group(1).replace(',', ''))
                            
                        return {
                            "id": yes_token,
                            "strike": strike,
                            "question": question
                        }
                        
            cursor = data.get('next_cursor')
            if not cursor or cursor == "LTE=":
                break
                
        except Exception as e:
            print(f"❌ CLOB Scan Error: {e}")
            break
            
    return None

async def run_market_monitor(bridge, log_func):
    client = ClobClient(host="https://clob.polymarket.com")
    failed_ids = [] # The Blacklist memory
    
    bridge.status = "📡 Scanning Orderbook for ACTIVE Markets..."
    market = find_best_bitcoin_market(failed_ids)
    
    if not market:
        bridge.status = "❌ No Active BTC Markets Found"
        return

    TOKEN_ID = market['id']
    STRIKE_PRICE = market['strike']
    EXPIRY_DATE = datetime(2026, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    bridge.status = f"🎯 Found: {market['question'][:30]}..."

    while True:
        try:
            # 1. Fetch live Polymarket Orderbook Price
            mid_data = client.get_midpoint(TOKEN_ID)
            bridge.market_poly = float(mid_data.get('mid', 0.50))
            
            # 2. Fetch live Binance Price
            res = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", timeout=5)
            bridge.market_actual = float(res.json()['price'])
            
            # 3. Brain Calculation
            now = datetime.now(timezone.utc)
            days_left = max(0.01, (EXPIRY_DATE - now).total_seconds() / 86400)
            bridge.forecast = calculate_fair_value(bridge.market_actual, STRIKE_PRICE, days_left)
            
            # 4. Dashboard Updates
            bridge.ev = calculate_ev(bridge.market_poly, bridge.forecast)
            bridge.last_update = now.strftime("%H:%M:%S")
            bridge.status = f"🟢 Tracking: ${STRIKE_PRICE:,.0f} Market"

            if bridge.automation_enabled and bridge.ev > 0.15:
                log_func(f"BTC>{STRIKE_PRICE}", "BUY", bridge.market_poly, 10, "AUTO")

            await asyncio.sleep(2)
            
        except Exception as e:
            err_msg = str(e)
            print(f"Loop Error: {err_msg}")
            
            # --- FIXED SELF HEALING LOGIC ---
            if "404" in err_msg or "No orderbook" in err_msg:
                print(f"Banning Bad ID: {TOKEN_ID[:10]}...")
                failed_ids.append(TOKEN_ID) # Add the broken ID to the Blacklist
                bridge.status = "⚠️ Market Expired. Hunting next..."
                
                # Hunt again, passing the blacklist so it finds a DIFFERENT market
                new_market = find_best_bitcoin_market(failed_ids)
                if new_market:
                    TOKEN_ID = new_market['id']
                    STRIKE_PRICE = new_market['strike']
                else:
                    bridge.status = "❌ Exhausted all BTC markets."
            else:
                bridge.status = f"⚠️ Lag: {err_msg[:15]}..."
                
            await asyncio.sleep(5)
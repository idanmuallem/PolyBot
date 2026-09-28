import os
import sys
import math
import json
import time
from datetime import datetime, timezone
import dateutil.parser

# Add repo root to sys.path so we can import brains and core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from curl_cffi import requests
import ccxt
import pandas as pd
from brains.crypto import HybridCryptoBrain
from hunters.parsers import extract_crypto_strike

def is_btc_eth_strike_market(market):
    title = market.get('question', '').lower()
    desc = market.get('description', '').lower()
    
    if not any(k in title or k in desc for k in ['btc', 'bitcoin', 'eth', 'ethereum']):
        return False
        
    if "binance 1 minute candle" not in desc and "binance 1-minute candle" not in desc and "coinbase spot price" not in desc:
        return False
        
    if not any(k in title or k in desc for k in ['exceed', 'trade below', 'reach', 'dip to', 'above', 'hit']):
        return False

    return True

def get_realized_vol(closes):
    if len(closes) < 2:
        return 0.5
    returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes)) if closes[i - 1] > 0 and closes[i] > 0]
    if len(returns) < 2:
        return 0.5
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(variance) * math.sqrt(365.0)

TRADES_URL = "https://data-api.polymarket.com/trades"
TRADES_PAGE_LIMIT = 500

def fetch_trades_page(condition_id, offset, limit=TRADES_PAGE_LIMIT):
    r = requests.get(
        TRADES_URL,
        params={"market": condition_id, "limit": limit, "offset": offset},
        impersonate="chrome120",
        timeout=15,
    )
    if r.status_code != 200:
        return []
    try:
        return r.json()
    except Exception:
        return []

def find_trade_near_time(condition_id, yes_token, target_ts, page_cache, max_offset=100000, limit=TRADES_PAGE_LIMIT):
    """Binary-search the trades endpoint (sorted newest-first, paginated by
    offset) for the YES-token trade whose timestamp is closest to target_ts.

    The /trades endpoint has no time-range filter, only offset pagination,
    and top-volume markets can have 10k+ trades concentrated near
    resolution -- a flat `limit=500` fetch only sees the most recent
    trades. Binary search over offset (page-aligned) finds a snapshot near
    any point in the market's lifetime in O(log n) requests instead of
    paginating through everything.
    """
    lo_step, hi_step = 0, max_offset // limit
    best = None

    def get_page(step):
        offset = step * limit
        page = page_cache.get(offset)
        if page is None:
            page = fetch_trades_page(condition_id, offset, limit)
            page_cache[offset] = page
        return page

    while lo_step <= hi_step:
        mid_step = (lo_step + hi_step) // 2
        page = get_page(mid_step)
        if not page:
            # past the end of available trades -> older data is further out
            hi_step = mid_step - 1
            continue

        yes_on_page = [tr for tr in page if tr.get('asset') == yes_token]
        if yes_on_page:
            closest = min(yes_on_page, key=lambda tr: abs(tr['timestamp'] - target_ts))
            if best is None or abs(closest['timestamp'] - target_ts) < abs(best['timestamp'] - target_ts):
                best = closest

        page_first_ts, page_last_ts = page[0]['timestamp'], page[-1]['timestamp']
        if target_ts > page_first_ts:
            hi_step = mid_step - 1
        elif target_ts < page_last_ts:
            lo_step = mid_step + 1
        else:
            break

    return best

def fetch_crypto_ohlcv(symbol, exchange, days_back=1000):
    print(f"Fetching historical daily OHLCV for {symbol}...")
    # 1000 days is the max limit for Binance daily
    candles = exchange.fetch_ohlcv(symbol, timeframe="1d", limit=days_back)
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    return df

def run_backtest():
    url = "https://gamma-api.polymarket.com/events"
    limit = 100
    all_events = []
    
    # 1. Fetch markets
    print("Fetching closed events from Gamma API...")
    for keyword in ["BTC", "ETH"]:
        offset = 0
        while True:
            params = {
                "closed": "true",
                "limit": limit,
                "offset": offset,
                "order": "volume",
                "ascending": "false",
                "q": keyword
            }
            resp = requests.get(url, params=params, impersonate="chrome120", timeout=15)
            if resp.status_code != 200:
                break
            
            events = resp.json()
            if not events:
                break
            
            all_events.extend(events)
            offset += limit
            if offset >= 500: # just get a few pages
                break

    unique_events = {e['id']: e for e in all_events}
    
    strike_markets = []
    for event_id, event in unique_events.items():
        if not event.get('markets'):
            continue
        for market in event['markets']:
            if market.get('closed') and market.get('resolvedBy'):
                if is_btc_eth_strike_market(market):
                    strike_markets.append(market)

    # Sort by volume and pick top 56
    strike_markets.sort(key=lambda x: float(x.get('volume', 0)), reverse=True)
    target_markets = strike_markets[:56]
    print(f"Selected {len(target_markets)} top volume BTC/ETH strike markets for backtest.")
    
    # 2. Fetch historical prices from CCXT
    exchange = ccxt.binance()
    ohlcv_data = {
        "BTC": fetch_crypto_ohlcv("BTC/USDT", exchange),
        "ETH": fetch_crypto_ohlcv("ETH/USDT", exchange)
    }

    brain = HybridCryptoBrain()
    dataset = []

    print("Fetching snapshots and re-evaluating...")
    for m_idx, m in enumerate(target_markets):
        print(f"[{m_idx+1}/{len(target_markets)}] Processing {m['question']}...")
        
        # Determine asset
        asset = "BTC" if "btc" in m['question'].lower() or "bitcoin" in m['question'].lower() else "ETH"
        
        # Strike price
        strike = extract_crypto_strike(m['question'], 0.0)
        if not strike:
            print("  Could not extract strike. Skipping.")
            continue
            
        # Outcome
        outcomes = json.loads(m.get('outcomes', '[]'))
        outcome_prices = json.loads(m.get('outcomePrices', '[]'))
        # Usually outcomes is ["Yes", "No"], and if Yes won, Yes price is 1
        yes_idx = -1
        for i, out in enumerate(outcomes):
            if out.lower() == 'yes':
                yes_idx = i
                break
        if yes_idx == -1:
            continue
            
        resolved_yes = (float(outcome_prices[yes_idx]) >= 0.99)
        outcome_val = 1 if resolved_yes else 0
        
        # Clob token id for Yes
        clob_ids = json.loads(m.get('clobTokenIds', '[]'))
        if not clob_ids:
            continue
        yes_token = clob_ids[yes_idx]

        end_date_raw = m.get('endDate') or m.get('closedTime') or m.get('umaEndDate')
        if not end_date_raw:
            print("  No end date available. Skipping.")
            continue
        end_date = dateutil.parser.parse(end_date_raw)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        # Fetch historical trades (prices-history is empty for closed markets).
        # High-volume markets have 10k+ trades concentrated near resolution, so
        # a flat limit=500 fetch only sees the last few days -- instead, binary
        # search the offset-paginated endpoint for trades near 5 target times
        # spread evenly across the market's actual lifetime.
        condition_id = m.get('conditionId')
        if not condition_id:
            print("  No conditionId. Skipping.")
            continue

        created_raw = m.get('createdAt')
        start_date = dateutil.parser.parse(created_raw) if created_raw else None
        if start_date and start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if not start_date or start_date >= end_date:
            print("  No usable start date. Skipping.")
            continue

        start_ts = start_date.timestamp()
        end_ts = end_date.timestamp()
        target_fracs = [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]
        target_times = [start_ts + f * (end_ts - start_ts) for f in target_fracs]

        page_cache = {}
        seen_trade_ids = set()
        snapshots = []
        for target_ts in target_times:
            trade = find_trade_near_time(condition_id, yes_token, target_ts, page_cache)
            if trade is None:
                continue
            key = trade.get('transactionHash', '') + str(trade['timestamp'])
            if key in seen_trade_ids:
                continue
            seen_trade_ids.add(key)
            snapshots.append(trade)

        if not snapshots:
            print("  No YES-token trades found near target times.")
            continue

        for snap in snapshots:
            snap_t = snap['timestamp']
            snap_dt = datetime.fromtimestamp(snap_t, tz=timezone.utc)
            market_prob = float(snap['price'])
            
            # TTE
            tte_days = (end_date - snap_dt).total_seconds() / 86400.0
            if tte_days <= 0:
                continue
                
            # Get historical spot and vol
            df = ohlcv_data[asset]
            # filter to before snap_dt
            past_df = df[df.index <= snap_dt]
            if past_df.empty:
                continue
                
            spot = past_df.iloc[-1]['close']
            
            # Vol 30d
            window_df = past_df.tail(31)
            vol = get_realized_vol(window_df['close'].tolist())
            
            # Call brain
            brain_prob = brain._price_first_passage(spot, strike, tte_days, vol)
            
            dataset.append({
                "market_id": m.get("id"),
                "question": m.get("question"),
                "asset": asset,
                "snapshot_time": snap_dt.isoformat(),
                "tte_days": round(tte_days, 2),
                "strike": strike,
                "spot": spot,
                "vol_30d": round(vol, 4),
                "market_prob": market_prob,
                "brain_prob": brain_prob,
                "outcome": outcome_val
            })
            
        time.sleep(0.1)

    print(f"Generated {len(dataset)} snapshots across {len(target_markets)} markets.")
    
    # Compute Brier scores and summarize
    df_res = pd.DataFrame(dataset)
    if df_res.empty:
        print("No dataset generated.")
        return
        
    df_res['divergence'] = df_res['brain_prob'] - df_res['market_prob']
    df_res['brier_brain'] = (df_res['brain_prob'] - df_res['outcome']) ** 2
    df_res['brier_market'] = (df_res['market_prob'] - df_res['outcome']) ** 2
    
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "backtest_resolved_markets.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"Saved dataset to {csv_path}")
    
    print("\n--- Summary ---")
    print(f"Total snapshots: {len(df_res)}")
    print(f"Brain average Brier:  {df_res['brier_brain'].mean():.4f}")
    print(f"Market average Brier: {df_res['brier_market'].mean():.4f}")

if __name__ == "__main__":
    run_backtest()

# Smart Cooldown Cache Implementation

## Overview
Implemented a **10-minute Smart Cooldown Cache** that forces the hunting system to explore alternative opportunities when a market has insufficient Expected Value (EV). Instead of getting stuck on "Bitcoin above 68,000", the system will automatically explore "Bitcoin above 70,000" or "Bitcoin above 72,000" for 10 minutes.

## Architecture

### 1. **Engine.py** - Cooldown State Management

#### MarketMonitor Class
```python
class MarketMonitor:
    """State keeper for the market monitoring engine."""
    
    def __init__(self):
        self.seen_markets = {}  # market_id: timestamp dictionary
    
    def _get_active_seen_ids(self) -> list:
        """Return list of market_ids currently in cooldown (< 10 minutes old).
        
        Removes expired entries (>= 600 seconds old) from the cache.
        """
```

**Key Features:**
- `seen_markets`: Dictionary storing `market_id: timestamp` pairs
- `_get_active_seen_ids()`: 
  - Gets current time using `time.time()`
  - Removes any market_ids older than 600 seconds (10 minutes)
  - Returns list of currently blocked market_ids

#### Cooldown Logic in Main Loop
```python
# Check EV threshold and manage cooldown
if ev < executor.risk_config.ev_threshold:
    # Low EV: Add to cooldown cache and re-hunt
    monitor.seen_markets[TOKEN_ID] = time.time()
    print(f"[ENGINE] Market {TOKEN_ID} has low EV ({ev:.4f} < {executor.risk_config.ev_threshold}). Entering 10m cooldown. Searching for new targets...")
    bridge.status = f"⏸️ Low EV on {ASSET_TYPE}. Entering 10m cooldown..."
    break  # Exit tracking loop, go back to hunt phase
```

**Behavior:**
- When found market has EV below threshold, it's added to cooldown cache
- Engine immediately breaks the tracking loop and searches for alternatives
- Market remains off-limits for exactly 10 minutes
- After 10 minutes, market becomes huntable again if still active

---

### 2. **Hunters/Base.py** - Abstract Interface Update

Updated `hunt()` method signature:
```python
@abstractmethod
def hunt(self, skip_ids: list = None) -> Optional[Dict[str, Any]]:
    """Hunt for a market matching this hunter's domain.
    
    The skip_ids parameter allows the engine to exclude markets currently in cooldown,
    forcing the hunter to explore alternative opportunities.
    
    Args:
        skip_ids: List of market_ids to skip (in cooldown). If None, defaults to [].
    
    Returns:
        Market dict or None.
    """
```

---

### 3. **Hunters/Crypto.py** - Skip Logic Implementation

#### Updated _scan_polymarket()
```python
def _scan_polymarket(self, anchor: float, symbol: str, topic_type: str, 
                    skip_ids: list = None, max_pages: int = 5):
    """Scan Polymarket for markets matching this crypto anchor.
    
    Args:
        skip_ids: List of market_ids to skip (in cooldown). Defaults to [].
    """
    if skip_ids is None:
        skip_ids = []
```

#### Skip Check in Market Loop
```python
# Get token ID early for skip_ids check
tokens = market.get("clobTokenIds")
if isinstance(tokens, str):
    try:
        tokens = json.loads(tokens)
    except Exception:
        tokens = None

if not (isinstance(tokens, list) and tokens):
    continue

market_id = str(tokens[0]).strip()

# Skip markets in cooldown cache
if market_id in skip_ids:
    print(f"[CryptoHunter] Skipping {market_id} (in 10m cooldown)")
    continue
```

#### Updated hunt()
```python
def hunt(self, skip_ids: list = None) -> Optional[Dict[str, Any]]:
    """Hunt for a crypto market.
    
    Respects skip_ids list to avoid markets in 10-minute cooldown.
    
    Args:
        skip_ids: List of market_ids to skip (in cooldown). Defaults to [].
    """
    if skip_ids is None:
        skip_ids = []
    
    for alias in aliases:
        found = self._scan_polymarket(anchor_price, symbol, alias, skip_ids=skip_ids)
```

---

### 4. **Hunters/Weather.py** - Skip Logic Implementation

Same pattern as CryptoHunter:

```python
def hunt(self, skip_ids: list = None) -> Optional[Dict[str, Any]]:
    """Hunt for a weather market.
    
    Respects skip_ids list to avoid markets in 10-minute cooldown.
    """
    if skip_ids is None:
        skip_ids = []
```

Early market_id extraction with skip check:
```python
# Rule 0: Get token ID early for skip_ids check
tokens = market.get("clobTokenIds")
if isinstance(tokens, str):
    try:
        tokens = json.loads(tokens)
    except Exception:
        tokens = None

if not (isinstance(tokens, list) and tokens):
    continue

market_id = str(tokens[0]).strip()

# Skip markets in cooldown cache
if market_id in skip_ids:
    print(f"[WeatherHunter] Skipping {market_id} (in 10m cooldown)")
    continue
```

---

### 5. **Hunters/Economy.py** - Skip Logic Implementation

Same pattern with FRED-based economy hunting:

```python
def hunt(self, skip_ids: list = None) -> Optional[Dict[str, Any]]:
    """Hunt for an economy market.
    
    Respects skip_ids list to avoid markets in 10-minute cooldown.
    """
    if skip_ids is None:
        skip_ids = []
    
    found = self._scan_polymarket(anchor_val, indicator, skip_ids=skip_ids)
```

---

## Workflow Diagram

```
┌─────────────────┐
│  Hunt Markets   │ ← Skip 10m cooldown markets
└────────┬────────┘
         │
         ├─ CryptoHunter (respects skip_ids)
         ├─ WeatherHunter (respects skip_ids)
         └─ EconomyHunter (respects skip_ids)
                │
         ┌──────▼──────┐
         │ Track Market │
         └──────┬───────┘
                │
         ┌──────▼────────────────┐
         │ Calculate Fair Value  │
         └──────┬────────────────┘
                │
         ┌──────▼──────────┐
         │ Calculate EV    │
         └──────┬──────────┘
                │
         ┌──────▼─────────────┐
         │ EV >= Threshold?   │
         └──┬────────────┬────┘
       YES  │           NO
            │            │
       ┌────▼────┐   ┌───▼──────────────┐
       │ Execute │   │ Add to Cooldown  │
       └─────────┘   │ Re-hunt          │
                     └──────────────────┘
```

---

## Example Scenarios

### Scenario 1: Bitcoin Market Exhaustion
1. **Hunt 1**: Find "Bitcoin above 68,000" (EV = 0.12, threshold = 0.15)
2. **Evaluation**: EV is too low (0.12 < 0.15)
3. **Action**: Add to cooldown cache with `time.time()`
4. **Hunt 2**: Skip "Bitcoin above 68,000", find "Bitcoin above 70,000" (EV = 0.18)
5. **Tracking**: Execute trades on "Bitcoin above 70,000"
6. **After 10 min**: "Bitcoin above 68,000" becomes huntable again

### Scenario 2: Weather Market Rotation
1. Find "Miami above 85°F" for 2 minutes
2. Market EV drops below threshold
3. Add to cooldown, search for "Miami above 87°F"
4. Track new market for up to 8 minutes
5. After 10 minutes total, "Miami above 85°F" available again

### Scenario 3: Economy Indicator Shift
1. Find "Fed Rate above 5.25%" market
2. EV calculation shows insufficient edge
3. Cooldown activated
4. EconomyHunter explores "Fed Rate above 5.5%" instead
5. Broader exploration of Fed Rate options

---

## Code Changes Summary

| File | Changes | Purpose |
|------|---------|---------|
| `engine.py` | Added `MarketMonitor` class + cooldown logic | State management |
| `hunters/base.py` | Updated `hunt()` signature to accept `skip_ids` | Interface contract |
| `hunters/crypto.py` | Added skip_ids checks + early token_id extraction | Skip low-EV markets |
| `hunters/weather.py` | Added skip_ids checks + early token_id extraction | Skip low-EV markets |
| `hunters/economy.py` | Added skip_ids checks + early token_id extraction | Skip low-EV markets |

---

## Performance Impact

**Positive:**
- Avoids wasting tracking cycles on low-EV markets
- Forces exploration of broader market opportunities
- Provides 10-minute recovery period for market conditions to change

**Minimal Overhead:**
- O(n) lookup in skip_ids list where n = number of cooldown markets
- Dictionary cleanup is O(m) where m = expired markets (typically < 100)
- Single `time.time()` call per hunt cycle

---

## Testing

To verify the cooldown cache works:

1. Run the dashboard
2. Monitor console logs for messages like:
   ```
   [ENGINE] Market abc123xyz has low EV (0.12 < 0.15). Entering 10m cooldown. Searching for new targets...
   [CryptoHunter] Skipping abc123xyz (in 10m cooldown)
   ```
3. Observe the engine hunts different markets after cooldown activation
4. Wait 10 minutes to see expired markets become huntable again

---

## Future Enhancements

1. **Configurable Cooldown Duration**: Allow custom cooldown periods per asset type
2. **Adaptive Thresholds**: Increase EV threshold after repeated low-EV discoveries
3. **Market Quality Scoring**: Track historical EV of markets to pre-filter poor opportunities
4. **Cooldown Analytics**: Dashboard widget showing active cooldowns and epoch remaining


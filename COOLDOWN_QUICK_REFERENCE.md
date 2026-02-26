# Smart Cooldown Cache - Quick Reference

## Installation & Verification

### 1. Updated Files
```
✅ engine.py           - Added MarketMonitor class + cooldown logic
✅ hunters/base.py     - Updated hunt() signature
✅ hunters/crypto.py   - Skip logic + early token extraction
✅ hunters/weather.py  - Skip logic + early token extraction
✅ hunters/economy.py  - Skip logic + early token extraction
📄 SMART_COOLDOWN_IMPLEMENTATION.md - Full documentation
```

### 2. Running the System

```bash
# Start the dashboard
cd c:\Users\97250\Desktop\Jobs\PolyBot
.venv\Scripts\streamlit run dashboard.py
```

### 3. Expected Console Output

**Normal Hunt:**
```
[ENGINE] Trying CryptoHunter... (skipping 0 cooldown markets)
[CryptoHunter] Starting hunt for 2 symbols (skipping 0 cooldown markets)
[CryptoHunter] Found market for BTCUSDT (alias=Bitcoin): 0x...abc
```

**Low EV Detection:**
```
[ENGINE] Tracking Crypto::BTCUSDT => 0x...abc (strike 68000.0)
[ENGINE] Market 0x...abc has low EV (0.12 < 0.15). Entering 10m cooldown. Searching for new targets...
```

**Skip Logic Active:**
```
[ENGINE] Trying CryptoHunter... (skipping 1 cooldown markets)
[CryptoHunter] Skipping 0x...abc (in 10m cooldown)
[CryptoHunter] Found market for BTCUSDT (alias=Bitcoin): 0x...xyz
```

---

## API Signature

### Engine
```python
class MarketMonitor:
    def __init__(self):
        self.seen_markets = {}
    
    def _get_active_seen_ids(self) -> list:
        # Returns currently blocked market_ids (< 10 min old)
        # Auto-cleans expired entries
```

### Hunters
```python
def hunt(self, skip_ids: list = None) -> Optional[Dict[str, Any]]:
    """
    Args:
        skip_ids: Market IDs to skip (in cooldown)
    
    Returns:
        Market dict or None
    """
```

---

## Cooldown Logic Flow

```
┌─────────────────────────────┐
│ Hunt for market             │
│ (excluding skip_ids)        │
└──────────────┬──────────────┘
               │
         ┌─────▼──────┐
         │ Found?     │
         └─┬───────┬──┘
        YES│       │NO
           │       │
    ┌──────▼──┐   └─ Wait 60s, retry
    │ Track   │
    │ Market  │
    └──────┬──┘
           │
    ┌──────▼──────────┐
    │ EV >= Threshold?│
    └─┬───────────┬───┘
   YES│          NO│
      │           │
 ┌────▼────┐  ┌───▼─────────────────────┐
 │ Execute │  │ Add market_id to cache  │
 │ Trades  │  │ Schedule next hunt      │
 └─────────┘  │ with skip_ids delivered │
              └─────────────────────────┘
```

---

## Configuration

### EV Threshold
```python
# In engine.py
executor = TradeExecutor(risk_config=RiskConfig(ev_threshold=0.15))
```

### Cooldown Duration
```python
# In engine.py, MarketMonitor._get_active_seen_ids()
if current_time - ts >= 600:  # 600 seconds = 10 minutes
    del self.seen_markets[mid]
```

To change:
- `0.15` (15%) → lower for more conservative execution
- `600` → milliseconds (3600 = 1 hour, 1800 = 30 min)

---

## Debugging

### Monitor Cooldown Cache Size
```bash
# Watch engine logs for skip count
[ENGINE] Trying CryptoHunter... (skipping N cooldown markets)
```

### Verify Skip Logic
```bash
# Should see these when market is in cooldown
[CryptoHunter] Skipping 0x... (in 10m cooldown)
[WeatherHunter] Skipping 0x... (in 10m cooldown)
[EconomyHunter] Skipping 0x... (in 10m cooldown)
```

### Check Cooldown Expiration
```bash
# Monitor for markets re-entering hunt pool after 10 minutes
# No skip message = market is out of cooldown
```

---

## Performance Notes

| Operation | Complexity | Frequency |
|-----------|-----------|-----------|
| Skip lookup | O(n) | Every market evaluation |
| Cooldown cleanup | O(m) | Every hunt cycle (~1 min) |
| Market add to cache | O(1) | Only on low EV |

Where:
- n = number of cooldown markets (typically < 10)
- m = number of expired markets (typically 0-2)

---

## Market Rotation Example

### Day 1: Bitcoin Abundance
```
Hour 0:00  Hunt → Bitcoin 68k (EV=0.12, cooldown 10min)
Hour 0:05  Hunt → Bitcoin 70k (EV=0.18, track)
Hour 0:08  EV drops → Bitcoin 70k (cooldown 10min)
Hour 0:12  Hunt → Bitcoin 72k (EV=0.16, track)
Hour 0:20  Hunt → Bitcoin 72k (still tracking)
Hour 0:27  Hunt → Bitcoin 68k back (cooldown expired after 10 min from hour 0:00)
```

### Market Diversity
```
Hunt 1: Bitcoin 68k (cooldown) ✓
Hunt 2: Bitcoin 70k (track) ✓
Hunt 3: Ethereum 3400 (cooldown) ✓
Hunt 4: Miami Temp 85F (track) ✓
...
```

---

## Troubleshooting

### Issue: Same market keeps getting hunted
**Solution:** Check EV threshold - may be too low letting poor markets execute
```python
# Try increasing threshold
RiskConfig(ev_threshold=0.20)  # Was 0.15
```

### Issue: System always hunting, never executing
**Solution:** Check cooldown isn't blocking ALL markets
```bash
# Verify skip count < total available markets
[ENGINE] Trying CryptoHunter... (skipping 1 cooldown markets)
# Should find alternative
```

### Issue: Cooldown not triggering
**Solution:** Verify EV calculation is working
```bash
# Look for EV values in logs
[ENGINE] Market ... has low EV (0.12 < 0.15)
```

---

## Next Steps

1. **Run dashboard** and monitor cooldown activation
2. **Verify alternative market discovery** (different strike prices)
3. **Adjust EV threshold** based on trading performance
4. **Add analytics** dashboard to visualize cooldown distribution
5. **Implement adaptive thresholds** per asset type


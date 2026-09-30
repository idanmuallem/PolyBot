# A1 trade-rate findings — and a correction to an earlier claim

## Correction first

In PHASE3_VALIDATION.md and the checklist I said `entry_k=0.5` is "less
bearish" than the old mechanism and worried it would SHRINK edges below
`min_ev` and produce too FEW trades. **Running the numbers shows the
opposite.** The real risk is the bot OVER-trading cheap-longshot YES bets.

## What the numbers show

Reconstructing the full entry path (raw brain -> logit_shrink(k=0.5) ->
40/60 market blend -> EV) at `min_ev=0.5`:

Synthetic frontier — the brain probability nearest the market that still
clears `min_ev` (smaller = trades more readily):

| market price | NEW div needed | OLD div needed |
|---|---|---|
| 0.05 | 0.000 | 0.272 |
| 0.10 | 0.000 | 0.398 |
| 0.15 | 0.056 | 0.480 |
| 0.20 | 0.201 | 0.534 |
| 0.25 | 0.373 | 0.568 |
| 0.30 | 0.512 | 0.586 |

On the historical far-OTM set (caveat: 94% NO, market ~0.003, NOT the
live-traded range): NEW clears `min_ev` on **20.7%** of rows vs OLD's 5.5%.

## Why — the mechanism

`logit_shrink` pulls the brain toward 0.5. On a CHEAP market (price < 0.5),
that pull lifts the 40%-weighted model component ABOVE the low market price,
manufacturing a YES edge **even when the brain agrees with the market**.
Hand-checked:

- market=0.10, brain=0.10 (zero divergence) -> post_prob=0.16 -> **ev_yes=+0.60**, tradable.
- market=0.05, brain=0.05 -> post_prob=0.105 -> **ev_yes=+1.09**, tradable.
- market=0.20, brain=0.20 -> post_prob=0.253 -> ev_yes=+0.27, NOT tradable (needs brain >= ~0.40).

So below ~0.12 the shrink alone clears `min_ev` regardless of the brain's
view; the effect fades out by ~0.20. The OLD mechanism did the reverse
(pushed low probabilities further down), so it only traded cheap markets
when the brain was strongly above the price.

## What this means for activation

- **`min_ev` is NOT too high — do not lower it.** The original A1 worry (zero
  trades) is wrong in direction. If anything the concern is the other way.
- **Expected behavior to anticipate, not mistake for a bug:** under
  `entry_k=0.5` the bot will readily take YES positions on cheap markets
  (≲0.12) with little or no brain conviction. Given cheap longshots
  historically resolve NO ~94% of the time, this could be systematically
  bad — which is exactly what the dry-run + `scripts/forward_calibration.py`
  are there to measure. Activating to observe this is fine (no capital at
  risk in dry-run) AS LONG AS you read a flood of cheap-YES trades as the
  known behavior of this setting, not a defect.
- **This may argue for a LOWER `entry_k`** (less shrink -> less manufactured
  cheap-YES edge), or for reconsidering the shrink-then-blend composition
  itself. Do NOT change it blind now — let the dry-run resolve some of these
  cheap-YES bets and let `forward_calibration.py` show whether they win. That
  is the calibration decision `entry_k` was always meant to wait for.

## Tooling

`scripts/trade_rate_check.py`:
- default: the synthetic frontier above (reflects live config).
- `--csv PATH`: trade-rate on a CSV of `brain_prob` + `market_prob`. Point it
  at a CURRENT live-market export from the instance for the real go/no-go
  (the historical CSV is far-OTM and unrepresentative).

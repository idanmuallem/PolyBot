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
- **This may argue for reconsidering the shrink-then-blend composition
  itself** (see the follow-up analysis below). Do NOT change it blind now.

## Follow-up: entry_k sweep on REAL live-zone data with outcomes (n=58)

Polymarket is unreachable from the build sandbox, so no true live export. The
best substitute: the backtest's own (brain, market, OUTCOME) triples filtered
to the live-traded zone market_prob in [0.05, 0.35] — 58 real rows, 3 YES /
55 NO. For each entry_k, simulate which trades the mechanism takes and whether
they actually won (min_ev=0.5):

| entry_k | trades | all YES? | win% | realized PnL (per $1/trade) |
|---|---|---|---|---|
| 0.2 | 33 | yes | 0% | -33.0 |
| 0.5 | 13 | yes | 0% | -13.0 |
| 0.8 | 4 | yes | 0% | -4.0 |
| 1.0 | 2 | yes | 0% | -2.0 |
| OLD (wang -0.75) | 0 | - | - | 0.0 |

CORRECTION to the bullet above: LOWER entry_k = MORE shrink toward 0.5 = MORE
manufactured edge (logit_shrink(0.10, 0.2)=0.39 vs (0.10, 0.8)=0.15). So a
lower entry_k is WORSE here, not better; higher k bleeds less. The earlier
"lower entry_k -> less shrink" phrasing was wrong in both directions.

Every trade the current mechanism takes in this zone is a YES bet, and on
this sample all lose — because it manufactures YES edge on cheap markets that
resolve NO ~95% of the time (favorite-longshot bias: cheap longshots are
overpriced, so the right side is NO, not YES). The OLD bug took zero trades
here (its down-push never cleared min_ev) — "safer" only by not trading.

### Proposed structural fix: shrink toward the MARKET, not toward 0.5

post = sigmoid( logit(market) + k*(logit(brain) - logit(market)) ), k in [0,1]

- k=0 -> post=market (defer fully to market, model ignored);
- k=1 -> post=brain (fully trust model);
- brain == market -> post == market -> ZERO edge, no manufactured trade.

Still provably monotonic (toward market). On the same 58 rows it cuts the
losing trades from 13 -> 2 at k=0.5, because it only acts on genuine
divergence instead of manufacturing edge from the cheap price alone. It also
subsumes model_weight (the interpolation IS the blend), collapsing two knobs
into one with correct semantics ("how much to trust the brain's disagreement
with the market").

### The harder truth underneath

Even the proposed fix still loses on this subset (2 trades, both YES, both
lose). Reason: the brain's UPWARD divergence from market in this zone is
itself unreliable — the market is better calibrated than the brain here,
exactly Phase 1's "no usable signal." No entry mechanism that trusts
brain-above-market divergence can profit while that holds. So the honest
posture is LOW trust in brain divergence until forward data earns it — which
under the toward-market design means a LOW k (safe: defers to market), the
opposite of what low k means today.

Caveat: n=58, 3 YES, one historical window. The STRUCTURAL findings
(manufactured cheap-YES; toward-market eliminates it; monotonicity) are
logical and robust; the P&L magnitudes are small-sample and only directional.

## Tooling

`scripts/trade_rate_check.py`:
- default: the synthetic frontier above (reflects live config).
- `--csv PATH`: trade-rate on a CSV of `brain_prob` + `market_prob`. Point it
  at a CURRENT live-market export from the instance for the real go/no-go
  (the historical CSV is far-OTM and unrepresentative).

# Phase 3 Validation — Entry-side logit-shrink vs old Wang Transform

**Read this before drawing the obvious wrong conclusion.** On this dataset
the new mechanism scores *worse* than the old one. That is expected and does
not mean the redesign was wrong — but it is a real result about the
`entry_k=0.5` *default*, and it is worth sitting with. Details below.

## The numbers (n=164 snapshots, 56 markets, 10 YES / 154 NO)

Full entry-side `post_prob` reconstructed end to end
(raw brain prob → transform/shrink → 40/60 market blend), scored against
resolved outcomes:

| Mechanism | Brier ↓ | Signed bias (pred − outcome) |
|---|---|---|
| OLD (wang_transform, λ=−0.75) | **0.0430** | +0.0394 |
| NEW (logit_shrink, k=0.5) | 0.0535 | +0.0811 |
| raw brain (no transform) | 0.0474 | +0.0519 |
| market alone | 0.0566 | +0.0719 |

The NEW mechanism has the **worst Brier of the four** and the **largest
positive bias** (most over-prediction of YES).

## Why this is a dataset artifact, not a regression

The sample is **94% NO outcomes** — these are far-OTM "moon" strikes that
almost never resolve YES. On a sample where the truth is almost always 0,
*any* mechanism that pushes probabilities toward 0 will score better on
Brier, regardless of whether it is better calibrated in general.

- The OLD transform's bug (λ=−0.75 pushing this system's 0.25–0.45 range
  *away* from 0.5, i.e. toward 0) happens to **align with a mostly-0
  sample** — so the bug flatters the Brier score here. It is being rewarded
  for being wrong in the same direction the sample is skewed.
- The NEW shrink pulls toward 0.5, i.e. *raises* these low probabilities —
  moving them **away from the mostly-0 outcomes** — so it scores worse.

This is precisely the confound Phase 1 already flagged: this dataset cannot
distinguish a genuinely better distrust mechanism from "just predict NO
harder." A lower Brier on a 94%-NO sample is not evidence of skill.

The redesign was never justified by Brier on this sample. Its justification
was **provable monotonicity** (no crossover point where the transform
reverses direction) and an **honest, documented default** in the absence of
a usable calibration signal. Both hold regardless of these numbers.

## Directional-bias check (the thing Phase 3 actually exists to test)

The old bug was directional (only "toward 0.5" on one side of a crossover).
Does the new mechanism reintroduce a directional skew?

Split by whether the brain is more bullish or bearish than the market:

| Subset | n | NEW signed bias | OLD signed bias |
|---|---|---|---|
| brain > market (bullish tilt) | 31 | +0.173 | +0.078 |
| brain < market (bearish tilt) | 133 | +0.060 | +0.030 |

The NEW mechanism is **uniformly less bearish** than the old one — it
over-predicts YES more on both sides. This is the direct, expected
consequence of replacing a toward-0 bug with a toward-0.5 shrink on a
NO-dominated sample. It is *not* a one-sided skew of the kind the old
crossover bug produced (both subsets move the same direction), so it does
not reintroduce that specific failure mode.

## Monotonicity confirmed on real data

| Mechanism | Moved toward 0.5 | 
|---|---|
| OLD | 64/164 (39%) |
| NEW | **164/164 (100%)** |

Direct empirical confirmation of the crossover bug (the old transform moved
the *majority* of real snapshots the wrong way) and of the fix (the new one
moves every real snapshot toward 0.5, as designed).

## Verdict

- **Code correctness / monotonicity: validated.** The new mechanism does
  exactly what it claims on real data.
- **Calibration on this sample: cannot be validated here** — the 94%-NO
  confound makes Brier uninformative, exactly as Phase 1 predicted. This is
  not a passing grade and should not be reported as one.
- **Genuine thing to watch:** `entry_k=0.5` makes the model **measurably
  less bearish on far-OTM markets** than the old accidental behavior. In the
  live strategy's terms, that means **less NO-side edge / different trade
  selection** than before. Whether that is good (the old NO-edge was partly
  the bug talking) or bad (those NO bets mostly won) **cannot be settled on
  a 94%-NO sample** — it needs forward dry-run trades that resolve across a
  less lopsided outcome mix. This is exactly the "revisit entry_k once
  forward data exists" note already in PROGRESS.md, now with a concrete
  reason: the default is less conservative than what it replaced.

**Recommendation:** the mechanism is correct and safe to keep in dry-run.
Do **not** treat this as calibration validation, and do **not** move toward
live capital on the strength of it. Let dry-run accumulate resolved trades
with a more balanced outcome mix, then re-run this exact split — if the
bullish-tilt bias (+0.17) persists on non-far-OTM markets, `entry_k` is too
high and should come down.

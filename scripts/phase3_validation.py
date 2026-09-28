"""Phase 3 validation: does the new entry-side logit-shrink mechanism
reintroduce a directional bias, and how does its calibration compare to the
old probit-shift Wang Transform?

Read-only analysis. Reconstructs the FULL entry-side pricing path
(raw brain probability -> shrink/transform -> market blend) under both the
OLD mechanism (wang_transform, lambda=-0.75) and the NEW mechanism
(logit_shrink, entry_k=0.5) on the same historical dataset used for Phase 1,
then scores each against the resolved outcomes.

Only the ENTRY side changed in this work, so this validates the entry
mechanism. The exit side (PricingEngine hierarchical model) was untouched
and is not re-scored here.
"""
import numpy as np
import pandas as pd

from brains.pricing_engine import logit_shrink, wang_transform
from core.trading_config import DEFAULT_ENTRY_K, DEFAULT_MODEL_WEIGHT

OLD_WANG_LAMBDA = -0.75  # the value this work replaced
NEW_ENTRY_K = DEFAULT_ENTRY_K  # 0.5
MODEL_WEIGHT = DEFAULT_MODEL_WEIGHT  # 0.40, unchanged by this work

df = pd.read_csv("data/backtest_resolved_markets.csv")

brain = df["brain_prob"].to_numpy()
market = df["market_prob"].to_numpy()
outcome = df["outcome"].to_numpy().astype(float)


def blend(model_component: np.ndarray) -> np.ndarray:
    """The market-blend step, identical for both mechanisms."""
    return MODEL_WEIGHT * model_component + (1.0 - MODEL_WEIGHT) * market


# Reconstruct the full entry-side post_prob under each mechanism.
old_model = np.array([wang_transform(float(p), OLD_WANG_LAMBDA) for p in brain])
new_model = np.array([logit_shrink(float(p), NEW_ENTRY_K) for p in brain])

post_old = blend(old_model)
post_new = blend(new_model)


def brier(pred: np.ndarray) -> float:
    return float(np.mean((pred - outcome) ** 2))


def signed_bias(pred: np.ndarray) -> float:
    # mean(pred - outcome): >0 means the estimate systematically sits ABOVE
    # the realized outcome (over-predicting YES), <0 means below.
    return float(np.mean(pred - outcome))


n = len(df)
n_yes = int(outcome.sum())

print(f"=== Phase 3 validation (n={n} snapshots, {n_yes} YES / {n - n_yes} NO) ===\n")
print(f"entry_k (new) = {NEW_ENTRY_K}, old wang_lambda = {OLD_WANG_LAMBDA}, model_weight = {MODEL_WEIGHT}\n")

print("Full entry-side post_prob (raw -> transform/shrink -> market blend):")
print(f"  OLD mechanism  Brier = {brier(post_old):.5f}   signed bias = {signed_bias(post_old):+.5f}")
print(f"  NEW mechanism  Brier = {brier(post_new):.5f}   signed bias = {signed_bias(post_new):+.5f}")
print(f"  (raw brain     Brier = {brier(brain):.5f}   signed bias = {signed_bias(brain):+.5f})")
print(f"  (market alone  Brier = {brier(market):.5f}   signed bias = {signed_bias(market):+.5f})\n")

# Directional bias broken out by which way the model diverges from market,
# since a bias that only appears on one side is the specific failure the
# old mechanism had.
print("Signed bias split by model-vs-market direction (NEW mechanism):")
bull = brain > market  # brain more bullish than market
bear = brain < market
for label, mask in (("brain > market (bullish tilt)", bull),
                     ("brain < market (bearish tilt)", bear)):
    if mask.sum() == 0:
        print(f"  {label}: n=0")
        continue
    print(f"  {label}: n={int(mask.sum())}  "
          f"signed bias = {float(np.mean(post_new[mask] - outcome[mask])):+.5f}  "
          f"Brier = {float(np.mean((post_new[mask] - outcome[mask])**2)):.5f}")

print("\nSame split, OLD mechanism (for contrast):")
for label, mask in (("brain > market (bullish tilt)", bull),
                     ("brain < market (bearish tilt)", bear)):
    if mask.sum() == 0:
        print(f"  {label}: n=0")
        continue
    print(f"  {label}: n={int(mask.sum())}  "
          f"signed bias = {float(np.mean(post_old[mask] - outcome[mask])):+.5f}  "
          f"Brier = {float(np.mean((post_old[mask] - outcome[mask])**2)):.5f}")

# Per-mechanism: how far does the model component itself sit from 0.5, on
# average, over the real input range? This is the direct "does the new one
# still push away from 0.5" check that Phase 1 flagged.
print("\nModel component vs raw brain prob (does the transform move toward 0.5?):")
moved_toward_old = np.abs(old_model - 0.5) < np.abs(brain - 0.5)
moved_toward_new = np.abs(new_model - 0.5) < np.abs(brain - 0.5)
print(f"  OLD: moved toward 0.5 in {int(moved_toward_old.sum())}/{n} snapshots "
      f"({100*moved_toward_old.mean():.1f}%)")
print(f"  NEW: moved toward 0.5 in {int(moved_toward_new.sum())}/{n} snapshots "
      f"({100*moved_toward_new.mean():.1f}%)")

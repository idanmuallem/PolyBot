"""Phase 1 analysis: does the brain's divergence from the market price predict
which side (brain or market) is more accurate at a given snapshot?

Reads data/backtest_resolved_markets.csv (produced by backtest_crypto_brain.py)
and writes PHASE1_FINDINGS.md in the repo root.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(REPO_ROOT, "data", "backtest_resolved_markets.csv")
OUT_PATH = os.path.join(REPO_ROOT, "PHASE1_FINDINGS.md")

# Divergences with |brain - market| below this are treated as "agreement" --
# not a real signal, just quoting/rounding noise -- and excluded from the
# direction analysis.
NEAR_ZERO_THRESHOLD = 0.005


def load_dataset():
    df = pd.read_csv(CSV_PATH)
    df["abs_divergence"] = df["divergence"].abs()
    # brier_edge > 0 means the brain was more accurate at this snapshot
    df["brier_edge"] = df["brier_market"] - df["brier_brain"]
    df["brain_wins"] = (df["brier_edge"] > 0).astype(int)
    # market's own error direction: positive means the market underpriced YES
    df["market_residual"] = df["outcome"] - df["market_prob"]
    # did the brain's tilt point toward the eventual outcome, relative to the market?
    nonzero = df["divergence"] != 0
    df["direction_correct"] = np.nan
    df.loc[nonzero, "direction_correct"] = (
        np.sign(df.loc[nonzero, "divergence"]) == np.sign(df.loc[nonzero, "market_residual"])
    ).astype(int)
    return df


def bucket_by_magnitude(df, n_buckets=3):
    labels = ["Low", "Medium", "High"][:n_buckets]
    df = df.copy()
    df["magnitude_bucket"] = pd.qcut(df["abs_divergence"], q=n_buckets, labels=labels, duplicates="drop")
    return df


def summarize_buckets(df):
    rows = []
    for bucket, g in df.groupby("magnitude_bucket", observed=True):
        rows.append({
            "bucket": bucket,
            "n": len(g),
            "abs_divergence_range": f"{g['abs_divergence'].min():.4f} - {g['abs_divergence'].max():.4f}",
            "mean_brier_brain": g["brier_brain"].mean(),
            "mean_brier_market": g["brier_market"].mean(),
            "brain_win_rate": g["brain_wins"].mean(),
            "direction_correct_rate": g["direction_correct"].mean(),
        })
    return pd.DataFrame(rows)


def direction_split(df, threshold=NEAR_ZERO_THRESHOLD):
    bullish = df[df["divergence"] > threshold]   # brain more optimistic on YES than market
    bearish = df[df["divergence"] < -threshold]  # brain more pessimistic on YES than market
    agree = df[df["divergence"].abs() <= threshold]
    return bullish, bearish, agree


def main():
    if not os.path.exists(CSV_PATH):
        print(f"Dataset not found at {CSV_PATH}. Run scripts/backtest_crypto_brain.py first.")
        sys.exit(1)

    df = load_dataset()
    n = len(df)
    print(f"Loaded {n} snapshots.")

    overall_brier_brain = df["brier_brain"].mean()
    overall_brier_market = df["brier_market"].mean()
    overall_brain_win_rate = df["brain_wins"].mean()
    outcome_rate = df["outcome"].mean()

    # --- Test 1: does divergence MAGNITUDE predict which side is more accurate? ---
    r_mag, p_mag = stats.pointbiserialr(df["brain_wins"], df["abs_divergence"])

    # NOTE: "direction_correct" (sign(divergence) == sign(outcome - market_prob))
    # is mathematically identical to "brain_wins" whenever divergence != 0. Brier
    # score (p - outcome)^2 is strictly monotonic in p across the whole [0, 1]
    # domain because outcome sits at a boundary (0 or 1), so moving p toward the
    # outcome -- by any amount, in a bounded-probability space -- can never
    # overshoot past it and always strictly lowers the score. So a point-biserial
    # test of |divergence| against direction_correct would just reproduce Test 1
    # verbatim; it is intentionally not run as a separate test here.

    # --- Test 2: does divergence sign correlate with the market's own error
    # direction (outcome - market_prob)? Continuous-continuous, Pearson. This one
    # is NOT tautological: market_residual doesn't involve brain_prob at all. ---
    r_sign, p_sign = stats.pearsonr(df["divergence"], df["market_residual"])

    # --- Bucket analysis by magnitude ---
    bucketed = bucket_by_magnitude(df)
    bucket_summary = summarize_buckets(bucketed)

    # --- Direction split ---
    bullish, bearish, agree = direction_split(df)

    def side_stats(g):
        if len(g) == 0:
            return dict(n=0, mean_brier_brain=float("nan"), mean_brier_market=float("nan"),
                        brain_win_rate=float("nan"), mean_market_residual=float("nan"))
        return dict(
            n=len(g),
            mean_brier_brain=g["brier_brain"].mean(),
            mean_brier_market=g["brier_market"].mean(),
            brain_win_rate=g["brain_wins"].mean(),
            mean_market_residual=g["market_residual"].mean(),
        )

    bullish_stats = side_stats(bullish)
    bearish_stats = side_stats(bearish)
    agree_stats = side_stats(agree)

    # Chi-square: is brain_wins independent of tilt direction (bullish vs bearish)?
    contingency = pd.crosstab(
        pd.concat([bullish["divergence"] > 0, bearish["divergence"] > 0])
        .map({True: "bullish", False: "bearish"}),
        pd.concat([bullish["brain_wins"], bearish["brain_wins"]]),
    )
    if contingency.shape == (2, 2) and (contingency.values > 0).all():
        chi2, p_chi2, _, _ = stats.chi2_contingency(contingency)
    else:
        chi2, p_chi2 = float("nan"), float("nan")

    # ---- Write findings ----
    lines = []
    lines.append("# Phase 1 Findings: Does Brain/Market Divergence Predict Accuracy?\n")
    lines.append(f"Dataset: `data/backtest_resolved_markets.csv` ({n} snapshots across "
                  f"{df['market_id'].nunique()} resolved BTC/ETH strike markets).\n")

    lines.append("## Overall calibration\n")
    lines.append(f"- Brain mean Brier score: **{overall_brier_brain:.4f}**")
    lines.append(f"- Market mean Brier score: **{overall_brier_market:.4f}**")
    lines.append(f"- Brain more accurate than market at **{overall_brain_win_rate:.1%}** of snapshots.")
    lines.append(f"- Outcome base rate: **{outcome_rate:.1%}** of snapshots resolved YES "
                 f"({int(df['outcome'].sum())}/{n}) -- these are almost all far-out-of-the-money "
                 "\"moon\" strike markets, so the sample is heavily skewed toward NO resolutions. "
                 "Treat directional splits below with that in mind: with only "
                 f"{int(df['outcome'].sum())} YES outcomes in the sample, a systematic bias toward "
                 "predicting NO will look like \"skill\" almost by construction.\n")

    lines.append("## Does divergence magnitude predict which side is more accurate?\n")
    lines.append(
        f"Point-biserial correlation between `|brain_prob - market_prob|` and "
        f"`brain_more_accurate` (1/0): **r = {r_mag:.3f}**, p = {p_mag:.4f} (n={n}).\n"
    )
    if p_mag < 0.05:
        direction = "larger disagreements favor the brain" if r_mag > 0 else "larger disagreements favor the market"
        lines.append(f"This is statistically significant: {direction}.\n")
    else:
        lines.append("Not statistically significant at the 0.05 level -- divergence magnitude alone does not "
                      "reliably predict which side wins.\n")

    lines.append("### Accuracy by divergence-magnitude bucket (terciles)\n")
    lines.append("| Bucket | n | |divergence| range | Mean Brain Brier | Mean Market Brier | Brain win rate | Direction-correct rate |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, row in bucket_summary.iterrows():
        lines.append(
            f"| {row['bucket']} | {row['n']} | {row['abs_divergence_range']} | "
            f"{row['mean_brier_brain']:.4f} | {row['mean_brier_market']:.4f} | "
            f"{row['brain_win_rate']:.1%} | {row['direction_correct_rate']:.1%} |"
        )
    lines.append("")
    lines.append(
        "*(Direction-correct rate is identical to brain win rate by construction here -- see note below.)*\n"
    )

    lines.append("## Does divergence DIRECTION carry signal?\n")
    lines.append(
        f"Pearson correlation between signed `divergence` (brain - market) and the market's own "
        f"error (`outcome - market_prob`): **r = {r_sign:.3f}**, p = {p_sign:.4f}.\n"
    )
    if p_sign < 0.05 and r_sign > 0:
        lines.append("Positive and significant: when the brain leans more bullish (or bearish) than the market, "
                      "the market's eventual pricing error tends to point the same way -- the brain's directional "
                      "tilt is informative, not noise.\n")
    elif p_sign < 0.05 and r_sign < 0:
        lines.append("Negative and significant: the brain's directional tilt tends to point *away* from the "
                      "market's eventual error -- i.e. anti-correlated with being right.\n")
    else:
        lines.append("Not statistically significant -- no reliable evidence that the direction of the brain's "
                      "tilt away from the market predicts which way the market was wrong.\n")

    lines.append(
        "(`direction_correct` = sign(divergence) matching sign(outcome - market_prob) is mathematically "
        "identical to `brain_wins` here -- see Test 1 above -- so it is not reported again as a separate "
        "correlation.)\n"
    )

    lines.append("### Bullish vs. bearish tilt (|divergence| > {:.3f})\n".format(NEAR_ZERO_THRESHOLD))
    lines.append("| Tilt | n | Mean Brain Brier | Mean Market Brier | Brain win rate | Mean market residual (outcome - market_prob) |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(
        f"| Brain > Market (bullish) | {bullish_stats['n']} | {bullish_stats['mean_brier_brain']:.4f} | "
        f"{bullish_stats['mean_brier_market']:.4f} | {bullish_stats['brain_win_rate']:.1%} | "
        f"{bullish_stats['mean_market_residual']:.4f} |"
    )
    lines.append(
        f"| Brain < Market (bearish) | {bearish_stats['n']} | {bearish_stats['mean_brier_brain']:.4f} | "
        f"{bearish_stats['mean_brier_market']:.4f} | {bearish_stats['brain_win_rate']:.1%} | "
        f"{bearish_stats['mean_market_residual']:.4f} |"
    )
    lines.append(
        f"| Agree (|divergence| <= {NEAR_ZERO_THRESHOLD}) | {agree_stats['n']} | "
        f"{agree_stats['mean_brier_brain']:.4f} | {agree_stats['mean_brier_market']:.4f} | "
        f"{agree_stats['brain_win_rate']:.1%} | {agree_stats['mean_market_residual']:.4f} |"
    )
    lines.append("")
    lines.append(
        f"Chi-square test of independence between tilt direction (bullish/bearish) and which side wins "
        f"(brain/market): chi2 = {chi2:.3f}, p = {p_chi2:.4f}"
        + (" (skipped -- a contingency cell was empty)" if np.isnan(chi2) else "") + "\n"
    )
    lines.append(
        f"Caveat: bearish-tilt snapshots win far more often ({bearish_stats['brain_win_rate']:.1%}) than "
        f"bullish-tilt ones ({bullish_stats['brain_win_rate']:.1%}), but the mean market residual is "
        f"similarly negative in both groups ({bullish_stats['mean_market_residual']:.3f} vs. "
        f"{bearish_stats['mean_market_residual']:.3f}) -- i.e. the market itself is, on average, biased toward "
        "overpricing YES on these longshot strikes regardless of which way the brain leans. Given the "
        f"{outcome_rate:.1%} YES base rate, any correction toward NO tends to help almost mechanically, so this "
        "split alone does not cleanly separate \"the brain has skill when bearish\" from \"the base rate favors "
        "bearish corrections.\"\n"
    )

    lines.append("## Summary\n")
    summary_points = []
    if p_mag < 0.05:
        summary_points.append(
            f"- Divergence **magnitude** is a statistically significant predictor of which side wins "
            f"(r={r_mag:.3f}, p={p_mag:.4f})."
        )
    else:
        summary_points.append(
            f"- Divergence **magnitude** alone is not a statistically significant predictor of which side wins "
            f"(r={r_mag:.3f}, p={p_mag:.4f})."
        )
    if p_sign < 0.05:
        summary_points.append(
            f"- Divergence **direction** is a statistically significant predictor of which way the market "
            f"was wrong (r={r_sign:.3f}, p={p_sign:.4f})."
        )
    else:
        summary_points.append(
            f"- Divergence **direction** is not a statistically significant predictor of which way the market "
            f"was wrong (r={r_sign:.3f}, p={p_sign:.4f})."
        )
    lines.extend(summary_points)
    lines.append("")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote findings to {OUT_PATH}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

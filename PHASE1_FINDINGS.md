# Phase 1 Findings: Does Brain/Market Divergence Predict Accuracy?

Dataset: `data/backtest_resolved_markets.csv` (164 snapshots across 56 resolved BTC/ETH strike markets).

## Overall calibration

- Brain mean Brier score: **0.0474**
- Market mean Brier score: **0.0565**
- Brain more accurate than market at **78.7%** of snapshots.
- Outcome base rate: **6.1%** of snapshots resolved YES (10/164) -- these are almost all far-out-of-the-money "moon" strike markets, so the sample is heavily skewed toward NO resolutions. Treat directional splits below with that in mind: with only 10 YES outcomes in the sample, a systematic bias toward predicting NO will look like "skill" almost by construction.

## Does divergence magnitude predict which side is more accurate?

Point-biserial correlation between `|brain_prob - market_prob|` and `brain_more_accurate` (1/0): **r = -0.103**, p = 0.1889 (n=164).

Not statistically significant at the 0.05 level -- divergence magnitude alone does not reliably predict which side wins.

### Accuracy by divergence-magnitude bucket (terciles)

| Bucket | n | |divergence| range | Mean Brain Brier | Mean Market Brier | Brain win rate | Direction-correct rate |
|---|---|---|---|---|---|---|
| Low | 55 | 0.0000 - 0.0044 | 0.0017 | 0.0018 | 98.2% | 98.2% |
| Medium | 54 | 0.0049 - 0.0513 | 0.0363 | 0.0352 | 68.5% | 68.5% |
| High | 55 | 0.0517 - 0.7000 | 0.1040 | 0.1322 | 69.1% | 69.1% |

*(Direction-correct rate is identical to brain win rate by construction here -- see note below.)*

## Does divergence DIRECTION carry signal?

Pearson correlation between signed `divergence` (brain - market) and the market's own error (`outcome - market_prob`): **r = 0.375**, p = 0.0000.

Positive and significant: when the brain leans more bullish (or bearish) than the market, the market's eventual pricing error tends to point the same way -- the brain's directional tilt is informative, not noise.

(`direction_correct` = sign(divergence) matching sign(outcome - market_prob) is mathematically identical to `brain_wins` here -- see Test 1 above -- so it is not reported again as a separate correlation.)

### Bullish vs. bearish tilt (|divergence| > 0.005)

| Tilt | n | Mean Brain Brier | Mean Market Brier | Brain win rate | Mean market residual (outcome - market_prob) |
|---|---|---|---|---|---|
| Brain > Market (bullish) | 30 | 0.1284 | 0.1202 | 10.0% | -0.1102 |
| Brain < Market (bearish) | 76 | 0.0504 | 0.0733 | 90.8% | -0.1041 |
| Agree (|divergence| <= 0.005) | 58 | 0.0017 | 0.0017 | 98.3% | -0.0098 |

Chi-square test of independence between tilt direction (bullish/bearish) and which side wins (brain/market): chi2 = 60.783, p = 0.0000

Caveat: bearish-tilt snapshots win far more often (90.8%) than bullish-tilt ones (10.0%), but the mean market residual is similarly negative in both groups (-0.110 vs. -0.104) -- i.e. the market itself is, on average, biased toward overpricing YES on these longshot strikes regardless of which way the brain leans. Given the 6.1% YES base rate, any correction toward NO tends to help almost mechanically, so this split alone does not cleanly separate "the brain has skill when bearish" from "the base rate favors bearish corrections."

## Summary

- Divergence **magnitude** alone is not a statistically significant predictor of which side wins (r=-0.103, p=0.1889).
- Divergence **direction** is a statistically significant predictor of which way the market was wrong (r=0.375, p=0.0000).

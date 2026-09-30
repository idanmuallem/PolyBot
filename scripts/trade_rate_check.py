"""Trade-rate check: will the bot actually trade under the current entry
pricing, or do the shrunk edges fall below min_ev and produce zero trades
(a wasted dry-run)?

entry_k=0.5 makes post_prob less extreme (closer to market) than the old
wang_lambda=-0.75 did, which SHRINKS edges. If the shrunk edges sit below
min_ev (default 0.50), the bot trades nothing. This quantifies that.

Two modes:

  --csv PATH   Reconstruct the full entry path (raw -> shrink -> market
               blend -> EV) on a CSV with brain_prob + market_prob columns,
               and report the % of rows that would clear min_ev, under the
               new mechanism vs the old one. Use it on a live export of
               evaluated markets from the instance for the real answer.

  (default)    Synthetic sensitivity grid over the LIVE-TRADED market-price
               range (~0.05-0.30). For each market price it finds the
               breakeven brain probability needed to clear min_ev, under new
               vs old pricing - showing how much more divergence the new
               mechanism demands before it will trade.

Reads entry_k, model_weight, min_ev from the live config so it reflects what
the instance would actually do. Read-only.

NOTE on the historical backtest CSV: data/backtest_resolved_markets.csv is
94% far-OTM (market prices ~0.003), NOT the 0.1-0.2 range the live strategy
trades. Running --csv on it answers a different question than "will the bot
trade on current live markets" - for that, export current evaluated markets
from the instance and point --csv at those.
"""
import argparse
import math
import sys

import numpy as np

from brains.pricing_engine import logit_shrink, wang_transform
from core.trading_config import TradingConfig, DEFAULT_ENTRY_K

OLD_WANG_LAMBDA = -0.75


def post_prob(model_component, market_price, model_weight):
    return model_weight * model_component + (1.0 - model_weight) * market_price


def best_ev(pp, price_yes):
    price_yes = min(max(price_yes, 1e-9), 1 - 1e-9)
    price_no = 1.0 - price_yes
    ev_yes = pp / price_yes - 1.0
    ev_no = (1.0 - pp) / price_no - 1.0
    return max(ev_yes, ev_no)


def run_csv(path, cfg):
    import pandas as pd
    df = pd.read_csv(path)
    brain = df["brain_prob"].to_numpy()
    market = df["market_prob"].to_numpy()

    def tradable_rate(transform_fn):
        n_trad = 0
        for b, m in zip(brain, market):
            mc = transform_fn(float(b))
            pp = post_prob(mc, float(m), cfg.model_weight)
            if best_ev(pp, float(m)) >= cfg.min_ev:
                n_trad += 1
        return n_trad, len(brain)

    new_t, n = tradable_rate(lambda b: logit_shrink(b, cfg.entry_k))
    old_t, _ = tradable_rate(lambda b: wang_transform(b, OLD_WANG_LAMBDA))
    print(f"=== Trade-rate on {path} (n={n}, min_ev={cfg.min_ev}, "
          f"entry_k={cfg.entry_k}, model_weight={cfg.model_weight}) ===\n")
    print(f"NEW (logit_shrink k={cfg.entry_k}): {new_t}/{n} clear min_ev "
          f"({100*new_t/n:.1f}%)")
    print(f"OLD (wang lambda={OLD_WANG_LAMBDA}): {old_t}/{n} clear min_ev "
          f"({100*old_t/n:.1f}%)")
    if "backtest_resolved_markets" in path:
        print("\nWARNING: this dataset is 94% far-OTM (market ~0.003), NOT the "
              "0.1-0.2 range the live strategy trades. Treat as mechanism "
              "illustration only, not a live trade-rate estimate.")


def breakeven_brain_prob(market_price, transform_fn, cfg):
    """Smallest brain prob (searching both directions from market) whose
    post-blend EV clears min_ev, or None if none in [1e-6, 1-1e-6]."""
    grid = np.linspace(1e-6, 1 - 1e-6, 4000)
    best = None
    for b in grid:
        mc = transform_fn(float(b))
        pp = post_prob(mc, market_price, cfg.model_weight)
        if best_ev(pp, market_price) >= cfg.min_ev:
            # record the brain prob closest to the market (smallest divergence)
            div = abs(b - market_price)
            if best is None or div < best[1]:
                best = (b, div)
    return best


def run_synthetic(cfg):
    print(f"=== Synthetic tradability frontier (min_ev={cfg.min_ev}, "
          f"entry_k={cfg.entry_k}, model_weight={cfg.model_weight}) ===\n")
    print("For each live-range market price: the brain probability nearest to "
          "the market that still clears min_ev, and the divergence it needs.\n")
    print(f"{'market':>7} | {'NEW brain@breakeven':>22} {'div':>7} | "
          f"{'OLD brain@breakeven':>22} {'div':>7}")
    print("-" * 74)
    for mkt in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        new_be = breakeven_brain_prob(mkt, lambda b: logit_shrink(b, cfg.entry_k), cfg)
        old_be = breakeven_brain_prob(mkt, lambda b: wang_transform(b, OLD_WANG_LAMBDA), cfg)
        def fmt(be):
            if be is None:
                return f"{'(never)':>22} {'-':>7}"
            return f"{be[0]:>22.4f} {be[1]:>7.4f}"
        print(f"{mkt:>7.2f} | {fmt(new_be)} | {fmt(old_be)}")
    print("\nReads: a larger required divergence (or '(never)') means the "
          "mechanism trades LESS often at that price. If NEW shows much larger "
          "divergences than OLD across the live range, expect a materially "
          "lower trade rate - decide whether to lower min_ev for this run "
          "BEFORE activating, or accept fewer trades.")
    print("\nThis is a sensitivity illustration, not a live estimate. The "
          "real go/no-go is --csv on a current live-market export from the "
          "instance.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="CSV with brain_prob + market_prob columns")
    args = ap.parse_args()

    # Use env config if available, else defaults - so it reflects the instance.
    try:
        cfg = TradingConfig.from_env()
    except Exception:
        cfg = TradingConfig(trading_mode="dry_run")

    if args.csv:
        run_csv(args.csv, cfg)
    else:
        run_synthetic(cfg)


if __name__ == "__main__":
    main()

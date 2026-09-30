"""Trade-rate / trade-quality check for the market-anchored entry mechanism.

Answers two questions before activation:
  1. Will the bot trade at all under the current entry_k, and in which
     direction (YES/NO)?
  2. If the CSV carries resolved outcomes, do those trades actually WIN?

Mechanism modeled: brains/pricing_engine.market_anchored_shrink, i.e.
post = sigmoid(logit(market) + entry_k*(logit(brain) - logit(market))), then
EV on the better side vs min_ev. Reads entry_k and min_ev from the live
config so it reflects what the instance would do.

Usage:
    python scripts/trade_rate_check.py --csv PATH   # needs brain_prob, market_prob; outcome optional
    python scripts/trade_rate_check.py              # synthetic frontier over the live-traded price range

NOTE: data/backtest_resolved_markets.csv is 94% far-OTM; filter to the
live-traded zone (market_prob in ~[0.05,0.35]) or, better, point --csv at a
current live-market export from the instance.
"""
import argparse

import numpy as np

from brains.pricing_engine import market_anchored_shrink
from core.trading_config import TradingConfig


def best_side_ev(post, price_yes):
    price_yes = min(max(price_yes, 1e-9), 1 - 1e-9)
    price_no = 1.0 - price_yes
    ev_yes = post / price_yes - 1.0
    ev_no = (1.0 - post) / price_no - 1.0
    return ("YES", ev_yes) if ev_yes >= ev_no else ("NO", ev_no)


def run_csv(path, cfg):
    import pandas as pd
    df = pd.read_csv(path)
    brain = df["brain_prob"].to_numpy()
    market = df["market_prob"].to_numpy()
    has_outcome = "outcome" in df.columns
    outcome = df["outcome"].to_numpy().astype(float) if has_outcome else None

    n_trade = n_yes = n_win = 0
    pnl = 0.0
    for i, (b, m) in enumerate(zip(brain, market)):
        post = market_anchored_shrink(float(b), float(m), cfg.entry_k)
        side, ev = best_side_ev(post, float(m))
        if ev < cfg.min_ev:
            continue
        n_trade += 1
        if side == "YES":
            n_yes += 1
        if has_outcome:
            p = float(m) if side == "YES" else 1.0 - float(m)
            p = min(max(p, 1e-9), 1 - 1e-9)
            win = (outcome[i] == 1.0) if side == "YES" else (outcome[i] == 0.0)
            n_win += int(win)
            pnl += (1.0 / p - 1.0) if win else -1.0

    print(f"=== {path} (n={len(df)}, entry_k={cfg.entry_k}, min_ev={cfg.min_ev}) ===\n")
    print(f"Trades clearing min_ev: {n_trade}/{len(df)} "
          f"({100*n_trade/len(df):.1f}%)  [YES={n_yes}, NO={n_trade-n_yes}]")
    if has_outcome and n_trade:
        print(f"Win rate: {n_win}/{n_trade} ({100*n_win/n_trade:.1f}%)   "
              f"realized PnL (per $1/trade): {pnl:+.2f}   avg {pnl/n_trade:+.3f}")
    elif has_outcome:
        print("No trades cleared min_ev, so no realized PnL.")
    if "backtest_resolved_markets" in path:
        print("\nWARNING: this dataset is far-OTM-heavy and not representative "
              "of the live-traded range; filter to market_prob in [0.05,0.35] "
              "or use a live export.")


def breakeven_divergence(market_price, cfg):
    """Smallest |brain - market| whose market-anchored EV clears min_ev."""
    grid = np.linspace(1e-6, 1 - 1e-6, 4000)
    best = None
    for b in grid:
        post = market_anchored_shrink(float(b), market_price, cfg.entry_k)
        _, ev = best_side_ev(post, market_price)
        if ev >= cfg.min_ev:
            div = abs(b - market_price)
            if best is None or div < best[1]:
                best = (b, div)
    return best


def run_synthetic(cfg):
    print(f"=== Synthetic frontier (market-anchored, entry_k={cfg.entry_k}, "
          f"min_ev={cfg.min_ev}) ===\n")
    print("For each market price: the brain probability nearest the market "
          "that still clears min_ev, and the divergence it needs. Larger "
          "divergence = trades less readily. Unlike the old shrink-toward-0.5 "
          "mechanism, agreement (div 0) never clears min_ev here.\n")
    print(f"{'market':>7} | {'brain@breakeven':>16} {'divergence':>11}")
    print("-" * 40)
    for mkt in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        be = breakeven_divergence(mkt, cfg)
        if be is None:
            print(f"{mkt:>7.2f} | {'(never)':>16} {'-':>11}")
        else:
            print(f"{mkt:>7.2f} | {be[0]:>16.4f} {be[1]:>11.4f}")
    print("\nThis is a sensitivity illustration. The real go/no-go is --csv on "
          "a current live-market export from the instance.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="CSV with brain_prob + market_prob (+ optional outcome)")
    args = ap.parse_args()
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

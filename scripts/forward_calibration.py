"""Forward calibration: join resolved dry-run trades back to their entry
snapshots and re-run the divergence-vs-correctness analysis on FORWARD data.

This is the tool the whole dry-run exists to feed. Phase 1/3 could only use
historical data that was 94% NO (far-OTM), which couldn't validate a distrust
mechanism. As the live dry-run accumulates trades that resolve, this script
reconstructs (brain_prob, market_prob, outcome) per resolved trade from what
the pipeline logs, and reports whether the brain's divergence from the market
predicts correctness - the question entry_k's value should eventually be set
from.

How the data is joined (see trading/decision_pipeline.py and
trading/paper_adapter.py):
  - Entry: a "TRACK" hunt_history row per executed trade, payload carrying
    pre_prob (brain raw), market_price_yes (market implied prob at entry),
    post_prob, wang_lambda (= entry_k), model_used, spot, tte_days.
  - Resolution: an "EXPIRED" hunt_history row per resolved position, payload
    carrying side (YES/NO) and price (payout per share: 1.0 win / 0.0 loss).
  - Each EXPIRED is matched to the most recent prior TRACK with the same
    token_id. The market's binary outcome is derived from the EXPIRED side +
    payout.

Usage:
    python scripts/forward_calibration.py [path/to/trades.db]

Read-only. Prints a clear message and exits cleanly when there are no
resolved trades yet (the expected state early in a dry-run).
"""
import ast
import json
import sqlite3
import sys

import numpy as np


def _parse_payload(text) -> dict:
    # Mirror ui/data_manager._parse_payload_value: rows are stored as
    # str(payload) (Python repr, single-quoted), NOT json.dumps - so json
    # fails and ast.literal_eval is the real path. Try both for safety.
    if isinstance(text, dict):
        return text
    s = str(text or "").strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return {}


def _market_resolved_yes(side: str, payout_price: float) -> bool:
    # payout_price is per-share: ~1.0 if the side we held won, ~0.0 if it lost.
    side = str(side or "").upper()
    won = payout_price >= 0.5
    if side == "YES":
        return won
    if side == "NO":
        return not won
    return False  # unknown side - caller filters these out


def load_resolved_trades(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT timestamp, level, token_id, payload FROM hunt_history "
            "WHERE level IN ('TRACK', 'EXPIRED') ORDER BY timestamp ASC, id ASC"
        ).fetchall()
    finally:
        conn.close()

    # Collect TRACK entries per token_id (chronological) and EXPIRED rows.
    tracks_by_token: dict[str, list] = {}
    expired = []
    for r in rows:
        payload = _parse_payload(r["payload"])
        rec = {"timestamp": r["timestamp"], "token_id": r["token_id"], **payload}
        if r["level"] == "TRACK":
            tracks_by_token.setdefault(r["token_id"], []).append(rec)
        else:
            expired.append(rec)

    joined = []
    dropped_unmapped = 0
    dropped_no_entry = 0
    for e in expired:
        token = str(e.get("token_id", ""))
        if token.startswith("__unmapped__"):
            dropped_unmapped += 1
            continue
        candidates = [
            t for t in tracks_by_token.get(token, [])
            if t["timestamp"] <= e["timestamp"]
        ]
        if not candidates:
            dropped_no_entry += 1
            continue
        entry = candidates[-1]  # most recent TRACK before this resolution
        pre = entry.get("pre_prob")
        mkt = entry.get("market_price_yes")
        if pre is None or mkt is None:
            # entries logged before the forward-calibration fields were added
            dropped_no_entry += 1
            continue
        joined.append({
            "token_id": token,
            "pre_prob": float(pre),
            "market_prob": float(mkt),
            "post_prob": float(entry.get("post_prob", np.nan)),
            "entry_k": entry.get("wang_lambda"),
            "model_used": entry.get("model_used"),
            "spot": entry.get("spot"),
            "tte_days": entry.get("tte_days"),
            "outcome": 1.0 if _market_resolved_yes(e.get("side"), float(e.get("price", 0.0))) else 0.0,
        })
    return joined, dropped_unmapped, dropped_no_entry


def analyze(joined):
    n = len(joined)
    pre = np.array([j["pre_prob"] for j in joined])
    mkt = np.array([j["market_prob"] for j in joined])
    out = np.array([j["outcome"] for j in joined])
    n_yes = int(out.sum())

    print(f"=== Forward calibration (n={n} resolved trades, {n_yes} YES / {n - n_yes} NO) ===\n")

    def brier(p):
        return float(np.mean((p - out) ** 2))

    print(f"Brain raw  Brier = {brier(pre):.5f}")
    print(f"Market     Brier = {brier(mkt):.5f}")
    print(f"Brain mean signed error = {float(np.mean(pre - out)):+.5f}")
    print(f"Market mean signed error = {float(np.mean(mkt - out)):+.5f}\n")

    # The core question: when brain and market diverge, which is closer?
    div = pre - mkt
    brain_closer = np.abs(pre - out) < np.abs(mkt - out)
    print(f"Brain closer to outcome than market: {int(brain_closer.sum())}/{n} "
          f"({100 * brain_closer.mean():.1f}%)")

    bull = pre > mkt
    bear = pre < mkt
    for label, mask in (("brain > market (bullish tilt)", bull),
                        ("brain < market (bearish tilt)", bear)):
        m = int(mask.sum())
        if m == 0:
            print(f"  {label}: n=0")
            continue
        print(f"  {label}: n={m}  brain_closer={100 * brain_closer[mask].mean():.1f}%")

    if n < 30:
        print("\nNOTE: n < 30 - not enough resolved trades to draw any "
              "conclusion yet. Keep the dry-run running.")
    if n_yes < 5 or (n - n_yes) < 5:
        print("NOTE: outcome mix is lopsided (need both YES and NO "
              "resolutions) - divergence-direction results are unreliable "
              "until both sides accumulate, same confound as PHASE1_FINDINGS.md.")


def main():
    db_path = sys.argv[1] if len(sys.argv) > 1 else "trades.db"
    try:
        joined, dropped_unmapped, dropped_no_entry = load_resolved_trades(db_path)
    except sqlite3.OperationalError as exc:
        print(f"Could not read {db_path}: {exc}")
        print("Point this at the live trades DB (e.g. /app/trades.db on the instance).")
        return

    if dropped_unmapped or dropped_no_entry:
        print(f"(skipped {dropped_unmapped} unmapped + {dropped_no_entry} "
              f"resolutions with no matching entry snapshot)\n")

    if not joined:
        print("No resolved trades with a matching entry snapshot yet - nothing "
              "to analyze. This is expected early in a dry-run; re-run once "
              "markets the bot entered have resolved.")
        return

    analyze(joined)


if __name__ == "__main__":
    main()

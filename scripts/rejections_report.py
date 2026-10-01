"""Why isn't the bot entering trades? Read-only breakdown of REJECTED rows.

Groups REJECTED / FILTERED rows by reason, split into "before" and "after" a
cutoff timestamp (default: when the market-anchored entry logic went live),
and for "EV below dynamic threshold" rejections shows how far the best
candidates were from clearing min_ev.

Usage (on the instance):
    docker exec polybot python3 scripts/rejections_report.py [CUTOFF] [/app/trades.db]
    CUTOFF format: "2026-09-30 20:30:00" (UTC, same clock as hunt_history)
"""
import ast
import json
import sqlite3
import sys
from collections import Counter

DEFAULT_CUTOFF = "2026-09-30 20:30:00"


def parse(text):
    s = str(text or "").strip()
    for fn in (json.loads, ast.literal_eval):
        try:
            v = fn(s)
            return v if isinstance(v, dict) else {}
        except Exception:
            pass
    return {}


def pct(vals, q):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(q * len(vals)))] if vals else float("nan")


def main():
    cutoff = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CUTOFF
    db = sys.argv[2] if len(sys.argv) > 2 else "/app/trades.db"
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT timestamp, level, payload FROM hunt_history "
        "WHERE level IN ('REJECTED','FILTERED','SCAN-SKIP') ORDER BY timestamp").fetchall()
    # normalise "T" separators so string comparison against the cutoff works
    norm = lambda t: str(t).replace("T", " ")

    for label, sel in (("BEFORE", lambda t: norm(t) < cutoff), ("AFTER", lambda t: norm(t) >= cutoff)):
        part = [(t, lv, parse(p)) for t, lv, p in rows if sel(t)]
        print(f"\n===== {label} cutoff {cutoff}: {len(part)} rows =====")
        reasons = Counter((lv, str(p.get("reason", "?"))[:60]) for _, lv, p in part)
        for (lv, reason), n in reasons.most_common(10):
            print(f"{n:>6}  {lv:<9} {reason}")

        ev_rows = [p for _, lv, p in part
                   if lv == "REJECTED" and "EV below" in str(p.get("reason", "")) and p.get("ev") is not None]
        if ev_rows:
            evs = [float(p["ev"]) for p in ev_rows]
            thr = ev_rows[-1].get("threshold")
            print(f"\nEV-below-threshold: n={len(evs)}  threshold={thr}")
            print(f"  ev percentiles  p50={pct(evs, .5):.3f}  p90={pct(evs, .9):.3f}  "
                  f"p99={pct(evs, .99):.3f}  max={max(evs):.3f}")
            print(f"  sides: {dict(Counter(str(p.get('side')) for p in ev_rows))}")
            print("  closest to clearing:")
            for p in sorted(ev_rows, key=lambda p: -float(p["ev"]))[:5]:
                print(f"    ev={float(p['ev']):.3f} side={p.get('side')} "
                      f"brain={p.get('pre_prob')} post={p.get('wang_fair_value')} "
                      f"k={p.get('wang_lambda')}  {str(p.get('market_name'))[:55]}")


if __name__ == "__main__":
    main()

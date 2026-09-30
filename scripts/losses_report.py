"""Where is the dry-run losing money? Read-only report from trades.db.

Joins each closed position's exit row (TAKE-PROFIT / STOP-LOSS /
WANG-EDGE-DECAY / EV-CONVERGENCE / EXPIRED, sold=True) back to its entry row
(DRY-RUN / PAPER-TRADE / LIVE-TRADE, same token_id) and breaks realized PnL
down by exit reason, side, entry-price bucket and asset type. Also prints the
portfolio curve and what is still open.

Usage (on the instance):
    docker exec polybot python3 scripts/losses_report.py [/app/trades.db]
"""
import ast
import json
import sqlite3
import sys
from collections import defaultdict

ENTRY_LEVELS = ("LIVE-TRADE", "DRY-RUN", "PAPER-TRADE")
EXIT_LEVELS = ("TAKE-PROFIT", "STOP-LOSS", "WANG-EDGE-DECAY", "EV-CONVERGENCE", "EXPIRED")


def parse(text):
    if isinstance(text, dict):
        return text
    s = str(text or "").strip()
    for fn in (json.loads, ast.literal_eval):
        try:
            v = fn(s)
            return v if isinstance(v, dict) else {}
        except Exception:
            pass
    return {}


def bucket(p):
    if p is None:
        return "?"
    for hi, label in ((0.05, "<0.05"), (0.10, "0.05-0.10"), (0.20, "0.10-0.20"),
                      (0.35, "0.20-0.35"), (0.65, "0.35-0.65"), (1.01, ">0.65")):
        if p < hi:
            return label
    return "?"


def table(title, groups):
    print(f"\n--- {title} ---")
    print(f"{'group':<22}{'n':>5}{'wins':>6}{'pnl$':>10}{'avg$':>9}")
    for k, v in sorted(groups.items(), key=lambda kv: sum(kv[1])):
        n = len(v)
        print(f"{str(k):<22}{n:>5}{sum(1 for x in v if x > 0):>6}{sum(v):>10.2f}{sum(v) / n:>9.2f}")


def main():
    db = sys.argv[1] if len(sys.argv) > 1 else "/app/trades.db"
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    print("=== Engine status ===")
    try:
        for r in conn.execute("SELECT * FROM engine_status"):
            print(dict(r))
    except Exception as e:
        print("n/a:", e)

    print("\n=== Portfolio value (paper_snapshots) ===")
    try:
        rows = conn.execute(
            "SELECT timestamp,cash,positions_value,total_value FROM paper_snapshots ORDER BY timestamp").fetchall()
        if rows:
            print("first:", tuple(rows[0]))
            print("last :", tuple(rows[-1]))
            print(f"min total: {min(r['total_value'] for r in rows):.2f}   "
                  f"max total: {max(r['total_value'] for r in rows):.2f}   snapshots: {len(rows)}")
        else:
            print("no snapshots")
    except Exception as e:
        print("n/a:", e)

    print("\n=== hunt_history rows by level ===")
    for r in conn.execute("SELECT level,COUNT(*) c,MIN(timestamp) a,MAX(timestamp) b "
                          "FROM hunt_history GROUP BY level ORDER BY c DESC"):
        print(f"{r['level']:<18}{r['c']:>7}   {r['a']}  ->  {r['b']}")

    rows = conn.execute(
        "SELECT timestamp,level,asset_type,token_id,payload FROM hunt_history "
        "WHERE level IN (%s) ORDER BY timestamp, id" % ",".join("?" * (len(ENTRY_LEVELS) + len(EXIT_LEVELS))),
        ENTRY_LEVELS + EXIT_LEVELS).fetchall()

    entries = defaultdict(list)
    closed = []
    for r in rows:
        p = parse(r["payload"])
        if r["level"] in ENTRY_LEVELS:
            entries[r["token_id"]].append((r["timestamp"], p, r["asset_type"]))
        elif p.get("sold"):
            closed.append((r, p))

    by_reason, by_side, by_bucket, by_asset = (defaultdict(list) for _ in range(4))
    unmatched = 0
    worst = []
    for r, p in closed:
        prior = [e for e in entries.get(r["token_id"], []) if e[0] <= r["timestamp"]]
        if not prior:
            unmatched += 1
            continue
        _, ep, asset = prior[-1]
        entry_price = ep.get("price")
        exit_price = p.get("price")
        shares = p.get("shares") or ep.get("shares")
        try:
            pnl = (float(exit_price) - float(entry_price)) * float(shares)
        except (TypeError, ValueError):
            unmatched += 1
            continue
        side = str(ep.get("side") or p.get("side") or "?").upper()
        by_reason[r["level"]].append(pnl)
        by_side[side].append(pnl)
        by_bucket[bucket(float(entry_price))].append(pnl)
        by_asset[str(asset or "?")[:20]].append(pnl)
        worst.append((pnl, r["level"], side, float(entry_price), float(exit_price),
                      str(ep.get("market_name") or ep.get("question") or "")[:50]))

    total = sum(sum(v) for v in by_reason.values())
    n = sum(len(v) for v in by_reason.values())
    print(f"\n=== Realized closed trades: n={n}  total PnL=${total:.2f}  (unmatched/skipped: {unmatched}) ===")
    if not n:
        print("No closed trades with a matching entry yet.")
    else:
        table("by exit reason", by_reason)
        table("by side", by_side)
        table("by entry price", by_bucket)
        table("by asset", by_asset)
        print("\n--- 8 worst trades ---")
        for pnl, lvl, side, ep_, xp, name in sorted(worst)[:8]:
            print(f"{pnl:>8.2f}  {lvl:<16}{side:<4} in={ep_:.3f} out={xp:.3f}  {name}")

    print("\n=== Open positions ===")
    try:
        op = conn.execute("SELECT side,COUNT(*) n,SUM(value) v,SUM(shares*initial_price) cost "
                          "FROM open_positions GROUP BY side").fetchall()
        for r in op:
            print(dict(r))
        if not op:
            print("none")
    except Exception as e:
        print("n/a:", e)


if __name__ == "__main__":
    main()

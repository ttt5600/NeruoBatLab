"""rt_review_lag helper: 5-min Kalshi tape (volume + VWAP per strike) for one KXRT event (read-only, local files).

    python scripts/rt_tape.py KXRT-HOWT 2026-11-11T12:30:00Z 2026-11-11T17:00:00Z 30,40,50,60
"""
import collections
import datetime as dt
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results" / "rt_review_lag"
ev, t0s, t1s, strikes = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
step = int(sys.argv[5]) if len(sys.argv) > 5 else 5
T = json.load(open(OUT / f"trades_{ev}.json"))


def p(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


t0, t1 = p(t0s), p(t1s)
B = collections.defaultdict(lambda: collections.defaultdict(list))
for x in T:
    t = p(x["created_time"])
    if not (t0 <= t < t1):
        continue
    b = t.replace(minute=t.minute // step * step, second=0, microsecond=0)
    k = x["ticker"].split("-")[-1]
    B[b][k].append((float(x["count_fp"]), float(x["yes_price_dollars"])))
    B[b]["_all"].append((float(x["count_fp"]), 0))
for b in sorted(B):
    row = [b.strftime("%m-%d %H:%M"), f"vol={sum(c for c, _ in B[b]['_all']):8.0f}"]
    for s in strikes:
        v = B[b].get(s)
        row.append(f">{s}:" + (f"{sum(c * q for c, q in v) / sum(c for c, _ in v):.2f}" if v else "  -  "))
    print("  ".join(row))

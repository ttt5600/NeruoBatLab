"""Validate the hourly-candle approximation against P2's exact trade-tape pools (results/winners_transfer/tailpool_result.json).
For each P2 series with exact TAIL >= $1k/30d: list its settled markets closing in [T_REF - days, T_REF] (T_REF = P2 run
start 2026-11-21T05:15Z, notes/winners_transfer.md log), run candle_pool, compare tail / mirror (= P2 tail_risk) / full."""
import json, sys
from datetime import datetime, timezone
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts")); sys.path.insert(0, str(HERE))
from kx_throttle import kget
import candle_pool as cp

T_REF = int(datetime(2026, 11, 21, 5, 15, tzinfo=timezone.utc).timestamp())
P2 = json.load(open(HERE.parents[0] / "winners_transfer" / "tailpool_result.json"))
out = HERE / "validate_p2_markets.jsonl"
rows = []
for s, r in P2.items():
    if r["tail_per30d"] < 1000: continue
    mk, cur = [], None
    while True:
        p = {"series_ticker": s, "status": "settled", "min_close_ts": T_REF - r["days"] * 86400, "limit": 1000}
        if cur: p["cursor"] = cur
        d = kget("/markets", p); mk += d.get("markets", []); cur = d.get("cursor")
        if not cur or not d.get("markets"): break
    mk = [m for m in mk if cp.ts(m["close_time"]) <= T_REF]
    cp.run(mk, str(out))
    print(s, len(mk), flush=True)
res = {}
for line in open(out):
    x = json.loads(line); s = x["series"]
    a = res.setdefault(s, dict(tail=0, mirror=0, full=0, tail_rng=0, mirror_rng=0, n=0, missing=0))
    for k in ("tail", "mirror", "full", "tail_rng", "mirror_rng"): a[k] += x[k]
    a["n"] += 1; a["missing"] += int(x["missing"])
cmp = []
for s, a in res.items():
    r = P2[s]; f = 30.0 / r["days"]
    row = dict(series=s, days=r["days"], n=a["n"], missing=a["missing"], exact_tail=r["tail_per30d"],
               candle_tail=round(a["tail"] * f, 2), rng_tail=round(a["tail_rng"] * f, 2),
               exact_mirror=r["tail_risk_per30d"], candle_mirror=round(a["mirror"] * f, 2), rng_mirror=round(a["mirror_rng"] * f, 2),
               exact_full=r["full_per30d"], candle_full=round(a["full"] * f, 2))
    for k in ("tail", "mirror", "full"):
        row["ratio_" + k] = round(row["candle_" + k] / row["exact_" + k], 3) if row["exact_" + k] else None
    for k in ("tail", "mirror"):
        row["ratio_rng_" + k] = round(row["rng_" + k] / row["exact_" + k], 3) if row["exact_" + k] else None
    cmp.append(row)
json.dump(cmp, open(HERE / "validate_p2.json", "w"), indent=1)
for row in sorted(cmp, key=lambda r: -r["exact_tail"]):
    print(row["series"], row["n"], row["exact_tail"], row["candle_tail"], row["ratio_tail"], row["ratio_rng_tail"],
          "| mirror", row["exact_mirror"], row["candle_mirror"], row["ratio_mirror"], row["ratio_rng_mirror"], "| full", row["ratio_full"])

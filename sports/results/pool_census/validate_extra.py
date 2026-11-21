"""Validate candle_pool (1-min for life <= 3 h; hourly otherwise) against the exact trade tape on market types P2 did not
cover: crypto 15-min / hourly, game winners (multi-day life, in-game action), tennis matches. Sample is fixed by a seeded
random draw from the enumeration (settled_90d.jsonl), not by outcome. Output validate_extra.json."""
import json, random, sys, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts")); sys.path.insert(0, str(HERE))
from kx_throttle import kget
import candle_pool as cp

random.seed(20261121)
SER = {"KXBTC15M": 3, "KXETH15M": 1, "KXBTCD": 1, "KXNBAGAME": 1, "KXNCAAMBGAME": 2, "KXATPCHALLENGERMATCH": 2,
       "KXNCAAMBTOTAL": 1, "KXNBAPTS": 1}
MAXPAGES = 120   # a market whose window tape exceeds this is dropped from the comparison (flagged), not truncated
by = collections.defaultdict(list)
for line in open(HERE / "settled_90d.jsonl"):
    m = json.loads(line)
    if m["result"] in ("yes", "no") and float(m["volume_fp"] or 0) > 0:
        by[m["event_ticker"].split("-")[0]].append(m)
picked = []
for s, k in SER.items():
    evs = sorted({m["event_ticker"] for m in by[s]})
    for e in random.sample(evs, min(k, len(evs))):
        picked += [m for m in by[s] if m["event_ticker"] == e]
print("markets", len(picked), flush=True)

def exact(m):
    ws, we, per = cp.plan(m)
    out = dict(tail=0.0, mirror=0.0, full=0.0, n=0)
    cur = None; pages = 0
    while True:
        pages += 1
        if pages > MAXPAGES:
            out["incomplete"] = True; break
        p = {"ticker": m["ticker"], "min_ts": ws, "max_ts": we, "limit": 1000}
        if cur: p["cursor"] = cur
        d = kget("/markets/trades", p); tr = d.get("trades", [])
        for t in tr:
            n = float(t["count_fp"]); y = float(t["yes_price_dollars"]); no = float(t["no_price_dollars"])
            assert abs(y + no - 1) < 1e-6, t
            pl = y if m["result"] == "no" else no
            out["full"] += n * pl; out["n"] += 1
            if pl <= 0.15: out["tail"] += n * pl
            if pl >= 0.85: out["mirror"] += n * pl
        cur = d.get("cursor")
        if not cur or not tr: break
    return out

cand_path = HERE / "validate_extra_candle.jsonl"
cp.run(picked, str(cand_path))
cand = {json.loads(l)["ticker"]: json.loads(l) for l in open(cand_path)}
agg = collections.defaultdict(lambda: collections.Counter())
for m in picked:
    s = m["event_ticker"].split("-")[0]; ex = exact(m); c = cand[m["ticker"]]
    print(m["ticker"], {k: round(v, 1) if isinstance(v, float) else v for k, v in ex.items()}, "candle", c["tail"], c["mirror"], c["full"], flush=True)
    if ex.get("incomplete"): continue
    a = agg[s]
    for k in ("tail", "mirror", "full"): a["ex_" + k] += ex[k]; a["c_" + k] += c[k]
    a["c_tail_rng"] += c["tail_rng"]; a["c_mirror_rng"] += c["mirror_rng"]; a["n"] += 1; a["trades"] += ex["n"]
res = {}
for s, a in agg.items():
    r = {k: round(v, 2) for k, v in a.items()}
    for k in ("tail", "mirror", "full"):
        r["ratio_" + k] = round(a["c_" + k] / a["ex_" + k], 3) if a["ex_" + k] else None
    for k in ("tail", "mirror"):
        r["ratio_rng_" + k] = round(a["c_%s_rng" % k] / a["ex_" + k], 3) if a["ex_" + k] else None
    res[s] = r; print(s, r, flush=True)
json.dump(res, open(HERE / "validate_extra.json", "w"), indent=1)

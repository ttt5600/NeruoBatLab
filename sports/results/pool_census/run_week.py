"""pool_census stage 2 driver over the WEEK sample: markets closing in [2026-11-14T00:00Z, 2026-11-21T00:00Z) (7 full UTC days,
Sat-Fri) from the global enumeration settled_90d.jsonl. Branch and bound: series ranked by the bound
TAIL/30d <= 0.15 x contracts_week x 30/7; measure in that order with candle_pool; stop when the next bound is below the
current K-th largest measured TAIL/30d, or when the call budget is spent. Heavy series (> CAP calls) are event-sampled:
the TOPN highest-volume events always, plus a seeded random sample of the rest; the rest is scaled by
N_rest / n_sampled (Horvitz-Thompson). Usage: python run_week.py [K=40] [CAP=30] [BUDGET=2500]"""
import json, random, sys, collections, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import candle_pool as cp

W0, W1 = "2026-11-14T00:00:00Z", "2026-11-21T00:00:00Z"
K = int(sys.argv[1]) if len(sys.argv) > 1 else 40
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 30
BUDGET = int(sys.argv[3]) if len(sys.argv) > 3 else 2500
TOPN = int(__import__("os").environ.get("TOPN", "10"))
F30 = 30.0 / 7.0
import os
TAG = os.environ.get("TAG", "week")      # pass 2 (non-sports) uses TAG=week2 -> separate output / plan files
OUT = HERE / (TAG + "_markets.jsonl"); PLAN = HERE / (TAG + "_plan.json")

seen = set(); by = collections.defaultdict(list)
for line in open(HERE / "settled_90d.jsonl"):
    m = json.loads(line)
    if not (W0 <= m["close_time"] < W1) or m["ticker"] in seen: continue
    seen.add(m["ticker"])
    if m["result"] in ("yes", "no") and float(m["volume_fp"] or 0) > 0:
        by[m["event_ticker"].split("-")[0]].append(m)
ser = []
for s, ms in by.items():
    v = sum(float(m["volume_fp"]) for m in ms)
    ser.append(dict(series=s, contracts_week=v, bound30=0.15 * v * F30, markets=len(ms), events=len({m["event_ticker"] for m in ms})))
ser.sort(key=lambda r: -r["bound30"])
json.dump(ser, open(HERE / "week_series_bounds.json", "w"))
import os
EXCL = set(filter(None, os.environ.get("EXCL", "").split(",")))   # pass 2: skip these categories, stop at MINB instead of K
MINB = float(os.environ.get("MINB", "0"))
if EXCL or MINB:
    cat = {x["ticker"]: x.get("category") for x in json.load(open(HERE / "series_all.json"))}
    MAXB = float(os.environ.get("MAXB", "1e18"))   # pass 2 leaves series above MAXB to pass 1
    ser = [r for r in ser if cat.get(r["series"]) not in EXCL and r["bound30"] < MAXB]
    K = 10 ** 9
print("series traded in week", len(ser), "markets", sum(r["markets"] for r in ser), flush=True)

plan = json.load(open(PLAN)) if PLAN.exists() else {}
def measured():
    tot = collections.defaultdict(float)
    if OUT.exists():
        for line in open(OUT):
            x = json.loads(line); w = plan.get(x["series"], {}).get("w", {}).get(x["event"], 1.0)
            tot[x["series"]] += x["tail"] * w * F30
    return tot
calls = sum(p.get("calls", 0) for p in plan.values())
random.seed(7)
for r in ser:
    s = r["series"]
    tot = measured()
    kth = sorted(tot.values(), reverse=True)[K - 1] if len(tot) >= K else 0.0
    if MINB and r["bound30"] < MINB:
        print("STOP: bound", round(r["bound30"]), "< MINB", MINB, "at", s, flush=True); break
    if r["bound30"] < kth:
        print("STOP: bound", round(r["bound30"]), "<", K, "th measured", round(kth), "at", s, flush=True); break
    if calls >= BUDGET:
        print("STOP: budget", calls, "at", s, "bound", round(r["bound30"]), "kth", round(kth), flush=True); break
    if s in plan and plan[s].get("done"): continue
    ms = by[s]
    nb = len(cp.batches(ms))
    w = {}
    if nb > CAP:
        ev_v = collections.Counter()
        for m in ms: ev_v[m["event_ticker"]] += float(m["volume_fp"])
        top = [e for e, _ in ev_v.most_common(TOPN)]
        rest = [e for e in ev_v if e not in top]
        random.shuffle(rest)
        chosen = list(top)
        for e in rest:
            sub = [m for m in ms if m["event_ticker"] in set(chosen + [e])]
            if len(cp.batches(sub)) > CAP: break
            chosen.append(e)
        samp = [e for e in chosen if e not in top]
        wt = (len(rest) / len(samp)) if samp else 0.0
        w = {e: (1.0 if e in top else wt) for e in chosen}
        ms = [m for m in ms if m["event_ticker"] in w]
        if rest and not samp:
            print("WARN", s, "no rest sampled; rest events", len(rest), flush=True)
    t = time.time()
    n = cp.run(ms, str(OUT), log_every=100)
    calls += n
    plan[s] = dict(done=True, calls=n, sampled=bool(w), n_events=r["events"], n_events_measured=len(w) if w else r["events"],
                   w=w, bound30=r["bound30"])
    json.dump(plan, open(PLAN, "w"))
    tot = measured()
    print(s, "bound", round(r["bound30"]), "tail30", round(tot.get(s, 0)), "calls", n, "total", calls, "sampled", bool(w),
          round(time.time() - t), "s", flush=True)
print("done; calls", calls, flush=True)

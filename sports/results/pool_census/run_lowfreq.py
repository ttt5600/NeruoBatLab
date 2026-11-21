"""pool_census stage 3: series with NO traded settled market in the week sample (monthly / weekly-off / seasonal / one-off,
and leagues paused in the 11-14..11-20 international break). Candidates = series_all.json with 0.075 x lifetime volume
>= MINB (0.075 = 0.15 x 1/2: all lifetime volume inside the 60-day window, per 30 d), in descending lifetime volume.
L1: list settled markets closing in the last 60 days (live endpoint; all after the historical cutoff) -> V60.
L2: candle pools (candle_pool) when 0.15 x V60 / 2 >= MINB; heavy series event-sampled like run_week (TOPN + random, HT).
Stops at MAXSER series or the call budget. Usage: python run_lowfreq.py MINB MAXSER BUDGET"""
import json, random, sys, time, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts")); sys.path.insert(0, str(HERE))
from kx_throttle import kget
import candle_pool as cp

MINB = float(sys.argv[1]); MAXSER = int(sys.argv[2]); BUDGET = int(sys.argv[3])
CAP, TOPN = 30, 10
T_END = int(time.time()); T0 = T_END - 60 * 86400
week = {r["series"] for r in json.load(open(HERE / "week_series_bounds.json"))}
S = [s for s in json.load(open(HERE / "series_all.json")) if s["ticker"] not in week
     and 0.075 * float(s.get("volume_fp") or 0) >= MINB]
import os
EXCL = set(filter(None, os.environ.get("EXCL", "").split(",")))
S = [s for s in S if s.get("category") not in EXCL]
S.sort(key=lambda s: -float(s["volume_fp"]))
OUT = HERE / "lowfreq_markets.jsonl"; PLAN = HERE / "lowfreq_plan.json"; L1 = HERE / "lowfreq_l1.json"
plan = json.load(open(PLAN)) if PLAN.exists() else {}
l1 = json.load(open(L1)) if L1.exists() else {}
calls = sum(p.get("calls", 0) for p in plan.values()) + sum(v.get("calls", 0) for v in l1.values())
random.seed(11)
print("candidates", len(S), "meta", json.dumps({"T0": T0, "T_END": T_END}), flush=True)
for i, s in enumerate(S[:MAXSER]):
    tk = s["ticker"]
    if calls >= BUDGET:
        print("STOP budget at", i, tk, flush=True); break
    if tk in l1 and (tk in plan or l1[tk]["bound30"] < MINB): continue
    mk, cur, nc = [], None, 0
    while True:
        p = {"series_ticker": tk, "status": "settled", "min_close_ts": T0, "max_close_ts": T_END, "limit": 1000}
        if cur: p["cursor"] = cur
        d = kget("/markets", p); nc += 1
        if d is None: break
        mk += d.get("markets", []); cur = d.get("cursor")
        if not cur or not d.get("markets"): break
    calls += nc
    ms = [m for m in mk if m.get("result") in ("yes", "no") and float(m.get("volume_fp") or 0) > 0]
    v60 = sum(float(m["volume_fp"]) for m in ms)
    l1[tk] = dict(calls=nc, markets=len(mk), traded=len(ms), events=len({m["event_ticker"] for m in ms}), v60=v60,
                  bound30=0.15 * v60 / 2, vlife=float(s["volume_fp"]))
    json.dump(l1, open(L1, "w"))
    if l1[tk]["bound30"] < MINB or not ms: continue
    nb = len(cp.batches(ms)); w = {}
    if nb > CAP:
        ev_v = collections.Counter()
        for m in ms: ev_v[m["event_ticker"]] += float(m["volume_fp"])
        top = [e for e, _ in ev_v.most_common(TOPN)]; rest = [e for e in ev_v if e not in top]; random.shuffle(rest)
        chosen = list(top)
        for e in rest:
            if len(cp.batches([m for m in ms if m["event_ticker"] in set(chosen + [e])])) > CAP: break
            chosen.append(e)
        samp = [e for e in chosen if e not in top]
        wt = (len(rest) / len(samp)) if samp else 0.0
        w = {e: (1.0 if e in top else wt) for e in chosen}
        ms = [m for m in ms if m["event_ticker"] in w]
    n = cp.run(ms, str(OUT), log_every=100); calls += n
    plan[tk] = dict(done=True, calls=n, sampled=bool(w), n_events=l1[tk]["events"], w=w, bound30=l1[tk]["bound30"])
    json.dump(plan, open(PLAN, "w"))
    print(i, tk, "v60", round(v60), "bound30", round(l1[tk]["bound30"]), "events", l1[tk]["events"], "calls", n, "total", calls, flush=True)
print("done calls", calls, flush=True)

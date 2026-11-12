"""Usage: q2_fetch.py (cached per game) then q2_score.py dev|forward.
Q2 score: pre-registered K1 (notes/nba_l32b_exec.md) on L32b live picks logged before 2026-11-11 (dev).
Also descriptive: same rule at the L32 decision tick, and with no re-check of the gap at S."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
S = json.load(open(Path.home() / ".claude/quant-sports/data/l32b_state.json"))
L = json.load(open(Path.home() / ".claude/quant-kalshi/data/open_totals_state.json"))
C = pd.read_parquet(OUT / "q2_candles.parquet")
M = pd.read_parquet(OUT / "q2_markets.parquet")
CUTOFF = pd.Timestamp("2026-11-11", tz="UTC")


def fee(p, n=100):
    return math.ceil(round(n * 0.07 * p * (1 - p) * 100, 6)) / 100 / n


def quote(key, T):
    t = int((T - EPOCH) // pd.Timedelta(seconds=1))
    c = C[(C.key == key) & (C.end <= t) & (C.end >= t - 3600) & C.bid.notna() & C.ask.notna()]
    if c.empty:
        return None
    return c.sort_values("end").groupby("strike").tail(1).set_index("strike")[["bid", "ask", "end"]]


def one(r, T, recheck=True):
    g = L[r["key"]]
    q = quote(r["key"], T)
    base = dict(key=r["key"], pick=r["pick"], T=T, open_line=r["open_line"], model=r["model"], c=r["c"])
    if q is None or q.empty:
        return {**base, "status": "no quote"}
    q = q[(q.bid > 0) & (q.ask < 1)]
    if q.empty:
        return {**base, "status": "no two-sided"}
    ks = np.array(sorted(q.index))
    dist = np.abs(ks - r["open_line"])
    cand = ks[dist == dist.min()]
    Sx = float(cand.min() if r["pick"] == "over" else cand.max())
    adj = r["model"] - r["c"] - Sx
    ok = adj >= 3 if r["pick"] == "over" else adj <= -3
    yb, ya = q.loc[Sx, "bid"], q.loc[Sx, "ask"]
    ask = ya if r["pick"] == "over" else 1 - yb
    f = fee(ask)
    fin = r.get("final_total")
    close = (g.get("pin_last") or {}).get("line")
    sgn = 1 if r["pick"] == "over" else -1
    out = {**base, "status": "bet" if (ok or not recheck) else "gap<3 at S", "S": Sx, "S_minus_open": Sx - r["open_line"],
           "adj_S": round(adj, 2), "yes_bid": yb, "yes_ask": ya, "ask": ask, "fee": f, "spread": ya - yb,
           "quote_age_min": (int((T - EPOCH) // pd.Timedelta(seconds=1)) - q.loc[Sx, "end"]) / 60,
           "nv_open": r["nv_prob"], "final": fin, "clv_S": None if close is None else (close - Sx) * sgn}
    if fin is not None and not (isinstance(fin, float) and np.isnan(fin)):
        win = (fin > Sx) == (r["pick"] == "over")
        out.update(result="win" if win else "loss", roi=(1 - ask - f) / (ask + f) if win else -1.0)
    return out


def summ(x, nm):
    x = x[x.status == "bet"]
    s = x[x.roi.notna()] if "roi" in x else x.iloc[:0]
    if not len(s):
        print(nm, "bets", len(x), "settled 0")
        return {}
    r = s.roi.values
    rng = np.random.default_rng(0)
    bs = np.array([rng.choice(r, len(r)).mean() for _ in range(4000)])
    cl = s.clv_S.dropna().values
    cb = np.array([rng.choice(cl, len(cl)).mean() for _ in range(4000)])
    res = dict(n=len(s), wins=int((s.result == "win").sum()), roi=r.mean(), lo=np.percentile(bs, 2.5),
               hi=np.percentile(bs, 97.5), p=(bs <= 0).mean(), clv=cl.mean(), clv_lo=np.percentile(cb, 2.5),
               clv_hi=np.percentile(cb, 97.5), cost=(s.ask + s.fee).mean(), spread=s.spread.mean())
    print(f"{nm}: settled {res['n']} ({res['wins']}W) ROI@ask+fee {res['roi']:+.1%} [{res['lo']:+.1%},{res['hi']:+.1%}] "
          f"p {res['p']:.3f}  CLV vs S {res['clv']:+.2f} [{res['clv_lo']:+.2f},{res['clv_hi']:+.2f}]  "
          f"mean cost {res['cost']:.3f} spread {res['spread'] * 100:.1f}c")
    return res


P = pd.DataFrame([r for r in S.values() if r.get("pick") and r["kind"] == "live"])
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "dev"   # dev: logged before 11-11; forward: from 11-12
P = P[pd.to_datetime(P.t) < CUTOFF] if SPLIT == "dev" else P[pd.to_datetime(P.t) >= pd.Timestamp("2026-11-12", tz="UTC")]
print("L32b live picks before cutoff:", len(P), " settled:", P.result.notna().sum() if "result" in P else 0)
rows = [one(r, pd.Timestamp(r["t"])) for _, r in P.iterrows()]
K = pd.DataFrame(rows)
print(K.status.value_counts().to_dict())
K.to_csv(OUT / f"q2_k1_rows_{SPLIT}.csv", index=False)
res = summ(K, "K1 PRIMARY (L32b tick, recheck gap at S)")
summ(K[K.pick == "over"], "   overs")
summ(K[K.pick == "under"], "   unders")
K2 = pd.DataFrame([one(r, pd.Timestamp(r["t"]), recheck=False) for _, r in P.iterrows()])
summ(K2, "descr: no recheck (all L32b picks at nearest strike)")
K3 = pd.DataFrame([one(r, pd.Timestamp(r["l32_decision_t"]) + pd.Timedelta(seconds=1), recheck=False)
                   for _, r in P.iterrows() if r.get("l32_decision_t")])
summ(K3, "descr: at L32 decision tick (open time), no recheck")
# same picks at the L32b no-vig open (reference)
s = P[P.result.isin(["win", "loss", "push"])]
print(f"reference L32b same picks: settled {len(s)} no-vig ROI {s.pnl_novig.mean():+.1%}  CLV {s.clv_pts.mean():+.2f}")
print("S - open: mean %+.2f, |.|>0 share %.2f; quote age min median %.1f; L32b tick - pin_open (h) median %.2f" % (
    K.S_minus_open.mean(), (K.S_minus_open != 0).mean(), K.quote_age_min.median(),
    ((pd.to_datetime(P.t) - pd.to_datetime(P.open_t)).dt.total_seconds() / 3600).median()))
json.dump({k: float(v) for k, v in res.items()}, open(OUT / f"q2_k1_summary_{SPLIT}.json", "w"), indent=1)

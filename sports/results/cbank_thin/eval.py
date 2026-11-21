"""cbank_thin dev eval: CB1 (consensus fade), P0 (Kalshi-favourite placebo), calibration. Run only after consensus.json frozen."""
import json, math, hashlib, sys
from pathlib import Path
import numpy as np, pandas as pd
R = Path("results/cbank_thin")
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "dev"
assert SPLIT == "dev", "sealed untouched"
q = pd.read_csv(R / "quotes_q.csv", parse_dates=["anchor"])
q = q[(q.split == SPLIT) & (q.series != "KXBOE")].copy()
cons = pd.DataFrame(json.load(open(R / "consensus.json")))
print("consensus.json sha256", hashlib.sha256((R / "consensus.json").read_bytes()).hexdigest()[:16])
out = pd.DataFrame(json.load(open(R / "outcomes_HIDDEN.json")))[["ticker", "result"]]
q = q.merge(out, on="ticker", how="left")
assert q.result.isin(["yes", "no"]).all(), q.result.value_counts()
q["yes"] = (q.result == "yes").astype(int)
chk = q.groupby(["event", "T"]).yes.sum()
print("events/T with exactly one YES:", (chk == 1).sum(), "of", len(chk), "| bad:", chk[chk != 1].to_dict())

def fee(p):  # Kalshi taker fee on a 10-contract order, rounded up to the cent, per contract
    return math.ceil(round(0.07 * 10 * p * (1 - p) * 100, 6)) / 100 / 10

def bets(df):
    d = df.copy(); d["p"] = 1 - d.bid; d["fee"] = d.p.map(fee); d["stake"] = d.p + d.fee
    d["pnl"] = np.where(d.yes == 0, 1 - d.stake, -d.stake); return d

def boot(d, B=10000, seed=0):
    ev = d.event.unique(); g = {e: (x.pnl.sum(), x.stake.sum()) for e, x in d.groupby("event")}
    rng = np.random.default_rng(seed); r = []
    for _ in range(B):
        s = rng.choice(ev, len(ev)); P = sum(g[e][0] for e in s); S = sum(g[e][1] for e in s); r.append(P / S)
    return np.percentile(r, [2.5, 97.5])

def report(name, d):
    if len(d) == 0: print(name, "n 0"); return
    roi = d.pnl.sum() / d.stake.sum(); lo, hi = boot(d)
    print(f"{name}: n {len(d)} bets / {d.event.nunique()} meetings, losses {int((d.yes==1).sum())}, mean NO price "
          f"{d.p.mean():.3f}, break-even loss rate {1-d.stake.mean():.3f}, ROI {roi:+.4f} [{lo:+.4f},{hi:+.4f}] (meeting bootstrap)")

# P0: favourite = highest mid; bet every other bucket with bid >= .04
p0 = bets(q[(~q.fav) & (q.bid >= 0.04)])
# CB1: consensus bucket with s >= .90 published before T
c = cons.dropna(subset=["median_bucket"]); c = c[c.s.fillna(0) >= 0.90][["event", "T", "median_bucket", "s"]]
m = q.merge(c, on=["event", "T"], how="inner")
cb1 = bets(m[(m.bucket != m.median_bucket) & (m.bid >= 0.04)])
report("P0 placebo (fade non-favourite, bid>=.04)", p0)
report("CB1 (fade non-consensus, s>=.90, bid>=.04)", cb1)
for T in ("D7", "D1"):
    report(f"  P0 {T}", p0[p0["T"] == T]); report(f"  CB1 {T}", cb1[cb1["T"] == T])
p0.to_csv(R / "bets_P0_dev.csv", index=False); cb1.to_csv(R / "bets_CB1_dev.csv", index=False)
# CB1 - P0 on matched meetings (events where both have bets)
ev = sorted(set(cb1.event) & set(p0.event))
if ev:
    rng = np.random.default_rng(1); diffs = []
    for _ in range(10000):
        s = rng.choice(ev, len(ev)); a = pd.concat([cb1[cb1.event == e] for e in s]); b = pd.concat([p0[p0.event == e] for e in s])
        diffs.append(a.pnl.sum() / a.stake.sum() - b.pnl.sum() / b.stake.sum())
    print("CB1-P0 matched meetings", len(ev), "diff CI", np.percentile(diffs, [2.5, 50, 97.5]).round(4))
# Calibration: Kalshi mid at D1, all quoted dev markets
d1 = q[q["T"] == "D1"].copy(); d1["bin"] = pd.cut(d1.mid, [0, .02, .05, .1, .2, .4, .6, .8, .9, .95, .98, 1.0])
cal = d1.groupby("bin", observed=True).agg(n=("yes", "size"), mid=("mid", "mean"), realised=("yes", "mean"), events=("event", "nunique"))
print("Kalshi D1 calibration (dev):\n", cal.round(3).to_string()); cal.to_csv(R / "calibration_D1_dev.csv")
fav = q[q.fav].groupby("T").agg(n=("yes", "size"), fav_mid=("mid", "mean"), fav_hit=("yes", "mean")); print("Kalshi favourite hit rate:\n", fav.round(3))
# Consensus surprise rate
cc = cons.dropna(subset=["median_bucket"]).merge(q[q.yes == 1][["event", "T", "bucket"]], on=["event", "T"], how="inner")
cc["hit"] = cc.median_bucket == cc.bucket
print("consensus hit rate by T:\n", cc.groupby("T").agg(n=("hit", "size"), hit=("hit", "mean")).round(3))
print("consensus s>=.90 hit:", cc[cc.s.fillna(0) >= .9].hit.agg(["size", "mean"]).to_dict())
kf = q[q.fav][["event", "T", "bucket"]].rename(columns={"bucket": "kfav"})
cc = cc.merge(kf, on=["event", "T"]); print("consensus == Kalshi favourite:", (cc.median_bucket == cc.kfav).mean().round(3), "n", len(cc))
print(cc[cc.median_bucket != cc.kfav][["event", "T", "median_bucket", "s", "kfav", "bucket"]].to_string())
cc.to_csv(R / "consensus_vs_outcome_dev.csv", index=False)

"""pm_full_census T1 score: DEV only (earliest 60% of events by PM US endDate). Sealed rows are never loaded into stats."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "scripts")
OUT = "results/pm_full_census/"
EDGE, LO, HI = 0.03, 0.05, 0.95
def fee(p, th): return th * p * (1 - p)

def load():
    d = pd.read_parquet(OUT + "xvus_panel.parquet")
    from pm_full_census_a3 import a3
    keep = a3(pd.read_parquet(OUT + "xvus_match.parquet"), pd.read_parquet(OUT + "xvus_gev.parquet"))
    d = d[d.slug.isin(set(keep.slug))]
    d["end_ts"] = (pd.to_datetime(d.end, utc=True) - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)  # traps: no int64//1e9
    ok = (d.ya + d.na >= 0.999) & (d.ya + d.na - 1 <= 0.10) & d.g.notna()
    ok &= d.t <= d.end_ts - 86400
    ok &= d.g_dec.isna() | (d.t < d.g_dec)
    ok &= (d.t - d.g_t) <= 3 * 86400  # reference no older than 3 days
    d = d[ok].copy()
    ev = d.groupby("question").end_ts.max().sort_values()
    cut = ev.iloc[int(len(ev) * 0.6) - 1]
    d["split"] = np.where(d.question.map(ev) <= cut, "dev", "sealed")
    return d, pd.to_datetime(cut, unit="s", utc=True)

def bets(d, ref="g", th=0.095, sign=+1, sens7=False):
    x = d.copy()
    if sens7: x = x[x.g_dec.isna() | (x.t < x.g_dec - 7 * 86400)]
    g = x[ref]
    ey = sign * (g - (x.ya + fee(x.ya, th))); en = sign * ((1 - g) - (x.na + fee(x.na, th)))
    x["side"] = np.where((ey >= EDGE) & x.ya.between(LO, HI), "yes", np.where((en >= EDGE) & x.na.between(LO, HI), "no", ""))
    x = x[(x.side != "") & x[ref].notna()].sort_values("t").groupby("slug").head(1)
    px = np.where(x.side == "yes", x.ya, x.na)
    x["cost"] = px + fee(px, th)
    x["pay"] = (x.side == x.win).astype(float)
    x["pnl"] = x.pay - x.cost
    return x

def boot(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    gs = x.groupby("question").agg(p=("pnl", "sum"), c=("cost", "sum"))
    P, C = gs.p.values, gs.c.values; k = len(gs)
    r = [P[i].sum() / C[i].sum() for i in (rng.integers(0, k, k) for _ in range(n))]
    return P.sum() / C.sum(), np.percentile(r, [2.5, 97.5]), k

def report(name, x):
    if len(x) == 0: print(name, "n=0"); return
    roi, ci, k = boot(x)
    top = x.groupby("question").pnl.sum().abs().max() / max(abs(x.pnl.sum()), 1e-9)
    print(f"{name:34s} n={len(x):5d} events={k:4d} ROI={roi*100:+.2f}% CI[{ci[0]*100:+.2f},{ci[1]*100:+.2f}] "
          f"pnl={x.pnl.sum():+.2f} max_event_share={top:.2f} yes/no={(x.side=='yes').sum()}/{(x.side=='no').sum()}")
    return roi, ci, k

if __name__ == "__main__":
    d, cut = load()
    print("valid rows", len(d), "dev/sealed rows", d.split.value_counts().to_dict(), "cut", cut)
    dev = d[d.split == "dev"]
    print("dev categories", dev.drop_duplicates("slug").category.value_counts().to_dict())
    B = bets(dev); B.to_parquet(OUT + "xvus_dev_bets.parquet")
    report("T1 main (fee .095)", B)
    report("T1 fee .0695", bets(dev, th=0.0695))
    report("T1 sens: drop 7d pre-decided", bets(dev, sens7=True))
    report("P1 placebo: global lagged 7d", bets(dev, ref="g7"))
    report("P2 placebo: reverse sign", bets(dev, sign=-1))
    for c, g in B.groupby("category"): report(f"  by cat {c}", g)

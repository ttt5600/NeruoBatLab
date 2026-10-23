import numpy as np, pandas as pd
lc = pd.read_parquet("/tmp/l2m_lc.parquet")
x = lc[lc.decision.isin(["INC", "CC"]) & ~lc.call_type.fillna("").str.contains("Personal")].copy()
x["miss"] = (x.decision == "INC").astype(float)
t = x.groupby(["call_type", "committer_trailing"]).miss.agg(["mean", "size"]).unstack()
t = t.dropna()
t = t[(t["size"] >= 15).all(axis=1)]
w = 1 / (1 / t["size"][False] + 1 / t["size"][True])
diff = (t["mean"][True] - t["mean"][False])
print(pd.DataFrame({"miss_lead": t["mean"][False], "miss_trail": t["mean"][True], "n_lead": t["size"][False], "n_trail": t["size"][True]}).round(3).sort_values("n_trail", ascending=False).head(12))
print("within-type weighted miss diff (trail-lead): %.3f over %d types" % ((w * diff).sum() / w.sum(), len(t)))
# cluster bootstrap by game of the weighted within-type diff
types = set(t.index); x = x[x.call_type.isin(types)]
games = x.gid.unique(); byg = x.groupby("gid")
idx = {g: i for i, g in enumerate(games)}
x["gi"] = x.gid.map(idx)
rng = np.random.default_rng(2); out = []
for _ in range(1000):
    pick = np.bincount(rng.integers(0, len(games), len(games)), minlength=len(games))
    wt = pick[x.gi.values]
    s = x.assign(w=wt, wm=wt * x.miss).groupby(["call_type", "committer_trailing"])[["w", "wm"]].sum().unstack()
    m = s["wm"] / s["w"]
    ww = 1 / (1 / s["w"][False] + 1 / s["w"][True])
    dd = m[True] - m[False]
    ok = np.isfinite(dd) & np.isfinite(ww)
    out.append((ww[ok] * dd[ok]).sum() / ww[ok].sum())
print("95% CI", np.quantile(out, [.025, .975]).round(3), "share<=0", np.mean(np.array(out) <= 0))

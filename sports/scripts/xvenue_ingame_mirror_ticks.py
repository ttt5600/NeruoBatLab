"""mirror direction (descriptive, NOT tradable for a US user on the intl book): after a Kalshi trade-price jump >= 4c in
10 s, how much does Polymarket's last trade move in the jump's direction over the next 10/30/60 s, measured from the
Poly price at the moment the Kalshi jump is complete? Compare with Poly half-spread (~0.5c) + a PMUS-style fee."""
import numpy as np
import pandas as pd

D = "data/xvenue_ingame/"
DUR = {"nba": 3.0, "nfl": 4.0, "nhl": 3.0, "cbb": 2.5}
m = pd.read_parquet(D + "tick_sample.parquet")
kt = pd.read_parquet(D + "kalshi_ticks_dev.parquet")
pt = pd.read_parquet(D + "poly_ticks_dev.parquet")
kt["s"] = (kt.ts - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)
pt["s"] = pt.ts.astype(float)
rows = []
for r in m.itertuples():
    a = kt[kt.cond == r.cond].sort_values("s")
    b = pt[pt.cond == r.cond].sort_values("s")
    if len(a) < 50 or len(b) < 50:
        continue
    t0 = r.start.timestamp()
    grid = np.arange(t0, t0 + DUR[r.lg] * 3600, 5)
    ka = pd.merge_asof(pd.DataFrame({"s": grid}), a[["s", "px"]], on="s").px.values
    pb = pd.merge_asof(pd.DataFrame({"s": grid}), b[["s", "px"]], on="s").px.values
    i = 2
    while i < len(ka) - 13:
        j = ka[i] - ka[i - 2]
        if np.isfinite(j) and abs(j) >= .04 and .05 < ka[i] < .95 and np.isfinite(pb[i]) and .05 < pb[i] < .95:
            sg = np.sign(j)
            rows.append({"cond": r.cond, "lg": r.lg, "jump": abs(j), "gap0": sg * (ka[i] - pb[i]),
                         "m10": sg * (pb[i + 2] - pb[i]), "m30": sg * (pb[i + 6] - pb[i]), "m60": sg * (pb[i + 12] - pb[i]),
                         "fee": 0.0695 * pb[i] * (1 - pb[i])})
            i += 12
        else:
            i += 1
e = pd.DataFrame(rows)
print("Kalshi-led jump events", len(e), "games", e.cond.nunique())
print(e[["jump", "gap0", "m10", "m30", "m60", "fee"]].describe().round(4).to_string())
e["net60"] = e.m60 - 0.005 - e.fee
g = e.groupby("cond")[["net60"]].agg(["sum", "count"])
s = g.values
rng = np.random.default_rng(0)
idx = rng.integers(0, len(s), (2000, len(s)))
v = s[idx].sum(1)
print("mean net60 (move - 0.5c half-spread - PMUS fee) %.4f  CI [%.4f, %.4f]" % (
    s[:, 0].sum() / s[:, 1].sum(), *np.percentile(v[:, 0] / v[:, 1], [2.5, 97.5])))
print("by lg:", e.groupby("lg").net60.agg(["count", "mean"]).round(4).to_dict())

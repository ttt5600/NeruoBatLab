"""Amendment A1 input: one-release level restatement across Wayback Metro ZHVI vintages (raw/wb/metro/*.csv from
wb_fetch.py). For each pair of vintages whose last months are m and m+1 (same file kind), per metro (SizeRank <= 300):
shift = log(value of month m-1 in new) - log(value of month m-1 in old)  [restatement of an already-published month]
rt_err = log(first-published m+1... ) is not needed here; we report the shift distribution per release and pooled.
Output results/home_values/metro_vintages.json (r = pooled q95 |shift| / 1.96)."""
import json
import numpy as np
import pandas as pd
from pathlib import Path

H = Path(__file__).resolve().parent
D = H / "raw" / "wb" / "metro"
V = {}
for f in sorted(D.glob("*.csv")):
    kind = f.stem.split("_")[0]
    d = pd.read_csv(f)
    d = d[d.SizeRank <= 300]
    cols = [c for c in d.columns if c[:2] in ("19", "20")]
    last = pd.Timestamp(cols[-1])
    V.setdefault((kind, last), (f.name, d.set_index("RegionID")[cols]))   # first snapshot per last-month
rel, pooled = [], []
keys = sorted(V)
for (k1, m1), (k2, m2) in zip(keys, keys[1:]):
    if k1 != k2 or (m2.to_period("M") - m1.to_period("M")).n != 1:
        continue
    (f1, a), (f2, b) = V[(k1, m1)], V[(k2, m2)]
    mm = a.columns[-1]
    s = (np.log(b[mm]) - np.log(a[mm])).dropna() * 100
    pooled += s.tolist()
    rel.append(dict(kind=k1, old=f1, new=f2, month=mm, n=int(len(s)), median=round(float(s.median()), 3),
                    q95_abs=round(float(s.abs().quantile(.95)), 3), us_102001=round(float(s.get(102001, np.nan)), 3)))
p = pd.Series(pooled)
out = dict(releases=rel, n_releases=len(rel), pooled_n=int(len(p)),
           pooled_q95_abs=float(p.abs().quantile(.95)) if len(p) else None,
           r_pct=float(p.abs().quantile(.95) / 1.96) if len(p) else None)
json.dump(out, open(H / "metro_vintages.json", "w"), indent=1)
print(json.dumps(out, indent=1))

"""POST-HOC (added after summary.json was read): adverse selection seen through CLV, the low-variance signal.

CLV gap = mean CLV(filled) - mean CLV(unfilled); CLV-implied ROI = p/cost - 1 with p = 0.5 + clv * phi(0)/sigma
(sigma = sd(final - book open): NBA 18.61, CFB 15.39, NFL total 13.09 / spread 12.51 from results/k50_strike/sigma.json;
CBB 16.69 from results/cbb_opener_model odds_2024 x games_2024, 2023-24 season). Date-clustered bootstrap as score.py.
-> clv_check.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from score import cboot  # noqa: E402

SIG = {("nba", "total"): 18.61, ("cfb", "total"): 15.39, ("nfl", "total"): 13.09, ("nfl", "spread"): 12.51,
       ("cbb", "total"): 16.69}
d = pd.read_csv(OUT / "rows.csv")
d["cpp"] = [0.398942 / SIG[(a, b)] for a, b in zip(d.lg, d.mk)]
d["p_clv"] = 0.5 + d.clv * d.cpp
d["roi_clv"] = d.p_clv / d.cost - 1
taker = d[d.x == 0].copy()
taker["roi_clv"] = taker.p_clv / taker.taker_cost - 1
R = {"taker_all": dict(n=len(taker), clv=cboot(taker, lambda s: s.clv.mean()),
                       roi_clv=cboot(taker, lambda s: s.roi_clv.mean()))}
print("taker", R["taker_all"])
for x in (0, 1, 2):
    for lg in ("pooled", "nba", "cfb", "nfl", "cbb"):
        g = d[d.x == x] if lg == "pooled" else d[(d.x == x) & (d.lg == lg)]
        f, u = g[g.fill], g[~g.fill]
        r = dict(n_filled=len(f), clv_filled=round(f.clv.mean(), 3), clv_unfilled=round(u.clv.mean(), 3) if len(u) else None,
                 clv_gap=cboot(g, lambda s: s[s.fill].clv.mean() - s[~s.fill].clv.mean()),
                 roi_clv_filled=cboot(f, lambda s: s.roi_clv.mean()) if f.date.nunique() > 1 else None,
                 roi_obs_filled=round(f.roi.mean(), 4))
        R[f"{lg}_x{x}"] = r
        print(lg, x, r)
json.dump(R, open(OUT / "clv_check.json", "w"), indent=1)

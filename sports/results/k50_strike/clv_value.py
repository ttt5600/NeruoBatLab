"""Descriptive (not judged): what ROI does the observed CLV buy at the price paid? Close-implied win prob =
Phi(clv / sigma) (normal around the closing line, prior-season sigma); implied ROI = p / cost - 1. Also the CLV
needed to break even at the mean cost. -> clv_value.json"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

OUT = Path(__file__).resolve().parent
SIG = json.load(open(OUT / "sigma.json"))
res = {}
for lg in ("nba", "cfb", "nfl"):
    d = pd.read_csv(OUT / f"rows_{lg}.csv")
    d = d[(d.status == "bet") & d.roi.notna() & d.clv.notna()]
    d["sig"] = [SIG[lg]["sd_total" if mk == "total" else "sd_spread"] for mk in d.mk]
    d["p_close"] = norm.cdf(d.clv / d.sig)
    d["roi_implied"] = d.p_close / d.cost - 1
    for rule, x in d.groupby("rule"):
        if not (rule == "lo" or x.primary.all()):
            continue
        sig = x.sig.mean()
        res[f"{lg}_{rule}"] = dict(n=len(x), roi_obs=round(x.roi.mean(), 4), roi_clv_implied=round(x.roi_implied.mean(), 4),
                                   clv=round(x.clv.mean(), 2), cost=round(x.cost.mean(), 4),
                                   clv_breakeven_pts=round(norm.ppf(x.cost.mean()) * sig, 2),
                                   cents_per_pt=round(100 * norm.pdf(0) / sig, 2))
print(json.dumps(res, indent=1))
json.dump(res, open(OUT / "clv_value.json", "w"), indent=1)

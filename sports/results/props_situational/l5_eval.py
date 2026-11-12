"""L5 2025 evaluation: replicate c on 2025, bootstrap (by date) the pre-registered bet rule, CLV."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

O = Path(__file__).resolve().parent
x = pd.read_parquet(O / "frame.parquet")
lg = lambda p: np.log(np.clip(p, .02, .98) / (1 - np.clip(p, .02, .98)))
b = np.array(json.load(open(O / "l5_check.json"))["coef"])
x = x[x.season == 2025].copy()
X = np.column_stack([np.ones(len(x)), lg(x.p_mkt), lg(x.p_emp) - lg(x.p_mkt)])
x["p_fit"] = 1 / (1 + np.exp(-X @ b))
nz = x.under_win.notna().to_numpy()
m = sm.GLM(x.under_win[nz].to_numpy(), X[nz], family=sm.families.Binomial()).fit(
    cov_type="cluster", cov_kwds={"groups": pd.factorize(x.day[nz])[0]})
out = {"refit2025_c": float(m.params[2]), "refit2025_ci": m.conf_int()[2].tolist()}
bt = x[x.p_fit >= x.p_mkt + 0.03].copy()
rng = np.random.default_rng(0)
days = bt.day.unique()
g = {d: v.pnl_under.to_numpy() for d, v in bt.groupby("day")}
B = np.array([np.concatenate([g[d] for d in rng.choice(days, len(days))]).mean() for _ in range(2000)])
bs = bt.sort_values("day")
h = len(bs) // 2
c = bt.dropna(subset=["p_close"])
out.update({"bets": len(bt), "dates": len(days), "roi": float(bt.pnl_under.mean()),
            "ci95": np.percentile(B, [2.5, 97.5]).tolist(), "p_roi_le_0": float((B <= 0).mean()),
            "half1": float(bs.pnl_under.iloc[:h].mean()), "half2": float(bs.pnl_under.iloc[h:].mean()),
            "win_rate": float(bt.under_win.mean()), "mean_dec_under": float(bt.dec_open_under.mean()),
            "clv_n": len(c), "clv_mean_pclose_minus_popen": float((c.p_close - c.p_mkt).mean()),
            "all_unders_2025_roi": float(x.pnl_under.mean()), "all_unders_2025_n": len(x)})
print(json.dumps(out, indent=1))
json.dump(out, open(O / "l5_eval.json", "w"), indent=1)

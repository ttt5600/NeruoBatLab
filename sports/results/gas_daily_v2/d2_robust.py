"""D2 robustness (descriptive, dev only): does the Benter blend improve Brier OUT OF SAMPLE across dates?
2-fold by time (first/second half of dev dates) + leave-one-month-out; coefficient by half."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy.special import logit, expit
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from edgelab.rules.gas_daily import _q, _ts
D = ROOT / "data" / "gas_daily_v2"
fr = pd.read_parquet(D / "gd_frame.parquet"); fr = fr[fr.result.isin(["yes", "no"]) & fr.p_m.notna()]
cs = json.loads((D / "gd_candles.json").read_text())
rows = []
for r in fr.itertuples(index=False):
    t = _ts(r.T); yb, ya, _ = _q(cs.get(r.ticker, []), t, t - 6 * 3600)
    rows.append((r.date, r.X, r.p_m, yb, ya, float(r.result == "yes")))
q = pd.DataFrame(rows, columns=["date", "X", "p", "yb", "ya", "y"])
q = q[q.yb.notna() & q.ya.notna() & (q.yb > 0) & (q.ya < 1)].copy()
q["mid"] = (q.yb + q.ya) / 2; q = q[(q.mid >= .03) & (q.mid <= .97)].copy()
q["lm"], q["lp"] = logit(q.mid), logit(q.p.clip(1e-4, 1 - 1e-4))
def fit(d): return sm.Logit(d.y.to_numpy(), np.c_[np.ones(len(d)), d.lm, d.lp]).fit(disp=0).params
dates = np.sort(q.date.unique()); half = dates[len(dates) // 2]
for name, tr, te in [("fit H1 -> test H2", q[q.date < half], q[q.date >= half]), ("fit H2 -> test H1", q[q.date >= half], q[q.date < half])]:
    b = fit(tr); pb = expit(b[0] + b[1] * te.lm + b[2] * te.lp)
    print(f"{name}: coef p_M {b[2]:+.3f}; test Brier mid {np.mean((te.mid - te.y)**2):.4f} blend {np.mean((pb - te.y)**2):.4f} n {len(te)}")
q["m"] = q.date.dt.to_period("M"); out = []
for m, te in q.groupby("m"):
    b = fit(q[q.m != m]); pb = expit(b[0] + b[1] * te.lm + b[2] * te.lp)
    out.append((str(m), len(te), np.mean((te.mid - te.y)**2), np.mean((pb - te.y)**2), b[2]))
o = pd.DataFrame(out, columns=["month", "n", "brier_mid", "brier_blend", "coef_pM"]); print(o.round(4).to_string())
print("LOMO pooled: mid", round((o.brier_mid * o.n).sum() / o.n.sum(), 4), "blend", round((o.brier_blend * o.n).sum() / o.n.sum(), 4))

"""Judges model: P(f1 wins | decision, round stats). Outcome-only (UFCStats); no Kalshi data.
Model choice on 2012-2021 fit / 2022-2024 validate; final refit on 2012-2024; TEST = 2025-01-01+ (scored once, printed).
Antisymmetric (no intercept): P(A) = 1 - P(B), so orientation does not matter."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
OUT = Path(__file__).resolve().parent
R = pd.read_parquet(OUT / "rounds.parquet")
R = R[R.dec & ~R.draw & R.y.notna() & (R.date >= "2012-01-01")].copy()
R = R[R.rn == R.rmax]
R["d_ctrlm"] = R.d_ctrl / 60.0
R["d_sig_miss"] = R.d_sig_a - R.d_sig_l
R["d_nonsig"] = R.d_tot_l - R.d_sig_l
X1 = ["d_kd", "d_sig_l", "d_sig_miss", "d_td_l", "d_ctrlm", "d_subatt", "d_rev", "d_nonsig", "d_head_l"]
SC = {c: R[c].std() for c in X1}
for c in X1: R[c + "_z"] = R[c] / SC[c]
XZ = [c + "_z" for c in X1]
fids = R.drop_duplicates("fid")[["fid", "date", "y", "sched5"]].set_index("fid")
# index rounds per fight
R = R.sort_values(["fid", "rnd"])
def pack(sub):
    f = sub.fid.values; u, inv = np.unique(f, return_inverse=True)
    return u, inv, sub[XZ].values, fids.loc[u, "y"].values
def nll_round(theta, inv, X, y, nf, l2=1e-3):
    w, k = theta[:-1], theta[-1]
    s = X @ w
    rs = np.bincount(inv, weights=2 * expit(s) - 1, minlength=nf)
    z = k * rs
    p = np.clip(expit(z), 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean() + l2 * (w @ w)
def fit_round(sub):
    u, inv, X, y = pack(sub)
    th0 = np.r_[np.zeros(len(XZ)), 1.0]; th0[1] = 1.0
    res = minimize(nll_round, th0, args=(inv, X, y, len(u)), method="L-BFGS-B")
    return res.x
def pred_round(th, sub):
    u, inv, X, y = pack(sub)
    s = X @ th[:-1]; rs = np.bincount(inv, weights=2 * expit(s) - 1, minlength=len(u))
    return u, expit(th[-1] * rs), y
def fit_tot(sub, cols):
    T = sub.groupby("fid")[cols].sum(); y = fids.loc[T.index, "y"].values; X = T.values
    def nll(w): p = np.clip(expit(X @ w), 1e-9, 1 - 1e-9); return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean() + 1e-3 * w @ w
    return minimize(nll, np.zeros(len(cols)), method="L-BFGS-B").x
def pred_tot(w, sub, cols):
    T = sub.groupby("fid")[cols].sum(); return T.index.values, expit(T.values @ w), fids.loc[T.index, "y"].values
def score(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return dict(n=len(y), ll=round(float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()), 4),
                brier=round(float(((p - y) ** 2).mean()), 4), acc=round(float(((p > .5) == (y == 1)).mean()), 4))
R["rw_sig"] = np.sign(R.d_sig_l)
fit = R[R.date < "2022-01-01"]; val = R[(R.date >= "2022-01-01") & (R.date < "2025-01-01")]; test = R[R.date >= "2025-01-01"]
print("fights fit/val/test", fit.fid.nunique(), val.fid.nunique(), test.fid.nunique())
print("base rate f1 wins (UFCStats lists winner first):", round(fids.y.mean(), 3))
res = {}
w = fit_tot(fit, ["rw_sig"]); u, p, y = pred_tot(w, val, ["rw_sig"]); res["MS rounds-won-by-sig"] = score(p, y)
w = fit_tot(fit, ["d_sig_l_z"]); u, p, y = pred_tot(w, val, ["d_sig_l_z"]); res["M1 total sig diff"] = score(p, y)
w = fit_tot(fit, XZ); u, p, y = pred_tot(w, val, XZ); res["MT totals all feats"] = score(p, y)
th = fit_round(fit); u, p, y = pred_round(th, val); res["MR round model"] = score(p, y)
for k, v in res.items(): print("VAL", k, v)
print("MR weights", dict(zip(X1 + ["k"], np.round(th, 3))))
if "--test" in sys.argv:
    trn = R[R.date < "2025-01-01"]
    th = fit_round(trn); u, p, y = pred_round(th, test); print("TEST MR", score(p, y))
    w = fit_tot(trn, ["rw_sig"]); u2, p2, y2 = pred_tot(w, test, ["rw_sig"]); print("TEST MS", score(p2, y2))
    w3 = fit_tot(trn, XZ); u3, p3, y3 = pred_tot(w3, test, XZ); print("TEST MT", score(p3, y3))
    # frozen model params + predictions for all 2025+ decisions
    np.save(OUT / "judges_MR_theta.npy", th)
    pd.Series(SC).to_json(OUT / "judges_scale.json")
    out = pd.DataFrame({"fid": u, "p_f1": p, "y_f1": y}).merge(pd.DataFrame({"fid": u3, "pT_f1": p3}), on="fid")
    out.to_parquet(OUT / "judges_test_preds.parquet")
    print("weights final", dict(zip(X1 + ["k"], np.round(th, 3))))

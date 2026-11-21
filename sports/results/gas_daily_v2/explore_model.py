"""DEV-ONLY model exploration on AAA values (no Kalshi prices). Expanding-window OOS forecasts of c_D = AAA(D)-AAA(D-1)
from info public at 08:00 ET on D-1: AAA up to D-1, RB=F daily settles up to D-2 (settle 14:30 ET D-2)."""
import pandas as pd, numpy as np
DEV_END = pd.Timestamp("2026-08-31")
s = pd.read_parquet("aaa_kalshi.parquet")[["day", "v"]].set_index("day").v
s = s[s.index <= DEV_END]
rb = pd.read_parquet("rb_daily.parquet").Close
rb.index = pd.to_datetime(rb.index).tz_localize(None).normalize()
c = s.diff() * 100  # cents
df = pd.DataFrame({"c": c})
df["c1"] = c.shift(1); df["c2"] = c.shift(2); df["c3"] = c.shift(3)
df["s7"] = c.shift(1).rolling(7).sum()
# RB known at D-1 08:00 ET: last settle with date <= D-2
def rb_asof(d, lag_td=0):
    r = rb[rb.index <= d - pd.Timedelta(days=2)]
    return r.iloc[-1 - lag_td] if len(r) > lag_td else np.nan
for k in (1, 3, 5, 10):
    df[f"drb{k}"] = [(rb_asof(d) - rb_asof(d, k)) * 100 for d in df.index]
df["drb5_10"] = df.drb10 - df.drb5
# margin: AAA(D-1) - RB(D-2)
df["m"] = [(s.get(d - pd.Timedelta(days=1), np.nan) - rb_asof(d)) * 100 for d in df.index]
df["mdev"] = df.m - df.m.shift(1).rolling(28, min_periods=14).mean()
df["mon"] = (df.index.dayofweek == 0).astype(float)
df["sun"] = (df.index.dayofweek == 6).astype(float)
df = df.dropna()
print("rows", len(df), df.index.min().date(), df.index.max().date())
specs = {
    "RW0": [], "PERSIST": None, "AR1": ["c1"], "AR2": ["c1", "c2"], "AR3": ["c1", "c2", "c3"],
    "AR2+RB": ["c1", "c2", "drb5", "drb5_10"], "AR2+RBup/dn": ["c1", "c2", "drb5", "drb5_10", "mdev"],
    "AR2+mdev": ["c1", "c2", "mdev"], "AR2+dow": ["c1", "c2", "mon", "sun"],
}
MIN = 30
res = {}
for name, X in specs.items():
    preds, ys = [], []
    for i in range(MIN, len(df)):
        tr, te = df.iloc[:i], df.iloc[i]
        if X is None:
            p = te.c1
        elif not X:
            p = 0.0
        else:
            A = np.c_[np.ones(len(tr)), tr[X].to_numpy()]
            b, *_ = np.linalg.lstsq(A, tr.c.to_numpy(), rcond=None)
            p = b[0] + te[X].to_numpy() @ b[1:]
        preds.append(p); ys.append(te.c)
    e = np.array(ys) - np.array(preds)
    res[name] = dict(n=len(e), rmse=np.sqrt(np.mean(e**2)), mae=np.mean(np.abs(e)), q99=np.quantile(np.abs(e), .99), mx=np.abs(e).max())
print(pd.DataFrame(res).T.round(3).to_string())
A = np.c_[np.ones(len(df)), df[["c1", "c2", "drb5", "drb5_10", "mdev"]].to_numpy()]
b, *_ = np.linalg.lstsq(A, df.c.to_numpy(), rcond=None); print("full-dev coefs const,c1,c2,drb5,drb5_10,mdev", b.round(3))
A = np.c_[np.ones(len(df)), df[["c1", "c2"]].to_numpy()]
b, *_ = np.linalg.lstsq(A, df.c.to_numpy(), rcond=None); r = df.c.to_numpy() - A @ b
print("AR2 coefs", b.round(3), "resid sd", r.std(ddof=3).round(3), "kurtosis", pd.Series(r).kurt().round(2))
print("resid quantiles", np.quantile(r, [.01, .05, .25, .5, .75, .95, .99]).round(2))

# ---- tail calibration of a t4 predictive for AR2+RB (expanding window, scale = in-sample resid sd), dev only
from scipy import stats
X = ["c1", "c2", "drb5", "drb5_10"]
z = []
for i in range(MIN, len(df)):
    tr, te = df.iloc[:i], df.iloc[i]
    A = np.c_[np.ones(len(tr)), tr[X].to_numpy()]
    b, *_ = np.linalg.lstsq(A, tr.c.to_numpy(), rcond=None)
    r = tr.c.to_numpy() - A @ b
    sd = r[-60:].std(ddof=1)
    p = b[0] + te[X].to_numpy() @ b[1:]
    z.append((te.c - p) / sd)
z = np.array(z)
sc = np.sqrt(4 / 2)  # t4 variance 2 -> unit-variance scaling
u = stats.t.cdf(z * sc, 4)
for q in (0.01, 0.03, 0.05):
    print(f"PIT < {q}: {np.mean(u < q):.3f}   PIT > {1-q}: {np.mean(u > 1-q):.3f}  (n {len(u)})")
print("std z", z.std().round(3))

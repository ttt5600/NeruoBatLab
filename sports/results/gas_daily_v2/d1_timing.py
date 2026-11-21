"""D1 (descriptive, prereg): book implied median of AAA(D) by hour ET of D-1 vs realized; vs model M mean (dev only)."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
D = ROOT / "data" / "gas_daily_v2"
fr = pd.read_parquet(D / "gd_frame.parquet"); fr = fr[fr.result.isin(["yes", "no"])]
cs = json.loads((D / "gd_candles.json").read_text())
def v(c, s):
    d = c.get(s) or {}; x = d.get("close_dollars", d.get("close")); return np.nan if x is None else float(x)
# per ticker arrays
arr = {}
for tk, cl in cs.items():
    if cl:
        arr[tk] = (np.array([c["end_period_ts"] for c in cl]), np.array([v(c, "yes_bid") for c in cl]), np.array([v(c, "yes_ask") for c in cl]))
def quote(tk, t):
    if tk not in arr: return np.nan, np.nan
    ts, yb, ya = arr[tk]; m = (ts <= t) & (ts > t - 6 * 3600)
    if not m.any(): return np.nan, np.nan
    i = np.where(m)[0][-1]; return yb[i], ya[i]
def implied_median(x, m):
    o = np.argsort(x); x = np.asarray(x)[o]; m = np.minimum.accumulate(np.asarray(m)[o])
    if m[0] < .5 or m[-1] >= .5: return np.nan
    i = np.where(m < .5)[0][0]
    return x[i] if m[i - 1] == m[i] else x[i - 1] + (m[i - 1] - .5) / (m[i - 1] - m[i]) * (x[i] - x[i - 1])
E = pd.Timestamp("1970-01-01", tz="UTC"); recs = []
for e, g in fr.groupby("event_ticker"):
    Dd = g.date.iloc[0]; val = float(g.expiration_value.iloc[0])
    base = pd.Timestamp(Dd.year, Dd.month, Dd.day, tz="America/New_York") - pd.Timedelta(days=1)
    for hh in range(9, 24):
        t = int(((base + pd.Timedelta(hours=hh)) - E) // pd.Timedelta(seconds=1))
        xs, ms = [], []
        for tk, X in zip(g.ticker, g.X):
            yb, ya = quote(tk, t)
            if np.isfinite(yb) and np.isfinite(ya):
                xs.append(X); ms.append((max(yb, 0) + ya) / 2)
        recs.append((e, Dd, hh, implied_median(xs, ms) if len(xs) >= 3 else np.nan, val, g.mu_m.iloc[0], g.X.min(), g.X.max()))
tm = pd.DataFrame(recs, columns=["ev", "date", "hour_et", "med", "v", "mu", "xlo", "xhi"])
tm["err_book"] = (tm.med - tm.v).abs() * 100; tm["err_model"] = (tm.mu - tm.v).abs() * 100
tm.to_parquet(Path(__file__).parent / "d1_timing.parquet", index=False)
sub = tm[tm.mu.notna() & tm.med.notna()]
print("D1 implied-median abs error (cents) by hour ET of D-1 (events with model M and a book median):")
print(sub.groupby("hour_et").agg(n=("err_book", "count"), book_mae=("err_book", "mean"), book_median=("err_book", "median"),
                                 model_mae=("err_model", "mean")).round(3).to_string())
w = sub.pivot_table(index="ev", columns="hour_et", values="err_book")
both = w[[10, 23]].dropna(); d = both[10] - both[23]
bs = [d.sample(len(d), replace=True, random_state=i).mean() for i in range(2000)]
print(f"paired 10h - 23h book error: {d.mean():+.3f} c [{np.quantile(bs,.025):+.3f}, {np.quantile(bs,.975):+.3f}] n {len(d)}")
m10 = sub[sub.hour_et == 10]; dm = (m10.err_model - m10.err_book)
bs = [dm.sample(len(dm), replace=True, random_state=i).mean() for i in range(2000)]
print(f"at 10h: model err - book err {dm.mean():+.3f} c [{np.quantile(bs,.025):+.3f}, {np.quantile(bs,.975):+.3f}] n {len(dm)}")

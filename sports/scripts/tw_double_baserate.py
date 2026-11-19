"""DBL (pre-registered in notes/trading_ways_survey.md): base rates of +100% / -50% calendar months, and
P(hit 2x before 0.5x within 1/3/6 months) going all-in from each month start. Descriptive only."""
import sys, json
import numpy as np, pandas as pd, yfinance as yf
from scipy.stats import beta
OUT = sys.argv[1]
TICKS = ["SPY", "QQQ", "TQQQ", "SOXL", "UPRO", "BTC-USD", "ETH-USD", "MSTR", "COIN", "GME"]
def cp(k, n):
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return [round(lo, 4), round(hi, 4)]
res = {}
for t in TICKS:
    px = yf.download(t, period="max", auto_adjust=True, progress=False)["Close"].squeeze().dropna()
    px.to_frame("close").to_parquet(f"{OUT}/data/{t}.parquet")
    m = px.resample("ME").last()
    r = m.pct_change().dropna()
    n = len(r); k2 = int((r >= 1.0).sum()); kh = int((r <= -0.5).sum())
    # ladder: from each month start (first trading day close), daily path, horizon in trading days
    starts = px.groupby(px.index.to_period("M")).head(1).index
    lad = {}
    for H, days in (("1m", 21), ("3m", 63), ("6m", 126)):
        hit2 = hit05 = tot = 0
        for s in starts:
            i = px.index.get_loc(s)
            if i + days >= len(px): continue
            path = px.iloc[i:i + days + 1].values / px.iloc[i]
            up = np.argmax(path >= 2.0) if (path >= 2.0).any() else 10**9
            dn = np.argmax(path <= 0.5) if (path <= 0.5).any() else 10**9
            tot += 1
            if up < dn: hit2 += 1
            elif dn < up: hit05 += 1
        lad[H] = {"n_starts": tot, "p_2x_first": round(hit2 / tot, 4), "ci": cp(hit2, tot),
                  "p_half_first": round(hit05 / tot, 4)}
    res[t] = {"first": str(px.index[0].date()), "last": str(px.index[-1].date()), "n_months": n,
              "months_ge_+100%": k2, "p": round(k2 / n, 4), "ci": cp(k2, n),
              "months_le_-50%": kh, "p_half": round(kh / n, 4), "ci_half": cp(kh, n),
              "median_m": round(float(r.median()), 4), "p95_m": round(float(r.quantile(.95)), 4),
              "max_m": round(float(r.max()), 4), "min_m": round(float(r.min()), 4), "ladder": lad}
    print(t, json.dumps(res[t]))
json.dump(res, open(f"{OUT}/dbl_baserates.json", "w"), indent=1)

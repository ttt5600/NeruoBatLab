"""FC1 funding carry backtest, pre-registered in notes/trading_ways_survey.md (2026-11-18). Frozen rule."""
import sys, json
import numpy as np, pandas as pd, yfinance as yf
D = sys.argv[1]; OUT = sys.argv[2]
CAP_MULT = 1.5  # capital per $1 notional at 2x perp leverage
DEV = ("2023-05-12", "2025-04-30"); HOLD = ("2025-05-01", "2026-11-17")
f = {c: pd.read_parquet(f"{D}/hl_funding_{c}.parquet") for c in ("BTC", "ETH")}
for c, df in f.items():
    df["h"] = df.ts.dt.floor("h")
    f[c] = df.groupby("h").fundingRate.sum()
fund = pd.concat(f, axis=1).fillna(0.0)
fund.index = fund.index.tz_convert(None)
fh = fund.mean(axis=1)  # 50/50 BTC/ETH per $ notional, hourly
daily_f = fh.resample("D").sum()  # funding earned on day d (hours of day d)
irx = yf.download("^IRX", start="2023-01-01", auto_adjust=True, progress=False)["Close"].squeeze()
irx.index = pd.to_datetime(irx.index).tz_localize(None)
rf_d = (irx.reindex(daily_f.index).ffill().bfill() / 100) / 365.0  # daily cash yield
# T2 signal: trailing 7-day mean of daily funding through day d-1 (decision at 00:00 UTC day d)
sig = daily_f.rolling(7).mean().shift(1)
def run(on, cost):
    on = on.fillna(False).astype(bool)
    entries = on & ~on.shift(1, fill_value=False)
    # T1 always-on: annual re-entry charged on Jan 1 and first day
    r = np.where(on, daily_f / CAP_MULT, rf_d)  # off -> T-bill on full capital
    r = pd.Series(r, index=daily_f.index) - entries * cost / CAP_MULT
    return r
out = {}
for cost in (0.005, 0.01, 0.02):
    t1_on = pd.Series(True, index=daily_f.index)
    t1 = run(t1_on, cost)
    jan1 = (t1.index.month == 1) & (t1.index.day == 1)
    t1 = t1 - jan1 * cost / CAP_MULT
    t2 = run(sig > 0, cost)
    for name, r in (("T1", t1), ("T2", t2)):
        for sp, (a, b) in (("dev", DEV), ("holdout", HOLD)):
            rr = r[a:b]; rf = rf_d[a:b]
            ex = (rr - rf)
            mex = ex.resample("ME").sum()
            rng = np.random.default_rng(0)
            boots = [rng.choice(mex.values, len(mex), replace=True).mean() for _ in range(5000)]
            key = f"{name}_{sp}_cost{cost}"
            out[key] = {"days": len(rr), "ann_ret_on_capital": round(rr.sum() / len(rr) * 365, 4),
                        "ann_rf": round(rf.sum() / len(rf) * 365, 4),
                        "ann_excess": round(ex.sum() / len(ex) * 365, 4),
                        "monthly_excess_mean": round(mex.mean(), 5),
                        "monthly_excess_ci90": [round(np.quantile(boots, .05), 5), round(np.quantile(boots, .95), 5)],
                        "n_months": len(mex),
                        "frac_days_on": round(float((sig > 0)[a:b].mean()) if name == "T2" else 1.0, 3),
                        "usd_per_month_on_1500": round(rr.sum() / len(rr) * 30.4 * 1500, 2)}
            print(key, out[key])
# yearly gross funding on notional, and liquidation check
yr = daily_f.groupby(daily_f.index.year).agg(["sum", "count"])
out["gross_funding_ann_by_year"] = {int(y): round(s / c * 365, 4) for y, (s, c) in yr.iterrows()}
liq = {}
for t in ("BTC-USD", "ETH-USD"):
    px = pd.read_parquet(f"{D}/{t}.parquet")["close"]; px = px["2023-05-12":]
    liq[t] = {"max_1d_rise": round(float(px.pct_change().max()), 4),
              "max_7d_rise": round(float((px / px.shift(7) - 1).max()), 4),
              "max_30d_rise": round(float((px / px.shift(30) - 1).max()), 4)}
out["liquidation_check"] = liq
print(json.dumps({k: out[k] for k in ("gross_funding_ann_by_year", "liquidation_check")}))
json.dump(out, open(OUT, "w"), indent=1)

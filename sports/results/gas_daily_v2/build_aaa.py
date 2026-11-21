"""AAA national daily series from Kalshi KXAAAGASD expiration_value + internal consistency checks. -> aaa_kalshi.parquet"""
import pandas as pd, numpy as np
df = pd.read_parquet("markets.parquet")
df = df[df.event_ticker.str.startswith("KXAAAGASD")].copy()
df["day"] = pd.to_datetime(df.event_ticker.str.split("-").str[1], format="%y%b%d")
df["v"] = pd.to_numeric(df.expiration_value, errors="coerce")
fin = df[df.status == "finalized"]
g = fin.groupby("day")
nv = g.v.nunique()
print("events finalized", fin.event_ticker.nunique(), "days with >1 distinct value", int((nv > 1).sum()), "no value", int(g.v.count().eq(0).sum()))
# result consistency: yes iff v > strike
fin = fin.assign(pred=np.where(fin.v > fin.floor_strike, "yes", "no"))
bad = fin[fin.pred != fin.result]
print("rungs", len(fin), "result != (v > strike):", len(bad))
print(bad[["ticker", "v", "floor_strike", "result"]].head(10).to_string())
ties = fin[np.isclose(fin.v, fin.floor_strike)]
print("ties v == strike:", len(ties), ties[["ticker","v","result"]].head().to_string())
s = g.agg(v=("v", "first"), open=("open_time", "min"), close=("close_time", "max"), settle=("settlement_ts", "max"),
          nrungs=("ticker", "size"), lo=("floor_strike", "min"), hi=("floor_strike", "max"),
          vol=("volume_fp", lambda x: pd.to_numeric(x, errors="coerce").sum())).reset_index()
for c in ("open", "close", "settle"):
    s[c] = pd.to_datetime(s[c], utc=True, format="ISO8601")
s["decimals"] = s.v.map(lambda x: len(f"{x:.6f}".rstrip("0").split(".")[1]))
s["prev_day"] = s.day.shift(1)
s["gap"] = (s.day - s.prev_day).dt.days
s["c"] = s.v.diff().where(s.gap == 1)
s.to_parquet("aaa_kalshi.parquet", index=False)
print(s.day.min().date(), s.day.max().date(), len(s), "gaps>1:", s[s.gap > 1][["day", "gap"]].to_string())
print("decimals", s.decimals.value_counts().to_dict())
s["close_et"] = s.close.dt.tz_convert("America/New_York").dt.strftime("%H:%M")
s["open_et"] = s.open.dt.tz_convert("America/New_York").dt.strftime("%H:%M")
s["settle_et"] = s.settle.dt.tz_convert("America/New_York").dt.strftime("%H:%M")
print("open ET", s.open_et.value_counts().head(5).to_dict()); print("close ET", s.close_et.value_counts().head(5).to_dict())
print("settle ET quantiles", s.settle.dt.tz_convert("America/New_York").dt.hour.describe().to_dict())
print("rungs", s.nrungs.describe().to_dict())
print("daily change c (cents) describe:", (s.c * 100).describe().round(3).to_dict())
print("abs c quantiles cents", (s.c.abs() * 100).quantile([.5, .9, .99]).round(3).to_dict())
print("lag-1 autocorr of c", round(s.c.autocorr(1), 3), "lag2", round(s.c.autocorr(2), 3), "lag7", round(s.c.autocorr(7), 3))
print("rung step", fin.groupby("day").floor_strike.apply(lambda x: np.round(np.diff(np.sort(x.unique())), 4).min() if len(x) > 1 else np.nan).value_counts().head().to_dict())

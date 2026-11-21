"""Look at the rows behind TAIL risk (cheap side WON) for one KXHIGH series, last N days, final-24h window."""
import json, sys, time
sys.path.insert(0, '.')
from tailpool import get, ts
import pandas as pd
series, days = sys.argv[1], int(sys.argv[2]); now = time.time(); t0 = now - days*86400
mk, cur = [], None
while True:
    p = {"series_ticker": series, "status": "settled", "min_close_ts": int(t0), "limit": 1000}
    if cur: p["cursor"] = cur
    d = get("/markets", p); mk += d.get("markets", []); cur = d.get("cursor")
    if not cur or not d.get("markets"): break
rows = []
for m in mk:
    if float(m.get("volume_fp") or 0) <= 0: continue
    c = ts(m["close_time"]); cur = None
    while True:
        p = {"ticker": m["ticker"], "min_ts": int(c-86400), "max_ts": int(c), "limit": 1000}
        if cur: p["cursor"] = cur
        d = get("/markets/trades", p); tr = d.get("trades", [])
        for t in tr:
            y = float(t["yes_price_dollars"]); n = float(t["count_fp"])
            pw = (1-y) if m["result"] == "no" else y
            rows.append(dict(ticker=m["ticker"], result=m["result"], exp=m.get("expiration_value"), close=m["close_time"],
                             t=t["created_time"], hrs_to_close=round((c-ts(t["created_time"]))/3600, 2), yes=y, n=n,
                             taker=t["taker_side"], cheap_won=pw <= 0.15, payout=n*(1-pw) if pw <= 0.15 else 0.0))
        cur = d.get("cursor")
        if not cur or not tr: break
df = pd.DataFrame(rows); df.to_parquet(f"riskrows_{series}.parquet")
w = df[df.cheap_won]
print(series, "trades", len(df), "cheap-won trades", len(w), "payout", round(w.payout.sum()))
g = w.groupby(["ticker", "result", "exp"]).agg(payout=("payout", "sum"), n=("n", "sum"), first=("hrs_to_close", "max"),
      last=("hrs_to_close", "min"), minyes=("yes", "min"), maxyes=("yes", "max")).sort_values("payout", ascending=False)
print(g.head(15).to_string())
print("payout by hrs_to_close bin:", w.groupby(pd.cut(w.hrs_to_close, [0, 2, 6, 12, 18, 24])).payout.sum().round().to_dict())

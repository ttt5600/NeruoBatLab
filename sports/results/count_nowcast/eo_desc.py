"""Descriptive (no money trial): KXEOWEEK repricing vs signing date, and NO-side price after week end (dev weeks)."""
import json, numpy as np, pandas as pd
from pathlib import Path
R = Path("results/count_nowcast")
fr = pd.DataFrame(json.load(open(R / "fr_eo.json"))).sort_values("publication_date").drop_duplicates("executive_order_number")
fr["sd"] = pd.to_datetime(fr.signing_date)
cc = json.load(open(R / "eo_candles.json"))
f = pd.read_parquet(R / "eo1_frame.parquet")
m = pd.read_parquet(R / "eo_rungs.parquet").set_index("ticker")
def q(tk):
    cs = cc.get(tk, [])
    return pd.DataFrame({"t": pd.to_datetime([x["end_period_ts"] for x in cs], unit="s", utc=True),
                         "ask": [float(x["yes_ask"]["close"] or "nan") for x in cs], "bid": [float(x["yes_bid"]["close"] or "nan") for x in cs]})
out = []
for r in f.itertuples():
    ws = m.loc[r.ticker, "ws"]; e = fr[(fr.sd >= ws) & (fr.sd <= r.we)].sort_values("sd")
    qq = q(r.ticker)
    if r.n > r.k:   # determined YES by FR: when did bid first reach .90 vs signing day (00:00 ET ~ 05:00Z) of the (k+1)th EO
        s = e.sd.iloc[int(r.k)].tz_localize("UTC") + pd.Timedelta(hours=5)
        hit = qq[qq.bid >= 0.90]
        out.append(dict(ticker=r.ticker, side="yes", sign=s, first_bid90=hit.t.min() if len(hit) else pd.NaT))
    else:          # NO by FR: YES quote at Sunday 17:00Z after week end
        T = r.we.tz_localize("UTC") + pd.Timedelta(days=1, hours=17)
        w = qq[qq.t <= T]
        out.append(dict(ticker=r.ticker, side="no", T=T, ask=w.ask.iloc[-1] if len(w) else np.nan, bid=w.bid.iloc[-1] if len(w) else np.nan,
                        result=r.result))
o = pd.DataFrame(out)
y = o[o.side == "yes"].copy(); y["h"] = (y.first_bid90 - y.sign).dt.total_seconds() / 3600
print("YES-determined rungs", len(y), "hours from signing-day 00:00ET to first bid>=.90:", y.h.describe().round(1).to_dict())
print("  share repriced before signing day (bid>=.90 earlier: lower rung or expectation):", round((y.h < 0).mean(), 3), " within 24h:", round((y.h <= 24).mean(), 3))
n = o[o.side == "no"]
print("NO-by-FR rungs", len(n), "result yes (FR mismatch):", (n.result == "yes").sum())
print("YES ask at Sun 17Z after week end:", n.ask.describe().round(3).to_dict())
print("YES bid at Sun 17Z:", n.bid.describe().round(3).to_dict())
print("NO-side buyable (YES bid >= .03 -> NO ask <= .97):", (n.bid >= 0.03).sum(), "; YES bid >= .10:", (n.bid >= 0.10).sum())
o.to_parquet(R / "eo_desc.parquet")

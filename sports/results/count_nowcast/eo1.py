"""EO1-Y: buy YES on KXEOWEEK rungs already determined by Federal Register publications (dev weeks only)."""
import sys, json, math
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from kx_throttle import kget
R = Path("results/count_nowcast"); E = pd.Timestamp("1970-01-01", tz="UTC")
DEV_END = pd.Timestamp("2026-08-31")
fr = pd.DataFrame(json.load(open(R / "fr_eo.json")))
fr = fr.sort_values("publication_date").drop_duplicates("executive_order_number", keep="first")   # corrections republish
fr["sd"] = pd.to_datetime(fr.signing_date); fr["T"] = pd.to_datetime(fr.publication_date).dt.tz_localize("UTC") + pd.Timedelta(hours=12)
m = pd.read_parquet(R / "eo_rungs.parquet"); m["k"] = m.k.fillna(0)
m["close_time"] = pd.to_datetime(m.close_time, utc=True, format="ISO8601"); m["open_time"] = pd.to_datetime(m.open_time, utc=True, format="ISO8601")
m = m[m.we <= DEV_END].copy()
print("dev rungs", len(m), "events", m.event_ticker.nunique(), m.we.min().date(), m.we.max().date())
cache = R / "eo_candles.json"
cc = json.loads(cache.read_text()) if cache.exists() else {}
for tk, o, c in zip(m.ticker, m.open_time, m.close_time):
    if tk in cc: continue
    p = {"start_ts": (o - E) // pd.Timedelta(seconds=1) - 3600, "end_ts": (c - E) // pd.Timedelta(seconds=1) + 3600, "period_interval": 60}
    cs = (kget(f"/series/KXEOWEEK/markets/{tk}/candlesticks", p) or {}).get("candlesticks", [])
    if not cs: cs = (kget(f"/historical/markets/{tk}/candlesticks", p) or {}).get("candlesticks", [])
    cc[tk] = cs
cache.write_text(json.dumps(cc))
def quotes(tk):
    cs = cc.get(tk, [])
    if not cs: return pd.DataFrame(columns=["t", "ask", "bid"])
    q = pd.DataFrame({"t": pd.to_datetime([x["end_period_ts"] for x in cs], unit="s", utc=True),
                      "ask": [float(x["yes_ask"]["close"]) if x["yes_ask"]["close"] is not None else np.nan for x in cs],
                      "bid": [float(x["yes_bid"]["close"]) if x["yes_bid"]["close"] is not None else np.nan for x in cs]})
    return q
fee = lambda p: math.ceil(round(0.07 * p * (1 - p) * 100, 6)) / 100
rows = []
for r in m.itertuples():
    e = fr[(fr.sd >= r.ws) & (fr.sd <= r.we)].sort_values("T")
    if len(e) <= r.k: Tk = pd.NaT
    else: Tk = e["T"].iloc[int(r.k)]
    q = quotes(r.ticker)
    row = dict(ticker=r.ticker, ev=r.event_ticker, we=r.we, k=r.k, n=len(e), result=r.result, close=r.close_time, Tk=Tk, status="not_determined")
    if pd.notna(Tk):
        if r.close_time <= Tk: row["status"] = "closed_before_T"
        else:
            w = q[(q.t > Tk - pd.Timedelta(hours=6)) & (q.t <= Tk + pd.Timedelta(hours=1))].dropna(subset=["ask"])
            w2 = q[q.t <= Tk + pd.Timedelta(hours=1)].dropna(subset=["ask"])        # sensitivity: last quote at any age
            row["ask_any"] = w2.ask.iloc[-1] if len(w2) else np.nan
            if not len(w): row["status"] = "no_candle"
            else:
                a = w.ask.iloc[-1]; row["ask"] = a
                row["status"] = "bet" if 0.50 <= a <= 0.98 else ("priced" if a > 0.98 else "ask<.50")
    rows.append(row)
f = pd.DataFrame(rows); f.to_parquet(R / "eo1_frame.parquet")
print(f.status.value_counts().to_string())
print("hours from T_k to close (determined rungs):", ((f.close - f.Tk).dt.total_seconds() / 3600).describe().round(1).to_dict())
print("ask_any at T_k for determined+open rungs:", f.ask_any.describe().round(3).to_dict())
b = f[f.status == "bet"].copy()
if len(b):
    b["cost"] = b.ask + b.ask.map(fee); b["pnl"] = (b.result == "yes") - b.cost
    print(b[["ticker", "k", "n", "Tk", "ask", "cost", "result", "pnl"]].to_string())
    print("bets", len(b), "ROI", round(b.pnl.sum() / b.cost.sum(), 4), "losses", (b.result != "yes").sum(),
          "market-null p", round(float(np.prod(b.ask)), 4))

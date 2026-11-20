"""Outcome-free capacity check: how often would a 1c-inside maker on one-sided far KXTSAD rungs (at T) have been hit?"""
import json
from pathlib import Path
import pandas as pd
OUT = Path(__file__).resolve().parent
t = pd.read_parquet(OUT / "tsa_dev_quotes.parquet")
tr = json.load(open(OUT / "tsa_trades_dev.json"))
rows = []
for r in t.itertuples():
    ao = pd.notna(r.ask) and 0 < r.ask < 1 and not (pd.notna(r.bid) and r.bid > 0)
    bo = pd.notna(r.bid) and r.bid > 0 and not (pd.notna(r.ask) and 0 < r.ask < 1)
    if not (ao or bo): continue
    x = pd.DataFrame(tr.get(r.ticker, []))
    n = c = 0
    if len(x):
        x["ts"] = pd.to_datetime(x.created_time, format="ISO8601"); x["yp"] = x.yes_price_dollars.astype(float); x["ct"] = x.count_fp.astype(float)
        x = x[(x.ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1) > r.T]
        if ao: h = x[(x.taker_side == "yes") & (x.yp >= r.ask - 0.01 - 1e-9)]
        else: h = x[(x.taker_side == "no") & (x.yp <= r.bid + 0.01 + 1e-9)]
        n, c = len(h), h.ct.sum()
    rows.append(dict(ticker=r.ticker, event=r.event_ticker, kind="ask_only" if ao else "bid_only", px=r.ask if ao else r.bid, hit=n > 0, trades=n, contracts=c))
f = pd.DataFrame(rows)
print(f.groupby("kind").agg(rungs=("hit", "size"), events=("event", "nunique"), filled_rungs=("hit", "sum"), contracts=("contracts", "sum")).to_string())
print(f[f.hit].groupby("kind").px.describe().round(3).to_string())

"""Probe: trade tape around close for a few DEV decision fights (read-only)."""
import sys, json
from pathlib import Path
import pandas as pd
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parents[1] / "scripts"))
from kx_throttle import kget
j = pd.read_parquet(OUT / "fights.parquet")
d = j[j.dec & (j.card < "2026-07-01")].sort_values("card")
def tape(tk, t0, t1):
    out = []
    for path in ("/markets/trades", "/historical/trades"):
        cur = None
        while True:
            p = {"ticker": tk, "min_ts": t0, "max_ts": t1, "limit": 1000}
            if cur: p["cursor"] = cur
            r = kget(path, p) or {}
            out += r.get("trades", []); cur = r.get("cursor")
            if not cur or not r.get("trades"): break
        if out: return out, path
    return out, None
for r in d.iloc[[5, 60, 150]].itertuples():
    ct = int(r.ct.timestamp())
    tr, path = tape(r.tA, ct - 40 * 60, ct + 300)
    print("=====", r.event_ticker, r.tA, r.nameA, "winner", r.winner, r.METHOD, r.fmt, "ct", r.ct, "st", r.st, path, len(tr))
    df = pd.DataFrame(tr)
    if not len(df): continue
    df["t"] = pd.to_datetime(df.created_time, format="ISO8601")
    df["px"] = df.yes_price_dollars.astype(float); df["n"] = df.count_fp.astype(float)
    df = df.sort_values("t")
    df["rel"] = (df.t - r.ct).dt.total_seconds()
    g = df.groupby((df.rel // 30) * 30).agg(n=("n", "sum"), k=("px", "size"), first=("px", "first"), last=("px", "last"), lo=("px", "min"), hi=("px", "max"))
    print(g.tail(40).to_string())
    print(df.columns.tolist())

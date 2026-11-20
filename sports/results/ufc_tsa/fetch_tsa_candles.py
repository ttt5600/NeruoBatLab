"""Hourly candles open..close for DEV KXTSAD markets (event date <= 2026-11-05). Read-only, restartable."""
import json, sys
from pathlib import Path
import pandas as pd
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parents[1] / "scripts")); sys.path.insert(0, str(OUT))
from kx_throttle import kget
from census import load
CACHE = OUT / "candles_tsad_dev.jsonl"
t = load("KXTSAD")
t = t[t.status == "finalized"].copy()
t["date"] = pd.to_datetime(t.event_ticker.str.split("-").str[1], format="%y%b%d")
t = t[t.date <= pd.Timestamp("2026-11-05")]
print("markets", len(t), "events", t.event_ticker.nunique(), flush=True)
done = {json.loads(l)["ticker"] for l in CACHE.open()} if CACHE.exists() else set()
E = pd.Timestamp("1970-01-01", tz="UTC")
with CACHE.open("a") as f:
    for r in t.itertuples():
        if r.ticker in done: continue
        p = {"start_ts": int((r.ot - E) // pd.Timedelta(seconds=1)), "end_ts": int((r.ct - E) // pd.Timedelta(seconds=1)), "period_interval": 60}
        c = (kget(f"/series/KXTSAD/markets/{r.ticker}/candlesticks", p) or {}).get("candlesticks") or []
        f.write(json.dumps({"ticker": r.ticker, "c": c}) + "\n")
print("done")

"""Hourly candles for each KXCS2GAME team market over [start-8h, start+1h]. Read-only; restartable (jsonl cache)."""
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from kx_throttle import kget
OUT = Path(__file__).resolve().parent
CACHE = OUT / "candles_raw.jsonl"
d = pd.read_parquet(OUT / "markets_KXCS2GAME.parquet")
ev = d.event_ticker.str.split("-").str[1]
d["start"] = pd.to_datetime(ev.str[:11], format="%y%b%d%H%M", errors="coerce").dt.tz_localize(
    "America/New_York", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
d["src"] = d["_src"]
d = d[d.start >= pd.Timestamp("2026-04-14", tz="UTC")].sort_values("start")
if "--dev-eligible" in sys.argv:   # dev window, events where both teams have >= 10 rated series results (preds.parquet)
    q = pd.read_parquet(OUT / "preds.parquet")
    q = q[(q.start >= pd.Timestamp("2026-04-15", tz="UTC")) & (q.start < pd.Timestamp("2026-10-19 13:45", tz="UTC")) & (q.n_series >= 10)]
    d = d[d.ticker.isin(set(q.ta) | set(q.tb))]
if "--sealed" in sys.argv:         # CS2F sealed window (prereg notes/cs2_model.md), all two-market events
    d = d[(d.start >= pd.Timestamp("2026-10-19 13:45", tz="UTC")) & (d.start < pd.Timestamp("2026-11-16", tz="UTC"))]
done = set()
if CACHE.exists():
    for line in CACHE.open():
        done.add(json.loads(line)["ticker"])
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
print("todo", len(d) - len(done & set(d.ticker)), flush=True)
with CACHE.open("a") as f:
    for i, r in enumerate(d.itertuples()):
        if r.ticker in done:
            continue
        s0 = int((r.start - EPOCH) // pd.Timedelta(seconds=1))
        p = {"start_ts": s0 - 8 * 3600, "end_ts": s0 + 3600, "period_interval": 60}
        paths = ([f"/historical/markets/{r.ticker}/candlesticks"] if r.src.startswith("/historical") else []) + \
                [f"/series/KXCS2GAME/markets/{r.ticker}/candlesticks"]
        c = []
        for path in paths:
            j = kget(path, p) or {}
            c = j.get("candlesticks") or []
            if c:
                break
        f.write(json.dumps({"ticker": r.ticker, "start_ts": s0, "c": c}) + "\n")
        if i % 500 == 0:
            f.flush(); print(i, r.start, len(c), flush=True)
print("done")

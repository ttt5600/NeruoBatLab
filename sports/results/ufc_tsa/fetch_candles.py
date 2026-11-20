"""Hourly candles over [T-6h, T] for DEV UFC markets (MOV, VICROUND, and FIGHT for the same fights). Read-only, restartable.
Adapted from results/kalshi_fg_markets/fetch_candles.py. SEALED (card >= 2026-08-01) is never fetched here."""
import json, sys
from pathlib import Path
import pandas as pd
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parents[1] / "scripts"))
from kx_throttle import kget
CACHE = OUT / "candles_ufc_dev.jsonl"
df = pd.read_parquet(OUT / "ufc_markets.parquet")
df = df[(df.split == "dev") & (df.ct > df.Tdec) & (df.ot < df.Tdec)]
fights = set(df[df.series != "KXUFCFIGHT"].fight)
df = df[(df.series != "KXUFCFIGHT") | df.fight.isin(fights)]
assert (df.card < pd.Timestamp("2026-08-01")).all()
df = df.assign(_o=df.series.map({"KXUFCFIGHT": 0, "KXUFCMOV": 1, "KXUFCVICROUND": 2})).sort_values(["_o", "card"])
print("markets to fetch", len(df), df.groupby("series").size().to_dict(), flush=True)
done = {json.loads(l)["ticker"] for l in CACHE.open()} if CACHE.exists() else set()
with CACHE.open("a") as f:
    for i, r in enumerate(df.itertuples()):
        if r.ticker in done: continue
        T = int((r.Tdec - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1))
        p = {"start_ts": T - 6 * 3600, "end_ts": T, "period_interval": 60}
        paths = ([f"/historical/markets/{r.ticker}/candlesticks"] if r.src_hist else []) + [f"/series/{r.series}/markets/{r.ticker}/candlesticks"]
        c = []
        for path in paths:
            c = (kget(path, p) or {}).get("candlesticks") or []
            if c: break
        f.write(json.dumps({"ticker": r.ticker, "T": T, "c": c}) + "\n"); f.flush()
        if i % 200 == 0: f.flush(); print(i, r.card, len(c), flush=True)
print("done", flush=True)

"""DEV decision fights only (card < 2026-07-01). Read-only GETs through kx_throttle. Restartable (jsonl caches).
Per fight: Kalshi milestone (start_date = Sportradar fight start), winner-market tapes (both tickers) on [ct-35m, ct+60s],
KXUFCDISTANCE market + tape (2026+), 1-min candles of both winner tickers on [ct-35m, ct+2m]."""
import json, sys
from pathlib import Path
import pandas as pd
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parents[1] / "scripts"))
from kx_throttle import kget
SPLIT = pd.Timestamp("2026-07-01")
j = pd.read_parquet(OUT / "fights.parquet")
d = j[j.dec & (j.card < SPLIT)].sort_values("card")
assert (d.card < SPLIT).all()
print("dev decision fights", len(d), flush=True)
# DISTANCE markets (all; metadata only)
dist = {}
for path in ("/markets", "/historical/markets"):
    cur = None
    while True:
        p = {"series_ticker": "KXUFCDISTANCE", "limit": 1000}
        if cur: p["cursor"] = cur
        r = kget(path, p) or {}
        for m in r.get("markets", []): dist.setdefault(m["event_ticker"], m)
        cur = r.get("cursor")
        if not cur or not r.get("markets"): break
json.dump(dist, open(OUT / "dist_markets.json", "w"))
print("dist events", len(dist), flush=True)
def tape(tk, t0, t1):
    for path in ("/markets/trades", "/historical/trades"):
        out, cur = [], None
        while True:
            p = {"ticker": tk, "min_ts": t0, "max_ts": t1, "limit": 1000}
            if cur: p["cursor"] = cur
            r = kget(path, p) or {}
            out += r.get("trades", []); cur = r.get("cursor")
            if not cur or not r.get("trades"): break
        if out: return out, path
    return [], None
def candles(series, tk, t0, t1):
    for path in (f"/historical/markets/{tk}/candlesticks", f"/series/{series}/markets/{tk}/candlesticks"):
        c = (kget(path, {"start_ts": t0, "end_ts": t1, "period_interval": 1}) or {}).get("candlesticks") or []
        if c: return c
    return []
CACHE = OUT / "dev_raw.jsonl"
done = {json.loads(l)["event_ticker"] for l in CACHE.open()} if CACHE.exists() else set()
with CACHE.open("a") as f:
    for i, r in enumerate(d.itertuples()):
        if r.event_ticker in done: continue
        ct = int(r.ct.timestamp())
        rec = {"event_ticker": r.event_ticker, "ct": ct}
        ms = (kget("/milestones", {"related_event_ticker": r.event_ticker, "limit": 5}) or {}).get("milestones", [])
        rec["milestone"] = ms[0] if ms else None
        for k, tk in (("A", r.tA), ("B", r.tB)):
            tr, path = tape(tk, ct - 35 * 60, ct + 60)
            rec[f"tape{k}"] = tr; rec[f"path{k}"] = path
            rec[f"cand{k}"] = candles("KXUFCFIGHT", tk, ct - 35 * 60, ct + 120)
        de = "KXUFCDISTANCE-" + r.fight
        if de in dist:
            m = dist[de]; rec["dist_ticker"] = m["ticker"]; rec["dist_close"] = m.get("close_time"); rec["dist_result"] = m.get("result")
            rec["tapeD"], _ = tape(m["ticker"], ct - 35 * 60, ct + 60)
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(i, r.event_ticker, len(rec["tapeA"]), len(rec["tapeB"]), len(rec.get("tapeD", [])), flush=True)
print("done", flush=True)

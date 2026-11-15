"""READ-ONLY census of Kalshi '<asset> after economic events' ladders (notes/econ_event_ladders.md). Public GETs only."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from kx_throttle import kget

FAM = ["KXGOLDEVENT","KXSILVEREVENT","KXCOPPEREVENT","KXBTCEVENT","KXETHEVENT","KXSOLEVENT",
       "KXEURUSDEVENT","KXGBPUSDEVENT","KXUSDJPYEVENT","KXAUDUSDEVENT","KXUSDCADEVENT","KXUSDMXNEVENT",
       "KX2YRRATEEVENT","KX5YRRATEEVENT","KX10YRRATEEVENT","KX30YRRATEEVENT"]
OUT = Path(__file__).parent.parent / "results" / "econ_event_ladders"
OUT.mkdir(parents=True, exist_ok=True)
out = {}
for s in FAM:
    ser = kget(f"/series/{s}") or {}
    evs, cur = [], None
    while True:
        p = {"series_ticker": s, "limit": 200, "with_nested_markets": "true"}
        if cur:
            p["cursor"] = cur
        r = kget("/events", p) or {}
        evs += r.get("events", [])
        cur = r.get("cursor")
        if not cur:
            break
    hist = kget("/historical/markets", {"series_ticker": s, "limit": 100}) or {}
    out[s] = {"series": ser.get("series", ser), "events": evs, "hist_markets": hist.get("markets", [])}
    print(s, len(evs), "events", sum(len(e.get("markets", [])) for e in evs), "mkts",
          len(hist.get("markets", [])), "hist", flush=True)
(OUT / "census_raw.json").write_text(json.dumps(out))

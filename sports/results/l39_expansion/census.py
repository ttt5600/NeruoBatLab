"""Census of candidate 'mathematically decidable' Kalshi weather series (read-only GETs via kx_throttle)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from kx_throttle import kget

CANDS = sys.argv[1].split(',')
out = {}
for s in CANDS:
    ser = kget(f"/series/{s}") or {}
    evs = []
    for st in ("settled", "open", "closed"):
        cur = None
        while True:
            p = {"series_ticker": s, "limit": 200, "status": st, "with_nested_markets": "true"}
            if cur:
                p["cursor"] = cur
            d = kget("/events", p) or {}
            evs += d.get("events", [])
            cur = d.get("cursor")
            if not cur or not d.get("events"):
                break
    rows = []
    for e in evs:
        ms = e.get("markets", [])
        vol = sum(float(m.get("volume_fp", m.get("volume", 0)) or 0) for m in ms)
        rows.append({"event": e["event_ticker"], "n_mk": len(ms), "vol": vol,
                     "status": sorted({str(m.get("status")) for m in ms}),
                     "strikes": sorted({(m.get("strike_type"), m.get("floor_strike"), m.get("cap_strike")) for m in ms}, key=str)[:14],
                     "close": max((m.get("close_time") or "") for m in ms) if ms else None})
    rules = None
    for e in evs:
        if e.get("markets"):
            rules = e["markets"][0].get("rules_primary")
            break
    out[s] = {"title": (ser.get("series") or {}).get("title"), "n_events": len(rows), "events": rows, "rules": rules,
              "settlement_sources": (ser.get("series") or {}).get("settlement_sources")}
    tot = sum(r["vol"] for r in rows)
    print(s, len(rows), "events, vol", round(tot), "|", (rules or "")[:220].replace("\n", " "), flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1, default=str)

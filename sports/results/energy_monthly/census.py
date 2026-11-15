"""energy_monthly census: series metadata + events per status (read-only Kalshi GETs via kx_throttle)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from kx_throttle import kget  # noqa: E402

OUT = Path(__file__).resolve().parent
SER = ["KXERCOTHOUM", "KXERCOTNORTHM", "KXERCOTWESTM", "KXTTFMON", "KXNATGASD", "KXBRENTD"]
res = {}
for s in SER:
    ser = kget(f"/series/{s}")
    res[s] = {"series": (ser or {}).get("series", {}), "events": {}}
    print("=====", s, json.dumps(res[s]["series"], default=str)[:700])
    for st in ["settled", "open", "closed"]:
        evs, cur = [], None
        while True:
            p = {"series_ticker": s, "status": st, "limit": 200, "with_nested_markets": "true"}
            if cur:
                p["cursor"] = cur
            ev = kget("/events", p) or {}
            evs += ev.get("events", [])
            cur = ev.get("cursor")
            if not cur or not ev.get("events"):
                break
        res[s]["events"][st] = evs
        print(st, len(evs), [e["event_ticker"] for e in evs][:8])
(OUT / "census.json").write_text(json.dumps(res, default=str))

"""rules_quirks: pull rules text for every open Kalshi series (read-only GETs via kx_throttle).
Output results/rules_quirks/series.json (all series meta) and rules_by_series.jsonl (one sample market per series).
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from kx_throttle import kget

OUT = Path(__file__).resolve().parents[1] / "results" / "rules_quirks"
OUT.mkdir(parents=True, exist_ok=True)

ser = kget("/series", {})
json.dump(ser, open(OUT / "series.json", "w"))
print("series", len(ser.get("series", [])), flush=True)

seen = {}
cur = None
pages = 0
while True:
    p = {"status": "open", "limit": 200, "with_nested_markets": "true"}
    if cur:
        p["cursor"] = cur
    r = kget("/events", p)
    pages += 1
    for ev in r.get("events", []):
        st = ev.get("series_ticker")
        mk = ev.get("markets") or []
        if not mk:
            continue
        rec = seen.setdefault(st, {"series_ticker": st, "n_events": 0, "n_markets": 0, "vol24": 0.0,
                                   "title": ev.get("title"), "category": ev.get("category"),
                                   "rules_primary": mk[0].get("rules_primary"),
                                   "rules_secondary": mk[0].get("rules_secondary"),
                                   "sample_market": mk[0].get("ticker")})
        rec["n_events"] += 1
        rec["n_markets"] += len(mk)
        for m in mk:
            try:
                rec["vol24"] += float(m.get("volume_24h_fp") or m.get("volume_24h") or 0)
            except Exception:
                pass
    cur = r.get("cursor")
    if pages % 20 == 0:
        print("pages", pages, "series", len(seen), flush=True)
        with open(OUT / "rules_by_series.jsonl", "w") as f:
            for v in seen.values():
                f.write(json.dumps(v) + "\n")
    if not cur or not r.get("events"):
        break
with open(OUT / "rules_by_series.jsonl", "w") as f:
    for v in seen.values():
        f.write(json.dumps(v) + "\n")
print("done pages", pages, "series", len(seen))

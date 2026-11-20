"""Read-only: series metadata + all markets (live /markets and /historical/markets) for 4 series."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from kx_throttle import kget
OUT = Path(__file__).resolve().parent
res = {}
for s in ["KXUFCMOV", "KXUFCVICROUND", "KXUFCFIGHT", "KXTSAD"]:
    meta = kget(f"/series/{s}")
    out = {"series": meta, "markets": [], "hist_markets": []}
    for key, path in (("markets", "/markets"), ("hist_markets", "/historical/markets")):
        cur = None
        while True:
            p = {"series_ticker": s, "limit": 1000}
            if cur: p["cursor"] = cur
            d = kget(path, p) or {}
            out[key] += d.get("markets", [])
            cur = d.get("cursor")
            if not cur or not d.get("markets"): break
    print(s, len(out["markets"]), len(out["hist_markets"]), flush=True)
    res[s] = out
json.dump(res, open(OUT / "census_raw.json", "w"))

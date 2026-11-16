"""Read-only: fetch every market (all statuses) for the Kalshi *HOME (Zillow ZHVI) series."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
from kx_throttle import kget

ser = [x["ticker"] for x in json.load(open(HERE / "series.json"))]
out = []
for s in ser:
    cur = None
    while True:
        p = {"series_ticker": s, "limit": 1000}
        if cur:
            p["cursor"] = cur
        r = kget("/markets", p)
        out += r["markets"]
        cur = r.get("cursor")
        if not cur:
            break
json.dump(out, open(HERE / "markets.json", "w"))
print(len(out))

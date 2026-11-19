"""Read-only census of Kalshi series touching mortgage rates / treasuries (agent mortgage_rates, 2026-11-18)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from kx_throttle import kget  # noqa: E402

OUT = Path(__file__).resolve().parent
KEYS = ["mort", "pmms", "freddie", "30-year", "30 year", "fixed", "treasury", "10y", "10-year", "home", "housing"]
s = kget("/series", {})
ser = s.get("series", []) if s else []
print("n series", len(ser))
hits = [x for x in ser if any(k in (x.get("ticker", "") + " " + x.get("title", "")).lower() for k in KEYS)]
for h in hits:
    print(h["ticker"], "|", h.get("title"), "|", h.get("frequency"), "|", h.get("category"))
json.dump(hits, open(OUT / "series_hits.json", "w"), indent=1)

"""Read-only: list Kalshi SNOW / RAIN monthly series, their open markets' rules station text."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from sports import wx  # noqa: E402
from sports.http import fetch  # noqa: E402

d = fetch(f"{wx.API}/series", {"limit": 10000}, cache=False)["series"]
snow = sorted(x["ticker"] for x in d if "SNOW" in x["ticker"])
print(len(snow), snow)
out = {}
for s in snow:
    m = fetch(f"{wx.API}/markets", {"series_ticker": s, "status": "open", "limit": 50}, cache=False).get("markets", [])
    if not m:
        continue
    ev = sorted({x["event_ticker"] for x in m})
    r = m[0].get("rules_primary", "")
    out[s] = {"events": ev, "rules": r}
    print(f"{s:16s} {ev} | {r[:260]}")
json.dump(out, open(Path(__file__).parent / "snow_open.json", "w"), indent=1)

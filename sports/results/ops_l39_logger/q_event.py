"""Read-only: dump event + one market JSON for a snow event, looking for the station name."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sports import wx  # noqa: E402
from sports.http import fetch  # noqa: E402

for ev in sys.argv[1:]:
    e = fetch(f"{wx.API}/events/{ev}", {"with_nested_markets": "true"}, cache=False)
    evd = e.get("event", {})
    mk = evd.pop("markets", None) or e.get("markets", [])
    print("==", ev, json.dumps(evd)[:1500])
    if mk:
        m = mk[0]
        print("  market keys:", sorted(m))
        for k, v in m.items():
            if isinstance(v, str) and any(w in v for w in ("KM", "CLI", "Airport", "International", "station")):
                print("  ", k, ":", v[:400])
    se = fetch(f"{wx.API}/series/{ev.split('-')[0]}", cache=False).get("series", {})
    print("  series:", json.dumps(se)[:1200])

"""Read-only census of the KXAAAGASCD* state daily gas-change series (state_gas agent)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from kx_throttle import kget  # noqa: E402

states = ("MS AZ WV LA IL MN CT ID ND VT MO AR MI NM OK NE CO WY OH SD UT WI AL NJ IN PA NV IA TN NC NH OR KS NY KY "
          "HI SC FL VA ME RI MA WA CA MD GA AK TX MT DE").split()
out = {}
for s in states:
    st = "KXAAAGASCD" + s
    ser = kget(f"/series/{st}")
    mk, cur = [], None
    while True:
        p = {"series_ticker": st, "limit": 1000}
        if cur:
            p["cursor"] = cur
        r = kget("/markets", p)
        if not r:
            break
        mk += r.get("markets", [])
        cur = r.get("cursor")
        if not cur:
            break
    out[s] = {"series": (ser or {}).get("series"), "markets": mk}
    print(s, len(mk), sorted({m["status"] for m in mk}), flush=True)
(ROOT / "results/state_gas/census_raw.json").write_text(json.dumps(out))

"""Read-only: list events + markets (structure/volume/settlement only) for mortgage-rate series."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from kx_throttle import kget  # noqa: E402

OUT = Path(__file__).resolve().parent
SERIES = sys.argv[1:] or ["FRM", "KXFRM", "KX30YMORTW", "KXFM30YMTG", "KXMORTGAGERATE", "KXMBAAPPS"]
allm = {}
for s in SERIES:
    ms, cur = [], None
    while True:
        p = {"series_ticker": s, "limit": 1000}
        if cur:
            p["cursor"] = cur
        r = kget("/markets", p)
        if not r:
            break
        ms += r.get("markets", [])
        cur = r.get("cursor")
        if not cur:
            break
    allm[s] = ms
    evs = sorted({m["event_ticker"] for m in ms})
    vol = sum(m.get("volume", 0) or 0 for m in ms)
    st = {}
    for m in ms:
        st[m.get("status")] = st.get(m.get("status"), 0) + 1
    print(f"{s}: {len(evs)} events, {len(ms)} markets, volume {vol}, status {st}")
    if evs:
        print("   first", evs[:3], "last", evs[-3:])
json.dump(allm, open(OUT / "markets.json", "w"), indent=1)

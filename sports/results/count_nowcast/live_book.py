"""Live open-market snapshot per series: spread on tradeable rungs, far-rung neglect share, fee type."""
import sys, json, statistics as st
sys.path.insert(0, "scripts")
from kx_throttle import kget
out = {}
for s in sys.argv[1:]:
    r = kget("/markets", {"series_ticker": s, "status": "open", "limit": 1000}) or {}
    mk = r.get("markets", [])
    sp, far, far0, fees = [], 0, 0, set()
    for m in mk:
        yb, ya = float(m.get("yes_bid_dollars") or 0), float(m.get("yes_ask_dollars") or 0)
        fees.add((m.get("fee_type") if "fee_type" in m else None))
        v24 = float(m.get("volume_24h_fp") or m.get("volume_24h") or 0)
        if 0 < yb and ya < 1 and ya > yb:
            sp.append(ya - yb)
        mid = (yb + ya) / 2 if ya > 0 else yb
        if 0.02 <= mid <= 0.15 or 0.85 <= mid <= 0.98:
            far += 1; far0 += (v24 == 0)
    ser = kget(f"/series/{s}") or {}
    out[s] = dict(n_open=len(mk), med_spread=round(st.median(sp), 3) if sp else None, n_twosided=len(sp),
                  far=far, far_no24h=far0, fee=ser.get("series", {}).get("fee_type"), fee_mult=ser.get("series", {}).get("fee_multiplier"))
    print(s, out[s])
json.dump(out, open("results/count_nowcast/live_book.json", "w"), indent=1)

"""Historical (settled, archived) market counts + current open ladders for candidate series. Read-only."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from kx_throttle import kget

out = {}
for s in sys.argv[1].split(','):
    rows = []
    for path, extra in (("/historical/markets", {}), ("/markets", {"status": "settled"}), ("/markets", {"status": "open"})):
        cur = None
        while True:
            p = {"series_ticker": s, "limit": 1000, **extra}
            if cur:
                p["cursor"] = cur
            d = kget(path, p) or {}
            ms = d.get("markets", [])
            for m in ms:
                m["_src"] = path + str(extra)
            rows += ms
            cur = d.get("cursor")
            if not cur or not ms:
                break
    seen = {}
    for m in rows:
        seen[m["ticker"]] = m
    ms = list(seen.values())
    evs = sorted({m["event_ticker"] for m in ms})
    vol = sum(float(m.get("volume_fp") or m.get("volume") or 0) for m in ms)
    opn = [m for m in ms if m.get("status") in ("active", "open")]
    out[s] = [{k: m.get(k) for k in ("ticker", "event_ticker", "status", "result", "strike_type", "floor_strike",
                                       "cap_strike", "yes_bid_dollars", "yes_ask_dollars", "no_ask_dollars", "yes_bid",
                                       "yes_ask", "volume_fp", "volume", "open_interest_fp", "close_time",
                                       "rules_primary", "_src")} for m in ms]
    print(f"{s}: {len(ms)} mkts, {len(evs)} events, vol {vol:.0f}, open {len(opn)}; events {evs[:3]}..{evs[-3:]}", flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1, default=str)

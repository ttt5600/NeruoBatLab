import sys, json
sys.path.insert(0, "/Users/jonathanwang/Desktop/vocalizations_lab/.claude/worktrees/quant-trader/sports/scripts")
from kx_throttle import kget
t = sys.argv[1] if len(sys.argv) > 1 else "KXTEMPNYCHS"
s = kget(f"/series/{t}", {})
print(json.dumps(s)[:900])
for st in ["settled", "closed"]:
    d = kget("/events", {"series_ticker": t, "status": st, "limit": 3})
    ev = d.get("events") or []
    print(st, len(ev), bool(d.get("cursor")))
    if ev: print(json.dumps(ev[0])[:500])
d = kget("/markets", {"series_ticker": t, "status": "settled", "limit": 2})
for m in d.get("markets") or []:
    print({k: m.get(k) for k in ["ticker", "open_time", "close_time", "created_time", "status"]})

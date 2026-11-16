import sys, json, datetime as dt
sys.path.insert(0, "/Users/jonathanwang/Desktop/vocalizations_lab/.claude/worktrees/quant-trader/sports/scripts")
from kx_throttle import kget
ts = int(dt.datetime(2026, 10, 29, tzinfo=dt.timezone.utc).timestamp())
for t in sys.argv[1:]:
    for st in ["settled", "closed"]:
        d = kget("/markets", {"series_ticker": t, "status": st, "max_close_ts": ts, "limit": 2})
        ms = (d or {}).get("markets") or []
        print(t, st, len(ms), [(m["ticker"], m.get("close_time"), m.get("status")) for m in ms][:2])
    d = kget("/markets", {"series_ticker": t, "max_close_ts": ts, "limit": 2})
    ms = (d or {}).get("markets") or []
    print(t, "nostatus", len(ms), [(m["ticker"], m.get("close_time"), m.get("status")) for m in ms][:2])

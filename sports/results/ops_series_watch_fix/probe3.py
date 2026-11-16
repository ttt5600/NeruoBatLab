import sys, json, datetime as dt
sys.path.insert(0, "/Users/jonathanwang/Desktop/vocalizations_lab/.claude/worktrees/quant-trader/sports/scripts")
from kx_throttle import kget
ts = int(dt.datetime(2026, 10, 29, tzinfo=dt.timezone.utc).timestamp())
for t in sys.argv[1:]:
    d = kget("/markets", {"series_ticker": t, "max_created_ts": ts, "limit": 2})
    ms = (d or {}).get("markets") or []
    print(t, "created<=", len(ms), [(m["ticker"], m.get("created_time"), m.get("open_time"), m.get("status")) for m in ms])
    d = kget("/markets", {"series_ticker": t, "limit": 1000})
    ms = (d or {}).get("markets") or []
    print(t, "all", len(ms), "min created", min((m.get("created_time") or "") for m in ms) if ms else None, bool(d.get("cursor")))

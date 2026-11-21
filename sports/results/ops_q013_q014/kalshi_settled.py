"""Fetch every settled KXTRUTHSOCIAL / KXTRUTHSOCIALD market (public GET) -> per-event window + expiration_value."""
import json, re, sys, time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests
API = "https://api.elections.kalshi.com/trade-api/v2"
UA = "quant-research-logger/1.0 (personal read-only research)"
NY = ZoneInfo("America/New_York")
_WIN = re.compile(r"begins at 12:00:00 AM ET on (\w{3} \d{1,2}, \d{4}) and ends at 11:59:59 PM ET on (\w{3} \d{1,2}, \d{4})")
out = {}
for s in ("KXTRUTHSOCIAL", "KXTRUTHSOCIALD"):
    cursor = None
    for _ in range(50):
        p = {"series_ticker": s, "status": "settled", "limit": 1000}
        if cursor: p["cursor"] = cursor
        r = requests.get(f"{API}/markets", params=p, headers={"User-Agent": UA}, timeout=30)
        r.raise_for_status(); d = r.json()
        for m in d.get("markets", []):
            e = out.setdefault(m["event_ticker"], {"series": s, "markets": []})
            g = _WIN.search(m.get("rules_secondary") or "")
            if g:
                d0 = datetime.strptime(g.group(1), "%b %d, %Y").replace(tzinfo=NY)
                d1 = datetime.strptime(g.group(2), "%b %d, %Y").replace(tzinfo=NY) + timedelta(days=1)
                e["win"] = [int(d0.timestamp()), int(d1.timestamp())]
            e["markets"].append({k: m.get(k) for k in ("ticker", "result", "expiration_value", "strike_type", "floor_strike", "cap_strike", "status", "close_time")})
        cursor = d.get("cursor")
        if not cursor: break
        time.sleep(0.5)
json.dump(out, open(sys.argv[1], "w"), indent=0)
nw = sum(1 for e in out.values() if "win" in e)
print(len(out), "events,", nw, "with parsed window")
from collections import Counter
print(Counter(e["series"] for e in out.values()))

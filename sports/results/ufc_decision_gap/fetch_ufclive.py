"""UFC.com live-stats CDN (public JSON behind ufc.com event pages; ufcLiveStatsApi baseUrl). Read-only, 1 req/s.
Saves event/live/{id}.json for ids in a range to ufclive/ (outcome + FightNightTracking wall-clock timestamps)."""
import json, sys, time
from pathlib import Path
import requests
OUT = Path(__file__).resolve().parent / "ufclive"
S = requests.Session(); S.headers["User-Agent"] = "Mozilla/5.0"
lo, hi = int(sys.argv[1]), int(sys.argv[2])
for i in range(lo, hi):
    p = OUT / f"ev_{i}.json"
    if p.exists() and p.stat().st_size > 1000: continue
    try:
        r = S.get(f"https://d29dxerjsp82wz.cloudfront.net/api/v3/event/live/{i}.json", timeout=30)
        if r.status_code == 200: p.write_bytes(r.content)
        d = r.json()["LiveEventDetail"] if r.status_code == 200 else {}
        print(i, r.status_code, d.get("Name"), d.get("StartTime"), flush=True)
    except Exception as e:
        print(i, "ERR", e, flush=True)
    time.sleep(1.0)

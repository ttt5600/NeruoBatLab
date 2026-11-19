"""pm_full_census: Gamma events CLOSED with endDate in the last 7 days (flow of short-lived markets). Compact on the fly."""
import json, sys, time, requests, pandas as pd
from datetime import datetime, timedelta, timezone
H = {"User-Agent": "Mozilla/5.0 (research; read-only)"}
lo = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
rows, cur = [], None
f = lambda x: float(x) if x not in (None, "") else None
while True:
    p = {"closed": "true", "limit": 500, "end_date_min": lo}
    if cur: p["after_cursor"] = cur
    for a in range(6):
        try:
            r = requests.get("https://gamma-api.polymarket.com/events/keyset", params=p, headers=H, timeout=60)
            if r.status_code == 200: break
        except requests.RequestException: pass
        time.sleep(2 ** a)
    d = r.json(); ev = d.get("events", [])
    for e in ev:
        rows.append(dict(eid=e["id"], title=e.get("title"), tslugs="|".join(t.get("slug") or "" for t in (e.get("tags") or [])),
                         endDate=e.get("endDate"), vol=f(e.get("volume")), n_mkts=len(e.get("markets") or [])))
    cur = d.get("next_cursor"); print(len(rows), file=sys.stderr)
    if not ev or not cur: break
    time.sleep(0.12)
pd.DataFrame(rows).to_parquet("results/pm_full_census/gamma_closed7d.parquet"); print(len(rows), lo)

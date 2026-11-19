"""pm_full_census: PM US CLOSED markets, compact (no descriptions). Read-only."""
import json, sys, time, requests, pandas as pd
H = {"User-Agent": "Mozilla/5.0 (research; read-only)"}
rows = []
import itertools
CATS = sys.argv[1].split(",")
for cat, off in ((c, 0) for c in CATS):
  extra = {"categories": cat} if cat != "sportsfut" else {"categories": "sports", "marketTypes": "futures"}
  while True:
      for a in range(6):
          try:
              r = requests.get("https://gateway.polymarket.us/v1/markets", params={"limit": 500, "offset": off, "closed": "true", **extra}, headers=H, timeout=90)
              if r.status_code == 200: break
          except requests.RequestException: pass
          time.sleep(2 ** a)
      ms = r.json().get("markets", [])
      for m in ms:
          win = None
          for s in m.get("marketSides") or []:
              if s.get("long") is True and s.get("price") in ("0", "1"): win = "yes" if s["price"] == "1" else "no"
          team = None
          for s in m.get("marketSides") or []:
              if s.get("long") is True and s.get("team"): team = s["team"].get("name")
          rows.append(dict(slug=m.get("slug"), question=m.get("question"), title=m.get("title"), category=m.get("category"),
                           sub=m.get("eventSubcategory"), mtype=m.get("marketType"), smt=m.get("sportsMarketType"),
                           start=m.get("startDate"), end=m.get("endDate"), created=m.get("createdAt"), updated=m.get("updatedAt"),
                           status=m.get("status"), win=win, team=team, fee=m.get("feeCoefficient")))
      print(cat, off, len(ms), file=sys.stderr)
      if not ms: break
      off += 500; time.sleep(0.1)
pd.DataFrame(rows).to_parquet(sys.argv[2]); print(len(rows))

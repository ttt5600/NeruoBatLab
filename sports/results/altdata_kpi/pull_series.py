"""Read-only: pull full Kalshi series list, save; find Carbon Arc / KPI / SPR families."""
import json, time, requests, sys
API="https://api.elections.kalshi.com/trade-api/v2"
def get(p, **q):
    for _ in range(6):
        r=requests.get(API+p,params=q,timeout=30)
        if r.status_code==429: time.sleep(2); continue
        r.raise_for_status(); return r.json()
out="results/altdata_kpi/series_all.json"
r=get("/series", include_volume="true")
ser=r.get("series",[])
json.dump(ser,open(out,"w"))
print(len(ser))

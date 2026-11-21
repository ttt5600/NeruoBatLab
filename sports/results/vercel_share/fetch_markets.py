"""Fetch all markets (any status) for given series -> mk/<SERIES>.json. Read-only Kalshi GETs."""
import requests, json, sys, time
API="https://api.elections.kalshi.com/trade-api/v2"
def get(path,params):
    for a in range(6):
        r=requests.get(API+path,params=params,timeout=30)
        if r.status_code==429: time.sleep(2*(a+1)); continue
        r.raise_for_status(); return r.json()
for s in sys.argv[1:]:
    mk,cur=[],None
    while True:
        p={"series_ticker":s,"limit":1000}
        if cur: p["cursor"]=cur
        d=get("/markets",p); mk+=d.get("markets",[]); cur=d.get("cursor")
        if not cur or not d.get("markets"): break
        time.sleep(0.15)
    json.dump(mk,open(f"mk/{s}.json","w"))
    print(s,len(mk),len({m['event_ticker'] for m in mk}),flush=True)

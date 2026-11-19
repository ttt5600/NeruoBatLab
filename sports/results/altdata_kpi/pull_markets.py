"""Read-only: all markets (any status) for listed series -> markets_<fam>.json"""
import json, time, requests, sys
API="https://api.elections.kalshi.com/trade-api/v2"
def get(p, **q):
    for _ in range(8):
        r=requests.get(API+p,params=q,timeout=30)
        if r.status_code==429: time.sleep(2); continue
        r.raise_for_status(); return r.json()
fam=sys.argv[1]; F=json.load(open("results/altdata_kpi/families.json"))
tick=[x[0] for x in F[fam]] if fam in F else sys.argv[2].split(",")
out={}
for t in tick:
    ms=[]; cur=None
    while True:
        q=dict(series_ticker=t,limit=1000)
        if cur: q["cursor"]=cur
        r=get("/markets",**q); ms+=r.get("markets",[]); cur=r.get("cursor")
        if not cur: break
    out[t]=ms; time.sleep(0.15)
json.dump(out,open(f"results/altdata_kpi/markets_{fam}.json","w"))
print(fam,len(out),sum(len(v) for v in out.values()))

"""Fetch Vercel AI Gateway leaderboard data (read-only GETs) and save with fetch timestamp.
- labs export (documented, CC-BY-4.0): /api/ai/leaderboard-export?dataset=labs&from=..&to=..
- weightClass (undocumented, called by the page's own JS): /api/ai/v5/gateway-model-leaderboard?groupBy=weightClass&from=..&to=..
Usage: python fetch_src.py FROM TO OUTDIR"""
import requests, json, sys, time
from datetime import datetime, timezone
f, t, out = sys.argv[1], sys.argv[2], sys.argv[3]
H={"User-Agent":"research-script/0.1 (contact: personal research)"}
now=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
r1=requests.get("https://vercel.com/api/ai/leaderboard-export",params={"dataset":"labs","from":f,"to":t,"format":"json"},headers=H,timeout=60)
r2=requests.get("https://vercel.com/api/ai/v5/gateway-model-leaderboard",params={"groupBy":"weightClass","from":f,"to":t},headers=H,timeout=60)
for name,r in [("labs",r1),("wc",r2)]:
    meta={"fetched_utc":now,"url":r.url,"status":r.status_code,"age":r.headers.get("Age"),"cache":r.headers.get("X-Vercel-Cache"),"date":r.headers.get("Date")}
    json.dump({"meta":meta,"data":r.json()},open(f"{out}/{name}_{f}_{t}_{now}.json","w"))
    print(name, meta)

import json,sys
sys.path.insert(0,'scripts')
from kx_throttle import kget
S=[s for s in json.load(open("results/cbank_thin/cb_series.json")) if s not in ("KXCANOREOY",)]
evs={}
for s in S:
    allv=[];cur=None
    while True:
        p={"series_ticker":s,"limit":200}
        if cur:p["cursor"]=cur
        r=kget("/events",p) or {}
        allv+=r.get("events",[]);cur=r.get("cursor")
        if not cur:break
    evs[s]=allv
    print(s,len(allv),[e["event_ticker"] for e in allv][:3],flush=True)
json.dump(evs,open("results/cbank_thin/events.json","w"),indent=1)

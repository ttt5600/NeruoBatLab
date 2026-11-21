import json,sys
sys.path.insert(0,'scripts')
from kx_throttle import kget
out=[];cur=None
while True:
    p={"limit":1000}
    if cur:p["cursor"]=cur
    r=kget("/series",p) or {}
    out+=r.get("series",[])
    cur=r.get("cursor")
    if not cur:break
print("total series",len(out))
keys=["rate","central bank","interest","policy","monetary","copom","banxico","rbi","repo","selic"]
hits=[s for s in out if s["ticker"].startswith("KXCBD") or any(k in (s.get("title","")+" "+s.get("category","")).lower() for k in keys)]
for s in hits: print(s["ticker"],"|",s.get("title"),"|",s.get("category"),"|",s.get("frequency"))
json.dump(hits,open("results/cbank_thin/series_hits.json","w"),indent=1)

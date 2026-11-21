"""Rebuild every settled Kalshi Vercel-family value from the current Vercel data (labs export + weightClass endpoint).
Kalshi value = share rounded to 1 decimal. Output: rebuild.csv + summary."""
import json, glob, re, sys
import pandas as pd
from datetime import datetime
labsf=sorted(glob.glob("src/labs_2025-10-01_*.json"))[-1]; wcf=sorted(glob.glob("src/wc_2025-10-01_*.json"))[-1]
labs=json.load(open(labsf))["data"]["rows"]; wc=json.load(open(wcf))["data"]
L={(r["date"],r["name"],r["metric"]):r["share_percent"] for r in labs}
W={}
for o in wc:
    for k,v in o["chef_values"]: W[(o["day"][:10],k,o["metric"])]=v
SER={"KXOPENSOURCESHARE":("W","Open Weights","tokens"),"KXOSSPEND":("W","Open Weights","cost"),
 "KXOPENVREQ":("L","openai","requests"),"KXOPENVSPEND":("L","openai","spend"),"KXMOONVSPEND":("L","moonshotai","spend"),
 "KXANTHVSPEND":("L","anthropic","spend"),"KXANTHVREQ":("L","anthropic","requests"),"KXGOOGVREQ":("L","google","requests"),
 "KXDEEPVREQ":("L","deepseek","requests"),"KXGOOGVSPEND":("L","google","spend"),"KXZAIVSPEND":("L","zai","spend"),"KXZAIVREQ":("L","zai","requests")}
rows=[]
for s,(src,name,metric) in SER.items():
    mk=json.load(open(f"mk/{s}.json")); ev={}
    for m in mk:
        if m["status"]=="finalized": ev.setdefault(m["event_ticker"],m)
    for e,m in ev.items():
        d=re.search(r"for (\w+ \d+, \d{4})",m["rules_primary"]).group(1)
        day=datetime.strptime(d,"%b %d, %Y").strftime("%Y-%m-%d")
        kv=m["expiration_value"].replace("%","")
        try: kv=float(kv)
        except: kv=None
        pv=(W if src=="W" else L).get((day,name,metric))
        rows.append(dict(series=s,event=e,day=day,kalshi=kv,page=pv,page_r1=None if pv is None else round(pv+1e-9,1),
                         diff=None if (pv is None or kv is None) else round(round(pv+1e-9,1)-kv,1)))
df=pd.DataFrame(rows).sort_values(["series","day"]); df.to_csv("rebuild.csv",index=False)
df["exact"]=df["diff"].abs()<0.05
print(df.groupby("series").agg(n=("event","size"),kalshi_num=("kalshi",lambda x:x.notna().sum()),page=("page",lambda x:x.notna().sum()),exact=("exact","sum"),
      within05=("diff",lambda x:(x.abs()<=0.5).sum()), maxabs=("diff",lambda x:x.abs().max())))
print("TOTAL exact", int(df.exact.sum()), "of", int(df.kalshi.notna().sum()), "numeric settled")
print(df[~df.exact & df.kalshi.notna()].to_string())

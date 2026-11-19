"""CA1 dev frame: pairs (prev settled value -> this event), far rungs by z; per prereg notes/altdata_kpi.md."""
import json,collections,re,math
import numpy as np, pandas as pd
M=json.load(open("results/altdata_kpi/markets_carbonarc.json"))
rows=[]
for t,ms in M.items():
    if re.search(r"(CCD|FTD)$",t): continue
    ev=collections.defaultdict(list)
    for m in ms: ev[m["event_ticker"]].append(m)
    E=[]
    for e,v in ev.items():
        settled=all(x["status"] in("settled","finalized") for x in v)
        vals={x.get("expiration_value") for x in v if x.get("expiration_value") not in (None,"")}
        val=None
        if settled and len(vals)==1:
            try: val=float(next(iter(vals)))
            except: val=None
        E.append(dict(series=t,event=e,close=min(x["close_time"] for x in v),open=min(x["open_time"] for x in v),settled=settled,val=val,mk=v))
    E.sort(key=lambda d:d["close"])
    for a,b in zip(E,E[1:]):
        if a["val"] is None: continue
        rows.append(dict(series=t,event=b["event"],prev_event=a["event"],v_prev=a["val"],v=b["val"],settled=b["settled"],
                         close=b["close"],open=b["open"],prev_close=a["close"],mk=b["mk"]))
P=pd.DataFrame(rows)
P["close"]=pd.to_datetime(P.close); P["dev"]=(P.close>="2026-11-01")&(P.close<"2026-12-01")&P.settled
print("pairs",len(P),"dev",P.dev.sum(),"dev with v",(P.dev&P.v.notna()).sum())
D=P[P.dev&P.v.notna()].copy()
D["r"]=np.log(D.v/D.v_prev)
s=1.4826*np.median(np.abs(D.r-np.median(D.r)))
print("s(MAD)",round(s,4),"median r",round(np.median(D.r),4),"sd r",round(D.r.std(),4),"n",len(D))
json.dump({"s":s,"n_pairs":int(len(D))},open("results/altdata_kpi/ca1_scale.json","w"))
far=[]
for _,p in P[P.dev].iterrows():
    for m in p.mk:
        if m.get("strike_type")!="greater" or m.get("floor_strike") is None: continue
        K=float(m["floor_strike"]); z=math.log(K/p.v_prev)/s
        if abs(z)>=2: far.append(dict(series=p.series,event=p.event,ticker=m["ticker"],K=K,v_prev=p.v_prev,z=z,
            side="no" if z>0 else "yes",result=m.get("result"),open=p.open,prev_close=p.prev_close,close=str(p.close)))
F=pd.DataFrame(far); F.to_csv("results/altdata_kpi/ca1_far_dev.csv",index=False)
print("far rungs",len(F),"events",F.event.nunique() if len(F) else 0)
D.drop(columns="mk").to_csv("results/altdata_kpi/ca1_pairs_dev.csv",index=False)

"""Daily source panel (UTC days) per Kalshi series from the current Vercel data -> panel.parquet (day x series share %).
Source-only statistics (no prices): lag-1/lag-2 autocorrelation, sd of daily change in pts and in logit, by window."""
import json, glob
import numpy as np, pandas as pd
SER={"KXOPENSOURCESHARE":("W","Open Weights","tokens"),"KXOSSPEND":("W","Open Weights","cost"),
 "KXOPENVREQ":("L","openai","requests"),"KXOPENVSPEND":("L","openai","spend"),"KXMOONVSPEND":("L","moonshotai","spend"),
 "KXANTHVSPEND":("L","anthropic","spend"),"KXANTHVREQ":("L","anthropic","requests"),"KXGOOGVREQ":("L","google","requests"),
 "KXDEEPVREQ":("L","deepseek","requests"),"KXGOOGVSPEND":("L","google","spend"),"KXZAIVSPEND":("L","zai","spend"),"KXZAIVREQ":("L","zai","requests")}
def load_panel(labsf=None, wcf=None):
    labsf=labsf or sorted(glob.glob("src/labs_2025-10-01_*.json"))[-1]; wcf=wcf or sorted(glob.glob("src/wc_2025-10-01_*.json"))[-1]
    labs=json.load(open(labsf))["data"]["rows"]; wc=json.load(open(wcf))["data"]
    L={(r["date"],r["name"],r["metric"]):r["share_percent"] for r in labs}
    W={}
    for o in wc:
        for k,v in o["chef_values"]: W[(o["day"][:10],k,o["metric"])]=v
    days=sorted({k[0] for k in L}|{k[0] for k in W})
    P=pd.DataFrame(index=pd.to_datetime(days))
    for s,(src,name,metric) in SER.items():
        D=W if src=="W" else L
        P[s]=[D.get((d,name,metric),np.nan) for d in days]
    return P
if __name__=="__main__":
    P=load_panel(); P.to_parquet("panel.parquet")
    P=P[P.index<"2026-11-21"]  # drop the current partial day
    lg=lambda v: np.log(v/(100-v))
    out=[]
    for s in P.columns:
        x=P[s].dropna(); x=x[x>0]
        for lab,(a,b) in {"fit 2025-10..2026-09-10":("2025-10-01","2026-09-10"),"dev 09-11..10-25":("2026-09-11","2026-10-25"),"sealed 10-26..11-20":("2026-10-26","2026-11-20")}.items():
            y=x[(x.index>=a)&(x.index<=b)]
            if len(y)<10: out.append(dict(series=s,window=lab,n=len(y))); continue
            d1=y.diff(); d2=y.diff(2); l1=lg(y).diff()
            out.append(dict(series=s,window=lab,n=len(y),first=str(y.index[0].date()),mean=round(y.mean(),1),sd_d1=round(d1.std(),2),sd_d2=round(d2.std(),2),
                            ac1=round(y.autocorr(1),2),ac2=round(y.autocorr(2),2),sd_logit_d1=round(l1.std(),3),
                            dow_range=round(y.groupby(y.index.dayofweek).mean().agg(lambda z:z.max()-z.min()),1)))
    df=pd.DataFrame(out); print(df.to_string()); df.to_csv("source_stats.csv",index=False)

"""Hourly candles for CA1 far dev rungs, window [T-3d, T]; T = max(open, prev_close)+24h."""
import json,time,requests,pandas as pd
API="https://api.elections.kalshi.com/trade-api/v2"
def get(p,**q):
    for _ in range(8):
        r=requests.get(API+p,params=q,timeout=30)
        if r.status_code==429: time.sleep(2); continue
        if r.status_code==404: return {}
        r.raise_for_status(); return r.json()
F=pd.read_csv("results/altdata_kpi/ca1_far_dev.csv")
out={}
for _,f in F.iterrows():
    T=max(pd.Timestamp(f.open),pd.Timestamp(f.prev_close))+pd.Timedelta(hours=24)
    Ts=int((T-pd.Timestamp("1970-01-01",tz="UTC"))//pd.Timedelta(seconds=1))
    q=dict(start_ts=Ts-3*86400,end_ts=Ts,period_interval=60)
    cs=get(f"/series/{f.series}/markets/{f.ticker}/candlesticks",**q).get("candlesticks",[])
    if not cs: cs=get(f"/historical/markets/{f.ticker}/candlesticks",**q).get("candlesticks",[])
    out[f.ticker]={"T":Ts,"c":cs}; time.sleep(0.1)
json.dump(out,open("results/altdata_kpi/ca1_candles.json","w"))
print(len(out),sum(bool(v["c"]) for v in out.values()))

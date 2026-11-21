import json,time,urllib.request,urllib.parse,pandas as pd
B="https://api.elections.kalshi.com/trade-api/v2"
def kget(path,p):
    u=B+path+'?'+urllib.parse.urlencode(p)
    for i in range(4):
        try: return json.load(urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"x"}),timeout=30))
        except Exception as e: time.sleep(2*(i+1))
    return None
ms=json.load(open('kx_tokenuse_markets_20261121.json'))['markets']
out={}
for m in ms:
    if m['status'] not in('finalized','settled'): continue
    tk=m['ticker']
    s=int(pd.Timestamp(m['open_time']).timestamp()); e=int(pd.Timestamp(m['close_time']).timestamp())+3600
    p={"start_ts":s,"end_ts":e,"period_interval":60}
    r=kget(f"/series/KXTOKENUSE/markets/{tk}/candlesticks",p); cs=(r or {}).get("candlesticks",[])
    if not cs:
        r=kget(f"/historical/markets/{tk}/candlesticks",p); cs=(r or {}).get("candlesticks",[])
    out[tk]=cs; time.sleep(0.12)
json.dump(out,open('candles_kxtokenuse.json','w'))
print(len(out), sum(bool(v) for v in out.values()))
k=next(k for k,v in out.items() if v); print(k, json.dumps(out[k][0])[:800])

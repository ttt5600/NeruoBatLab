"""cbank_thin: markets + hourly candles (close-8d .. close) for every closed event. Read-only. Outcomes -> separate file."""
import json,sys
from pathlib import Path
sys.path.insert(0,'scripts')
from kx_throttle import kget
import pandas as pd
R=Path("results/cbank_thin")
E=json.load(open(R/"events.json"))
now=pd.Timestamp("2026-11-20T00:00Z")
cd=json.loads((R/"candles.json").read_text()) if (R/"candles.json").exists() else {}
mk,out=[],[]
for s,evs in E.items():
    for e in evs:
        et=e["event_ticker"]
        ms=(kget("/markets",{"event_ticker":et}) or {}).get("markets",[]) or (kget("/historical/markets",{"event_ticker":et}) or {}).get("markets",[])
        for m in ms:
            ct=pd.Timestamp(m["close_time"])
            row={k:m.get(k) for k in ["ticker","event_ticker","open_time","close_time","yes_sub_title","status","volume","volume_fp","open_interest","rules_primary"]}
            row["series"]=s; mk.append(row)
            out.append({"ticker":m["ticker"],"result":m.get("result"),"status":m.get("status")})
            if ct>now or m["ticker"] in cd: continue
            t1=int(ct.timestamp()); p={"start_ts":t1-8*86400,"end_ts":t1,"period_interval":60}
            cs=(kget(f"/series/{s}/markets/{m['ticker']}/candlesticks",p) or {}).get("candlesticks",[])
            if not cs: cs=(kget(f"/historical/markets/{m['ticker']}/candlesticks",p) or {}).get("candlesticks",[])
            cd[m["ticker"]]=cs
        print(et,len(ms),flush=True)
(R/"candles.json").write_text(json.dumps(cd))
json.dump(mk,open(R/"markets.json","w"),indent=1)
json.dump(out,open(R/"outcomes_HIDDEN.json","w"))

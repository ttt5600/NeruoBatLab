"""Forward snapshot logger (read-only GETs; no orders). Each run: Vercel labs export + weightClass for the last 3 UTC days
(export cache-busted by a nonce param: same-URL requests are CDN HITs for up to 24 h) and every open Kalshi market of the Vercel series (bid/ask/last/vol).
Appends one JSON line per run to fwd/snapshots.jsonl. Usage: python poll.py [N_RUNS] [SLEEP_S]"""
import requests, json, sys, time
from datetime import datetime, timezone, timedelta
API="https://api.elections.kalshi.com/trade-api/v2"; H={"User-Agent":"research-script/0.1"}
SER=["KXOPENSOURCESHARE","KXOSSPEND","KXOPENVREQ","KXOPENVSPEND","KXMOONVSPEND","KXANTHVSPEND","KXANTHVREQ","KXGOOGVREQ",
     "KXDEEPVREQ","KXGOOGVSPEND","KXZAIVSPEND","KXZAIVREQ"]
KEEP=("ticker","event_ticker","floor_strike","yes_bid_dollars","yes_ask_dollars","last_price_dollars","volume_fp","volume_24h_fp",
      "open_interest_fp","close_time","status","yes_bid_size_fp","yes_ask_size_fp")
def run():
    now=datetime.now(timezone.utc); f=(now-timedelta(days=2)).strftime("%Y-%m-%d"); t=now.strftime("%Y-%m-%d")
    rec={"ts":now.isoformat()}
    try:
        labs=requests.get("https://vercel.com/api/ai/leaderboard-export",params={"dataset":"labs","from":f,"to":t,"format":"json","nocache":str(int(time.time()))},headers=H,timeout=60).json()
        rec["labs"]=[r for r in labs["rows"] if r["metric"] in("requests","spend","tokens")]
    except Exception as e: rec["labs_err"]=str(e)
    try:
        rec["wc"]=requests.get("https://vercel.com/api/ai/v5/gateway-model-leaderboard",params={"groupBy":"weightClass","from":f,"to":t},headers=H,timeout=60).json()
    except Exception as e: rec["wc_err"]=str(e)
    mk=[]
    for s in SER:
        try:
            d=requests.get(API+"/markets",params={"series_ticker":s,"status":"open","limit":200},timeout=30).json()
            mk+=[{k:m.get(k) for k in KEEP} for m in d.get("markets",[])]
        except Exception as e: rec.setdefault("kerr",[]).append(f"{s}:{e}")
        time.sleep(0.2)
    rec["mk"]=mk
    with open("fwd/snapshots.jsonl","a") as fh: fh.write(json.dumps(rec)+"\n")
    return rec
if __name__=="__main__":
    n=int(sys.argv[1]) if len(sys.argv)>1 else 1; sl=int(sys.argv[2]) if len(sys.argv)>2 else 600
    for i in range(n):
        r=run(); print(r["ts"], len(r.get("labs",[])), len(r.get("mk",[])), flush=True)
        if i<n-1: time.sleep(sl)

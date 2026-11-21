"""Build quote frame at T = close-7d, close-1d (no outcomes). -> frame_quotes.csv + census."""
import json,math
from pathlib import Path
import pandas as pd
R=Path("results/cbank_thin")
mk=pd.DataFrame(json.load(open(R/"markets.json")))
cd=json.load(open(R/"candles.json"))
mk["close"]=pd.to_datetime(mk["close_time"],utc=True)
mk=mk[mk.close<pd.Timestamp("2026-11-20",tz="UTC")]
import re as _re
def _anchor(r):
    # Amendment 2026-11-20 (before outcomes): close_time is sometimes AFTER the announcement (KXCBDPOLAND-26SEP02 closes
    # 09-09; AUSTRALIA-26FEB03 closes 02-04). Anchor = min(close, ticker date 00:00 UTC); 00:00 UTC precedes every bank's
    # same-date announcement (Asia/Pacific announce >= 01:00 UTC). Canada tickers carry no day -> close.
    m=_re.search(r"-(\d{2})([A-Z]{3})(\d{2})$",r.event_ticker)
    if not m: return r.close
    d=pd.Timestamp(f"20{m.group(1)}-{m.group(2)}-{m.group(3)}",tz="UTC")
    return min(r.close,d)
mk["anchor"]=mk.apply(_anchor,axis=1)
def px(c,k):
    v=c.get(k,{}) or {}
    for kk in ("close_dollars","close"):
        if v.get(kk) is not None:
            return float(v[kk])  # historical candles: "close" already in dollars (checked MEXICO-26FEB05-HOLD 0.97/0.98)
    return None
rows=[]
for _,m in mk.iterrows():
    cs=cd.get(m.ticker,[])
    vol=sum(float(c.get("volume_fp",c.get("volume",0)) or 0) for c in cs)
    for lab,dd in (("D7",7),("D1",1)):
        T=m.anchor-pd.Timedelta(days=dd); t=int(T.timestamp())  # Timestamp.timestamp() is seconds (no [us] trap)
        prior=[c for c in cs if c["end_period_ts"]<=t and c["end_period_ts"]>t-86400]
        r={"ticker":m.ticker,"event":m.event_ticker,"series":m.series,"bucket":m.yes_sub_title,"close":m.close,"anchor":m.anchor,"T":lab,
           "vol8d":vol,"mkt_volume":m.volume_fp if m.volume_fp is not None else m.volume}
        if prior:
            c=prior[-1]; r.update(bid=px(c,"yes_bid"),ask=px(c,"yes_ask"),last=px(c,"price"),
                                  vol24=sum(float(x.get("volume_fp",x.get("volume",0)) or 0) for x in prior))
        rows.append(r)
f=pd.DataFrame(rows); f.to_csv(R/"frame_quotes.csv",index=False)
print("markets closed",len(mk),"events",mk.event_ticker.nunique(),"series",mk.series.nunique())
q=f.dropna(subset=["bid"]); print("quoted market-T cells",len(q),"of",len(f))
q=q.assign(spread=q.ask-q.bid)
print(q.groupby("T").spread.describe())
ev=f.groupby(["series","event"]).agg(close=("close","first"),vol=("mkt_volume",lambda s: pd.to_numeric(s,errors="coerce").sum())).reset_index()
print(ev.groupby("series").agg(n=("event","size"),vol_med=("vol","median"),first=("close","min")).sort_values("n"))
print("events by month"); print(ev.close.dt.strftime("%Y-%m").value_counts().sort_index())

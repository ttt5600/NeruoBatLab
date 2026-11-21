"""Census of candidate count/chart series: settled markets/events, volume, recency. Read-only public GETs."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, "scripts")
from kx_throttle import kget
import pandas as pd
D = Path("results/count_nowcast/mk"); D.mkdir(parents=True, exist_ok=True)
SER = sys.argv[1:]
def pull(ep, ser, extra):
    out, cur = [], None
    while True:
        p = {"series_ticker": ser, "limit": 1000, **extra}
        if cur: p["cursor"] = cur
        r = kget(ep, p) or {}
        out += r.get("markets", [])
        cur = r.get("cursor")
        if not cur or not r.get("markets"): break
    return out
rows = []
for s in SER:
    f = D / f"{s}.json"
    if f.exists():
        mk = json.load(open(f))
    else:
        mk = pull("/markets", s, {}) + pull("/historical/markets", s, {})
        json.dump(mk, open(f, "w"))
    if not mk:
        rows.append(dict(series=s, n_mk=0)); continue
    m = pd.DataFrame(mk).drop_duplicates("ticker")
    m["close_time"] = pd.to_datetime(m["close_time"], utc=True, format="ISO8601")
    vol = pd.to_numeric(m.get("volume_fp", m.get("volume")), errors="coerce").fillna(0)
    m["vol"] = vol
    st = m[m.status.isin(["settled", "finalized", "determined"])]
    now = pd.Timestamp.now(tz="UTC")
    rows.append(dict(series=s, n_mk=len(m), n_ev=m.event_ticker.nunique(), settled_ev=st.event_ticker.nunique(),
                     ev_90d=st[st.close_time > now - pd.Timedelta(days=90)].event_ticker.nunique(),
                     first=str(m.close_time.min())[:10], last=str(m.close_time.max())[:10],
                     open_ev=m[m.status.isin(["active", "open", "initialized"])].event_ticker.nunique(),
                     vol_tot=int(vol.sum()), vol_med_ev=int(m.groupby("event_ticker").vol.sum().median())))
df = pd.DataFrame(rows)
print(df.to_string())
df.to_csv("results/count_nowcast/census_%d.csv" % int(time.time()), index=False)

"""P4 SETTLEERR (pre-registered in notes/winners_transfer.md). Read-only Kalshi + IEM GETs."""
import json, sys, time, re, io, os
sys.path.insert(0, '.')
from tailpool import get, ts
import pandas as pd, numpy as np, requests
from zoneinfo import ZoneInfo
d = json.load(open('../forum_strategy_scout/kalshi_series.json')); s = d['series'] if isinstance(d, dict) else d
ser = sorted(x['ticker'] for x in s if re.match(r'^KX(HIGH|LOW)', x['ticker']))
t0 = ts('2026-08-14T00:00:00Z')
if not os.path.exists('p4_markets.parquet'):
    rows = []
    for se in ser:
        cur = None
        while True:
            p = {"series_ticker": se, "status": "settled", "min_close_ts": int(t0), "limit": 1000}
            if cur: p["cursor"] = cur
            r = get("/markets", p); mk = r.get("markets", []); cur = r.get("cursor")
            for m in mk:
                g = re.search(r'\(CLI([A-Z]{3})\)', m.get('rules_primary', '') or '')
                rows.append(dict(series=se, event=m["event_ticker"], ticker=m["ticker"], result=m["result"], exp=m.get("expiration_value"),
                                 close=m["close_time"], settle=m.get("settlement_ts"), st=m.get("strike_type"), floor=m.get("floor_strike"),
                                 cap=m.get("cap_strike"), last=float(m.get("last_price_dollars") or 0), cli=g.group(1) if g else None,
                                 vol=float(m.get("volume_fp") or 0)))
            if not cur or not mk: break
    pd.DataFrame(rows).to_parquet('p4_markets.parquet')
mk = pd.read_parquet('p4_markets.parquet')
print('markets', len(mk), 'missing cli', mk.cli.isna().sum(), 'stations', mk.cli.nunique())
# station tz (standard offset) from series knowledge: use IEM station metadata via network lookup is heavy; infer from Kalshi close hour
os.makedirs('iem', exist_ok=True)
for st in sorted(mk.cli.dropna().unique()):
    f = f'iem/{st}.csv'
    if os.path.exists(f) and os.path.getsize(f) > 1000: continue
    url = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=%s&data=tmpf&year1=2026&month1=8&day1=12"
           "&year2=2026&month2=11&day2=21&tz=Etc/UTC&format=onlycomma&latlon=no&missing=M&report_type=3&report_type=4" % st)
    for a in range(5):
        r = requests.get(url, timeout=120)
        if r.status_code == 200 and not r.text.startswith('Too many'): open(f, 'w').write(r.text); break
        time.sleep(15 * (a + 1))
    time.sleep(4)
    print('iem', st, os.path.getsize(f) if os.path.exists(f) else 0, flush=True)

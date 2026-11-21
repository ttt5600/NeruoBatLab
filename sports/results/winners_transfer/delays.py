"""Census of delayed weather settlements (material-error holds) since the TWC switch 2026-08-14. Read-only GETs."""
import json, sys, time, re
sys.path.insert(0, '.')
from tailpool import get, ts
import pandas as pd
d = json.load(open('../forum_strategy_scout/kalshi_series.json')); s = d['series'] if isinstance(d, dict) else d
ser = sorted(x['ticker'] for x in s if re.match(r'^KX(HIGH|LOW)', x['ticker']))
t0 = ts('2026-08-14T00:00:00Z'); rows = []
for se in ser:
    cur = None
    while True:
        p = {"series_ticker": se, "status": "settled", "min_close_ts": int(t0), "limit": 1000}
        if cur: p["cursor"] = cur
        r = get("/markets", p); mk = r.get("markets", []); cur = r.get("cursor")
        for m in mk:
            if not m.get("settlement_ts"): continue
            rows.append(dict(series=se, event=m["event_ticker"], ticker=m["ticker"], result=m["result"], exp=m.get("expiration_value"),
                             close=m["close_time"], settle=m["settlement_ts"], delay_h=(ts(m["settlement_ts"]) - ts(m["close_time"]))/3600,
                             vol=float(m.get("volume_fp") or 0), last=m.get("last_price_dollars")))
        if not cur or not mk: break
df = pd.DataFrame(rows); df.to_parquet('delays.parquet')
ev = df.groupby(['series', 'event']).agg(delay_h=('delay_h', 'max'), vol=('vol', 'sum'), exp=('exp', 'first'), close=('close', 'first')).reset_index()
print('series', len(ser), 'markets', len(df), 'events', len(ev))
print('event delay quantiles (h):', ev.delay_h.quantile([.5, .9, .95, .99]).round(1).to_dict())
big = ev[ev.delay_h > 36].sort_values('delay_h', ascending=False)
print('events with delay > 36 h:', len(big)); print(big.head(40).to_string())

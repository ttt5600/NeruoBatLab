"""Read-only: last 1-min Kalshi candle (yes bid/ask close) at or before ESPN tip for every settled pick -> pre-tip mid."""
import time, requests, pandas as pd, numpy as np
API = "https://api.elections.kalshi.com/trade-api/v2"
OUT = 'results/props_forward_1121/'
S = pd.read_parquet(OUT + 'frame.parquet')
S = S[S.pick.notna() & S.graded].copy()
EPOCH = pd.Timestamp('1970-01-01', tz='UTC')
S['tip_s'] = (S.tip_utc - EPOCH) // pd.Timedelta(seconds=1)       # traps.py: never int64 // 1e9 on [us]
last = [0.0]
def kget(path, params):
    for k in range(6):
        w = 0.55 - (time.time() - last[0])
        if w > 0: time.sleep(w)
        last[0] = time.time()
        try: r = requests.get(API + path, params=params, timeout=20)
        except requests.RequestException: time.sleep(2 ** k); continue
        if r.status_code == 429 or r.status_code >= 500: time.sleep(min(2 ** (k + 1), 30)); continue
        if r.status_code == 404: return {}
        r.raise_for_status(); return r.json()
    raise RuntimeError(path)
f = lambda d, a: (float(d[a]['close_dollars']) if d.get(a) and d[a].get('close_dollars') is not None else np.nan)
out = []
for i, r in enumerate(S.itertuples()):
    ser = r.ticker.split('-')[0]
    j = kget(f'/series/{ser}/markets/{r.ticker}/candlesticks', {'start_ts': int(r.tip_s) - 3600, 'end_ts': int(r.tip_s), 'period_interval': 1})
    cs = [c for c in j.get('candlesticks', []) if c['end_period_ts'] <= r.tip_s]
    row = {'ticker': r.ticker, 'tip_s': int(r.tip_s), 'n_candles': len(cs)}
    for c in reversed(cs):
        b, a = f(c, 'yes_bid'), f(c, 'yes_ask')
        if b == b and a == a and 0 < b < a < 1:
            row.update(close_bid=b, close_ask=a, close_mid=(a + b) / 2, candle_end=c['end_period_ts']); break
    out.append(row)
    if i % 200 == 0: print(i, len(S), flush=True)
pd.DataFrame(out).to_parquet(OUT + 'candles_close.parquet'); print('done', len(out))

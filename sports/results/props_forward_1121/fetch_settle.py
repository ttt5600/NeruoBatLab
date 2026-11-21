"""Read-only: Kalshi settlement result + settlement value for every logged event ticker (cross-check of ESPN grading)."""
import json, time, requests, pandas as pd
API = "https://api.elections.kalshi.com/trade-api/v2"
D = '/Users/jonathanwang/.claude/quant-sports/data/'
R = [json.loads(l) for l in open(D + 'kalshi_prop_log.jsonl') if l.strip()]
ets = sorted({r['event_ticker'] for r in R if r.get('type') == 'rung'})
last = [0.0]
def kget(path, params):
    for k in range(6):
        w = 0.55 - (time.time() - last[0])
        if w > 0: time.sleep(w)
        last[0] = time.time()
        try:
            r = requests.get(API + path, params=params, timeout=20)
        except requests.RequestException:
            time.sleep(2 ** k); continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(min(2 ** (k + 1), 30)); continue
        r.raise_for_status(); return r.json()
    raise RuntimeError(path)
out = []
for i, et in enumerate(ets):
    cur = None
    while True:
        j = kget('/markets', {'event_ticker': et, 'limit': 1000, **({'cursor': cur} if cur else {})})
        for m in j.get('markets', []):
            out.append({'ticker': m['ticker'], 'status': m.get('status'), 'result': m.get('result'),
                        'settle_val': m.get('settlement_value_dollars'), 'expiration_value': m.get('expiration_value'),
                        'settlement_ts': m.get('settlement_ts'), 'close_time': m.get('close_time')})
        cur = j.get('cursor')
        if not cur: break
    if i % 50 == 0: print(i, len(ets), len(out), flush=True)
pd.DataFrame(out).to_parquet('results/props_forward_1121/kalshi_settle.parquet')
print('done', len(out))

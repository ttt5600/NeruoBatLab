import json, glob, os, datetime as dt, sys
# Descriptive: for finalized KXPOKEMON markets, YES bid/ask/last at ~T-14d, T-7d, T-2d before close vs result.
# P&L of buying NO at the NO ask (= 1 - yes_bid) at T-k, before fees. Not an edge test (n~20, 2 monthly cohorts).
d = sys.argv[1]
m = {x['ticker']: x for x in json.load(open(os.path.join(d, 'kalshi_markets_KXPOKEMON.json')))['markets']}
rows = []
for f in sorted(glob.glob(os.path.join(d, 'kalshi_candles', '*.json'))):
    t = os.path.basename(f)[:-5]; x = m[t]
    close = dt.datetime.fromisoformat(x['close_time'].replace('Z', '+00:00')).timestamp()
    cs = json.load(open(f))['candlesticks']
    out = {'t': t, 'res': x['result'], 'strike': x['floor_strike'], 'exp': x['expiration_value'], 'vol': float(x['volume_fp'] or 0)}
    for k in (14, 7, 2):
        tgt = close - k * 86400
        c = [c for c in cs if c['end_period_ts'] <= tgt]
        if not c: out[f'b{k}'] = out[f'a{k}'] = None; continue
        c = c[-1]
        out[f'b{k}'] = float(c['yes_bid'].get('close_dollars') or 0); out[f'a{k}'] = float(c['yes_ask'].get('close_dollars') or 1)
    rows.append(out)
for r in rows:
    print(r['t'], r['res'], 'strike', r['strike'], 'exp', r['exp'], '| T-14 bid/ask', r['b14'], r['a14'], '| T-7', r['b7'], r['a7'], '| T-2', r['b2'], r['a2'])
for k in (14, 7, 2):
    pnl = []; 
    for r in rows:
        if r[f'b{k}'] is None or 'SEPCHA-' in r['t']: continue
        no_ask = 1 - r[f'b{k}']
        pnl.append((1 if r['res'] == 'no' else 0) - no_ask)
    print(f'T-{k}: buy NO at no_ask, n={len(pnl)}, wins={sum(p>0 for p in pnl)}, sum pnl/contract={sum(pnl):+.2f}, mean={sum(pnl)/len(pnl):+.3f} (pre-fee; excludes SEPCHA anomaly)')

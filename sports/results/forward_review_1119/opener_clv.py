"""Opener trials: pre-reg CLV (book close vs S, or CLV_c for nhla) AND descriptive Kalshi-own CLV
(close mid of the side bought - entry mid; and net = close mid - (ask+fee)), date-clustered bootstrap (10k, seed 1119)."""
import json, os, numpy as np
from collections import defaultdict
D = os.path.expanduser('~/.claude/quant-sports/data/kx_forward')
rng = np.random.default_rng(1119)
def boot(vals, clus, B=10000, ratio=None):
    g = defaultdict(list)
    for v, c in zip(vals, clus): g[c].append(v)
    keys = list(g); k = len(keys)
    if k < 2: return (float('nan'),)*2
    arr = [np.array(g[x]) for x in keys]
    out = []
    for _ in range(B):
        idx = rng.integers(0, k, k)
        if ratio is None:
            s = np.concatenate([arr[i] for i in idx]); out.append(s.mean())
        else:
            s = np.concatenate([arr[i] for i in idx]); out.append(s[:,0].sum()/s[:,1].sum())
    return np.percentile(out, 2.5), np.percentile(out, 97.5)
def keyof(r, t):
    return {'k1': r.get('key'), 'kcfb': r.get('event'), 'k50nfl': r.get('bet_key'), 'nhla': r.get('key')}[t]
for t, pre in [('k1', 'clv_S'), ('kcfb', 'clv'), ('k50nfl', 'clv'), ('nhla', 'clv_c')]:
    R = [json.loads(l) for l in open(f'{D}/{t}.jsonl') if l.strip()]
    picks = {keyof(r, t): r for r in R if r.get('kind') == 'pick' and r.get('status') == 'bet'}
    sets = [r for r in R if r.get('kind') == 'settle']
    rows = []
    for s in sets:
        p = picks.get(keyof(s, t))
        if p is None: print(t, 'NO PICK for', keyof(s, t)); continue
        if s.get('kalshi_status') not in ('finalized', 'settled', None) or s.get('result') not in ('yes', 'no'): print(t, 'skip void/unsettled', keyof(s,t)); continue
        cost = p['ask'] + p['fee']
        if 'yes_bid' in p and p.get('yes_bid') is not None:
            ym = (p['yes_bid'] + p['yes_ask']) / 2; em = ym if p['side'] == 'yes' else 1 - ym
        else:
            em = p['ask'] - p.get('spread_c', 0) / 200
        cm = s.get('close_mid')
        pnl = (1 - cost) if s['won'] else -cost
        date = (p.get('event_start') or '')[:10]
        rows.append(dict(date=date, pre=s.get(pre), clvk=None if cm is None else cm - em, clvn=None if cm is None else cm - cost, pnl=pnl, cost=cost, em=em, cm=cm))
    n = len(rows); cl = [r['date'] for r in rows]
    pre_v = [r['pre'] for r in rows if r['pre'] is not None]
    print(f'== {t}: settled {n}, game-dates {len(set(cl))}')
    print(f'  pre-reg CLV ({pre}) mean {np.mean(pre_v):+.3f} CI {boot(pre_v, [r["date"] for r in rows if r["pre"] is not None])}')
    ck = [(r['clvk'], r['date']) for r in rows if r['clvk'] is not None]
    if ck:
        v, c = zip(*ck); print(f'  Kalshi-own CLV (close mid - entry mid, prob) mean {np.mean(v):+.4f} CI {boot(v, c)} n {len(v)}')
        v2 = [r['clvn'] for r in rows if r['clvn'] is not None]; print(f'  Kalshi-own net (close mid - ask - fee) mean {np.mean(v2):+.4f} CI {boot(v2, c)}')
    pc = [(r['pnl'], r['cost']) for r in rows]
    print(f'  ROI {sum(x for x,_ in pc)/sum(y for _,y in pc):+.3f} date-clustered CI {boot(pc, cl, ratio=True)}')
    print('  sample:', [(r['date'], round(r['em'],3), r['cm']) for r in rows[:4]])

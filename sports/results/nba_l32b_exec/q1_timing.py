"""Q1: KXNBATOTAL listing / first tight vs Pinnacle first sighting; Kalshi first implied vs Pinnacle open & close."""
import json
import numpy as np
import pandas as pd

L = json.load(open('/Users/jonathanwang/.claude/quant-kalshi/data/open_totals_state.json'))
T = json.load(open('/Users/jonathanwang/.claude/quant-kalshi/data/open_totals_tight.json'))
mk = {m['key']: m for m in T['markets'].values()}
rows = []
for k, g in L.items():
    if not g.get('pin_open') or k not in mk:
        continue
    m = mk[k]
    po = pd.Timestamp(g['pin_open']['t'])
    start = pd.Timestamp(g['start'])
    pre = (g.get('decision') or {}).get('why') == 'pre-existing at first logger run'
    ot = pd.Timestamp(m['open_time']) if m.get('open_time') else None
    ft = m.get('first_tight')
    fs = m.get('first_two_sided')
    rows.append(dict(
        key=k, pre=pre, pin_open_t=po, start=start, kal_open_time=ot, kal_censored=m['listing_censored'],
        h_kal_list_minus_pin=(ot - po).total_seconds() / 3600 if ot is not None else np.nan,
        h_pin_before_tip=(start - po).total_seconds() / 3600,
        h_tight_minus_pin=(pd.Timestamp(ft['t']) - po).total_seconds() / 3600 if ft and not ft['censored'] else np.nan,
        h_tight_minus_pin_any=(pd.Timestamp(ft['t']) - po).total_seconds() / 3600 if ft else np.nan,
        tight_spread=ft['spread'] if ft else np.nan, first2_spread=fs['spread'] if fs else np.nan,
        first2_censored=fs['censored'] if fs else None,
        pin_open=g['pin_open']['line'], pin_close=(g.get('pin_last') or {}).get('line'),
        kal_first_impl=(g.get('kal_open') or {}).get('implied'),
        kal_first_t=pd.Timestamp(g['kal_open']['t']) if g.get('kal_open') else None))
d = pd.DataFrame(rows)
d = d[~d.pre]
print('games (not pre-existing) with pin_open + kalshi event:', len(d))
print('pin first sighting, h before tip: median %.1f  p10 %.1f p90 %.1f' % tuple(np.nanpercentile(d.h_pin_before_tip, [50, 10, 90])))
x = d.h_kal_list_minus_pin.dropna()
print('Kalshi open_time minus Pinnacle first sighting (h): n %d median %.1f p10 %.1f p90 %.1f; share Kalshi listed first %.2f'
      % (len(x), x.median(), x.quantile(.1), x.quantile(.9), (x < 0).mean()))
y = d.h_tight_minus_pin.dropna()
print('first tight (uncensored) minus pin first sighting (h): n %d median %.1f p10 %.1f p90 %.1f'
      % (len(y), y.median(), y.quantile(.1), y.quantile(.9)) if len(y) else 'no uncensored tight')
z = d.h_tight_minus_pin_any.dropna()
print('first tight (ANY, incl censored) minus pin first sighting (h): n %d median %.2f p10 %.2f p90 %.2f; share <= 0 %.2f'
      % (len(z), z.median(), z.quantile(.1), z.quantile(.9), (z <= 0).mean()))
print('first two-sided spread median (c):', d.first2_spread.median() * 100, ' first-tight spread median', d.tight_spread.median() * 100)
s = d.dropna(subset=['kal_first_impl', 'pin_close'])
same = s[s.kal_first_t == s.pin_open_t]
print('kal implied seen at the SAME tick as pin open: %d of %d' % (len(same), len(s)))
for nm, q in [('all', s), ('same tick', same)]:
    a = (q.kal_first_impl - q.pin_open).abs()
    b = (q.kal_first_impl - q.pin_close).abs()
    dd = (a - b).values
    rng = np.random.default_rng(0)
    bs = [rng.choice(dd, len(dd)).mean() for _ in range(4000)]
    print(f'{nm}: n {len(q)}  |kal_first - pin_open| mean {a.mean():.2f}  |kal_first - pin_close| mean {b.mean():.2f}  '
          f'diff {dd.mean():+.2f} [{np.percentile(bs, 2.5):+.2f},{np.percentile(bs, 97.5):+.2f}]  '
          f'closer-to-open share {(a < b).mean():.2f} ties {(a == b).mean():.2f}')
    print(f'   |pin_open - pin_close| mean {(q.pin_open - q.pin_close).abs().mean():.2f}; '
          f'kal_first - pin_open mean {(q.kal_first_impl - q.pin_open).mean():+.2f}')
d.to_csv('results/nba_l32b_exec/q1_timing.csv', index=False)

"""E5 forward paper audit: ROI at ask / mid / maker-proxy / all-at-bid, station-day cluster bootstrap. Read-only."""
import json, numpy as np, pandas as pd
P = '/Users/jonathanwang/.claude/quant-kalshi/data/forecastex_paper.jsonl'
OUT = '/Users/jonathanwang/Desktop/vocalizations_lab/.claude/worktrees/quant-trader/sports/results/e5_forecastex_audit/'
rows = [json.loads(l) for l in open(P) if l.strip()]
t = pd.DataFrame([r for r in rows if r.get('kind') == 'pick' and r.get('trial') == 'FX1-fwd'])
print('picks', len(t), 'kinds', pd.Series([r.get('kind') for r in rows]).value_counts().to_dict())
print('dup tickers', t.duplicated('ticker').sum())
s = t[t.yes_won.notna()].copy()
s['won'] = np.where(s.side == 'YES', s.yes_won == 1, s.yes_won == 0)
s['cl'] = s.code + '|' + s.event_date
print('settled', len(s), 'date range', s.event_date.min(), s.event_date.max())
rng = np.random.default_rng(0)


def score(name, price, mask=None):
    d = s.copy(); d['c'] = price + 0.01
    d = d[d.c.notna() & (d.c < 1)]
    if len(d) == 0:
        print(name, 'n 0'); return
    d['pnl'] = np.where(d.won, 1 - d.c, -d.c)
    roi = d.pnl.sum() / d.c.sum()
    g = d.groupby('cl').agg(p=('pnl', 'sum'), c=('c', 'sum'))
    idx = rng.integers(0, len(g), (5000, len(g)))
    bs = g.p.values[idx].sum(1) / g.c.values[idx].sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    be = 1 - d.c.mean()
    print(f'{name:<28} n {len(d):4d} clusters {len(g):3d} losses {(~d.won).sum():3d} lossrate {(~d.won).mean():.3f} '
          f'(b/e {be:.3f}) meancost {d.c.mean():.3f} units {d.pnl.sum():+.2f} ROI {roi:+.2%} CI [{lo:+.2%},{hi:+.2%}]')
    return d


score('A at ask', s.side_ask)
score('MID', (s.side_bid + s.side_ask) / 2)
filled = s.low_d < s.side_bid
score('MK proxy (low_d<bid)', s.side_bid.where(filled))
score('ALL at bid (upper bound)', s.side_bid)
score('bid, print not through', s.side_bid.where(~filled & s.low_d.notna()))
score('bid, no day-D print', s.side_bid.where(s.low_d.isna()))
print('bid present', round(s.side_bid.notna().mean(), 3), 'ask present', round(s.side_ask.notna().mean(), 3),
      'low_d present', round(s.low_d.notna().mean(), 3), 'last_d1 present', round(s.last_d1.notna().mean(), 3))
for lab, m in [('filled', filled), ('print, not through', ~filled & s.low_d.notna()), ('no print', s.low_d.isna())]:
    x = s[m & s.side_bid.notna()]
    print(f'  {lab:<20} n {len(x)} loss rate {(~x.won).mean():.3f} mean bid {x.side_bid.mean():.3f}')
print('overall settled loss rate', round((~s.won).mean(), 4), 'n', len(s))
print(s.groupby('event_date').agg(n=('won', 'size'), losses=('won', lambda w: (~w).sum()),
                                  bid=('side_bid', 'mean'), ask_n=('side_ask', 'count')).to_string())
s.to_csv(OUT + 'settled_picks.csv', index=False)

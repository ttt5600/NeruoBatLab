import pandas as pd, numpy as np, re, json, os
OFF = dict(ATL=-5, BOS=-5, DCA=-5, EWR=-5, MIA=-5, NYC=-5, PHL=-5, TTN=-5, SDF=-5, AUS=-6, DFW=-6, HOU=-6, MDW=-6, MSP=-6,
           MSY=-6, OKC=-6, SAT=-6, DEN=-7, PHX=-7, LAS=-8, LAX=-8, SAN=-8, SEA=-8, SFO=-8)
mk = pd.read_parquet('p4_markets.parquet'); mk = mk[mk.cli.notna() & mk.result.isin(['yes', 'no'])]
obs = {}
for st in OFF:
    f = f'iem/{st}.csv'
    if not os.path.exists(f): continue
    o = pd.read_csv(f, comment='#'); o = o[o.tmpf != 'M']; o['tmpf'] = o.tmpf.astype(float)
    o['t'] = pd.to_datetime(o.valid, utc=True); obs[st] = o[['t', 'tmpf']].sort_values('t')
mon = dict(JAN=1, FEB=2, MAR=3, APR=4, MAY=5, JUN=6, JUL=7, AUG=8, SEP=9, OCT=10, NOV=11, DEC=12)
rows = []
for ev, g in mk.groupby('event'):
    st = g.cli.iloc[0]
    if st not in obs: continue
    m = re.search(r'-(\d\d)([A-Z]{3})(\d\d)$', ev)
    day = pd.Timestamp(2000 + int(m.group(1)), mon[m.group(2)], int(m.group(3)), tz='UTC')
    ws = day - pd.Timedelta(hours=OFF[st]); we = ws + pd.Timedelta(days=1)
    close = pd.Timestamp(g.close.iloc[0]); T = min(close - pd.Timedelta(hours=1), we)
    o = obs[st]; o = o[(o.t >= ws) & (o.t <= T)]
    if len(o) < 12: continue
    hi = g.series.iloc[0].startswith('KXHIGH')
    v = int(round(o.tmpf.max() if hi else o.tmpf.min()))
    gg = g.sort_values('last', ascending=False); fav = gg.iloc[0]
    if fav['last'] < 0.85: continue
    if fav.st == 'between': lo, up = fav.floor, fav.cap
    elif fav.st == 'greater': lo, up = fav.floor + 1, 999
    else: lo, up = -999, fav.cap - 1
    d = 0 if lo <= v <= up else (lo - v if v < lo else v - up)
    win = g[g.result == 'yes']
    rows.append(dict(event=ev, st=st, hi=hi, v=v, fav=fav.ticker, fav_last=fav['last'], lo=lo, up=up, d=d,
                     direction='fav_above_obs' if v < lo else ('fav_below_obs' if v > up else 'inside'),
                     flip=fav.result == 'no', exp=g.exp.iloc[0], winner_last=win['last'].max() if len(win) else np.nan,
                     delay_h=(pd.to_datetime(g.settle, utc=True).max() - close).total_seconds() / 3600, nobs=len(o)))
r = pd.DataFrame(rows); r.to_parquet('p4_rows.parquet')
print('events evaluated', len(r), 'flips', r.flip.sum(), 'delayed>36h', (r.delay_h > 36).sum())
for thr in [2, 3, 4]:
    fl = r[r.d >= thr]
    print(f'd>={thr}: flags {len(fl)}, flips among flags {fl.flip.sum()} (precision {fl.flip.mean() if len(fl) else 0:.3f}), '
          f'flips caught {fl.flip.sum()}/{r.flip.sum()}, by direction {fl.groupby("direction").flip.agg(["sum","count"]).to_dict()}')
print(r[r.flip | (r.delay_h > 36)].sort_values('delay_h', ascending=False)[['event', 'v', 'fav', 'fav_last', 'lo', 'up', 'd', 'flip', 'exp', 'winner_last', 'delay_h']].to_string())
print('d distribution:', r.d.value_counts().sort_index().to_dict())

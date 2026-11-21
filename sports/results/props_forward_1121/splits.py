"""Descriptive splits of the prop-ladder forward trial (NOT tests; the verdict is in score.py)."""
import json, numpy as np, pandas as pd
OUT = 'results/props_forward_1121/'
M = pd.read_parquet(OUT + 'frame.parquet'); S = pd.read_parquet(OUT + 'settled_picks.parquet')
Q = pd.read_parquet(OUT + 'calib_rows.parquet')
D = '/Users/jonathanwang/.claude/quant-sports/data/'
GM = pd.DataFrame([r for r in map(json.loads, open(D + 'kalshi_prop_log.jsonl')) if r['type'] == 'game'])

def cboot(df, num, den=None, n=10000, seed=0):
    if len(df) == 0: return dict(est=np.nan, lo=np.nan, hi=np.nan)
    g = df.groupby('event_id').agg(s=(num, 'sum'), c=((den or num), 'sum' if den else 'size'))
    s, c = g.s.to_numpy(float), g.c.to_numpy(float)
    k = np.random.default_rng(seed).integers(0, len(g), (n, len(g)))
    bs = s[k].sum(1) / c[k].sum(1)
    return dict(est=float(s.sum() / c.sum()), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)))
def line(df, tag):
    r = cboot(df, 'prof_x', 'cost_x'); c = cboot(df.dropna(subset=['clv_re']), 'clv_re')
    w = int(df.win.sum()) if len(df) else 0
    edge = (df.sim - df.mid).mean() if len(df) else np.nan
    print(f"  {tag:<34} n {len(df):>5} g {df.event_id.nunique():>3} W-L {w}-{len(df)-w:<5} ROI {r['est']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}]"
          f"  CLV {c['est']:+.4f} [{c['lo']:+.4f},{c['hi']:+.4f}]  mean px {df.px.mean():.3f} hit {df.win.mean():.3f}")
    return dict(n=len(df), roi=r, clv=c)
res = {}
print('[E] by stat x side (exact fee)')
for st in ('pts', 'reb', 'ast', None):
    for sd in ('YES', 'NO', None):
        m_ = pd.Series(True, index=S.index)
        if st: m_ &= S.stat == st
        if sd: m_ &= S.pick == sd
        x = S[m_]
        if len(x): res[f'{st}_{sd}'] = line(x, f'{st or "all"} {sd or "both"}')

# ---- stars (market-implied pts median >= 20) and upper rungs (mid < .5)
def implied_median(df):
    df = df.sort_values('k'); k, m = df.k.to_numpy(float), df.mid.to_numpy(float)
    if (m >= .5).all(): return k[-1]
    if (m < .5).all(): return k[0] - 1
    i = np.argmax(m < .5)            # first rung below .5 (mids should decline in k)
    k0, k1, m0, m1 = k[i - 1], k[i], m[i - 1], m[i]
    return k0 + (m0 - .5) / (m0 - m1) * (k1 - k0) if m0 != m1 else k0
med = M[M.stat == 'pts'].groupby(['event_id', 'player']).apply(implied_median, include_groups=False).rename('pts_med').reset_index()
M2 = M.merge(med, on=['event_id', 'player'], how='left', validate='many_to_one')
S2 = S.merge(med, on=['event_id', 'player'], how='left', validate='many_to_one')
Q2 = Q.merge(med, on=['event_id', 'player'], how='left', validate='many_to_one')
S2['star'] = S2.pts_med >= 20; Q2['star'] = Q2.pts_med >= 20; S2['upper'] = S2.mid < .5; Q2['upper'] = Q2.mid < .5
print(f"\n[F] stars: {int((med.pts_med >= 20).sum())} player-games of {len(med)} with a pts ladder")
for st_, up_ in ((True, True), (True, False), (False, True), (False, False)):
    x = Q2[(Q2.star == st_) & (Q2.upper == up_)]
    print(f"  calib star={st_!s:<5} upper={up_!s:<5} n {len(x):>5}: mean sim {x.sim.mean():.3f} mid {x.mid.mean():.3f} hit {x.y.mean():.3f}"
          f"  sim-mid {(x.sim - x.mid).mean():+.3f}; Brier sim {((x.sim - x.y)**2).mean():.4f} mid {((x.mid - x.y)**2).mean():.4f}")
    for sd in ('YES', 'NO'):
        z = S2[(S2.star == st_) & (S2.upper == up_) & (S2.pick == sd)]
        if len(z): res[f'star{st_}_upper{up_}_{sd}'] = line(z, f'   picks star={st_} upper={up_} {sd}')
print(f"  share of NO picks that are star upper rungs: {((S2.pick=='NO') & S2.star & S2.upper).sum()}/{(S2.pick=='NO').sum()}")

# ---- top-5 contributing games
gp = S.groupby(['event_id', 'game']).agg(n=('prof_x', 'size'), units=('prof_x', 'sum'), staked=('cost_x', 'sum')).reset_index()
gp['tip'] = S.groupby('event_id').tip_utc.first().reindex(gp.event_id).dt.tz_convert('US/Eastern').dt.strftime('%m-%d').to_numpy()
print('\n[G] top-5 games by units (best) and worst-5:')
print(gp.sort_values('units', ascending=False).head(5).round(2).to_string(index=False))
print(gp.sort_values('units').head(5).round(2).to_string(index=False))
for drop in ('best', 'worst'):
    e = gp.sort_values('units', ascending=(drop == 'worst')).event_id.iloc[0]
    x = S[S.event_id != e]; r = cboot(x, 'prof_x', 'cost_x')
    print(f"  ROI excluding the {drop} game: {r['est']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]")
pl = S.groupby('player').agg(n=('prof_x', 'size'), units=('prof_x', 'sum')).sort_values('n', ascending=False)
print('  most-picked players:', pl.head(8).round(2).to_dict('index'))

# ---- POST-HOC: minutes errors, blowouts, voids
g = pd.read_parquet(D + 'espn/nba/games_2027.parquet')[['event_id', 'home_score', 'away_score']]
g['event_id'] = g.event_id.astype(str); g['margin'] = (g.home_score - g.away_score).abs()
S3 = S.merge(g[['event_id', 'margin']], on='event_id', how='left', validate='many_to_one')
S3['merr'] = S3.minutes - S3.min_proj
print('\n[H] POST-HOC minutes errors (ESPN minutes - min_proj), settled picks')
print(f"  merr all: mean {S3.merr.mean():+.2f}, sd {S3.merr.std():.2f}; |merr|>=10: {int((S3.merr.abs()>=10).sum())}/{len(S3)}")
for tag, x in (('winners', S3[S3.win == 1]), ('losers', S3[S3.win == 0])):
    print(f"  {tag}: n {len(x)} mean merr {x.merr.mean():+.2f}; share |merr|>=10 {(x.merr.abs()>=10).mean():.3f}; "
          f"share merr<=-10 {(x.merr<=-10).mean():.3f}; share merr>=+10 {(x.merr>=10).mean():.3f}")
for sd in ('YES', 'NO'):
    for cond, tag in ((S3.merr.abs() < 5, '|merr|<5'), ((S3.merr.abs() >= 5) & (S3.merr.abs() < 10), '5<=|merr|<10'), (S3.merr.abs() >= 10, '|merr|>=10')):
        line(S3[cond & (S3.pick == sd)], f'{sd} {tag}')
print('  blowouts (final margin >= 20):')
for b_, tag in ((S3.margin >= 20, 'margin>=20'), (S3.margin < 20, 'margin<20')):
    line(S3[b_], tag)
# voided picks: what Kalshi actually paid (fair price) -- needs kalshi_settle.parquet
try:
    K = pd.read_parquet(OUT + 'kalshi_settle.parquet')
    V = M[M.pick.notna() & M.void.fillna(False)].merge(K, on='ticker', how='left', validate='one_to_one')
    V['sv'] = pd.to_numeric(V.settle_val, errors='coerce')
    V['prof_fair'] = np.where(V.pick == 'YES', V.sv, 1 - V.sv) - V.cost_x
    print(f"  void picks {len(V)}: Kalshi result counts {V.result.value_counts(dropna=False).to_dict()}; settle values "
          f"{V.sv.describe()[['min','50%','max']].round(3).to_dict()}; P&L if settled at Kalshi value {V.prof_fair.sum():+.2f} on {V.cost_x.sum():.2f}")
    T = pd.concat([S.assign(pp=S.prof_x), V.assign(pp=V.prof_fair)])
    r = cboot(T, 'pp', 'cost_x'); print(f"  ROI incl. voids at Kalshi fair value: {r['est']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] n {len(T)}")
except Exception as ex:
    print('  (kalshi_settle not ready:', ex, ')')

# ---- POST-HOC: exclude questionable-status players (game row q lists at log time)
GM['event_id'] = GM.event_id.astype(str)
qset = {(r.event_id, n) for r in GM.itertuples() for n in (list(r.q_home or []) + list(r.q_away or []))}
import re, unicodedata
def nn(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r'[^a-z ]', '', s.replace('-', ' ')); return ' '.join(t for t in s.split() if t not in ('jr','sr','ii','iii','iv','v'))
qn = {(e, nn(n)) for e, n in qset}
S['quest'] = [(e, nn(p)) in qn for e, p in zip(S.event_id.astype(str), S.player)]
print(f"\n[I] POST-HOC questionable: {len(qset)} questionable player-games listed; picks on them {int(S.quest.sum())}")
line(S[S.quest], 'questionable players')
line(S[~S.quest], 'excluding questionable (post-hoc)')
json.dump(res, open(OUT + 'splits.json', 'w'), indent=1, default=str)

import json, sys
import pandas as pd, numpy as np
sys.path.insert(0, '/Users/jonathanwang/agent-harness/lib')
D = '/Users/jonathanwang/.claude/quant-sports/data/'
L = [json.loads(l) for l in open(D + 'kalshi_prop_log.jsonl') if l.strip()]
G = [json.loads(l) for l in open(D + 'kalshi_prop_grades.jsonl') if l.strip()]
print('log rows', len(L), 'types', pd.Series([r.get('type') for r in L]).value_counts().to_dict())
gm = pd.DataFrame([r for r in L if r['type'] == 'game'])
R = pd.DataFrame([r for r in L if r['type'] == 'rung'])
for c in ('ts', 'tip_utc'):
    gm[c] = pd.to_datetime(gm[c], utc=True); R[c] = pd.to_datetime(R[c], utc=True)
print('game rows', len(gm), 'unique event_id', gm.event_id.nunique())
print('rung rows', len(R), 'unique ticker', R.ticker.nunique())
dup = R[R.ticker.duplicated(keep=False)]
print('dup tickers rows', len(dup), dup.event_id.nunique() if len(dup) else 0)
# minutes to tip, computed with Timedelta (not int64 // 1e9)
mtt = (gm.tip_utc - gm.ts) / pd.Timedelta(minutes=1)
print('game mins_to_tip (recomputed) min/median/max', mtt.min().round(1), mtt.median().round(1), mtt.max().round(1))
print('rung mins_to_tip logged min/max', R.mins_to_tip.min(), R.mins_to_tip.max())
print('any rung ts >= tip:', int((R.ts >= R.tip_utc).sum()))
gm['etdate'] = gm.tip_utc.dt.tz_convert('US/Eastern').dt.date
R['etdate'] = R.tip_utc.dt.tz_convert('US/Eastern').dt.date
per = gm.groupby('etdate').agg(games=('event_id', 'size'), rungs=('n_rungs', 'sum'), picks=('n_picks', 'sum'))
# ESPN slate
g = pd.read_parquet('/Users/jonathanwang/.claude/quant-sports/data/espn/nba/games_2027.parquet')
g['start_utc'] = pd.to_datetime(g.start_utc, utc=True)
g['etdate'] = g.start_utc.dt.tz_convert('US/Eastern').dt.date
sl = g.groupby('etdate').agg(slate=('event_id', 'size'))
per = sl.join(per, how='left').fillna(0)
per = per[(per.index >= pd.Timestamp('2026-11-01').date()) & (per.index <= pd.Timestamp('2026-11-20').date())]
# tip times PT for missed games
gm_ids = set(gm.event_id.astype(str))
g['logged'] = g.event_id.astype(str).isin(gm_ids)
miss = g[(~g.logged) & (g.etdate >= pd.Timestamp('2026-11-01').date()) & (g.etdate <= pd.Timestamp('2026-11-20').date())]
mm = miss.groupby('etdate').apply(lambda x: ', '.join(f"{r.away}@{r.home} {r.start_utc.tz_convert('US/Pacific'):%H:%M}" for r in x.itertuples()))
per['missed'] = mm
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 150)
print(per.to_string())
print('sim_tag counts', gm.sim_tag.value_counts().to_dict())
print('unmatched names per game: mean', gm.unmatched.apply(len).mean(), 'max', gm.unmatched.apply(len).max())
print('rungs with sim null', int(R.sim.isna().sum()), 'absent', int(R.absent.fillna(False).astype(bool).sum()))
# grades
GD = pd.DataFrame(G)
print('grade rows', len(GD), 'unique tickers', GD.ticker.nunique(), 'pending col' , 'pending' in GD)
dg = GD[GD.ticker.duplicated(keep=False)]
print('dup grade rows', len(dg))
if len(dg):
    conf = dg.groupby('ticker').agg(ny=('y', lambda s: s.dropna().nunique()), nv=('void', 'nunique'))
    print('conflicting dup grades y', int((conf.ny > 1).sum()), 'void', int((conf.nv > 1).sum()))
GD['graded_ts'] = pd.to_datetime(GD.graded_ts, utc=True)
print('graded batches (PT date):', GD.groupby(GD.graded_ts.dt.tz_convert('US/Pacific').dt.date).size().to_dict())
ungraded = R[~R.ticker.isin(GD.ticker)]
print('logged rungs not graded:', len(ungraded), 'games', ungraded.event_id.nunique(), sorted(ungraded.etdate.unique()))
# grades for tickers not in log
print('grade tickers not in log', int((~GD.ticker.isin(R.ticker)).sum()))

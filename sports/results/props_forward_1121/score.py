"""Score the Kalshi NBA prop-ladder forward paper trial against notes/kalshi_prop_ladders.md (props_forward_1121).
Independent re-implementation (does not import the logger); the logger's own `report` is run separately as a check."""
import json, sys
import numpy as np, pandas as pd
D = '/Users/jonathanwang/.claude/quant-sports/data/'
OUT = 'results/props_forward_1121/'
EDGE, MIN_VOL, MIN_PROJ, FEE_K = 0.08, 100.0, 20.0, 0.07
L = [json.loads(l) for l in open(D + 'kalshi_prop_log.jsonl') if l.strip()]
GM = pd.DataFrame([r for r in L if r['type'] == 'game'])
R = pd.DataFrame([r for r in L if r['type'] == 'rung'])
G = pd.DataFrame([json.loads(l) for l in open(D + 'kalshi_prop_grades.jsonl') if l.strip()])
for c in ('ts', 'tip_utc'):
    R[c] = pd.to_datetime(R[c], utc=True); GM[c] = pd.to_datetime(GM[c], utc=True)
G['graded_ts'] = pd.to_datetime(G.graded_ts, utc=True)
M = R.merge(G.drop(columns=['event_id']), on='ticker', how='left', validate='one_to_one')
M['graded'] = M.graded_ts.notna()
M['void'] = M['void'].astype('boolean')

def flag(sim, bid, ask, vol, mp):
    if pd.isna(sim) or pd.isna(mp) or vol < MIN_VOL or mp < MIN_PROJ:
        return None
    if sim >= round(ask + EDGE, 6) - 1e-9: return 'YES'
    if sim <= round(bid - EDGE, 6) + 1e-9: return 'NO'
    return None
M['pick_re'] = [flag(*x) for x in M[['sim', 'bid', 'ask', 'volume', 'min_proj']].itertuples(index=False)]
mis = (M.pick.fillna('-') != M.pick_re.fillna('-'))
print(f"[A] logged picks {int(M.pick.notna().sum())}; recomputed {int(M.pick_re.notna().sum())}; mismatches {int(mis.sum())}")
# absent players can't be picked (min_proj 0) -- check
print(f"    picks on absent players: {int((M.pick.notna() & M.absent.fillna(False).astype(bool)).sum())}")

yes = M.pick == 'YES'
px = np.where(yes, M.ask, 1 - M.bid)
M['px'] = px
M['fee_exact'] = FEE_K * px * (1 - px)
M['fee_round'] = np.ceil(np.round(100 * FEE_K * px * (1 - px), 8)) / 100
M['win'] = np.where(yes, M.y, 1 - M.y)
M['cost_x'] = M.px + M.fee_exact
M['cost_r'] = M.px + M.fee_round
M['prof_x'] = M.win - M.cost_x
M['prof_r'] = M.win - M.cost_r
M['clv_re'] = np.where(yes, M.close_yes - M.ask, M.bid - M.close_yes)
P_all = M[M.pick.notna()].copy()
S = P_all[P_all.graded & ~P_all.void.fillna(False) & P_all.y.notna()].copy()
print(f"[B] picks {len(P_all)}: graded {int(P_all.graded.sum())}, void {int(P_all.void.fillna(False).sum())}, "
      f"settled non-void {len(S)}, games {S.event_id.nunique()}; pending (11-20 games) {int((~P_all.graded).sum())}")
# consistency with the logger's stored profit / clv
d1 = (S.prof_x - S.profit).abs().max(); d2 = (S.clv_re - S.clv).abs().max()
print(f"    max |my profit - logged profit| {d1:.2e}; max |my clv - logged clv| {d2:.2e}")

def cboot(df, num, den=None, n=10000, seed=0):
    g = df.groupby('event_id').agg(s=(num, 'sum'), c=((den or num), 'sum' if den else 'size'))
    s, c = g.s.to_numpy(float), g.c.to_numpy(float)
    k = np.random.default_rng(seed).integers(0, len(g), (n, len(g)))
    bs = s[k].sum(1) / c[k].sum(1)
    return dict(est=float(s.sum() / c.sum()), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)),
                p_le0=float((bs <= 0).mean()), p_ge0=float((bs >= 0).mean()), n=int(c.sum()) if not den else len(df), games=len(g))

def judge(df, tag):
    roi = cboot(df, 'prof_x', 'cost_x'); roir = cboot(df, 'prof_r', 'cost_r')
    C = df.dropna(subset=['clv_re']); cv = cboot(C, 'clv_re')
    w = int(df.win.sum())
    if roi['est'] > 0 and roi['p_le0'] < .05 and cv['est'] > 0: v = 'PASS'
    elif roi['hi'] < 0 or (cv['est'] < 0 and cv['p_ge0'] < .05): v = 'KILL'
    else: v = 'NOT DISTINGUISHABLE'
    print(f"  {tag}: n {len(df)} (YES {int((df.pick=='YES').sum())} / NO {int((df.pick=='NO').sum())}), games {df.event_id.nunique()}, "
          f"W-L {w}-{len(df)-w}, units {df.prof_x.sum():+.2f} on {df.cost_x.sum():.2f}")
    print(f"     ROI exact-fee {roi['est']:+.4f} [{roi['lo']:+.4f}, {roi['hi']:+.4f}] p(ROI<=0) {roi['p_le0']:.4f};"
          f"  rounded-fee {roir['est']:+.4f} [{roir['lo']:+.4f}, {roir['hi']:+.4f}]")
    print(f"     CLV(last trade pre-tip) mean {cv['est']:+.4f} [{cv['lo']:+.4f}, {cv['hi']:+.4f}] n {len(C)} p(CLV>=0) {cv['p_ge0']:.4f}"
          f"  -> verdict {v}")
    return dict(n=len(df), games=int(df.event_id.nunique()), wins=w, units=float(df.prof_x.sum()), staked=float(df.cost_x.sum()),
                roi=roi, roi_rounded=roir, clv=cv, clv_n=len(C), verdict=v)

# ---- judge-at-n: grading batches in order
S['batch'] = S.graded_ts.dt.tz_convert('US/Pacific').dt.floor('h')
out = {}
batches = sorted(S.batch.unique())
cum = []
first150 = first300 = None
for b in batches:
    x = S[S.batch <= b]
    cum.append((b, len(x), x.event_id.nunique()))
    if first150 is None and len(x) >= 150 and x.event_id.nunique() >= 20: first150 = b
    if first300 is None and len(x) >= 300 and x.event_id.nunique() >= 20: first300 = b
print('[C] cumulative settled picks by grading batch:', [(f"{b:%m-%d %H}", n, g) for b, n, g in cum])
print(f"    first look (>=150 & >=20 games) at batch {first150}; second look (>=300) at batch {first300}")
out['first_look'] = judge(S[S.batch <= first150], f'FIRST LOOK @ {first150:%Y-%m-%d %H:%M} PT')
if first300 is not None:
    out['second_look'] = judge(S[S.batch <= first300], f'SECOND LOOK @ {first300:%Y-%m-%d %H:%M} PT')
out['full'] = judge(S, 'FULL SAMPLE (all graded through 11-19 games)')

# ---- calibration: all settled non-void rungs with a sim price, not absent
Q = M[M.graded & ~M.void.fillna(False) & M.y.notna() & M.sim.notna() & ~M.absent.fillna(False).astype(bool)].copy()
Q['b_sim'] = (Q.sim - Q.y) ** 2; Q['b_mid'] = (Q.mid - Q.y) ** 2; Q['d_b'] = Q.b_sim - Q.b_mid
cl = lambda p: np.clip(p, .01, .99)
Q['l_sim'] = -(Q.y * np.log(cl(Q.sim)) + (1 - Q.y) * np.log(1 - cl(Q.sim)))
Q['l_mid'] = -(Q.y * np.log(cl(Q.mid)) + (1 - Q.y) * np.log(1 - cl(Q.mid)))
Q['d_l'] = Q.l_sim - Q.l_mid
db, dl = cboot(Q, 'd_b'), cboot(Q, 'd_l')
print(f"[D] CALIBRATION rows {len(Q)}, games {Q.event_id.nunique()}: Brier sim {Q.b_sim.mean():.4f} mid {Q.b_mid.mean():.4f} "
      f"diff {db['est']:+.4f} [{db['lo']:+.4f}, {db['hi']:+.4f}];  logloss sim {Q.l_sim.mean():.4f} mid {Q.l_mid.mean():.4f} "
      f"diff {dl['est']:+.4f} [{dl['lo']:+.4f}, {dl['hi']:+.4f}]")
out['calib'] = dict(n=len(Q), games=int(Q.event_id.nunique()), brier_sim=float(Q.b_sim.mean()), brier_mid=float(Q.b_mid.mean()),
                    brier_diff=db, ll_sim=float(Q.l_sim.mean()), ll_mid=float(Q.l_mid.mean()), ll_diff=dl, by_stat={})
for st in ('pts', 'reb', 'ast'):
    x = Q[Q.stat == st]; a, b = cboot(x, 'd_b'), cboot(x, 'd_l')
    print(f"    {st}: n {len(x)} Brier sim {x.b_sim.mean():.4f} mid {x.b_mid.mean():.4f} diff {a['est']:+.4f} [{a['lo']:+.4f}, {a['hi']:+.4f}]"
          f"  LL diff {b['est']:+.4f} [{b['lo']:+.4f}, {b['hi']:+.4f}]")
    out['calib']['by_stat'][st] = dict(n=len(x), brier_diff=a, ll_diff=b)
bins = [0, .05, .15, .25, .35, .45, .55, .65, .75, .85, .95, 1.0001]
Q['bs'] = pd.cut(Q.sim, bins, right=False); Q['bm'] = pd.cut(Q.mid, bins, right=False)
cs = Q.groupby('bs', observed=False).agg(n_sim=('y', 'size'), sim=('sim', 'mean'), hit_s=('y', 'mean'))
cm = Q.groupby('bm', observed=False).agg(n_mid=('y', 'size'), mid=('mid', 'mean'), hit_m=('y', 'mean'))
cs.index = cs.index.astype(str); cm.index = cm.index.astype(str)
print(pd.concat([cs, cm], axis=1).round(3).to_string())
# same on the vol>=100 & proj>=20 universe (the rungs eligible to be picked)
E = Q[(Q.volume >= MIN_VOL) & (Q.min_proj >= MIN_PROJ)]
a = cboot(E, 'd_b')
print(f"    eligible universe (vol>=100, proj>=20): n {len(E)} Brier diff {a['est']:+.4f} [{a['lo']:+.4f}, {a['hi']:+.4f}]")
out['calib']['eligible'] = dict(n=len(E), brier_diff=a)
M.to_parquet(OUT + 'frame.parquet'); S.to_parquet(OUT + 'settled_picks.parquet'); Q.drop(columns=['bs', 'bm']).to_parquet(OUT + 'calib_rows.parquet')
json.dump(out, open(OUT + 'score.json', 'w'), indent=1, default=str)
print('wrote', OUT + 'score.json')

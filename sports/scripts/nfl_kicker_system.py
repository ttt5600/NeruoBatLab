"""NFL kicker OVER 3.5 points system (@linemakersports, Oct 2) -- PRE-REGISTERED DEV-ONLY test.

Pre-registration (written before any outcome was computed, 2026-10-29):

Claim: bet kicker OVER 3.5 kicker points when
  (a) the opposing defense allows >= 7.0 kicker points per game,
  (b) the kicker's offense averages >= 3.0 red-zone attempts per game,
  (c) the kicker is >= 90% accurate on field goals.
"Wins 90%."

Data: nflverse pbp, DEV seasons 2018-2024, regular season only. 2025-26 NOT scored (sealed elsewhere).
Unit: one kicker-team-game. Kicker = the team's kicker with the most FG+XP attempts in that game; if the
team attempted no FG/XP, the kicker of the team's most recent prior game (kicker gets 0 points).
Outcome: kicker points = 3*FG made + 1*XP made (incl. OT); hit = points >= 4.
Features (season-to-date, games strictly BEFORE this one, same season, no lookahead):
  (a) opp_kpa  = mean kicker points scored by the opponent's opponents' kickers in the opponent's prior games.
  (b) off_rz   = mean per game of the kicker's team's drives with >= 1 scrimmage/FG play at yardline_100 <= 20
                 ("red-zone attempts" read as red-zone trips; the claim does not define it).
  (c) fg_pct   = kicker's FG made / FG attempted in his prior games this season (any team); needs >= 1 FGA.
  Eligibility: kicker, his team and the opponent each have >= 4 prior games this season.
Thresholds exactly as stated: opp_kpa >= 7, off_rz >= 3, fg_pct >= 0.90.

Report:
  1. hit rate (pts >= 4) qualifying vs all eligible vs non-qualifying, Wilson 95% CIs, n, per-season.
     Matched baseline: logistic P(hit | implied team total from nflverse spread_line/total_line) fit on all
     eligible games; qualifying residual = hit - predicted (does the system add anything beyond the total?).
  2. break-even American odds implied by the qualifying hit rate.
  3. Market: ESPN core odds 2024 has 'Total Kicking Points' (ESPN BET, main line 4.5-9.5, NOT 3.5).
     (i) ROI betting OVER the main line at openers in qualifying vs all priced games;
     (ii) market-implied P(pts >= 4): logistic of hit on logit(de-vigged opener P(over)) + line, fit on ALL
          priced 2024 games; qualifying residual vs that. No over-3.5 alt prices exist in our data.
One trial; ledger dataset nfl_kicker_system. Pass bar: residual vs market > 0 with p < 0.05 AND hit rate
above the break-even of the assumed alt price. No orders.
"""
import json, hashlib, datetime as dt
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

SEASONS = range(2018, 2025)
COLS = ['game_id', 'season_type', 'week', 'game_date', 'home_team', 'away_team', 'posteam', 'defteam',
        'play_type', 'kicker_player_id', 'kicker_player_name', 'field_goal_result', 'extra_point_result',
        'field_goal_attempt', 'extra_point_attempt', 'yardline_100', 'fixed_drive', 'spread_line', 'total_line']


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def american(p):
    return -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p


def build_season(s):
    p = pd.read_parquet(f'data/pbp/play_by_play_{s}.parquet', columns=COLS)
    p = p[p.season_type == 'REG']
    games = p.groupby('game_id').agg(week=('week', 'first'), date=('game_date', 'first'), home=('home_team', 'first'),
                                     away=('away_team', 'first'), spread=('spread_line', 'first'),
                                     total=('total_line', 'first')).reset_index()
    tg = pd.concat([games.assign(team=games.home, opp=games.away, ishome=1),
                    games.assign(team=games.away, opp=games.home, ishome=0)])
    tg['itt'] = np.where(tg.ishome == 1, tg.total / 2 + tg.spread / 2, tg.total / 2 - tg.spread / 2)
    k = p[(p.field_goal_attempt == 1) | (p.extra_point_attempt == 1)].copy()
    k['pts'] = 3 * (k.field_goal_result == 'made') + 1 * (k.extra_point_result == 'good')
    k['fga'] = (k.field_goal_attempt == 1).astype(int); k['fgm'] = (k.field_goal_result == 'made').astype(int)
    kk = k.groupby(['game_id', 'posteam', 'kicker_player_id']).agg(
        pts=('pts', 'sum'), fga=('fga', 'sum'), fgm=('fgm', 'sum'), n=('pts', 'size'),
        name=('kicker_player_name', 'first')).reset_index()
    team_pts = kk.groupby(['game_id', 'posteam']).pts.sum().rename('team_kpts')  # all kickers of team
    main = kk.sort_values('n', ascending=False).drop_duplicates(['game_id', 'posteam'])
    tg = tg.merge(main.rename(columns={'posteam': 'team'}), on=['game_id', 'team'], how='left')
    tg = tg.merge(team_pts.reset_index().rename(columns={'posteam': 'team'}), on=['game_id', 'team'], how='left')
    tg = tg.sort_values(['team', 'week'])
    tg['kicker_player_id'] = tg.groupby('team').kicker_player_id.ffill()
    tg[['pts', 'fga', 'fgm', 'team_kpts']] = tg[['pts', 'fga', 'fgm', 'team_kpts']].fillna(0)
    sc = p[p.play_type.isin(['run', 'pass', 'field_goal', 'qb_kneel', 'qb_spike']) & (p.yardline_100 <= 20)]
    rz = sc.groupby(['game_id', 'posteam']).fixed_drive.nunique().rename('rz').reset_index()
    tg = tg.merge(rz.rename(columns={'posteam': 'team'}), on=['game_id', 'team'], how='left').fillna({'rz': 0})
    tg = tg.sort_values(['team', 'week'])
    g = tg.groupby('team')
    tg['team_prior'] = g.cumcount()
    tg['off_rz'] = g.rz.transform(lambda x: x.shift().expanding().mean())
    # kicker points allowed by a defense = opponent's team_kpts in that game
    a = tg[['game_id', 'team', 'week']].merge(
        tg[['game_id', 'opp', 'team_kpts']].rename(columns={'opp': 'team', 'team_kpts': 'allowed'}),
        on=['game_id', 'team']).sort_values(['team', 'week'])
    a['def_kpa'] = a.groupby('team').allowed.transform(lambda x: x.shift().expanding().mean())
    a['def_prior'] = a.groupby('team').cumcount()
    tg = tg.merge(a[['team', 'week', 'def_kpa', 'def_prior']].rename(
        columns={'team': 'opp', 'def_kpa': 'opp_kpa', 'def_prior': 'opp_prior'}), on=['opp', 'week'], how='left')
    tg = tg.sort_values(['kicker_player_id', 'week'])
    gk = tg.groupby('kicker_player_id')
    tg['k_prior'] = gk.cumcount()
    cfga = gk.fga.transform(lambda x: x.shift().fillna(0).cumsum())
    cfgm = gk.fgm.transform(lambda x: x.shift().fillna(0).cumsum())
    tg['fg_pct'] = np.where(cfga > 0, cfgm / cfga.replace(0, np.nan), np.nan)
    tg['season'] = s
    return tg


def main():
    df = pd.concat([build_season(s) for s in SEASONS], ignore_index=True)
    df['hit'] = (df.pts >= 4).astype(int)
    el = df[(df.k_prior >= 4) & (df.team_prior >= 4) & (df.opp_prior >= 4) & df.fg_pct.notna()].copy()
    el['qual'] = ((el.opp_kpa >= 7) & (el.off_rz >= 3) & (el.fg_pct >= 0.90)).astype(int)
    lr = LogisticRegression(C=1e6).fit(el[['itt']].fillna(el.itt.mean()), el.hit)
    el['pred_itt'] = lr.predict_proba(el[['itt']].fillna(el.itt.mean()))[:, 1]
    out = {}
    for lab, d in [('all_eligible', el), ('qualifying', el[el.qual == 1]), ('non_qualifying', el[el.qual == 0])]:
        k, n = int(d.hit.sum()), len(d)
        out[lab] = dict(n=n, hits=k, rate=k / n, ci=wilson(k, n), mean_pts=d.pts.mean(),
                        pred_itt=d.pred_itt.mean())
    q = el[el.qual == 1]
    res = q.hit - q.pred_itt
    out['qual_resid_vs_itt'] = dict(mean=res.mean(), p_one_sided=stats.ttest_1samp(res, 0, alternative='greater').pvalue)
    out['per_season'] = {int(s): dict(n=len(d), rate=d.hit.mean(), base=el[el.season == s].hit.mean(),
                                      resid=(d.hit - d.pred_itt).mean())
                         for s, d in q.groupby('season')}
    out['breakeven_american_qual'] = american(out['qualifying']['rate'])
    out['breakeven_american_all'] = american(out['all_eligible']['rate'])

    # --- market: ESPN 2024 Total Kicking Points (ESPN BET openers)
    e = pd.read_parquet('data/espn/nfl/graded_2024.parquet')
    e = e[e.market == 'Total Kicking Points (incl. overtime)']
    w = e.pivot_table(index=['event_id', 'athlete_id', 'line_open'], columns='side',
                      values=['dec_open', 'actual', 'won', 'push'], aggfunc='first').reset_index()
    w.columns = ['_'.join(c).strip('_') for c in w.columns]
    w = w.dropna(subset=['dec_open_over', 'dec_open_under'])
    po, pu = 1 / w.dec_open_over, 1 / w.dec_open_under
    w['p_over'] = po / (po + pu); w['actual'] = w.actual_over
    box = pd.read_parquet('data/espn/nfl/box_2024.parquet')[['event_id', 'athlete_id', 'team_id']].drop_duplicates(
        ['event_id', 'athlete_id'])
    gm = pd.read_parquet('data/espn/nfl/games_2024.parquet')
    w['event_id'] = w.event_id.astype(str); w['athlete_id'] = w.athlete_id.astype(str)
    box['event_id'] = box.event_id.astype(str); box['athlete_id'] = box.athlete_id.astype(str)
    gm['event_id'] = gm.event_id.astype(str)
    w = w.merge(box, on=['event_id', 'athlete_id'], how='left')
    w = w.merge(gm[['event_id', 'start_utc', 'home', 'away', 'home_team_id', 'away_team_id', 'season_type']], on='event_id')
    tid = w.team_id.astype(str)
    w['team'] = np.where(tid == w.home_team_id.astype(str), w.home, np.where(tid == w.away_team_id.astype(str), w.away, None))
    fix = {'WSH': 'WAS', 'LAR': 'LA'}
    w['team'] = w.team.replace(fix); w['home'] = w.home.replace(fix); w['away'] = w.away.replace(fix)
    e24 = el[el.season == 2024].copy()
    e24['key'] = e24.home + '_' + e24.away + '_' + e24.date.astype(str)
    w['date_et'] = (pd.to_datetime(w.start_utc).dt.tz_convert('America/New_York')).dt.strftime('%Y-%m-%d')
    w['key'] = w.home + '_' + w.away + '_' + w.date_et
    m = w.merge(e24[['key', 'team', 'qual', 'pts', 'hit', 'pred_itt']], on=['key', 'team'], how='inner')
    m['agree'] = (m.actual == m.pts)
    m['hit4'] = (m.actual >= 4).astype(int)
    X = np.c_[np.log(m.p_over / (1 - m.p_over)), m.line_open]
    mk = LogisticRegression(C=1e6).fit(X, m.hit4)
    m['p4_mkt'] = mk.predict_proba(X)[:, 1]
    prof = np.where(m.push_over == 1, 0, np.where(m.won_over == 1, m.dec_open_over - 1, -1))
    m['prof_over'] = prof
    mq = m[m.qual == 1]
    out['market_2024'] = dict(
        priced_matched=len(m), actual_pts_agree=float(m.agree.mean()), unmatched_priced=len(w) - len(m),
        line_open_counts=m.line_open.value_counts().to_dict(),
        all=dict(n=len(m), hit4=m.hit4.mean(), p4_mkt=m.p4_mkt.mean(), over_roi=m.prof_over.mean()),
        qual=dict(n=len(mq), hit4=mq.hit4.mean(), p4_mkt=mq.p4_mkt.mean(), over_roi=mq.prof_over.mean(),
                  resid=(mq.hit4 - mq.p4_mkt).mean(),
                  p_one_sided=stats.ttest_1samp(mq.hit4 - mq.p4_mkt, 0, alternative='greater').pvalue if len(mq) > 2 else None,
                  fair_american=american(mq.p4_mkt.mean())),
    )
    print(json.dumps(out, indent=1, default=float))
    rec = dict(ts=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'), strategy='kicker_over35_linemaker',
               sha=hashlib.sha1(open(__file__, 'rb').read()).hexdigest()[:12], dataset='nfl_kicker_system',
               hypothesis='pre-registered in scripts/nfl_kicker_system.py; kicker pts>=4 when opp allows>=7 kpts, off rz>=3, fg%>=.90',
               source='social media @linemakersports Oct 2', status='ok', split='DEV', trial_number=1, study_trials=1,
               bets=out['qualifying']['n'], hit=out['qualifying']['rate'], base=out['all_eligible']['rate'],
               resid_vs_itt=out['qual_resid_vs_itt']['mean'], p=out['qual_resid_vs_itt']['p_one_sided'],
               breakeven_american=out['breakeven_american_qual'], market_2024=out['market_2024'])
    return out, rec, el


if __name__ == '__main__':
    import sys
    out, rec, el = main()
    if '--log' in sys.argv:
        with open('lab/ledger.jsonl', 'a') as f:
            f.write(json.dumps(rec, default=float) + '\n')

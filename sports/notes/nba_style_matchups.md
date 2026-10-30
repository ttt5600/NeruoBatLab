# NBA style matchups: shot-zone profile x opponent zone defense (+ passing)

User hypothesis 2026-10-30: markets under-price STYLE MATCHUPS -- a team's shot-location profile (rim 0-3 ft, short,
mid, long 2, corner 3, above-break 3) against the opponent's defense by zone (frequency and FG% allowed), rim
protection and passing (assist share, turnovers / steals = passing-lane defense); also player zone profile x
opponent zone defense for player points props.

Not a redo: SITUATIONAL_FACTORS F "Style matchups (3PA, FT rate, ORB, TOV, transition)" was box-score level
(best 3PA x 3PA allowed +0.23 p .07, DEAD); "Paint points allowed share" is a marginal team trait (model feature).
Zone-by-zone interaction from play-by-play shot locations has not been tested.

Script `scripts/nba_style_matchups.py` (parse / calib / reliability / build / dist / score). Data `data/nba_pbp/`
(gitignored). Results `results/nba_style_matchups/`. Ledger `lab/ledger.jsonl`, dataset `nba_style_matchups`.

## PRE-REGISTRATION (written 2026-10-30 BEFORE the reliability check, before any signal distribution, and before any
## signal was joined to an outcome, a line or a prop price)

What had been looked at before writing (structure only): ESPN summary play schema; coverage and calibration
(`calib`, results/nba_style_matchups/calib.json) -- no outcome, line or signal was computed.

### Data
- ESPN site summary `plays`, all regular-season finals 2020-21..2025-26 (every one already in data/cache; no fetch).
  stats.nba.com shotchartdetail timed out (curl 25 s, no response) -> not used; passing comes from ESPN box (AST, TOV,
  STL) and play text (assisted makes).
- FGA = shootingPlay and not a free throw. Distance = the text "N-foot" when present (65-73% of FGA), else the
  coordinate distance from the rim at (x 25, y 1). Three = text "three point" or made for 3, OR a miss in the corner
  box (y <= 9, |x - 25| >= 22) or >= 24 ft (ESPN omits "three point" on many missed threes). Zones: threes -> corner (y <= 9) /
  above-break; twos -> rim <= 3 ft, short 4-9, mid 10-15, long 2 >= 16.
- Lines: `lines_{S}.parquet`, one book per event (DraftKings if both MLs else ESPN BET; plain fields = close), as in
  nba_clock_calendar_arena. Props: `graded_{2023..2025}` at the OPEN (line_open / dec_open; "current" is partly
  in-game), DK 2022-23 & 2023-24, ESPN BET 2024-25; one quote per player-game (over/under pair at the open line with
  no-vig p_over closest to .5), pushes leave; joined to pbp on ESPN (event_id, athlete_id).
- Splits: history 2020-21 (thresholds, EB priors only); DEV 2021-22..2024-25 (props 2022-23..2024-25); SEALED
  2025-26, scored once only for trials that pass DEV (user authorised).

### Point-in-time profiles (strictly games dated before the game's ET date; same season)
- Team offense per zone z: FGA share m_O(z), FG% -> points per shot p_O(z) = value(z) x FG%. Team defense per zone:
  share of opponents' FGA m_D(z) and their points per shot p_D(z). Passing: offense assisted share (AST/FGM, box),
  offense TOV/poss, defense forced TOV/poss, defense STL/poss, rim protection (blocked share of opponents' rim FGA).
- Empirical-Bayes shrink toward league: rate = (count + k L) / (n + k). k = mean binomial variance / between-team
  variance, method of moments on full-season team totals of season s-1 (2020-21 for 2021-22). If the between-team
  variance estimate is <= 0, the metric's deviation is set to 0. League L = current season to date when >= 300
  team-games, else season s-1.
- **Reliability gate (run FIRST, `reliability`, no outcomes):** for each team-season of the DEV seasons, split its
  games by parity (odd/even game number), compute the raw metric on each half, demean within season, Pearson r over
  team-seasons, 95% CI by team-season bootstrap (2,000). A metric is STABLE if r >= 0.30 and its CI lower bound > 0.
  Only stable metrics enter the matchup term; an unstable component's deviation is set to 0 (the gate is fixed
  before the gate is run; nothing about it changes after).

### Matchup term
With deviations from league a = m_O - L_m, c = p_O - L_p (offense T), b = m_D - L_m, e = p_D - L_p (defense D):
the zone-additive model expected points per shot = sum_z (L_m + a + b)(L_p + c + e). Subtracting the offense-only and
defense-only marginals (what a plain rating already carries) leaves the pure style interaction
**I(T vs D) = FGA_L x sum_z [ a_z e_z + b_z c_z ]** in points per game (FGA_L = league FGA per team-game to date).
Game terms: I_total = I(H vs A) + I(A vs H); I_margin = I(H vs A) - I(A vs H).
Player term (props): the player's zone mix and points per shot over his last 82 team-games played (any season,
strictly earlier), shrunk to league with player-level EB k (player-seasons of s-1 with >= 100 FGA); FGA_p = his mean
FGA in his last 20 games; needs >= 10 prior games. adj_p = FGA_p x sum_z [a_pz e_z(D) + b_z(D) c_pz].
Passing interaction (descriptive in M4 only, not a trial): z(off assisted share) x z(def steal rate).

### Trials (k = 4, Bonferroni one-sided bar p < 0.05/4 = **0.0125**; flat 1u)
- **M1 totals**: matchup-adjusted total = close total + I_total (the close carries the marginals; I_total is the
  claimed un-priced part). OVER if I_total >= X1_s, UNDER if I_total <= -X1_s, at the close total and price.
  X1_s = 80th percentile of |I_total| over all games of season s-1 (walk-forward, PIT values; ~20% of games).
- **M2 sides**: home ATS if I_margin >= X2_s, away ATS if <= -X2_s, at the close spread and price; X2_s the same
  construction on |I_margin|.
- **M3 player points props at OPEN**: OVER if adj_p >= Y_s, UNDER if adj_p <= -Y_s; Y_s = 90th percentile of |adj_p|
  over all eligible player-games of season s-1 (box/pbp, minutes > 0).
- **M4 information**: OLS actual margin ~ 1 + close home margin + I_margin, and actual total ~ 1 + close total +
  I_total, all DEV games with a close. Coefficient gamma on I; each regression at bar/2 = 0.00625 one-sided (gamma > 0);
  M4 passes if either passes AND its gamma > 0 in both season halves.
- Metric: observed ROI at the stated prices (pushes 0, kept), 95% CI = percentile of a game-clustered bootstrap
  (10,000, seed 2026); props clustered by game AND by date (p = the larger). One-sided p = share of resampled ROI <= 0.
  DEV PASS: p < 0.0125 AND ROI > 0 in both halves (game lines: H1 = 2021-22 + 2022-23, H2 = 2023-24 + 2024-25;
  props: median bet date). SEALED (passed trials only): p < 0.05 / n_passed and ROI > 0.
- Break-even at -110 is 52.38%; props hold ~7% at the open. A DEV pass is a bug until bet rows are inspected
  (PIT shift, team-id joins, spread sign, home/away swap).
- KILL: 0/4 DEV passes -> zone/passing style matchups DEAD; one line for SITUATIONAL_FACTORS F.

### nba_sim --matchup (descriptive, not a trial)
`nba_sim.py validate --season 2024-25 --matchup` adds I_margin and I_total (coefficient 1, not fitted) to the sim's
mean margin and total. Reported: MAE vs close and vs actual with and without, paired game-bootstrap difference.
Improvement claimed only if the paired CI excludes 0.

Prior: low. Prior style-interaction tests were 0 for N; zone defense FG% (especially 3P% allowed) is known to be
mostly noise, so the gate will likely zero part of the term; the market sets totals from pace x efficiency already.

## Result
(filled in below after the run; nothing above edited after scoring)

### Run 2026-10-30: **DEAD, 0/4 pass** (bar p < 0.0125). SEALED 2025-26 not scored (nothing passed).
Outputs: results/nba_style_matchups/{calib,reliability,build_info,score}.json, bets_M{1,2,3}_dev.csv.
Ledger: 4 rows, dataset nba_style_matchups (style_M1..M4_dev), sha 6ed1fac69355 = the script at scoring time;
live_terms()/ensure_season_shots() for `nba_sim game --matchup` were added afterwards (scoring code unchanged).

**Coverage / calibration** (calib.json). Summaries for every regular-season final 2020-21..2025-26 were already
cached (1081 / 1231 / 1231 / 1232 / 1234 / 1235 games, all parsed, 0 missing). Shooter is on the play's team in 99.99-100% of
FGA. pbp FGA = box FGA exactly in 97-99% of team-games (dev), FGM 99.8-100%, made 3s 99.8-100%, assists 99.4-99.8%.
3PA after the missed-three fix: exact 94-96%, within 1 in 99.8-99.9% (before the fix, text alone was exact in only
11-23%: ESPN omits "three point" on many missed threes). Coordinates: rim at (25, ~1); coordinate vs text distance MAE 0.66 ft
(2021-23) / 0.39 ft (2024-26). 3PT check: corner-three text distances 1st/5th pct 22/22 ft, median 23 (line is 22 ft);
above-break 1st pct 24, 5th 24-25, median 26 (line 23.75; ESPN rounds); 0.00-0.04% of threes listed under 22 ft; no twos > 24 ft after the fix. Turnovers from pbp match box only
~30-45% exactly (team turnovers), so passing metrics use box AST/TOV/STL.
**Odd:** 2025-26 (sealed) pbp FGA = box exactly in only 61% of team-games (mean +0.47 FGA/team-game); dev seasons 97-99%.
Not used here; check before any 2025-26 pbp work.

**Reliability gate** (DEV team-seasons n = 120, odd/even games, season-demeaned):
| zone | off freq | off FG% | def freq allowed | def FG% allowed |
|---|---|---|---|---|
| rim | .853 | .760 | .840 | **.553** |
| short 4-9 | .785 | .618 | .831 | **.405** |
| mid 10-15 | .922 | .428 | .745 | .139 [-.007, .288] FAIL |
| long 2 | .945 | .307 | .568 | .113 FAIL |
| corner 3 | .790 | .082 FAIL | .847 | -.020 FAIL |
| above-break 3 | .930 | .357 | .718 | .182 [.037, .323] FAIL (< .30) |
Passing: off assisted share .826, def assisted share allowed .669, off TOV/poss .777, def forced TOV .763, def
STL/poss .707, rim protection (blocked share of opp rim FGA) .643 -- all stable. Lesson: WHERE a defense lets you
shoot is a stable trait; how well you shoot there against it is stable only at the rim and short range.
Defensive 3P% allowed is noise (corner -0.02).

**Size of the term (before outcomes).** SD of I_total 0.24-0.26 pts per game, I_margin 0.19-0.23; the plain
zone marginal term has SD ~2.3-2.5 and correlates ~0 (|r| <= .05) with I. Walk-forward thresholds X1 0.28-0.31,
X2 0.24-0.27 pts; mean |I| of the bets 0.44 (M1) / 0.37 (M2). Player adj_p SD 0.045-0.060 pts (Y 0.07-0.09).
An interaction is a product of two small deviations (mix +-3-5 pp x points-per-shot +-0.05-0.1), so it is
second-order. A whole-percentile signal is ~1/3 of a point.

| trial | bets | ROI (game-cluster 95% CI) | p | halves | pass |
|---|---|---|---|---|---|
| M1 totals, close | 993 | -3.16% [-8.92, +2.67] | .857 | -3.6% / -2.8% | no |
| M2 sides ATS, close | 1,034 | -2.81% [-8.56, +2.97] | .828 | -4.0% / -1.8% | no |
| M3 player points props, OPEN | 6,571 | -7.69% [-9.95, -5.46] (date CI [-9.95, -5.42]) | 1.00 | -6.1% / -9.9% | no |
| M4 margin gamma | n 4,920 | -0.25 [-2.07, +1.58] (OLS se 0.90) | .601 | +0.22 / -0.67 | no |
| M4 total gamma | n 4,918 | +0.39 [-1.56, +2.30] (OLS se 1.03) | .347 | -0.40 / +1.11 | no |
M3 hit rate minus no-vig p: -0.6 pp; mean hold 6.9%. Props join: 46,889 open quotes, 43,859 matched (the 3,030
unmatched are playoffs/play-in 3,019 + 11), graded actual == box points 100%. M4: close coefficients 1.013 / 1.010.

Descriptive (not trials): (actual - close total) by I_total quintile +0.75 / +0.48 / +0.50 / +0.50 / +0.70 (U-shaped,
not monotone); margin residual by I_margin quintile -0.08 / +0.10 / -0.38 / -0.51 / +0.36. Passing interaction
(off assisted share x opp steal rate) vs close residual: margin +0.05 pts/SD [-0.34, +0.44], total -0.34 [-0.82,
+0.16]. Plain zone marginal (offense + defense, no interaction) vs close margin residual +0.000 pts/SD [-0.34,
+0.39] -- the market fully prices zone profiles as team traits. Post hoc only: M1 overs +2.8% (503) vs unders
-9.3% (490); not a rule.

**Power caveat (honest reading of M4):** gamma CIs (+-1.8 to 1.9) include both 0 and 1. So the data cannot say whether the market
ignores this term or prices it. But it does not matter for betting: even fully unpriced (gamma = 1), a bet at the
threshold carries ~0.3-0.45 pts of edge, ~1 pp of cover probability, against the 2.4 pp needed at -110.
The term is too small to bet, whether or not it is priced.

**nba_sim --matchup** (validate 2024-25, ACT set, n 1,147, paired, model means, diff = with - without):
spread vs close 2.308 -> 2.321 (+0.013 [-0.000, +0.026]); spread vs actual 10.844 -> 10.849 (+0.006 [-0.008,
+0.019]); total vs close 3.036 -> 3.041 (+0.005 [-0.010, +0.021]); total vs actual 14.429 -> 14.423 (-0.006
[-0.021, +0.010]) (results/nba_sim/validate_2025_matchup.json; a first run before a filename bug fix gave the same to
+-0.002). No improvement (all CIs cross 0; it moves slightly AWAY from the close on spreads). Keep the flag off by default.

**Why it fails (the lesson).** The hypothesis needs two things: a stable style interaction, and a market that
ignores it. Part 1 half-holds. Zone frequencies are very stable on both ends, but defensive shooting allowed is
stable only near the rim. And the interaction (a product of deviations) is ~0.25 pts per game: an order of magnitude
smaller than the marginal team traits the close already prices (zone marginal vs close residual = 0.000/SD).
Part 2 cannot be resolved with this power, and is moot at -110. Player level is smaller still (0.05 pts),
so M3 just pays the ~7% prop hold.

KILL applied: zone/passing style matchups DEAD. Suggested SITUATIONAL_FACTORS F line: "Shot-zone x zone-defense
interaction (pbp, reliability-gated) + passing x steals | NBA | YES | M1 -3.2%/993, M2 -2.8%/1,034, M3 props -7.7%/6,571,
M4 gamma CIs include 0 (and 1); term SD ~0.25 pts | nba_style_matchups.md | DEAD (too small)".

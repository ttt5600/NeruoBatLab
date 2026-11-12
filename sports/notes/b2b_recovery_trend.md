# b2b_recovery_trend -- did the NBA back-to-back penalty shrink faster than the market's B2B adjustment? (2026-11-12)

Agent topic `b2b_recovery_trend`, research subagent for the quant-trader lead. DEV ONLY. 2025-26 NBA is held out and is
never loaded (rows with season >= 2025 dropped at read). No orders, no git.

## Question (user hypothesis 11-12)
Recovery science / load management improved, so the real B2B penalty shrank; if the closing line still marks B2B teams
down by the old amount, then backing B2B teams ATS (or B2B-game overs) became +EV in recent seasons.

Prior work: scripts/nba_angles.py N1 rest_edge_ats (FADE the B2B team vs rested opp, dev 2007-08..2018-19) and N4 b2b_unders
lost about the vig (-1.5..-3.9%). Those were fixed angles, never a time trend, and never on 2019-20+.
Note: nba_angles.py's 2017 prereg called 2019-20+ "sealed"; since then the lead / other agents (L30 paint, early-season)
used 2019-20..2024-25 as dev, and the lead's brief for this agent names 2019-20..2024-25 as dev. Following the brief.

## Data
- Lines + scores: data/nba_odds/OddsData.sqlite (SBR scrape, kyleskom repo). Tables odds_YYYY-YY_new for 2007-08..2022-23
  (ISO dates), `2023-24`, `2024-25`. Closing spread (unsigned -> signed from the ML favourite, as in nba_angles.py),
  closing OU, Points, Win_Margin. All spread/total bets priced at -110 (file has no spread/total prices).
- Regular season only. Season end dates: data/nba_games.parquet (ESPN, game_type) for 2015-16+, hard-coded official
  regular-season end dates for 2007-08..2014-15.
- Rest computed from each team's previous game date in the same season table (validated against Days_Rest_* columns).
- Travel: great-circle miles between yesterday's venue and today's venue (hard-coded arena coordinates by franchise).

## Definitions
- B2B = team played the previous calendar day (rest = 1). 3in4 = this is the team's 3rd game in 4 nights.
- Bubble = 2020-07-30..2020-10-11 games (neutral site, no travel): EXCLUDED everywhere (pre-registered).
- COVID 2020-21 (no / limited fans, compressed schedule): INCLUDED in the main analysis; sensitivity reported excluding it.

## Analysis (pre-registered 2026-11-12, before any outcome was computed)
Per season s (2007-08..2024-25) and pooled with a linear season trend, OLS with team-season strength fixed effects
(home row +1 / away row -1 dummies for margin; +1/+1 for totals), SEs clustered by game date:
- (a) realized: home margin ~ HCA + team-season FE + b_r * (B2B_home - B2B_away) [+ 3in4 diff, travel diff in a 2nd spec]
- (b) market:   closing home spread ~ same RHS -> b_m (market's B2B adjustment, positive spread = home favoured)
- (c) residual: (margin - spread) ~ same RHS -> b_c = b_r - b_m (ATS residual of the B2B differential)
- totals: Points / OU / Points-OU ~ team-season total FE + n_B2B (0,1,2)
Trend = coefficient on d x (season - 2016) in the pooled model.

### Success (ALL needed for "ready-for-sealed-ask")
S1 realized B2B penalty trends toward 0: pooled trend in b_r > 0 (B2B-team perspective penalty shrinking), and
S2 residual trend for the B2B team is positive with 95% clustered CI excluding 0 (market adjustment shrinks slower), and
S3 a pre-registered bet rule passes the edgelab dev bar on dev 2019-20..2024-25 (bubble excluded):
   B1 (family nba_sides): team on the 2nd night of a B2B whose opponent is NOT on a B2B -> back the B2B team ATS at the
      close, -110, pushes = 0.
   B2 (family nba_totals): game with >= 1 team on a B2B -> OVER at the closing total, -110.
   Bar = edgelab family bar after registration (nba_sides ~0.05/199, nba_totals ~0.05/33); also halves positive.
### Kill
Dead if S2's CI includes 0 or is negative AND both B1, B2 have dev ROI CIs that include or sit below 0.
"Lead" if S2 passes but no rule passes the bar, or a rule has p < 0.05 but not the bar.
### Sensitivity (reported, not decision-making)
Excluding 2020-21; B2B-team-on-road only; 3in4; travel > 1000 mi.

## NHL (only if cheap)
Same realized vs market trend on goals / closing ML-implied if a lines file with rest exists. Notes/nhl_simple_model.md
already finds b2b -0.24 goals, priced (dev, vs opening line).

## Log
- 2026-11-12 prereg written before any computation.
- Build (`scripts/b2b_recovery_trend.py build` -> results/b2b_recovery_trend/games.parquet, build_info.json): 21,454 regular-season
  games 2007-08..2024-25 (bubble out, 21 bad-line rows dropped by a line-validity filter: OU<150/>300, swapped OU/spread, the
  2012-04-21 160/5 placeholders). Own rest agrees with the file's Days_Rest 99.85%. corr(spread, margin) .46.
  Team names are consistent within each season (30 per season), so rest is not split across franchise names.
- Wrong turn: the first trend run had junk OU/spread rows (OU 1955.5, spread 216) -- 2016 and 2019 market CIs blew up to
  +-5..10 pts. Filter added (no outcome used); headline numbers moved < 0.1 pt.

## Results (`scripts/b2b_recovery_trend.py trend` -> results/b2b_recovery_trend/trend.json)
Coefficient = effect on the B2B team (pts of margin), team-season FE, SEs clustered by date. Pooled with linear trend
(level at 2016-17, slope per season):

| quantity | level 2016 | trend / season |
|---|---|---|
| realized margin | -1.99 [-2.31,-1.66] | -0.053 [-0.117,+0.011] (not distinguishable; point estimate = penalty GROWING) |
| market (closing spread) | -1.35 [-1.43,-1.27] | -0.058 [-0.073,-0.042] (market adjustment GREW) |
| ATS residual (margin-spread) | -0.63 [-0.97,-0.30] | +0.004 [-0.060,+0.069] (not distinguishable from 0) |
| totals realized / market / residual (per B2B team) | -0.15 / -0.17 / +0.02 | -0.059 / -0.025 / -0.035, all CIs cross 0 |

Per season: realized penalty ~ -1.1..-1.9 in 2012-2016, -2.4..-3.1 in 2021-2024; market -0.8..-1.3 in 2010-15, -2.0..-2.3
in 2022-24. Excluding 2020-21: same (residual trend -0.004 [-0.070,+0.062]). With 3in4 and travel controls: realized
3in4 -0.34 [-0.66,-0.03], travel -0.71/1000 mi [-1.60,+0.17]; market prices both (3in4 -0.13, travel -0.33); residual
trend +0.003 [-0.062,+0.068].
**S1 FAIL** (realized penalty did not shrink), **S2 FAIL** (residual trend CI includes 0).

## Bet rules (edgelab, registered before evaluation: b2b_t1_back_b2b_ats-b513951f, b2b_t2_b2b_overs-0b5012cc)
- B1 back B2B team ATS, dev 2019-20..2024-25: n 1,787, ROI -8.06% [-12.58,-3.42], halves -7.2% / -8.9% -> dead.
  By season: 2019 -6.5%, 2020 +2.0%, 2021 -16.3%, 2022 -3.5%, 2023 -10.1%, 2024 -13.2%.
- B2 overs in B2B games: n 2,113, ROI -3.78% [-7.68,+0.08], halves -4.1% / -3.4% -> dead.
- Post hoc (NOT a trial, do not bet): the mirror -- fade the B2B team -- = -0.9% [-5.3,+3.7] 2019-24, +2.0% [-3.5,+7.5]
  2021-24 (date-cluster bootstrap). Below vig; N1 rest_edge_ats (fade) already lost on 2007-18.

## NHL
Not done: data/nhl_games.parquet has no lines (home_ml/total_line all null) -- not cheap. notes/nhl_simple_model.md
already has b2b -0.24 goals priced vs the opening line.

## Verdict: DEAD
The mechanism the user proposed is backwards in this data: the realized B2B penalty did not shrink (point estimate grew
from ~-1.5 to ~-2.7 pts; plausibly because load management now means B2B nights are when stars sit), and the closing
spread grew with it (about -1.0 -> -2.1). The B2B team still underperforms the close by ~0.6 pts, which is less than the
-110 vig needs. No sealed ask.

## Learnings
### The NBA back-to-back penalty did not shrink with recovery science -- it grew, and the market tracked it
Realized B2B margin effect: trend -0.053 pts/season [-0.117,+0.011]; ~-1.5 pts (2007-12) -> ~-2.7 (2021-24). Closing-spread
adjustment trend -0.058 [-0.073,-0.042]. Residual trend +0.004 [-0.060,+0.069]. Source results/b2b_recovery_trend/trend.json.
Why it matters: "the edge changed over time" ideas need the market's adjustment measured on the same design; here both
moved together.
### Backing B2B teams ATS loses more than the vig in 2019-25
B1 -8.06% [-12.58,-3.42] on 1,787 bets (edgelab b2b_t1_back_b2b_ats, dev). B2B teams cover ~48% at the close; the
fade is ~-1% after vig. B2B overs -3.78% [-7.68,+0.08].
### SBR OddsData.sqlite has junk OU/spread rows
OU 1955.5 / 1.0, spread 216, and 2012-04-21 160/5 placeholders; 21 rows in 2007-25. They blew per-season totals CIs up to
+-10 pts. Filter OU in [150,300] and |spread|<=25 unless |ML|>=1000 before any regression on that file.

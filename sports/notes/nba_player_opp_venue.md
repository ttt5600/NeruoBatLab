# Q: do NBA players beat their prop line vs specific opponents / at specific arenas / vs former teams?

User hypothesis 2026-10-29: "Jokic always performs against the Pelicans" -- some players reliably over/under-perform
against particular opponents or in particular arenas, and NBA player prop lines underweight it. Added the same day
(coordinator relay): "revenge games" vs a former team.

Not on the dead list: FUND_BOOK 3d's "NBA matchups 0/69 trait cells" is TEAM-level (comebacks/coaching); P1/P2/R_REB/L6b
are form, distribution and referee models. Player x opponent and player x arena history has not been tested.

## PRE-REGISTRATION (written 2026-10-29 BEFORE any signal was joined to a prop line or outcome)

What I had looked at before writing this (structure / data quality only, no signal):
- Prop coverage: DEV props are ESPN's 2022-23 and 2023-24 (DraftKings + 6 other books; points/rebounds/assists only)
  and 2024-25 (ESPN BET only; adds PRA). SEALED 2025-26 exists only for Oct-Nov 2025 (ESPN BET, 299 games):
  ESPN BET shut down and DK records for Dec 2025 - Jun 2026 games return 404, so the sealed set is ~1/4 of a season.
- **"current" is NOT a usable close.** Aggregate |actual - line| for points (over side), current vs open:
  2023 DraftKings 3.63 vs 4.84 with 33% of lines moved >= 3 pts; Caesars the same; 2025 ESPN BET 4.90 vs 5.38 (19%
  moved >= 3); 2024 all books ~0.03 better and ~2% moved >= 3. A line that "learns" 1.2 points of the outcome was set
  in-game. Grading a pre-game signal at those prices is lookahead in the price and would be untradeable.
  => **Decision price = ESPN's OPENING line and price** (line_open / dec_open), all seasons. Its timestamp is
  unknown but it is pre-game. This departs from the brief ("treat current as close"), for the reason above.
  Descriptive only: the same trials at "current" on 2023-24 (the one season where current looks pre-game).
- Box scores share ESPN athlete ids with props (no name matching). Match rate is reported.

Data: box scores ESPN 2018-19 .. 2025-26 (history for signals; 2018-19..2021-22 are history only), regular season,
play-in and playoffs; preseason dropped. Minutes > 0 only.

Book per season (one obs per player-game, protocol rule 4): DraftKings 2022-23 and 2023-24, ESPN BET 2024-25 and
sealed 2025-26. For each player-game-market keep the over/under pair at the opening line whose proportional
de-vigged p_over is closest to 0.5. Pushes leave the sample (lines are .5 almost always).

Signals (all computed from games on dates STRICTLY before the game date):
- Baseline b = player's mean stat over his last 20 games (any season); needs >= 10 prior games.
- Residual r = stat - b, per past game.
- S_opp = sum(r over prior games vs this opponent franchise, ESPN team id) / (n_opp + k), **k = 10** (empirical-Bayes
  shrinkage toward 0 with a prior worth 10 games; fixed in advance; the data-implied k is reported, not used).
  Eligible: n_opp >= 3.
- S_venue = same over prior games at this venue_id (arena). Tested in the player's AWAY games only (at his own
  arena S_venue is his home split, which the market prices). Eligible: n_venue >= 3.
- Revenge R = 1 if the player played >= 20 games (minutes > 0, regular season) for this opponent in some PRIOR season
  and is not on it now. "First meeting" = first game vs that team after his last game for it; later = the rest.
- Decile cutoffs: 10th / 90th percentile of the signal among ALL eligible box player-games of the PREVIOUS season
  (walk-forward; 2022-23 uses 2021-22). Over if S >= p90, under if S <= p10.
- z = S / SD(S among eligible prior-season player-games).

Trials (k = 5, Bonferroni one-sided bar p < 0.05/5 = **0.01**), DEV = 2022-23 .. 2024-25:
- T1 points, S_opp deciles: over top / under bottom at the open price. ROI.
- T2 points, S_venue deciles, away games only. ROI.
- T3 all stats pooled (points, rebounds, assists; + PRA in 2024-25), S_opp deciles per stat. ROI.
- T4 points, logistic over_hit ~ logit(p_open_novig) + z_opp, eligible player-games; coefficient on z_opp > 0.
- T5 revenge: OVER points at the open in R = 1 games. ROI. Reported alongside (not trials): PRA (2024-25 only),
  first meeting vs later, over-hit minus no-vig p, mean (actual - line).
p = share of 4000 bootstrap resamples with units <= 0 (coef <= 0 for T4). Clustered by **player-game** (as asked) AND
separately by **game date**; a trial passes only if BOTH p < 0.01, AND ROI (coef) > 0 in both time halves (split at
the median bet date). Intervals reported are the observed ROI with the date-cluster 2.5/97.5 percentiles.
SEALED 2025-26 (Oct-Nov only) is scored once, and only with the user's explicit yes, if a DEV trial passes.

Reliability check (descriptive, box scores only, all seasons): for (player, opponent) pairs with >= 3 meetings in each
half, correlate the mean points residual in even-ending vs odd-ending seasons (Pearson, pair bootstrap CI). Same for
venues. If the true effect has variance tau^2 and a meeting has noise sigma^2, cov(halves) estimates tau^2 and
k_implied = sigma^2 / tau^2.

Power note (before results): points residual SD per game ~7; a real 1-point persistent edge at the line moves hit
rate ~5 pp, which would clear a ~4.5% hold. With k = 10 and 3-8 meetings, S is mostly shrunk noise unless tau is large.
A pass is a priori unlikely: books set lines from recent form, and opponent defence is already in the line.

Kill: all 5 fail -> DEAD, add one line to FUND_BOOK 3d.

## RESULTS (run 2026-10-29; DEAD, 0/5 pass, bar p < 0.01)

Script `scripts/nba_player_opp_venue.py` (sha d05ace32b474 in the ledger). Output `results/nba_player_opp_venue/`
(results_open.json, reliability.json, bets_T*_open.csv). Ledger: `lab/ledger.jsonl`, dataset nba_player_opp_venue, 5 rows.
SEALED 2025-26 not loaded.

Join audit: 148,059 opening-line quotes (DK 22-23/23-24, ESPN BET 24-25), 148,024 matched to a box row on ESPN
(event_id, athlete_id) = 99.98%; graded actual == box stat on 100% of matched rows. 14 box rows dropped (2019-20 GS
listed twice on 2020-03-01; history only). All-Star exhibition games removed. Base rates at the open (points):
over hit 48.3% vs no-vig p_over 49.7%, mean hold 6.9% -- the known over-shading.

| Trial | Bets / n | ROI (date-cluster 95% CI) | hit - no-vig p | p (pg / date) | halves | Pass |
|---|---|---|---|---|---|---|
| T1 points S_opp deciles | 9,279 | -5.3% [-7.2, -3.3] | +0.8 pp | 1.00 / 1.00 | -6.1% / -4.3% | no |
| T2 points S_venue deciles, away | 3,668 | -4.4% [-7.5, -1.2] | +1.3 pp | .999 / .998 | -5.5% / -3.2% | no |
| T3 pts/reb/ast/PRA S_opp deciles | 29,687 | -5.2% [-6.4, -4.0] | +1.0 pp | 1.00 / 1.00 | -6.3% / -4.0% | no |
| T4 logit coef on z(S_opp), points | 40,551 | coef +0.020 [+0.001, +0.039] | ~+0.5 pp per SD | .017 / .018 | +0.004 / +0.042 | no (p > .01) |
| T5 revenge, over points | 1,491 | -6.7% [-11.2, -2.1] | -0.1 pp | .996 / .999 | -6.9% / -6.4% | no |

Notes on the table:
- T4 is the only positive estimate; it misses the bar (p .018 vs .01) and is worth ~0.5 pp of hit rate per SD,
  a tenth of the 6.9% hold. In 2023-24 alone it is -0.004 [-0.034, +0.027]; the second half carries it.
- The "hit - p" gains in T1-T3 come from the UNDER side (T1 unders +2.6 pp, T2 unders +4.0 pp, ROI -2.0% and +0.8%),
  which is mostly the base over-shading (unders beat no-vig p by ~1.4 pp everywhere). Overs: T1 -8.4%, T2 -9.5%.
  Post-hoc, not a trial.
- T5 descriptives: first meeting after leaving 173 bets ROI -1.3% (p .57); later meetings 1,318 bets -7.4%;
  mean actual - line +0.24 pts in revenge games vs +0.14 in all other points quotes (first meetings +0.29).
  PRA (2024-25 only) 494 bets -4.3%.
- Price check (descriptive, 2023-24 only, the one season where "current" looks pre-game): T1 at current -6.5%
  (3,049), T4 coef -0.013 [-0.043, +0.017]. Same conclusion as at the open.

Reliability (box scores 2018-19..2024-25, points residual vs last-20 baseline, even vs odd seasons, >= 3 meetings per
half): player x opponent r = +0.017 [-0.005, +0.039] over 10,159 pairs (~5.6 meetings per half); player x away arena
r = +0.024 [-0.009, +0.057] over 3,995 pairs. Both not distinguishable from zero. Implied true-effect variance
tau^2 = 0.13 pts^2 (tau ~0.36 pts) vs per-game sigma^2 = 37.7, so the data-implied prior is k ~ 287 games (venue
~143), not the k = 10 used here: a player-opponent "tendency" is ~1/3 of a point and needs hundreds of meetings to see.

Illustration, Jokic vs NOP: 22 meetings 2018-19..2024-25 with a baseline, mean points residual +3.25 (SD 6.95, SE 1.48)
-- about 2.2 SE, which is what the most extreme of ~10,000 player-opponent pairs looks like by chance. S_opp with
k = 10 sat at +2.1..+2.7 in 2022-23..2024-25. He went over the open in 5 of the 6 meetings that had a DK quote
(2022-23..2023-24); his three 2024-25 NOP games have no ESPN BET points quote in the data (he has quotes in 55
other 2024-25 games).

## Why it fails (the lesson)
A per-matchup sample is tiny (2-4 meetings a season) and a points game has SD ~6. The split-half check says the
persistent part is ~0.36 pts, ~5% of one game's noise, so the "Jokic always kills the Pelicans" pattern is almost all
selection from thousands of pairs. Whatever is real is far below the ~7% prop hold, and the decile strategy just pays
the vig (-5%). Revenge games: no over-performance against the line.

## Data note (found on the way)
ESPN "current" prop prices are in-game for 2022-23 DK/Caesars (|actual - line| 3.6 vs 4.8 at open; 33% of lines
moved >= 3 pts) and partly for 2024-25 ESPN BET. This was already known (nba_props_model.py: "dec/line can be
in-game; we use ONLY line_open"); P1/P2/L6b all used the open. It contradicts the "treat current as close" shorthand,
so anything that grades NBA props at "current" for 2022-23 or 2024-25 is grading at in-game prices. 2025-26 props exist only for Oct-Nov 2025 (ESPN BET); after that
ESPN returns 404 for DK historical props, so a full sealed 2025-26 season is not available from ESPN.

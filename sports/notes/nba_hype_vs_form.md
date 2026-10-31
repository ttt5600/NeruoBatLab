# NBA hype vs form: are early-season lines anchored to PRESEASON expectations?

PRE-REGISTERED 2026-10-30, written BEFORE any outcome was joined to any price, form or preseason variable.
What had been looked at before writing: the preseason win-total tables (team, total, odds, book, as-of date), ESPN game
schedules (first game dates, team codes), ESPN lines column coverage (which seasons/books have `*_open` fields). No
outcome, ROI, coefficient or gap-vs-line relation has been computed.
Script: `scripts/nba_hype_vs_form.py` (one run: builds features, freezes thresholds from DEV prices/form/totals only,
joins outcomes, scores DEV, scores SEALED only for DEV passes). Ledger dataset `nba_hype_vs_form` (lab/ledger.jsonl).

## Question (user hypothesis 10-30)
"The edge is underpriced teams and overhyped teams at the moment": early in the season lines stay anchored to PRESEASON
expectations, so teams playing far above their hype are underpriced and teams far below it are overhyped.
Closest prior work: notes/nba_early_season_bias.md (form vs CLOSE, gp <= 15: DEAD; gp <= 8 sensitivity +22.1%/137 bets,
p .022, carried by 2024-25 -> forward logger F1). What is new here: an explicit hype anchor (preseason win total) and a
shrinkage model that blends hype and form, priced at the OPENING line where one exists. Prior expectation: no pass
(full-season NBA sides efficient; L2 team ratings dead; the close already prices form at gp <= 15).

## Data
- Preseason anchor: `data/preseason/nba_win_totals_sportsoddshistory.csv`, scraped 2026-10-30 from
  https://www.sportsoddshistory.com/nba-win/?y=YYYY-YYYY&sa=nba&t=win&o=t ("Lines courtesy of BetMGM"), 30 teams x
  2019-20..2026-27. As-of dates vs first regular-season game (ESPN, UTC): 2021-22 Oct 19 / Oct 19; 2022-23 Oct 18 / Oct 18;
  2023-24 Oct 24 / Oct 24; 2024-25 **Oct 23 / Oct 22** (one day after opening night; at most 1 game per team can have been
  seen; window starts at game 4, so judged immaterial but disclosed); 2025-26 Oct 21 / Oct 21. 2026-27 has no as-of date
  (used only for the descriptive current list). The line (win total) is used; over/under juice ignored (fixed now).
- Games: ESPN `data/espn/nba/games_{season}.parquet` regular season finals (loaders reused from nba_early_season_bias.py).
- Prices (one book per event): seasons 2023-24, 2024-25, 2025-26 have `*_open` fields -> price = OPENING quote
  (`home_ml_open`/`away_ml_open`, M = -`home_spread_open`), DraftKings if it has both open MLs and an open spread, else
  ESPN BET; if neither has an opener, fall back to the close of the early-season study's book (flagged `price=close`).
  2021-22 and 2022-23 have NO opening fields at all -> price = the plain field (= close) as in the early-season study.
  So DEV half 1 is priced at the close and half 2 at the open; this confound is disclosed, not fixable. Bets are paid at the
  ML price actually quoted (incl. vig). ESPN "open" timing is not verified as tradeable; treated as "earliest available".

## Splits
- DEV = 2021-22, 2022-23, 2023-24, 2024-25 (the early-season study's DEV seasons). Halves by season:
  half1 = 2021-22 + 2022-23, half2 = 2023-24 + 2024-25.
- SEALED = 2025-26, same window. The early-season study pre-declared it as sealed but never scored it (no DEV pass), so it
  is disjoint from anything that study scored. Not virgin for the project (O1, L32, L2 scored full-season 2025-26).
  Scored ONCE in the same run, only for DEV passes.

## Point-in-time features (games with ET date strictly before the game date)
- gp_t, adjMOV_t exactly as nba_early_season_bias.py (`form_asof`).
- Hype in points: E_t = a_s + b_s * (WT_t / 82 - 0.5), where (a_s, b_s) is an OLS fit of full-season MOV per game on
  full-season win pct over all team-seasons in ESPN seasons BEFORE season s (2019-20 onward; 2020-21 uses its 72-game
  pct). This is a units map (wins -> points), fitted on prior seasons only.
- gap_t = adjMOV_t - E_t. gap > 0 = "underpriced" (playing above hype), < 0 = "overhyped".
- Shrinkage rating r_t = E_t + w_t * gap_t, w_t = gp_t / (gp_t + K), K = 20 fixed now (a preseason prior worth ~20 games of
  margin data, a common rule of thumb; NOT fitted).
- Model home margin P = H_s + r_h - r_a; H_s = mean M over that season's window games (prices only, as before).
- Z = P - M (edge in points vs the earliest price). D = gap_h - gap_a.
- Unit test: form values for 60 sampled rows/season recomputed from data truncated to dates < game date must match.

## Window (fixed now)
Games 4-20 of the season for BOTH teams: gp_h and gp_a in [3, 19] before the game.

## Trials (k = 4, Bonferroni one-sided p < 0.0125; flat 1u; one bet per game; ML at the earliest available price)
- H1 SHRINKAGE vs LINE: bet home if Z >= X, away if Z <= -X. X = 2/3 quantile of |Z| over DEV window games (no outcomes).
- H2 REGRESSION: OLS (margin - M) ~ D over all DEV window games; slope > 0.
- H3 FADE OVERHYPED: H1 bets in which the team bet AGAINST has gap <= -G, G = median |gap_t| over DEV window team-games
  (no outcomes).
- H4 POPULAR-TEAM FADE: pre-declared list {LAL, GS, BOS, NY, CHI, PHI}; window games with exactly one listed team; bet
  the other team ML. (Does not use the hype gap; as briefed.)

## Metric, bar
- ROI = OBSERVED mean profit per bet; 95% CI and one-sided p (share of resampled means <= 0) from a GAME-DATE-clustered
  bootstrap, 4000 resamples, seed 2026. H2: slope with date-clustered bootstrap CI and p (share <= 0).
- DEV PASS: p < 0.0125 AND ROI (slope for H2) > 0 in both halves.
- SEALED (only passed trials): p < 0.05 / (#trials taken to sealed) and ROI (slope) > 0. Small n -> a non-pass there is
  "not distinguishable", not a refutation.
- CI crossing 0 -> wording "not distinguishable". Too-good guard: ROI > +10% on > 200 bets -> inspect rows (as-of date
  after season start, line timestamps, team joins) before believing it. Any DEV pass is a bug until rows are inspected.
- KILL: no DEV pass and H2 CI includes 0 or is negative -> file DEAD.

## Power (stated before scoring)
Per-bet ML profit SD ~1.33 (early-season study). With ~n bets the SE of ROI is 1.33/sqrt(n); detecting a true ROI at
one-sided .0125 with 80% power needs ROI ~ (2.24 + 0.84) * 1.33 / sqrt(n). Build-only run (no outcomes, `--build-only`):
DEV window 973 games (243/245/245/240 by season), frozen X = 2.638 pts, G = 3.324 pts, X0 (twin) = 5.306; H1 would bet 325
(77/96/54/98) -> MDE ~ +22.7% ROI. So only a very large mispricing is detectable; a fail is weak evidence of absence.
All-Star exhibition entities (LEB, DUR, ...) have no win total and drop out of the window; lookahead test 0 mismatches.

## Descriptive extras (not trials)
- Overlap with F1: apply the F1 rule historically (both gp 3..8, R = (adj_h - adj_a) - (M - 2.06), |R| >= 5.3659, side by
  sign) to the same rows and count H1 bets that F1 would also have made on the same side.
- Form-only twin: H1 with K = 0 (w = 1, hype ignored) in the same window, own 2/3-quantile threshold, to show whether hype
  changes anything. Reported, never a trial.
- Current 2026-27 list: gap_t for every team from 2026-27 finals to date with the 2026-27 BetMGM totals; top 5 / bottom 5.

## Result
(filled in below this line after the run; nothing above edited after scoring)

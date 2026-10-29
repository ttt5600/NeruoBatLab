# L2 -- team_rating.py vs the closing moneyline / spread (NBA 2025-26 held out)

PRE-REGISTERED 2026-10-29, written BEFORE any model output was joined to any outcome or market price.
Script: `scripts/l2_team_rating_sides.py` (`build` = ratings only, no prices/outcomes; `score` = the one scoring run).
Ledger dataset: `l2_team_rating_sides` (lab/ledger.jsonl), one row per trial.

## Question
Does the Bayesian team rating (`scripts/team_rating.py`, as used by the `rating <team>` wrapper) contain information
about NBA game winners beyond the closing sportsbook price, and does betting its disagreements with the close make
money? (Motivating: 10-29 it said ATL 61% vs Kalshi 52% / Pinnacle 51.9%; MEM 35% vs Kalshi ~28%.)

## Data and hold-out
- Games: ESPN `data/espn/nba/games_2026.parquet` = 2025-26, regular season, STATUS_FINAL. Outcome = home_score > away_score.
- Hold-out check (from the code, not recall): `team_rating.fit_params(cur)` fits sigma/h on seasons 2022-2025 and
  b/tau on S=2023..2025 (`range(2023, 2026)`), i.e. 2021-22..2024-25. With cur=2026 the ridge loop is
  `range(2022, 2026)`, so 2025-26 games never enter sigma, tau, b, h or r_last (= 2024-25 ridge). 2025-26 is held out
  from the FIT. Caveats written now: (a) the player-value weights (frozen BPM-lite, results/nba_simple_model_frozen.json)
  are documented as DEV-only; (b) 2025-26 is NOT virgin for the project -- earlier sealed tests (O1 NBA simple model vs
  opener, L32 totals) scored on it, and team_rating.py was written 10-28 by people who had seen those results. No
  team_rating parameter was tuned on 2025-26 as far as the code shows.
- Prices: ESPN keyless core odds `data/espn/nba/lines_2026.parquet` (collected by collect.py). One row per event:
  DraftKings if present, else ESPN BET (ESPN BET covers 10-21..11-30, DK 11-24..04-12; 44 overlap events -> DK).
  `*_close` fields are used as the CLOSE only (ESPN last_updated is post-kickoff; never treated as a tradeable pre-game
  quote other than "the closing price").
- No Pinnacle and no Kalshi history in this test.

## Model, point in time (no lookahead)
For every ET game date d, using only games with ET date < d, exactly the script's functions/formulas:
- P = fit_params(2026): sigma, tau, b, h, r_last.
- PRIOR_t = b * r_last[t] + roster delta. Roster delta = strength(full set at d) - strength(team's last-10-games set of
  last season), strength = nsl.strength_fn on player state built from box scores dated < d. Historical ESPN roster
  snapshots do not exist, so "full" uses the script's own fallback (roster API unavailable -> players in the team's last
  10 games whose latest team is t). Consequence: before a team's first game, offseason acquisitions are still credited to
  their old team -> roster delta ~0 in week 1; this is a handicap vs the live model.
- SEASON = opponent-adjusted margin (opponents' PRIOR as of d), 3P-luck adjusted (K=750, team_rating formulas), blended
  with w = n/(n + (sigma/tau)^2).
- AVAILABILITY: point-in-time injury reports are NOT available historically -> run WITHOUT availability (avail = 0
  for every team). This is a weaker model than the live one; a pass/fail here says nothing about the availability term.
- spread_home = blend_h - blend_a + h*(not neutral); P_model(home) = Phi(spread/sigma). Model ATS: P(home covers close
  spread L) = Phi((spread_home + L)/sigma).
- Parity check before scoring: on >= 2 historical dates, team_rating.main()'s own printout (roster/injury feeds stubbed
  empty, kalshi off) must match this reconstruction to 0.1 pt.

## Market probabilities
Home/away close ML -> implied probs -> de-vig. PRIMARY de-vig for bet selection = POWER (edge_math.devig_power);
proportional and Shin reported as sensitivity (not separate trials). Same for spread prices.

## Trials (k = 4, flat 1u, one bet per game at most, paid at the CLOSE price incl. vig; pushes = 0, kept as bets)
- T1 ML: bet the side with model - devig_close >= 0.05 (home side if P_m - P_c >= .05, away if P_c - P_m >= .05).
- T2 ML: same, threshold 0.03.
- T3 ML: T1 restricted to games where BOTH teams have >= 10 games played before this date.
- T4 SPREAD: bet the side with model cover prob - devig close cover prob >= 0.05 (paid at the close spread odds).
Scored games: all 2025-26 regular-season finals with the needed close prices (ML for T1-T3, spread line+odds for T4).

## Metric, bar, kill
- ROI at close price; 95% interval from a game-level bootstrap (10,000 resamples, seed 2026; one bet per game, so
  game-clustered = bet-level); one-sided p = share of resampled means <= 0. Report the OBSERVED ROI, not the resample mean.
  Date-clustered bootstrap reported as a sensitivity.
- PASS for a trial: p < 0.05/4 = 0.0125 AND both chronological halves of its bets (by bet order) have ROI > 0.
- INFORMATION test (all ML-priced games): logistic regression y ~ logit(P_close_power) + logit(P_model); report the
  coefficient on logit(P_model) with 95% CI (model-based and game-bootstrap). Also Brier and log-loss of model vs
  power-devig close with paired bootstrap of the difference.
- KILL (for L2 as a sides edge on this model): model coefficient CI includes 0 AND T1 ROI CI includes 0.
- Calibration table: bucket by (P_model - P_close) for the home side: <=-.10, -.10..-.05, -.05..-.03, -.03..0, 0...03,
  .03...05, .05...10, >.10 -> n, mean P_model, mean P_close, actual home win rate.
- A result that clears the bar is treated as a bug until rows are inspected for lookahead/misalignment.

## Result
(filled in after `score`, below this line; nothing above is edited after scoring)

### Scored 2026-10-29 (`l2_team_rating_sides.py score --ledger`; results/l2_team_rating_sides/score.json, bets_*.csv; ledger rows l2_T1..T4)
Parity: on 2025-12-15 DET@BOS and 2026-02-20 CLE@CHA the reconstruction matched team_rating.main() (feeds stubbed) to
the printed 0.1 pt (BOS +5.6 / DET +4.5 / BOS by 3.1; CHA -0.2 / CLE +5.1 / CHA gets 3.3).
Games: 1,235 rated; 1,219 with ML close; 1,221 with spread close. Fit params: sigma 13.58, tau 3.57, b 0.600, h 2.01.

| trial | n | ROI (observed) | 95% CI (game boot) | p one-sided | halves | pass |
|---|---|---|---|---|---|---|
| T1 ML gap>=5c | 789 | -13.87% | [-23.80, -3.51] | .9954 | +6.48% / -34.16% | no |
| T2 ML gap>=3c | 974 | -11.26% | [-20.09, -2.09] | .9909 | +8.34% / -30.85% | no |
| T3 T1 after 10 games | 679 | -15.25% | [-26.28, -4.19] | .9958 | +4.70% / -35.15% | no |
| T4 spread gap>=5c | 792 | -4.94% | [-11.67, +1.70] | .9280 | +5.11% / -14.99% | no |
De-vig sensitivity (T1 ROI): proportional -12.84%, Shin -13.60%. Date-clustered T1 CI [-24.54, -2.63].
Information: logit(y) = a + 0.925 logit(close) + 0.115 logit(model); model coef 95% CI [-0.213, +0.442] (boot
[-0.236, +0.447]), n 1,219. Brier model .20480 vs close .19408 (diff +.01072 [+.00507, +.01623]); log-loss .59752 vs
.56832 (diff +.02920 [+.01578, +.04241]) -> the close is significantly better; the model adds nothing detectable.
Calibration (home side, bucket of P_model - P_close): >= +.10: n 226, model .472, close .301, actual .310;
<= -.10: n 222, model .590, close .753, actual .716. The model is far off in both tails; the close is closer (its .753 vs .716 is itself 3.7c rich on home favourites).

VERDICT: FAIL. The literal kill rule (coef CI incl 0 AND T1 ROI CI incl 0) did NOT fire only because T1's ROI CI is
entirely BELOW zero -- a pre-registration wording gap; the evidence against is stronger than the kill case.
Mechanism (rows inspected): 85% of T1 bets are underdogs (669 of 789, ROI -15.8%); the biggest gaps are games the market
priced off news the model does not see (DEN@PHI 01-05 close -14.5 vs model DEN by 3.4; final-day 04-12 games
ORL@BOS / CHA@NY with BOS/NY resting). Model spread SD 7.0-7.2 vs close SD 9.7-11.3 in Mar-Apr (tanking/rest), and the
T1 loss concentrates there (Feb -28%, Mar -32%, Apr -44%).
POST-HOC DESCRIPTIVE D1 (added after scoring, not a trial, LOOKAHEAD upper bound: availability = who actually played):
model coef .159 [-.186, +.504]; Brier .20299 vs close .19408 (diff CI [+.0034, +.0143]); T1-analog 776 bets ROI -11.81%
[-22.15, -1.12]. Even perfect knowledge of who played, run through this player-strength scale (actual-availability adj
SD ~0.4 pts), does not close the gap -> the availability term is too weak, not just missing.
Caveat: no critic subagent has reviewed this yet.

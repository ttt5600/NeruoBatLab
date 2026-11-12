# props_situational -- L7 situational NBA prop model + L5 shrink-P2-toward-market (agent 2026-11-12)

Dev only. No orders, no git, no sealed scoring.

## Data (found before any outcome was computed)
- Prop lines: `data/espn/nba/graded_{season}.parquet` (ESPN core API, keyless; `scripts/collect.py`). Columns: provider,
  athlete_id, market_n, side, line_open, dec_open (OPEN), line, dec (last snapshot; `hours_before` median -3.2 h =
  frozen ~at tip, used as CLOSE), actual. This is the P1 (130k) / P2 (60,826) source (`scripts/nba_props_model.py`
  D, DEV).
  Rows: 2022 4,312 (stub), 2023 900,981 (7 books), 2024 989,476 (6 books), 2025 325,589 (ESPN BET only),
  2026 72,102 (ESPN BET), 2027 37,139 (DraftKings, current season = forward).
- Split (season = end year): **DEV = 2023, 2024, 2025** (Oct 2022 - Jun 2025). **SEALED = 2026 (2025-26)** per the
  frozen protocol in nba_props_model.py (and already spent once by R_REB sealed, ledger line 154). 2027 = forward.
  edgelab vault.yaml has no ESPN-props dataset; family nba_props = 31 trials -> bar p < 0.05/31 = 0.00161
  (`Registry().family_bar('nba_props')`); a new trial makes it 0.05/32 = 0.00156.
- Box scores `box_{s}`, schedule/venue `games_{s}`, game lines (spread open/close) `lines_{s}`.

## How this differs from the dead list
Dead: P1 form model, R_REB, player x opponent/venue/revenge, matchup graph, injury->teammate, hot/cold streaks,
ref crew -> FT props (L6b), team-level rest/b2b ATS/totals angles. All of those are either player-form or
team-level. This test uses ONLY schedule/geography situational variables at the PLAYER-PROP level, measured against
the prop market's own de-vigged opening probability (not raw ROI), so it asks "does the prop line under-adjust for
fatigue / travel / altitude / nightlife / blowout risk". Refs are excluded (L6b dead).

## Pre-registration (written 2026-11-12 before computing any outcome)
Universe: regular-season dev games; markets points, rebounds, assists, 3PM, PRA; one row per (player, game, market) at
the season's anchor book (most rows: DraftKings 2023 & 2024, ESPN BET 2025) with both over and under dec_open at the
same line_open. Players with prior-20-game mean minutes >= 20 (as P2).
Outcome (primary): e = 1{actual < line_open} - p_mkt_under, p_mkt_under = de-vigged open (multiplicative). Pushes dropped.
Features (team schedule from games_{s}, all known before the prop opens):
  F1 b2b: player's team played the previous calendar day (US/Eastern dates)
  F2 rest3: >= 3 days since team's last game
  F3 three_in_four: 3rd game in 4 nights
  F4 travel: log1p(great-circle km from previous game city to this city)
  F5 tz_east: time-zone hours moved EAST since previous game (signed, east positive)
  F6 alt_visitor: player's team is AWAY at DEN or UTA
  F7a night_own: team b2b AND previous game in NY/BKN/LA (LAL/LAC home)/MIA (cities fixed now)
  F7b night_opp: OPPONENT b2b after NY/BKN/LA/MIA (prediction: own player OVER)
  F8 blowout: |home_spread_open| >= 10 (anchor/any book open spread; close NOT used)
  F9 opp_b2b: opponent b2b
  F10 away: player's team away
Feature check: OLS e ~ feature (one at a time) + market dummies, SEs clustered by game date. A feature "moves the
residual" if its 95% CI excludes 0 AND p < 0.005 (Bonferroni over 10). If NONE pass -> STOP, verdict dead.
Model (only if >= 1 passes): e_hat = OLS on passing features fit 2023-2024, applied to 2025 (and 2023-24 by
leave-season-out). Bet UNDER at the anchor book's dec_open when e_hat >= k, k = 0.03 (fixed now). One trial registered
in edgelab family nba_props. Success = edgelab dev pass (p < family bar ~0.00156, both halves > 0, n >= 200) at the
recorded open odds. CLV reported (close de-vigged p_under vs open). Kill = anything else.
L5 check: logit(P(under)) = a + b*logit(p_mkt) + c*(logit p_emp - logit p_mkt), p_emp = P2 prior-20 empirical share,
fit 2023-24, CI on c clustered by date. Shrinkage toward the market only helps if c > 0 with CI excluding 0; then
evaluate bets with fitted P >= p_mkt + 0.03 on 2025 at the anchor book. If c's CI includes 0 or c <= 0 -> L5 dead
(P2's edge is 0 once the market is in the model).

## Log
- 2026-11-12 build: `scripts/props_situational.py build` -> results/props_situational/frame.parquet, 141,746 rows
  (2023 47,479 / 2024 46,185 / 2025 48,082), 0 duplicate keys; b2b share .168, three_in_four .251, tz_east [-3,+3],
  alt_visitor .034, night_own .029; blowout spread_open missing for 2023 (n 93,999). p_close available 58%.
  Deviation: 11 features were listed (F7a/F7b split), so Bonferroni bar is .0045, not .005 -- changes nothing.
- Feature check (`... check` -> results/props_situational/feature_check.json; OLS e ~ f + market FE, CR1 by date,
  459 dates). **0/11 pass.** Only tz_east has a 95% CI excluding 0: +0.0043 per hour east [+0.0007,+0.0078] p .019
  (by season .0011/.0108/.0012 -- one season carries it; 1 of 11 at 5% is what chance gives). away +0.0063
  [-0.0001,+0.0127] p .055. b2b +0.0033 [-0.0050,+0.0116]; alt_visitor -0.0012 [-0.021,+0.018]; night_own +0.0072
  [-0.011,+0.025]; night_opp +0.0079 [-0.011,+0.027]; blowout -0.0028 [-0.014,+0.0085]. Even tz_east at 3 h = +1.3 pp
  of under probability vs a ~4.5% vig. **L7 STOPPED per pre-reg: dead. No model fit, no edgelab trial registered.**
  Baseline: mean e = +0.0128 (unders win 1.3 pp more than the de-vigged open says) but all-unders ROI -4.32% at the
  anchor book.
- L5 (`... l5` -> l5_check.json; `results/props_situational/l5_eval.py` -> l5_eval.json). Calibration fit 2023-24
  (n 93,664): c = +0.094 [+0.047,+0.142] p .0001, b(market) 1.006, intercept +0.055. Replicates on 2025 refit:
  c = +0.112 [+0.078,+0.145]. So P2's empirical distribution DOES carry information beyond the open line (optimal
  shrink weight ~10% toward P2, 90% market). Pre-registered bet rule on 2025 (p_fit >= p_mkt + .03, ESPN BET open):
  5,569 bets, 160 dates, ROI -0.17% [-2.51,+2.31] (date bootstrap) p(ROI<=0) .56, halves +1.66% / -1.99%; win 56.6%
  at mean dec 1.828; all-unders 2025 baseline -6.75%/48,082. CLV descriptive: close de-vigged p_under exceeds open by
  +1.05 pp (n 2,463 with same line at close). **L5 dead at real prices: real signal, ~6.6 pts of selection, not past
  vig.** Not registered in edgelab (no ESPN-props vault dataset exists); lead should log it as 1 nba_props trial.

## Verdict
L7 situational prop model: DEAD (0/11 features move the residual past the bar). L5 shrink-toward-market: DEAD as a bet
(-0.17% [-2.5,+2.3]), but the information result is real and replicated.

## Learnings
### Situational schedule variables do not move NBA prop residuals
0/11 features (b2b, rest, 3-in-4, travel, time zones, altitude, nightlife, blowout spread, opp b2b, away) pass
p < .0045 against the de-vigged OPEN prop price, 141,746 props 2022-25 (results/props_situational/feature_check.json).
Largest: tz_east +0.43 pp per hour east [+0.07,+0.78], driven by one season. Prop books price schedule spots.
### P2's empirical distribution adds real but sub-vig information to the prop open
Logit calibration coefficient c = +0.094 [+0.047,+0.142] (2023-24) replicates at +0.112 [+0.078,+0.145] on 2025, but
the best bets (fitted P >= market + 3 pp) return -0.17% [-2.5,+2.3] on 5,569 bets (l5_eval.json). Shrinkage cannot
turn a ~10%-weight signal into a profit against ~4.5% hold at a single book.
### Unders beat the de-vigged price, not the vig
Unders win 1.3 pp more often than the multiplicatively de-vigged open implies (mean e +0.0128), yet all-unders lose
4.3% (2023-25 anchor) and 6.75% (2025 ESPN BET). The over/under hold is skewed onto overs; de-vig by the multiplicative
method understates under fairness. Any prop model must beat this, not 50%.

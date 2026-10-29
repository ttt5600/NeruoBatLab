# NBA early-season closing-price bias (is the close slow to price current-season form?)

PRE-REGISTERED 2026-10-29, written BEFORE any outcome was joined to any price or form variable.
(What had been looked at before writing: file schemas, which line providers/columns exist per season, and the rate at which
ESPN's plain `home_ml`/`spread` fields equal the `*_close` fields. No outcome, ROI or coefficient had been computed.)
Script: `scripts/nba_early_season_bias.py` (one run: builds features, scores DEV, scores SEALED only if a DEV trial passes).
Ledger dataset: `nba_early_season_bias` (lab/ledger.jsonl), one row per trial (+ sealed rows if any).

## Question
User hypothesis: early in the NBA season the market is anchored on last season / reputation and is slow to price what
is visible in the first weeks (motivating example, 2026-27 not in this test: 10-28 OKC@DAL, DAL ~37% on Kalshi, OKC -5.5,
DAL won 120-96). We test whether the CLOSING price is systematically biased early season -- NOT whether a model beats it.
Prior from this project (FUND_BOOK 3d): full-season NBA sides are efficient (home dog ML, big-spread dog, Elo
disagreement all dead); L2 team-rating model lost to the 2025-26 close. Prior expectation: no pass.

## Data
- Games: `data/espn/nba/games_{2022..2026}.parquet` (ESPN season label = year the season ends; 2022 = 2021-22),
  `season_type == regular-season`, `status == STATUS_FINAL`. Outcome = home_score > away_score (from ESPN finals).
- Prices: `data/espn/nba/lines_{season}.parquet`, ONE book per event: DraftKings ("DraftKings" or "Draft Kings") if it has
  both MLs, else ESPN BET. Fields used: the plain `home_ml`, `away_ml`, `spread` (home line), `home_spread_odds`,
  `away_spread_odds`. Reason: 2021-22 and DK 2022-23 have NO `*_close` columns, and the 2022-23 ESPN BET
  `home_spread_close` column holds odds, not the line (collection bug). Checked before writing: where `*_close` exists the
  plain field equals it 98-100% of the time (2024 DK ml .986, 2025 ESPN BET .999, 2026 DK 1.000), i.e. ESPN's plain field
  is the final (= closing) quote. Treated as the CLOSE only; never as a tradeable pre-game quote.
  Caveat: 2021-22 / 2022-23 DK plain = close is inferred from the later seasons, not verifiable.
  Book mix: DEV mostly DK (2021-22..2023-24) and ESPN BET (2024-25); SEALED 2025-26 early season = ESPN BET (DK starts 11-24).
- De-vig: POWER (edge_math.devig_power) for probabilities; bets are paid at the actual close price incl. vig.

## Splits
- DEV = 2021-22, 2022-23, 2023-24, 2024-25. Dev halves BY SEASON: H1 = 2021-22 + 2022-23, H2 = 2023-24 + 2024-25.
- SEALED = 2025-26, games with ET date < first regular-season ET date + 30 days, and inside the window below.
  Scored ONCE, in the same run, and ONLY for trials that pass DEV. (User authorised sealed tests.) 2025-26 is not virgin
  for the project (O1, L32, L2 scored on it) but no early-season form test has been run on it.

## Point-in-time features (no lookahead)
For game g of team t on ET date d, using only that team's regular-season games with ET date < d:
- gp = games played before d.
- MOV = mean(team margin) over those games.
- adjMOV (opponent-adjusted, one step, no parameters) = MOV + mean over those games of the opponent's MOV, where each
  opponent's MOV is also computed from games with ET date < d. No home/away adjustment.
- REP (T2 only) = team's full previous regular-season MOV per game (ESPN data; so 2021-22 has no REP -> T2 DEV = 2022-23..2024-25,
  T2 H1 = 2022-23 only).
- Market home margin M = -spread (home line). H_s = mean(M) over that season's window games (prices only, no outcomes).
- A unit test in the script asserts, on rows, that every form value is unchanged when all games on/after d are deleted.

## Window
Primary: both teams gp <= 15. Form trials (T1, T2, T5) additionally require both gp >= 3 (form from 0-2 games is not
defined/noise; fixed now). T3/T4 use gp 0..15. Sensitivity reported (not a trial): both gp <= 8.

## Trials (k = 5, Bonferroni bar one-sided p < 0.01; flat 1u; one bet per game; pushes = 0 and kept)
- T1 FORM vs MARKET (ML at close): R = (adjMOV_h - adjMOV_a) - (M - H_s). X = 2/3 quantile of |R| over DEV window games
  (prices + form only, no outcomes; frozen and reused for SEALED). Bet home if R >= X, away if R <= -X.
- T2 REPUTATION FADE (ML at close): D_t = REP_t - adjMOV_t (reputation minus current start). G = D_h - D_a.
  Y = 2/3 quantile of |G| over DEV window games. Bet AWAY if G >= Y (home over-reputed), HOME if G <= -Y.
- T3 HOME UNDERDOG ML at close: home power-devig prob < 0.5; bet home ML.
- T4 BIG UNDERDOG ATS: |close spread| >= 5; bet the underdog ATS at -110 (primary, as briefed); actual close spread odds
  reported as sensitivity.
- T5 INFORMATION TEST: logistic y_home ~ logit(p_close_home, power) + F, F = (adjMOV_h - adjMOV_a)/10, on DEV form window.
  Statistic = coefficient on F; one-sided test (coef > 0). CIs: model-based (Wald) and game bootstrap (10,000; one row per
  game so game-clustered = row; date-clustered as sensitivity).

## Metric, bar
- ROI (T1-T4): OBSERVED mean profit per 1u bet at the close price; 95% CI from game-clustered bootstrap (10,000, seed 2026;
  one bet per game so = bet-level); date-clustered CI as sensitivity. One-sided p = share of resampled means <= 0.
- PASS (DEV): p < 0.01 AND ROI > 0 in both season halves. T5 PASS: bootstrap one-sided p (share of coef <= 0) < 0.01 AND
  coef > 0 fit separately in each half.
- SEALED (only passed trials): PASS if one-sided p < 0.05 / (number of trials taken to SEALED) and ROI > 0
  (T5: coef > 0 with the same p rule). Expected n is small (~1 month); a non-pass there is "not distinguishable", not a refutation.
- A DEV pass is treated as a bug until bet rows are inspected (lookahead, join, sign of spread).
- KILL: no DEV pass and T5 coef CI includes 0 or is negative -> idea filed DEAD in FUND_BOOK 3d.

## Result
(filled in below this line after the run; nothing above edited after scoring)

### Scored 2026-10-29 (`nba_early_season_bias.py --ledger`; results/nba_early_season_bias/score.json, bets_dev_T*.csv, dev_form_window.csv; ledger rows nesb_T1..T5)
Checks: lookahead test (60 sampled rows/season recomputed from data truncated to games before the date) 0 mismatches every
season; spread-sign vs ML favourite agreement .991-1.000; books: 2021-22/2022-23 DK, 2023-24 DK (2 ESPN BET), 2024-25 ESPN BET.
Frozen from DEV prices+form only: X (T1) = 5.37 pts, Y (T2) = 7.37 pts. DEV window games 935; form window (gp 3..15) 740.

| trial | n | ROI (observed) | 95% CI (game boot) | p one-sided | H1 / H2 (by season) | pass |
|---|---|---|---|---|---|---|
| T1 form vs market ML | 247 | +3.14% | [-11.21, +18.23] | .341 | -6.46% / +13.63% | no |
| T2 reputation fade ML | 185 | +2.77% | [-12.04, +17.42] | .364 | -3.71% (22-23 only) / +5.44% | no |
| T3 home dog ML | 347 | -0.58% | [-14.23, +13.88] | .531 | +1.06% / -2.01% | no |
| T4 dog >= 5 ATS @-110 | 487 | +0.26% | [-8.19, +8.66] | .471 | +3.26% / -2.58% | no |
T1 by season: 21-22 -12.0% (60), 22-23 -1.7% (69), 23-24 -10.3% (51), 24-25 +31.9% (67) -> the pooled + is one season.
T4 at actual close spread odds: +0.50% [-8.01, +8.87]. Date-clustered CIs are of the same width (score.json).
T5: y ~ 0.796 logit(close) + 0.164 F (F = adjMOV diff / 10), n 740; form coef Wald 95% [-0.091, +0.418] (two-sided p .21),
game bootstrap [-0.099, +0.423] one-sided p .105, date-clustered [-0.105, +0.399]; halves -0.015 / +0.374. Not distinguishable from 0.
SEALED: not scored (no DEV pass), as pre-registered.

Sensitivity <= 8 games (pre-stated as sensitivity, NOT a trial): T1 137 bets +22.1% [+0.6, +44.5] p .022 (by season -3.5%,
+20.5%, -4.4%, +63.4%: again driven by 2024-25); T2 93 bets +19.2% [-3.0, +42.3] p .046; T3 -1.4%; T4 -2.4%;
T5 form coef 0.322, boot [+0.015, +0.680] p .021 (1,000 resamples), n 328. None clears .01; with ~5 extra looks this is a lead at most.
Descriptive: early-window home teams won .581 vs power-devig .563 (diff +.0174 [-.0124, +.0463], n 935) -- not distinguishable;
rest of season .547 vs .563. Close-implied favourites early: mean p .667, won .672.

VERDICT: FAIL on all 5 (none near the .01 bar; T5 coef CI includes 0). KILL rule fires -> DEAD as specified.
The close is not detectably slow to price current-season form at <= 15 games. The only hint is the very first ~8 games
(T1/T5 sensitivity, p ~ .02, carried by 2024-25 alone), which would need its own pre-registered forward test (e.g. 2026-27
weeks 1-3 on paper) before anyone believes it. Caveat: no critic subagent has reviewed this yet.

## FORWARD PAPER TEST F1 -- pre-registered 2026-10-29 (~23:40Z), before any 2026-27 signal was computed
Origin: the <= 8-game sensitivity above (T1 137 bets +22.1% [+0.6, +44.5], T5 coef .322). That was one of ~5 extra looks
in a FAILED study, so the prior is that it is noise (carried by 2024-25). Paper test only, no orders.
Logger: `scripts/nba_early_form_logger.py` (tick / report), log `data/nba_early_form_log.jsonl`, one row per game.

Rule (frozen; nothing re-fit on 2026-27):
- Form exactly as DEV: adjMOV = MOV + mean of opponents' MOV, all from 2026-27 regular-season finals with ET date < game date.
- Eligibility: both teams have played 3..8 regular-season games before the game date (gp >= 3 as in the DEV form window).
- R = (adjMOV_h - adjMOV_a) - (M - H). M = -(home spread of ESPN's first listed book at the tick, normally DraftKings).
  H = 2.06, frozen as the mean of the four DEV season H_s values (2.53 / 2.17 / 1.55 / 2.00). DEV used each season's own
  mean of closing prices; forward cannot, so this is the one adaptation.
- Fire when |R| >= X = 5.3659 pts (frozen DEV value). Side = home if R >= X, away if R <= -X. Flat $1.
Entry: Kalshi KXNBAGAME YES ask for that side at the tick, plus the Kalshi taker fee (edge_math.fee_per_contract).
Target tick 60-90 min pre-tip. Ticks are run by hand, so each row stores mins_to_tip and `in_window`. Any row logged >= 15 min
before tip counts for the primary score. The time is reported, and an in-window-only cut is a sensitivity.
CLV (secondary): ESPN closing ML (same book) power de-vig prob on the bet side minus the Kalshi ask paid (also vs Kalshi mid).
Window: forward only, from 2026-10-29 until no eligible games remain (~ early-mid November). Earlier 2026-27 games are
EXCLUDED: no decision-time Kalshi price was recorded for them before their outcomes. The only pre-tip record,
data/l2_rating_log.jsonl, starts with tonight's 3 games. Every eligible game is logged whether or not the rule fires, so F
(form diff) can later be pooled across seasons for a T5-style coefficient.

Power, stated honestly: DEV had 74-76 eligible games per season after day 8, and the rule fired on 23-38 of them. Per-bet
profit SD was 1.33. So 2026-27 gives roughly 25-40 bets, and the SE of ROI is ~0.21-0.27. Even the DEV point estimate (+22%)
would be only ~1 SE, so this season CANNOT confirm the edge. 80% power for one-sided .05 at +22% needs
~(2.49*1.33/.22)^2 ~ 230 bets, about 7 seasons of this window.
Decision rules (judged once, at the end of the window):
- KILL if ROI at Kalshi ask + fee <= 0 OR mean CLV (vs ask) <= 0 -> file F1 DEAD, no further seasons.
- CONTINUE (log again in 2027-28 and pool) if ROI > 0 AND mean CLV > 0 but the bootstrap one-sided p >= .05.
- PASS (eligible to ask the user about live $1 units) only if one-sided p < .05 on ROI AND mean CLV > 0 with its 95% CI
  above 0. Not expected this season.

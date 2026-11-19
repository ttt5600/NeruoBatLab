# kalshi_fg_markets -- are Kalshi NFL/NCAAF field-goal markets mispriced vs a points->FG model? (agent, 2026-11-19)

User idea "kicker stats". Do-not-redo read: notes/nfl_kicker_system.md, notes/nfl_kicker_wx_matchups.md, FUND_BOOK 3d
(row "NFL FG props (defense style + script)": -8.2u/53, market wins). Hypothesis: Kalshi FG ladders are priced less
carefully than the game total/spread they derive from, so a model that only translates sportsbook points -> FGs may beat them.

## 1. Census (2026-11-19, read-only GETs; script results/kalshi_fg_markets/census_series.py -> census_raw.json)
All 8 series exist; all fee_type quadratic, multiplier 1 (taker 0.07*C*p*(1-p), cent-rounded up).
| series | events | markets | history | verdict |
|---|---|---|---|---|
| KXNFLGAMEFG | 0 | 0 | none | drop |
| KXNFLFG (team N+ made FGs, strikes 1+/2+/3+/4+, 8 mkts per game) | 161 | 1288 (1200 finalized) | occurrence 2026-09-10 .. 2026-11-23; median vol/market 2951 ct; median by strike 1+ 1095, 2+ 5544, 3+ 4398, 4+ 2746; total 8.54M ct | **only tradeable series** |
| KXNFLTEAMFG | 0 | 0 | none | drop |
| KXNFLLONGFG | 0 | 0 | none | drop |
| KXNFLLONGESTFG (season longest FG 64..70+) | 1 | 5 | season-long, 1 event, total vol 3426 | drop (no history) |
| KXNFL60YARDFGS (season count of 60+ yd FGs) | 1 | 6 | season-long, 1 event, vol 1055 | drop (no history) |
| KXNCAAFTEAMFG (team N+ FGs) | 33 | 204 finalized | 2026-08-29 .. 2026-09-13 only; median vol by strike 115-256 ct | drop (2 weeks, thin) |
| KXNCAAFTOTALFG | 1 | 6 | 1 game, total vol 19 | drop |
Rules (KXNFLFG): "If <team> records N+ field goals in the <A> vs <B> game ..." = made FGs incl. OT; postponed >48h -> special.

## 2. PRE-REGISTRATION (written before any Kalshi quote is joined to outcomes or a return is computed)
- Series: KXNFLFG only. Unit = team-game (2 per event), 4 strikes.
- Split (by NFL schedule week, 2026): DEV = weeks 1-7 (gamedays 2026-09-09 .. 2026-10-26, 107 games, 71%).
  SEALED = weeks 8-10 (2026-10-29 .. 2026-11-16, 43 games): not fetched, not scored; opened only on lead/user yes.
  FORWARD = week 11+.
- Model M1 (fit ONLY on nflverse pbp 2018-2025 REG+POST, no 2026 data): per team-game made FGs (pbp
  field_goal_result=='made' by posteam). Team implied points from nflverse schedule lines: home = (total_line +
  spread_line)/2, away = (total_line - spread_line)/2. Indoor = roof in {dome, closed}. For k=1..4: separate logistic
  P(FG>=k) ~ implied + implied^2 + indoor; ladder forced monotone (cumulative min over k). No kicker/wind terms (one model,
  one trial). Known small lookahead: nflverse lines for played games are the close (kickoff), decision is T-60; noted.
- Decision time T = kickoff - 60 min; kickoff = schedule gameday+gametime in America/New_York (per date) -> UTC,
  cross-checked vs Kalshi occurrence_datetime. Quote = last 1-minute candle ending <= T (fetched [T-180min, T]); none -> no bet.
- Rule R1: for each market, YES cost = yes_ask, NO cost = 1 - yes_bid; valid if cost in [0.03, 0.97] and yes_bid > 0.
  fee = 10-lot quadratic taker, cent-rounded per order, per contract. edge = p_side - cost - fee. Per team-game take the
  single max-edge (strike, side); bet 1 ct if edge >= 0.04. Hold to settlement (Kalshi result).
- Metric: ROI = sum pnl / sum (cost+fee). Bootstrap 10,000 resamples clustered by game, and separately by ET gameday.
- PASS (-> ask lead for sealed): n >= 50 bets AND ROI > 0 with BOTH clustered 95% CIs lower bound > 0 AND observed ROI above
  the 95th pct of placebo (model ladders shuffled across dev team-games, 200 shuffles, same rule).
- KILL: ROI <= 0, or placebo pct < 80. Anything in between = "not distinguishable", no sealed request.
- Descriptive (not trials): Brier/log-loss of M1 vs Kalshi mid at T per strike; calibration of Kalshi mid vs outcome per strike;
  M1 calibration on 2026 dev outcomes.
- Edgelab: register trial kfg1_points_model_t60 (family kalshi_sports_pregame) before scoring dev returns.

## 3. Log
- 2026-11-19 model fit (results/kalshi_fg_markets/model.py). Pooled 2018-25 (4454 team-games): base P(>=1..4) .820/.494/.220/.074.
  Implied points barely move FG counts (in-sample by implied bin, P(>=2) ranges .40 (<=17 pts) .. .53 and falls again >29 pts:
  high scorers score TDs). Season drift is upward: P(>=2) 2018 .468 -> 2024 .537 / 2025 .528; P(>=4) .052 -> .091.
  **AMENDMENT (before any Kalshi quote/outcome join; only training data seen): M1 fit window changed to 2023-2025** (1710
  team-games; base .830/.520/.232/.088) because the pooled fit is biased low for 2026. Same features. preds -> preds_2026.parquet.
- Edgelab: registered kfg1_points_model_t60-751dc009 (family kalshi_sports_pregame, 22 trials -> bar p < 2.27e-3) and
  dataset kalshi_nfl_fg in edgelab/vault.yaml BEFORE candles were fetched. Spec edgelab/specs/kfg1_points_model_t60.yaml.
- Candles: results/kalshi_fg_markets/fetch_candles.py (dev weeks 1-7 only; sealed weeks 8-10 NOT fetched). 1288/1288
  markets matched one-to-one to nflverse games (JAC->JAX, LAR->LA; ticker date +-1 day). Kalshi occurrence_datetime is
  ~kickoff+3h, so kickoff comes from the schedule. 856 dev markets, 835 with a quote at T; staleness median 4 min, p75 14 min.
- Mechanics (dev.py --mech): asserts T = kickoff - 3600 s for all rows, quote end <= T, traps.check_epoch_seconds passed.

## 4. DEV RESULT (results/kalshi_fg_markets/dev.py -> result_dev.json, bets_dev.parquet)
- Liquidity at T: median spread 1+ 0.14, 2+ 0.02, 3+ 0.08, 4+ 0.08. Only the 2+ rung is tight.
- Calibration per strike (n, outcome rate, Kalshi mid, M1 p): 1+ 213 .854/.767/.834; 2+ 213 .521/.512/.528;
  3+ 212 .269/.218/.234; 4+ 197 .086/.080/.089. Brier(mid) - Brier(M1), game-clustered 95% CI: all +0.0063 [+0.0016,+0.0109];
  1+ +0.0177 [+0.0083,+0.0266] (mid depressed by the 0.14 spread, not a fair price); 2+ +0.0067 [-0.0025,+0.0159];
  3+ -0.0004 [-0.0073,+0.0066]; 4+ +0.0006 [-0.0019,+0.0032]; spread <= .03 only +0.0048 [-0.0013,+0.0112].
  -> On rungs that trade tight, M1 is NOT distinguishable from the Kalshi mid.
- KFG1 rule: n 56 bets (30 yes / 26 no), 36 wins, ROI +23.0%, game-clustered CI [-0.7%, +47.3%] (p .028),
  ET-gameday-clustered CI [+8.8%, +41.1%] (p .0002, only ~20 date clusters -> likely too narrow). Mean edge 0.070.
  Mostly 2+ rung (38 of 56 bets). By week pnl: +.82 +.36 +2.78 +.85 +1.20 +1.12 -.41 (units of $1/ct).
- Placebo (ladders shuffled across team-games x200): mean ROI +5.0%, p95 +16.1%; observed at 99.5th pct.
  Positive placebo mean = part of the profit is the base-rate ladder shape vs Kalshi (e.g. 3+ priced .218 vs .269 hit), not team info.
- Rows inspected (bets_dev.parquet): quotes pre-T, no post-kickoff quotes, prices plausible; 4 bets with quotes >30 min stale (pnl +0.43).
- **Verdict per prereg: NOT a pass** (game-clustered CI lower bound < 0; edgelab bar p < 2.27e-3 not met). ROI > 0 and placebo
  pct >= 80 -> "not distinguishable". Per prereg: no sealed request. Lead-level only. n 56 is the binding constraint.
- Proposal (lead/user decides): forward PAPER log of KFG1 from week 11 (no orders) at ~10 bets/week; or the user may choose to open
  the sealed weeks 8-10 (43 games, ~22 bets expected) -- that window alone cannot reach the bar.

## Learnings
### Kalshi KXNFLFG is the only FG series with history
161 events since 2026-09-10, 8 markets each (team 1+..4+ made FGs), median volume ~3k ct/market; NCAAF FG series ran
2 weeks only (Aug 29-Sep 13 2026, ~100-250 ct/market); season-long longest-FG / 60-yd count have 1 event each. Source census_raw.json.
### Sportsbook points barely predict team FG counts
nflverse 2023-25: P(>=2 FG) .36-.56 across implied-point bins, non-monotone (high scorers score TDs); base rate drifted up
(P>=2 .468 in 2018 -> .537 in 2024). A points->FG model is mostly a recent-era base rate. Source results/kalshi_fg_markets/model.py.
### KFG1 points-model vs Kalshi FG ladder: +23% ROI on 56 bets, not distinguishable
Game-clustered CI [-0.7%, +47%]; placebo mean +5% (base-rate shape alone profits a little); on tight (2+) rungs M1 Brier is not
distinguishable from the mid. Too few events for the 22-trial family bar. Source result_dev.json.
### Few date clusters give over-tight bootstrap CIs
ET-gameday clustering (~20 clusters) gave CI [+8.8%,+41%] while game clustering gave [-0.7%,+47%]; with < ~30 clusters
the cluster bootstrap understates variance -- report the more conservative one.

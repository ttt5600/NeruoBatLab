# Player simulation vs simple team rating, NBA game lines, scored vs OPEN and CLOSE (2026-10-22)

User ask (10-22): "Look at the LATEST research. Simulating player scores might be a good way to predict games (the
RotoBot way). Otherwise plain analytics could work well too; simple models often beat convoluted ones."

- Script: `scripts/nba_player_sim.py`. The rules were pre-registered in its docstring before any join to a market.
- Ledger dataset: `nba_player_sim`, 6 trials, all logged 2026-10-22, so the bar is p < 0.0083.
- Data: DEV only, seasons 2021-22..2024-25. NBA 2026 (2025-26) is sealed and was not read; the script asserts this.
- Scored frame for diagnostics: `data/lab/nba_player_sim_scored.parquet` (gitignored).

## Verdict
1. **The simple model wins. The player sim loses on every metric.**
   - Margin MAE: simple 10.26 / 11.11 / 10.99 vs sim 10.46 / 11.19 / 11.31 (2023 / 2024 / 2025).
   - Log loss: simple 0.6194 vs sim 0.6391.
   - The sim loses money against both the open and the close.
2. **Neither model beats the CLOSE.**
   - Log loss vs the close: simple +0.027 worse [+0.020, +0.035]; sim +0.047 worse.
   - Stacking coefficient on the simple model, given the close: +0.01 [-0.17, +0.19].
   - Betting the gap at the closing price: simple -2.3% (p .78); sim -9.0%.
   - This matches notes/nba_public_models.md and the ~45 earlier closing-line angles.
3. **The simple model does carry information the OPENER lacks.** This is the first NBA game-line result to pass a bar.
   - Stacked with the ESPN BET opener, its coefficient is +0.30 [+0.09, +0.52].
   - For bets where the model disagrees with the open by at least 4 points, the line moved toward the model by
     **+1.53 probability points on average** (CLV). That is 1,512 bets with p < 1/4000, positive in both seasons.
     It also holds for home bets, away bets, favourites and underdogs, net of the blind open->close home drift
     (t 8.8). It grows with the threshold: +1.3 pts at x = .02, +2.3 pts at x = .10.
   - The CLV trial passes the dataset bar (.0083) and the whole-ledger bar (0.05/167 = .0003).
4. **But it does not yet make money after the opener's vig.**
   - Realised ROI at the opening price: +3.8%, CI [-3.9%, +11.8%], p .17. Not a pass.
   - Expected ROI implied by the close (close-fair prob × opening decimal odds − 1): +1.1%, CI [-0.05%, +2.35%].
     That is borderline. The opener holds about 4.1% overround, so +1.5 pts of CLV only just clears it.
   - Post-hoc only, not a trial: at larger gaps the close-implied EV rises (+1.8% at x = .06, +2.8% at .08, +3.6% at
     .10). This is the lead to pre-register for sealed data.
5. **Why the sim failed: summing independent player draws overstates game variance.**
   - Its implied sigma is about 23.7 points; the real figure is about 12.5.
   - Each player's points were drawn independently. In reality teammates share shots, so their points are
     negatively correlated.
   - The probabilities therefore get compressed toward 0.5, and the sim bets underdogs 91% of the time.
   - Even rescaled post-hoc (mu/12.5), its mean margin is worse than the ridge: log loss 0.6349 vs 0.6194,
     corr(mu_A, mu_B) 0.78.
   - It also lacks rest/travel and has no injury feed (see limitations).
   - Lesson: a player sim needs (a) a team-level possession/shot budget so player outcomes compete for the same
     shots, and (b) real availability news. Without both it is a noisier version of a team rating.

**Status: DEV ONLY. Not an edge yet.** The opener CLV is real on dev. The money test (ROI or close-implied EV at the
open) has not passed. Proposed sealed test below; it needs the user's explicit yes.

## 1. Literature, latest first (research agent, web, 2026-10-22)
Kaggle, ScienceDirect, De Gruyter, Nature and MDPI blocked full-text fetches. Rows marked (snippet) rest on search
snippets only and are UNVERIFIED.

**Market benchmark:**
- The Las Vegas opening spread in 2006-14 had margin RMSE ~11.7 and picked 69.2% of winners (Manner, JQAS 2016,
  [PDF](https://static.uni-graz.at/fileadmin/_Persoenliche_Webseite/manner_hans/Publikationen/NBA_revision2.pdf)).
- Bookmaker favourites won 69% (Pinnacle closing odds 2010-14) (Hubáček & Šír,
  [arXiv 2010.12508](https://arxiv.org/pdf/2010.12508), 2020).
- Our close: 66-70% accuracy, margin MAE 9.8-10.6, de-vigged log loss 0.592.

**MARKET-TESTED evidence:**
| Source | Date | Method | Result vs line |
|---|---|---|---|
| Manner, JQAS 12(1) | 2016 | Kalman dynamic team strength; OLS ratings | **Opening** spread: every model significantly worse (MSE 143.1 / 142.2 vs 137.5). A 50/50 blend with the line: 69.4% vs 69.2% correct. The simple OLS beat the Kalman model |
| Hubáček, Šourek & Železný, IJF 35(2); arXiv 2010.12508 | 2019/2020 | Neural net decorrelated from the book | Mostly Pinnacle **closing** odds, 9 seasons. Accuracy 61-69%, below the book. Best profit +1.7% needs portfolio staking; uniform staking lost at almost every setting |
| Walsh & Joshi, arXiv 2303.06021 v4 | 2024-02 | ML models chosen by calibration | Westgate **closing** ML, ONE season (2018-19): +34.7% vs -35.2% when chosen by accuracy. Single season, fragile ([arXiv](https://arxiv.org/pdf/2303.06021)) |
| Player-absence study, Finance Research Letters | 2015 | Line movement on player absences | Openers err when players are absent; the close fixes it; nothing profitable at the close (snippet, [link](https://www.sciencedirect.com/science/article/abs/pii/S1544612315000227)). **This is the same pattern our CLV result shows** |
| Štrumbelj & Vračar, IJF 28(2) | 2012 | Possession-level Markov **simulation** | "Comparable" to logit/ratings models; the bookmaker odds were best (snippet, [link](https://iaorifors.com/paper/76013)) |
| Informal GitHub test, 2022-23..2025-26 | 2026-10-18 | Ratings model vs posted line, n = 4,374 | Line RMSE 13.30 vs model 13.83; model-vs-line ≥2 pts went 50.5% ATS ([issue](https://github.com/isaacrome21-sys/SportsEdge/issues/2395)). Informal; not a certified close |

**ACCURACY-ONLY claims (never tested against a line):**
- Elo alone: 67.2% accuracy, beating Four Factors (64.0%); simple and complex nets were about equal (Migliorati 2021,
  [arXiv 2111.09695](https://arxiv.org/pdf/2111.09695)).
- 538 Elo with a season-dependent K beat plain Elo, but not significantly per season (Harvard Sports Analysis
  2019-01, snippet).
- DraftKings 2024 Data+AI Summit deck: the book itself prices with Monte Carlo engines on Kalman player/team
  ratings, and shows no results
  ([deck](https://microsites.databricks.com/sites/default/files/dais/2024/D241751B_How%20To%20(Maybe)%20Make%20Money%20Betting%20On%20The%20NBA%20Using%20Rocket%20Science%20V3_1717677883986001pwuO.pdf)).
  **The player-sim architecture is what the market already uses.**
- Agent-based NBA simulation (Erazo, WSC 2023, [arXiv 2304.09918](https://arxiv.org/abs/2304.09918)): no figures.
- 2024-26 papers reporting 74-76% accuracy (Sci Rep 2025, PLOS ONE 2024): these use same-game statistics, i.e.
  leakage (UNVERIFIED for Sci Rep).
- DARKO, EPM, LEBRON, BPM, Evan Miya: no public out-of-sample or market test found.
- Minutes models: practitioner heuristics only (e.g. RotoGrinders' 0.75 × season + 0.25 × last 5). No peer-reviewed
  NBA minutes-forecast accuracy study was found.
- MIT Sloan 2024-26: no game-prediction or player-simulation papers found.

**Kaggle March Mania (NCAA):**
- 2026 winner (H. Horan): XGBoost on seed, quality wins and an efficiency rating, plus EvanMiya player ratings
  adjusted by hand for injuries. CV Brier 0.159; leaderboard 0.1097 on 126 games, 1st of 3,485. Used no betting
  lines ([writeup](https://github.com/harrisonhoran/kaggle-march-mania-2026-1st-place/blob/main/kagglewriteup.md)).
- 2024 winner: R Monte Carlo on third-party ratings plus judgment
  ([mlcontests](https://mlcontests.com/state-of-machine-learning-competitions-2024/)).
- 2025 winner: UNVERIFIED.
- Lopez & Matthews (2014 winners, [arXiv 1412.0248](https://arxiv.org/pdf/1412.0248)): the Vegas spread plus
  Pomeroy won, and luck dominates the leaderboard.

**Literature synthesis:**
- No published player-sim backtest against the spread or close exists.
- Simple ratings match or beat complex ones on accuracy (Manner, Migliorati).
- Every market test loses at the close.
- The only inefficiency documented is at the **opener**, around player news.
- Our result reproduces all of these.

## 2. Models (frozen hyperparameters, tuned on 2021-22 games from 2021-12-01 only)
- **A SIMPLE:** weighted ridge on margins with these terms: team ratings (penalty λ 5); a home term; back-to-back
  flags for each team; rest difference capped at 3; travel in thousands of km from the previous game. Recency
  half-life 60 days. P = Φ(mu/12.5). Tuning grid MAE 11.15-11.41; the choice barely matters.
- **B PLAYER SIM** (2,000 sims per game):
  - Roster = players who appeared for the team in its last 10 games.
  - P(play | consecutive team games missed), estimated on 2022: 0 → .862, 1 → .412, 2 → .293, 3-5 → .210,
    6-9 → .126.
  - Minutes = EWMA(0.3) of past minutes, rescaled to 240.
  - Points per minute shrunk with 300 prior minutes, half-life 60 days. Points drawn as 2·Poisson.
  - Plus a team defence adjustment, home 1.59, and a team shock with sd 6. The chosen sd was at the bottom edge of
    the pre-registered grid {6, 8, 10, 12}; the sim wanted less noise because the player noise was already too large.
- Sanity check: box-score points sum to the final score in 99.8% of games.

## 3. Results (scored = regular season, non-neutral, both teams ≥5 games played; 3,445 games 2023-25; 2,284 with ESPN BET open, 2024-25)
| Trial (x = .04) | Bets | Metric | Value | 95% CI (sum) | p | Seasons + | Pass |
|---|---|---|---|---|---|---|---|
| S1 simple_open_roi | 1,512 | ROI at opener | +3.79% | [-59.3, +177.9]u | .169 | 2/2 | no |
| S2 sim_open_roi | 1,784 | ROI at opener | -5.81% | [-245.8, +46.6]u | .91 | 0/2 | no |
| **S3 simple_open_clv** | 1,512 | mean CLV (prob pts) | **+0.0153** | [+18.1, +28.2] | <.00025 | 2/2 | **YES** |
| **S4 sim_open_clv** | 1,784 | mean CLV | **+0.0068** | [+6.9, +17.5] | <.00025 | 2/2 | **YES** |
| S5 simple_close_roi | 2,355 | ROI at close | -2.30% | [-197, +87]u | .78 | 0/3 | no |
| S6 sim_close_roi | 2,744 | ROI at close | -9.02% | [-415, -84]u | .999 | 0/3 | no |

**Descriptive, on the same games** (dLL > 0 means the model is worse; stack = the model's coefficient given the market):
- Simple vs close: dLL +0.0271 [+0.0195, +0.0346]; stack +0.01 [-0.17, +0.19].
- Simple vs open: dLL +0.0137 [+0.0046, +0.0225]; stack **+0.30 [+0.09, +0.52]**.
- Sim vs close: dLL +0.0468; stack -0.18. Sim vs open: dLL +0.0384; stack +0.14 [-0.19, +0.47].
- ESPN BET open vs ESPN BET close: the open is worse by +0.0127 [+0.0073, +0.0185]. The close is the better forecast.
- corr(model − open, close − open): simple +0.257, sim +0.133. Mean |close − open| = 4.5 prob pts.
- Blind baselines (CLV): always-home -0.35 pts (t -2.6); always-favourite +0.12 pts (n.s.).

## 4. Limitations / traps
- **No historical injury feed.** ESPN's summary `injuries` block for a 2024 game returned CURRENT entries dated
  2026-10-18 (checked on event 401656363), so it is not point-in-time. The sim used recent appearances only.
  Using who actually played would be a leak and was not done. That leaves the sim's main theoretical advantage,
  faster injury response, untested.
- **ESPN's "open" timestamp is undocumented.** It is ESPN BET only and 2 seasons. Whether that price was bettable,
  and at what limits, is UNVERIFIED. Openers usually carry low limits.
- Proportional de-vig (the opener's overround is ~4.1%). Ratings start each season from the previous season, and
  the first 5 games of each season are not scored.
- The CLV is measured against the ESPN BET close, i.e. the same book.

## 5. Proposed next step (needs user's yes; NOT run)
**Sealed test O1** on NBA 2025-26 (season 2026):
- Model A frozen exactly as above (H 60, λ 5, σ 12.5, refit daily, no tuning).
- Rule: |p_A − p_open| ≥ 0.06, the ESPN BET opener (fixed now, before any sealed look).
- PRIMARY metric = close-implied EV at the opening price, one-sided game bootstrap, pass p < 0.05 and > 0.
- Secondary: realised ROI and CLV.
- First a schema-only check that lines_2026 contains `*_open` columns, run with the user's OK.

Going forward, an **opener logger** (gap_logger-style: record each book's first posted NBA line and Kalshi's first
price) is needed before any of this is bettable.

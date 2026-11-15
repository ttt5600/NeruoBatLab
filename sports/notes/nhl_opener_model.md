# nhl_opener_model -- de-biased xG/goalie NHL totals model vs Kalshi KXNHLTOTAL ladders (agent, 2026-11-15)

## Question
Kalshi NHL total ladders list ~70 h pre-tip and sit on the opener (timing gate D -0.048 goals [-0.069,-0.026],
notes/kalshi_openers_xleague.md). Does a de-biased NHL totals model built on xG (MoneyPuck), goalie GSAx and rest
carry CLV vs Kalshi's OWN 50c rung and make money buying that rung at ask + taker fee? Two decision times:
(A) Kalshi first-tight hour (~1 day out, starting goalie NOT known -> projected goalie), (B) tip - 60 min with the
actual starter (goalie confirmation is the news that moves NHL totals).

## Disclosed before pre-registering
- L34 (scripts/nhl_simple_model.py; ridge goals + save%-GSAA + b2b + shots) had dev CLV vs the ESPN BET opener
  (totals +0.073 implied goals, p<1/4000) but was KILLED forward: 140 settled picks vs the PINNACLE opener (open_lines
  logger), CLV -0.0015 prob [-0.0050,+0.0020] (notes/scorecards.md). Reading: ESPN BET's opener was soft; Pinnacle's
  opener already contains a goals/goalie/rest model. ACT (actual starter) vs STRICT added ~nothing in L34 dev.
- nhl_luck: PDO luck is priced (~90%) by the close; blind 6.5 unders do not replicate.
- Across leagues opener models carry ~0.8-0.9 pts of CLV vs Kalshi's own line, taker break-even ~1.2-1.4 pts
  (k50_strike, cbb_opener_model); maker at the 50c rung is adversely selected (maker_fairfill, dead).
- Seen of the Kalshi NHL data: only schemas and 2 header rows of q1_nhl_rows.csv. No model, no gap, no Kalshi outcome
  computed by me.
- Seal: NHL 2025-26 is sealed for scoring. Following the cbb_opener_model precedent I use 2025-26 RESULTS
  (MoneyPuck team/goalie xG game logs) ONLY as the prior input for 2026-27 predictions; no 2025-26 line is read,
  and no accuracy or market number is computed on 2025-26.

## Data
- MoneyPuck game-by-game team files (situation 'all': xGoalsFor/Against, goalsFor/Against, shots) and goalie files
  (xGoals against, goals against, icetime -> starter = most icetime per team-game). Seasons 2021-22 .. 2026-27.
- NHL schedule (api-web.nhle.com) / ESPN for tip times. Kalshi KXNHLTOTAL hourly candles 10-01..11-11 already
  pulled (results/kalshi_openers_xleague/nhl_*.parquet) + refetch 11-12..11-14 via kx_throttle.
- Dev book check: ESPN BET total open/close with prices, 2023-24 + 2024-25 (data/espn/nhl/lines_2024/2025).

## Model (fixed a priori; no tuning against any market number)
Per team t vs opponent u, at decision date D, using only games with ET date < D:
- O_t = 0.7*xGF/g + 0.3*GF/g (all situations), D_u = xGA/g (shot quality allowed, goalie-free), exponentially
  weighted by age (half-life 45 days), shrunk with 10 pseudo-games toward 0.5 x (last season final) + 0.5 x league.
- Goalie: per-goalie GSAx rate s = sum(xGA - GA) / (sum xGA + 100 pseudo-xG), half-life 365 days, across seasons.
  Opponent goalie factor G_u = 1 - s. (A) projected starter = share of the team's last 10 starts (weighted G);
  (B) actual starter.
- Rest: back-to-back second leg: own goals -0.24, opponent goals +0.21 (L34 coefficients fit on 2021-24, not on any
  Kalshi data).
- lam_t = L * (O_t/L) * (D_u/L) * G_u * home(+/-2%) + rest, L = league goals/team-game (same weighting).
  Total: P(over x.5) from independent Poisson(lam_h), Poisson(lam_a) with a tie adding 1 goal (OT/SO winner);
  model median m = kx.implied() over the ladder x = 2.5..10.5 (same scale as Kalshi's ladder median).

## Kalshi test (2026-27, games 10-07..11-14; nothing tuned on it; next forward games = holdout)
- Ladder at hour h = last hourly candle ending <= h, two-sided rungs only; L_K(h) = kx.implied ladder median.
  S50 = two-sided rung with mid nearest 0.50 (tie -> narrower spread).
- Trial A `nhl_om_a_kft`: T = Kalshi first-tight hour (anchor spread <= 4c) and >= 2 h pre-tip; model (A).
- Trial B `nhl_om_b_goalie`: T = last candle ending <= tip - 60 min; model (B) with actual starters (proxy for the
  morning-skate confirmation; small lookahead on warmup surprises, flagged).
- De-bias: c(g) = mean(m - L_K(T)) over the most recent 100 Kalshi-test games with tip < T (>= 30 required, else no bet).
  g = m - c - L_K(T). Over if g >= k, under if g <= -k. **k = 0.40 goals** (a priori: 1 goal ~ 16c at the 50c rung;
  taker friction ~2.8c ~ 0.18 goals of true edge; model gaps shrink ~50% vs the market). Descriptive k 0.25/0.5/0.6.
- Buy S50 on our side at the ask (YES) or 1 - bid (NO) + Kalshi taker fee ceil(0.07*100*P*(1-P))/100 per contract
  (100-ct order). Graded by the Kalshi result of that rung.
- ROI per $: win (1-cost)/cost, loss -1. CLV_c (primary CLV, cents) = Kalshi mid of the bought rung at the last candle
  <= tip minus entry mid, signed. CLV_g (goals) = L_K(last) - L_K(T) signed; CLV vs DK close median also reported.
- Bootstrap clustered by ET game date, 4,000 resamples, seed 0; one-sided p = P(mean <= 0).
- Verdict per trial: PASS (-> sealed/real-money ask) only if ROI p < edgelab nhl_lines bar AND CLV_c > 0.
  GO forward paper if ROI > 0 and CLV_c > 0 (points) and n >= 30. KILL if ROI CI upper < 0, or CLV_c <= 0 with n >= 40.
  Otherwise NEEDS DATA.
- Descriptive (not judged): goalie absorption -- regress L_K(h) - L_K(T_A) on the goalie delta m(B) - m(A) for
  h = tip-24h .. tip-1h: when does Kalshi price the confirmed starter?
- Dev book check (descriptive gate for "is the model any good"): 2023-24/2024-25 vs ESPN BET: CLV (close median -
  open median, signed) for |g| >= 0.4 and the stack coefficient of (final - close median) on (m - c - close median).

## Results

### Edgelab (registered 2026-11-15 ~10:25Z, before kalshi.py / dev.py were run)
nhl_om_a_kft-754cfc46, nhl_om_b_goalie-e983d6c5 (family nhl_lines, 28 trials all-time -> bar p < 1.79e-03; A was
registered at 27 -> 1.85e-03). rule forward:unwired. Specs edgelab/specs/nhl_om_*.yaml.

### Data (results/nhl_opener_model/)
- MoneyPuck: 33 team + 176 goalie game-by-game CSVs (fetch_mp.py -> mp/), 2026-27 through 2026-11-13. Starter in
  history = most goalie icetime per team-game. 2026-27 starters from NHL API boxscore `starter` flag
  (fetch_starters.py -> starters_2027.parquet, 317 regular-season games 10-07..11-14, 0 missing).
- Kalshi: 290 events 10-01..11-11 (kalshi_openers_xleague) + 27 events 11-12..11-14 refetched through kx_throttle
  (`KX_TAG=_nhlom q1_fetch.py nhl`, log kfetch.log). 317 regular-season events, all with candles.
- ESPN historical openers: ESPN BET only in 2023-24/2024-25 (as L34 found); 2025-26 sealed (not read); 2026-27 DK.

### Dev book check vs ESPN BET, 2023-24 + 2024-25 (dev.py -> dev.json, dev_rows.parquet; descriptive)
2,548 games (2,482 with c; c mean -0.12 goals). MAE vs final: de-biased model 1.862, ESPN BET open median 1.875,
close median 1.872. CLV (close med - open med, signed), date-clustered: k0.4 n482 +0.166 goals [+0.144,+0.188]
(seasons +0.157 / +0.176; overs 54%); k0.25 +0.125, k0.5 +0.198, k0.6 +0.223 (monotone). Stack: (final - close med)
on (m - c - close med) slope 0.46 [0.18, 0.74] -- the model adds to ESPN BET's close too (ESPN BET was a soft book).

### Kalshi test 2026-27 (kalshi.py -> kalshi.json, kalshi_rows.pkl, bets_A.csv, bets_B.csv)
T_A median 32 h pre-tip (mostly 12:00-13:00 ET the day before); T_B 1 h pre-tip. c in [-0.12, +0.08] (A).
| trial | n | W | ROI @ ask+fee [95%], p | P&L/ct [95%] | entry mid / cost | CLV_c bought rung, cents [95%] | CLV goals Kalshi | CLV vs DK close |
|---|---|---|---|---|---|---|---|---|
| **A first tight, k0.4** | 38 (23 dates) | 21 | **+5.8% [-25.6, +37.2], p .37** | +3.1c [-12.9, +19.2] | 0.496 / 0.521 | **+1.67 [+0.75, +2.67], p<.001** | +0.214 [+0.127, +0.313] | +0.214 [+0.138, +0.308] |
| A k0.25 (descr.) | 112 | 61 | +3.2% [-15.3, +23.7] | +1.3c | 0.498 / 0.532 | +1.29 [+0.81, +1.86] | +0.122 | +0.127 |
| A k0.5 (descr.) | 27 | 17 | +20.5% [-19.2, +61.0] | +10.4c | 0.495 / 0.525 | +2.20 [+1.07, +3.31] | +0.288 | +0.261 |
| A k0.6 (descr.) | 7 | 5 | +29% | | 0.507 / 0.546 | +0.79 [-1.0, +3.2] | +0.523 | +0.508 |
| **B tip-60 actual goalie, k0.4** | 36 (21 dates) | 17 | **-7.2% [-46.0, +29.4], p .66** | -4.0c [-23.6, +14.4] | 0.495 / 0.512 | **+0.10 [-0.18, +0.35]** | +0.004 [-0.012, +0.018] | +0.027 [-0.004, +0.058] |
| B k0.25 (descr.) | 103 | 49 | -8.6% [-28.9, +12.3] | -4.7c | 0.498 / 0.523 | +0.03 [-0.13, +0.20] | -0.001 | -0.002 |
- Rows inspected (bets_A.csv): 38 distinct events, anchor spreads <= 4c, entry mids 0.43-0.62.
  Early ladders (>48 h out) are sometimes non-monotone, so L_K can sit far from S50 (e.g. 26OCT19SJTOR L_K 5.38 vs
  S50 6.5 at 0.51) -- which raised the worry below.
- **Is A's CLV real information or reversion of a noisy L_K?** (checks.py -> checks.json, post hoc, not pre-registered)
  Regress (L_close - L_K) on the model gap (m - c - L_K) AND a placebo gap (trailing mean L_K - L_K, no team info),
  date-cluster bootstrap: A model 0.34 [0.21, 0.46], placebo 0.04 [-0.03, 0.10]; placebo bets |gap|>=0.4 n67 CLV +0.04
  goals. B model 0.01 [-0.01, 0.03]. DK open -> DK close move on (m - c - DK open): slope 0.39 (A), corr 0.42.
  -> A's CLV is model information, not noise reversion: by tip Kalshi moves ~1/3 of the model gap toward the model.
- Lookahead check: D_A = ET date of T_A, inputs gameDate < D_A. 11 of 317 T_A fall 00:00-02:59 ET, when a late
  west-coast game from D_A-1 may still be in progress (one game in a 45-day-half-life rating; negligible, flagged).
- Goalie absorption (descriptive; slope of L_K(tip-h) - L_K(T_A) on goalie delta = m(actual starters) - m(projected),
  same D; sd of delta 0.15 goals, |delta| > 0.1 in 50% of games): h24 0.14, h12 0.21, h8 0.28, h6 0.37, h4 0.40,
  h2 0.43, h1 0.45, close 0.49; final total on delta 0.86 (noisy). Kalshi prices the confirmed starter mostly between
  ~12 h and ~6 h before tip (morning skate) and has absorbed it by T-60 -- hence B has no edge.
- Not yet critic-checked (global rule: a finding is `confirmed` only after a fresh critic). Status here = lead.

## Verdict
- Pre-reg A (k0.4): ROI +5.8% p .37 -> not a PASS (bar 1.85e-03); ROI > 0, CLV_c > 0, n 38 >= 30 -> **GO forward paper**.
- Pre-reg B (k0.4): ROI -7.2%, CLV_c +0.10c [-0.18,+0.35], n 36 < 40 -> NEEDS DATA by the letter; de facto no edge:
  at T-60 the model (with the actual goalie) has nothing Kalshi lacks.
- Unlike L34-vs-Pinnacle, this model DOES lead Kalshi's first-tight ladder (CLV +1.67c / +0.21 goals, clean of the
  reversion artefact). But the edge is smaller than taker friction: entry mid 0.496 -> cost 0.521 = 2.5c of half-spread
  + fee vs 1.67c of CLV, so CLV-implied P&L ~ -0.9c/ct (~ -1.6% ROI). The observed +5.8% is within luck (P&L CI
  +/-16c). Same binding constraint as NBA/CFB/CBB: ~65% of what taker friction needs.
- Why A works where L34 died: Kalshi tightens ~32 h out, ~5.6 h BEFORE Pinnacle's first number; Pinnacle's opener
  already contains a team/goalie model, Kalshi's early ladder (like ESPN BET's opener) does not fully.

## Forward rule (paper, no orders) -- nhl_om_a_kft-754cfc46
Daily, per NHL game: at Kalshi's first hourly candle with an anchor (mid nearest 50c) spread <= 4c and >= 2 h before tip,
compute m with model.py (inputs gameDate < ET date of that hour, projected goalie = last-10 start shares, b2b from the
schedule); c = trailing mean(m - L_K) over the last 100 scored games; g = m - c - L_K; |g| >= 0.40 -> buy S50 on our side
at ask + taker fee. Grade on the Kalshi result; CLV_c = rung close mid - entry mid. Rerun: `fetch_mp.py --refresh`,
`fetch_starters.py` (extend end date), Kalshi refetch with a new KX_TAG added to TAGS in kalshi.py, then `kalshi.py`.
Judge at 100 settled: ROI > 0 one-sided p < .10 AND CLV_c > 0 -> sealed/real-money ask; kill ROI CI upper < 0 after 60
or CLV_c <= 0 after 100. ~1 bet/day at k0.4 -> ~100 by mid-February.

## Proposals (fresh prereg needed, not run)
1. Edge must clear ~2.5c friction: A at k0.5 had CLV_c +2.2c (descriptive, forking path) -- forward-log k0.5 alongside.
2. Goalie-confirmation timing: Kalshi absorbs the starter between T-12 and T-6 h. A logger that timestamps confirmed
   starters (team reporters / DailyFaceoff) and snapshots the Kalshi ladder at that minute would test whether Kalshi lags
   the news (cf. news_lag NFL: dead there). Needs timestamps we do not have historically.
3. Earlier entry at Kalshi's very first two-sided ladder (~70 h): bigger model gap, wider spread; test cost first.

## Learnings

### An xG + GSAx NHL totals model leads Kalshi's first-tight ladder
At Kalshi first tight (~32 h pre-tip), de-biased MoneyPuck xG/GSAx/b2b model, |g| >= 0.4 goals: CLV +1.67c [+0.75,+2.67]
on the bought rung, +0.21 goals [+0.13,+0.31], n38; Kalshi moves 0.34 [0.21,0.46] of the model gap by tip; a no-info
placebo gap moves 0.04. Source results/nhl_opener_model/kalshi.json, checks.json. Matters: L34 had ~0 CLV vs Pinnacle's
opener; Kalshi tightens before Pinnacle posts, so its early ladder is the soft NHL number.

### NHL model CLV vs Kalshi is ~2/3 of taker friction
Entry mid 0.496, cost (ask + fee) 0.521 = 2.5c friction vs 1.67c CLV -> CLV-implied ~ -0.9c/contract. ROI +5.8%
[-25.6,+37.2] n38 is luck-width. Source results/nhl_opener_model/kalshi.json. Same constraint as NBA/CFB/CBB.

### Kalshi prices the confirmed NHL starter by T-60; absorption happens T-12 to T-6 h
Slope of Kalshi ladder move on the goalie delta: 0.14 (T-24h), 0.21 (12h), 0.37 (6h), 0.45 (1h), 0.49 (close).
Model with the actual starter at T-60: CLV +0.10c [-0.18,+0.35] n36. Source kalshi.json goalie_absorption. Matters: any
goalie-news edge must trade at confirmation time (morning skate), not at T-60.

### Kalshi early ladder medians are noisy -- check CLV against a placebo gap
Early (>48 h) NHL ladders are sometimes non-monotone; L_K can sit ~1 goal from the 50c rung. Any rule that bets
g = m - L_K mechanically earns "CLV" from L_K reversion. Placebo (trailing mean L_K - L_K) here moved only 0.04, so the
model CLV survives. Source results/nhl_opener_model/checks.json.

### A DataFrame column named T collides with .T (transpose)
`x.T[i]` returned the transpose, not the column; name time columns Tdec or use x["T"]. Cost one crash in kalshi.py.

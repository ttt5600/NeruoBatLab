# clutch_market: does Kalshi's in-game NBA price misprice comebacks for "clutch reputation" teams? (2026-11-12)

User belief 11-12 (strong): "clutch factor" is real; some teams come back more than others and the market misses it.
Agent: clutch_market (research subagent, quant-trader lead). DEV ONLY. No orders, no git, no sealed rows.
Script: `scripts/clutch_market.py`. Results: `results/clutch_market/`.

## What is already known (read, not redone)
- notes/nba_comebacks_matchups.md (10-25): comeback OUTCOMES depend only on deficit x time x pregame spread; 0/69 team-trait
  cells survive (3PA rate, pace, bench, star usage, leader's defense/FT/bench drop-off ...). Coach "closes games well" p .61.
- notes/late_game_calibration.md: Kalshi in-game calibrated in every late bin (0-60c); model adds nothing (coef -0.06).
- L37 mid-game comeback rule SEALED FAIL (Kalshi Brier .1284 < model .1319). L38 trailing-longshot fade DEAD in dev.
- nba_ingame_lineup Gate B: on-floor quality / gone players / FOUL TROUBLE / starter minutes vs Kalshi: coef +0.116
  [-0.179,+0.434], n.d. -> foul trouble and bench-on-floor are already tested against Kalshi as a bundle.
- So the open angle is the MARKET conditional on the PERCEIVED trait (a team's comeback / close-game record).

## PRE-REGISTRATION (written 2026-11-12 before any outcome was joined to a trait or a price)

Data
- Game states: `data/nba_ingame_lineup_states.parquet` (ESPN pbp, every play, all periods: eid, period, clock_s, hs, as,
  wall UTC, home_t, away_t, p0, home_won, F_home/F_away, date, season, gtype), seasons 2021-22..2025-26.
  Rows dated >= 2026-01-15 are dropped AT LOAD (sealed) -- they are never read, not even for traits.
- Kalshi: KXNBAGAME 1-min candles per side, `data/nba_ingame.parquet` (2025-26) + `data/nba_late_candles_2025po.parquet`
  (2025 playoffs). DEV = games dated 2025-04-15 .. 2026-01-14 (same boundary as L37/L38/late_wp/lineup Gate B).
  SEALED = 2026-01-15 .. 2026-06-30 (user's yes only; not touched).

Trait (team-level, from games dated STRICTLY BEFORE the snapshot game's date; regular + playoffs; crosses seasons)
- T_CB (primary, "comeback record" -- what the public sees): over the team's last 82 prior games, the games in which
  the team trailed by >= 10 at any point; trait = share of those it WON. Needs >= 10 such games, else NaN (snapshot dropped).
- T_CL (secondary, "clutch record", NBA.com clutch definition): over the last 82 prior games, the games with
  |margin| <= 5 at any play with <= 300 s left in Q4, or any OT; trait = share WON. Needs >= 10, else NaN.
- Terciles: cut points = 1/3 and 2/3 quantiles of the trait over the dev SNAPSHOT rows (trait values only, no outcome);
  frozen numerically into the rule after `build`.

Snapshots (the trailing team's own market)
- For every 1-min candle (end c) of each side's market: state = last ESPN play with wall <= c - 60 s (60 s latency).
  Trailer = the side whose team is behind in that state. Two-sided quote (bid >= .01, ask <= .99), ask - bid <= .03.
- Cells: deficit D1 6-10, D2 11-15, D3 16+ ; phase P_MID = period 2-3, P_Q4 = period 4 (regulation only; Q1 and OT excluded).
- One snapshot per game x cell: the FIRST qualifying candle in that cell. Price = trailer mid (calibration);
  bets use the real ask of the side bought on that candle.

Primary measure (decides lead vs dead for the "market misprices clutch" claim)
- r = trailer won - trailer mid. Mean r by T_CB tercile; DIFF = mean r(top) - mean r(bottom).
  95% CI by game-cluster bootstrap (2000). Plus logistic: won ~ 1 + logit(mid) + z(T_CB), coefficient on z(T_CB)
  with game-cluster bootstrap CI. Same two for T_CL (secondary).
- LEAD if the T_CB DIFF CI excludes 0 (either sign: positive = clutch under-priced -> buy; negative = over-priced -> fade)
  AND the matching money trial below passes the edgelab dev gates. DEAD if the DIFF CI crosses 0.
- Power (stated before looking): ~650 dev games, residual sd ~0.3-0.4 clustered -> DIFF SE ~2-2.5c, MDE ~5c.
  A real 1-2c effect would not be detectable here; "not distinguishable" will not mean "zero".

Money trials (edgelab family kalshi_ingame, dataset nba_clutch_ingame, taker, fee 0.07 p(1-p), hold to settlement,
first qualifying snapshot per game across all 6 cells in time order, price in [0.03, 0.97]):
- CM1 clutch_buy   : trailer T_CB in TOP tercile -> buy trailer YES at its ask.
- CM2 clutch_fade  : trailer T_CB in TOP tercile -> buy the LEADER's YES at the leader market's ask (same candle end).
- CM3 low_buy      : trailer T_CB in BOTTOM tercile -> buy trailer YES at ask.
- CM4 low_fade     : trailer T_CB in BOTTOM tercile -> buy leader YES at ask.
  Family bar after registration = 0.05 / (29 + 4) = 1.5e-3. Kill: ROI CI below zero after 100.

Secondary "any reason the market ignores" (descriptive, one test): r vs foul trouble difference
F_leader - F_trailer (top-2 players in foul trouble, from lineup states) at the snapshot; game-cluster CI on the slope.
Bench depth / on-floor quality is NOT re-run: Gate B (nba_ingame_lineup) already tested it against Kalshi (n.d.).

## Log
- build (`scripts/clutch_market.py build`, results/clutch_market/build.json): 693 dev games, 676 with Kalshi candles,
  2,507 snapshots in 673 games (D1_MID 621, D2_MID 520, D3_MID 343, D1_Q4 460, D2_Q4 344, D3_Q4 219). Every snapshot has
  both traits (82-game window crosses into 2024-25). Frozen cuts T_CB 0.200 / 0.3243; T_CL 0.472 / 0.536.
  Audit: state lag 60-173 s (median 71 s); trailer mid falls with deficit/phase as expected (D1_MID .33 .. D3_Q4 .03);
  t_ask + l_ask median 1.01. T_CB tracks team quality (OKC .65, WSH .07) -- fine, the test is on the market residual.
- calib (`scripts/clutch_market.py calib`, results/clutch_market/calib.json; game-cluster bootstrap 2000):
  overall trailer won .197 vs mid .194, r +0.3c [-2.1, +2.7] (calibrated again).
  T_CB terciles r (won - mid): bottom -0.0c [-3.5,+3.7] (n 839/262 g), middle +2.2c [-2.0,+6.5], top -1.3c [-5.7,+3.6].
  PRIMARY DIFF top - bottom = -1.2c [-7.1, +4.8] -> NOT DISTINGUISHABLE. Logit coef z(T_CB) | logit(mid) +0.021 [-0.18, +0.21].
  By phase: MID -2.8c [-9.5,+4.3]; Q4 +0.9c [-5.1,+7.4].
  T_CL: DIFF -0.8c [-6.8, +5.4]; coef +0.044 [-0.18, +0.27]; MID -2.1c [-9.3,+4.7]; Q4 +1.2c [-4.8,+7.5].
  Realised MDE was ~6c, not the ~5c guessed (snapshots within a game are correlated; 6 cells/game ~ 1 effective obs).
- foul trouble (secondary): dF = leader - trailer top-2 players in foul trouble is nonzero in only 95/2,507 snapshots;
  slope -0.7c per unit [-12.2, +11.6]; leader-in-trouble r +6.6c [-11.8, +28.9] (n 31). Data too thin; n.d.
- edgelab (dataset nba_clutch_ingame, family kalshi_ingame now 33 trials, bar p < 1.52e-3), dev, taker + fee:
  CM1 top_buy  n 258 ROI -17.56% [-30.97, -3.91] p .99 (px .386, won .329, -7.0c/contract) DEAD
  CM2 top_fade n 258 ROI  +4.16% [-4.26, +12.91] p .16 (px .630, won .671, +2.7c/contract) DEAD (p >> bar)
  CM3 low_buy  n 258 ROI  -0.85% [-20.38, +20.59] p .54 DEAD
  CM4 low_fade n 250 ROI  -4.55% [-10.93, +1.46] p .92 DEAD
  Per-contract numbers recomputed outside the evaluator from the rule rows (same direction/size).
- Not critic-checked (negative result); data rows audited by hand (sample printed in build step).

## Verdict: DEAD (dev)
The market does not misprice comebacks by clutch reputation, in either direction, at a size this sample can see (~6c).
Point estimates lean toward clutch-reputation trailers being slightly OVER-priced early (CM1 loses 7c/contract on the
first 6-10-down Q2-Q3 snapshot; top-tercile r -1.3c overall, MID -2.8c), i.e. the reverse of the user's belief, but the
pre-registered primary DIFF crosses 0 and the fade (CM2) is p .16 vs bar .0015. Not a lead.
Post-hoc idea only (would be a NEW trial, forward only): fade top-T_CB trailers 6-10 down in Q2-Q3. Too small to clear
the family bar on any realistic forward n; not recommended.

## Learnings
### Kalshi in-game NBA prices do not misprice "clutch reputation" teams
Trailer won - mid, top minus bottom tercile of prior-82-game comeback record: -1.2c [-7.1, +4.8], 2,507 dev snapshots /
673 games (results/clutch_market/calib.json). Close-game record: -0.8c [-6.8, +5.4]. Adds the market side to the 10-25
finding that comeback OUTCOMES carry no team trait: neither the outcome nor the price has a clutch term.
### Comeback / clutch record is mostly team quality
T_CB ranged .07 (WSH) to .65 (OKC); a team's comeback rate is largely how good it is, which the pregame and in-game
price already hold. Strip strength before calling something a "clutch" trait.
### Several snapshots per game shrink the effective sample to about the game count
6 deficit/phase cells per game gave a ~6c MDE on 2,507 snapshots, about what 673 independent observations give.
Size in-game calibration tests by games, not rows.
### A mirror pair of taker rules is not a free two-sided test
CM1 (buy trailer) loses 7.0c/contract while its mirror CM2 (buy leader, same games) gains only 2.7c, because both pay
spread + fee (edgelab status clutch_cm1/cm2).

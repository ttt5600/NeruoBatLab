# kalshi_openers_xleague -- does Kalshi list early and price the OPENER outside the NBA? (agent, 2026-11-12)

## Question
notes/nba_l32b_exec.md: KXNBATOTAL lists ~3.4 h before our first Pinnacle sighting and prices the opener (|K - open| 0.33
vs |K - close| 1.82), so the de-biased L32b opener model is executable (K1 forward). Repeat for:
NFL totals + spreads (KXNFLTOTAL/KXNFLSPREAD), NHL totals (KXNHLTOTAL), CFB totals + spreads (KXNCAAFTOTAL/SPREAD),
CBB totals + spreads (KXNCAAMBTOTAL/SPREAD -- confirmed listed 11-12). Then, where Kalshi tracks the opener AND an
existing model leads the opener, price that model's picks at Kalshi ask+fee.

## Disclosed before pre-registering
- Known: L33 NFL forward CLV +0.50 pts n15 (lead brief); L34 NHL killed CLV ~0 n110 (scorecards); L35 Kalshi early NFL
  ML first prices move +3.3c toward the model but maker/taker not profitable; L36 CFB totals sealed: CLV +1.53 pts
  [+1.22,+1.86] holds OOS, ROI at real DK/ESPN-BET open +0.7% [-10.7,+11.5] (vig eats it).
- Q004 (pick_tight: model vs the first tight Kalshi anchor) exists in the NFL/NHL loggers; NBA Q004 died from a level-biased
  model. I have NOT looked at NFL/NHL pick_tight outcomes or any CFB/CBB Kalshi price before writing this.
- Data: open_lines_{nfl,nhl}_state/tight/picks (15-min logger since 10-26; archived games keep only CLV, so Pinnacle
  open times exist only for picked games + current games). ESPN core odds give DraftKings open/close (no timestamps)
  for every league. Kalshi candlesticks are public and historical.

## Pre-registration (written before any Kalshi-vs-open/close number for these leagues was computed)
### Q1 timing gate (per league x market; games 2026-10-01 .. 2026-11-11 [CBB 11-03..], FBS/D-I games Kalshi listed)
- AMENDMENT 17:40Z, before any outcome: window widened from 10-26 to 10-01 for power (DK open/close + Kalshi candles
  exist back to then; the Pinnacle reference B stays 10-26+ since the logger started then). Dev window for Q2 likewise.
- Kalshi ladder from public candlesticks (hourly from open_time to tip, 1-min at specific ticks). Implied line = strike where
  the YES mid ladder crosses 0.5 (linear interpolation between adjacent strikes; spreads: home-margin ladder built from
  both teams' "wins by over s" markets). FIRST TIGHT = first hourly candle whose anchor (strike with mid nearest 50c) has
  a two-sided spread <= 4c.
- Book reference A (all leagues): ESPN DraftKings open and close (fallback: first provider with open+close).
  Book reference B (NFL/NHL only, where the logger has it): Pinnacle first sighting time/line and last pre-start line.
- Metrics: hours from Kalshi open_time and first tight to tip; Kalshi first sighting minus Pinnacle first sighting (B);
  D = |K_first_tight - open| - |K_first_tight - close| (points; NHL totals in goals via the ladder), mean with game
  bootstrap 95% CI (4,000, seed 0); share closer to open; depth at first tight from the logger (NFL/NHL only).
- GATE (passes = "Kalshi tracks the opener, executable"): n >= 15 games with |open - close| > 0 AND D CI upper < 0 AND
  median first tight >= 12 h before tip. Otherwise "does not track the opener" (or "underpowered" if n < 15).

### Q2 model at Kalshi prices (only for gate-passing league x market with a model that has opener CLV)
- NHL: L34 killed (CLV ~0) -> no model step regardless of the gate. CBB: no model -> timing only.
- NFL (L33, nfl_simple_model live rule via open_lines_logger decisions): K-NFL rule = at the logger decision tick,
  S = Kalshi strike nearest the Pinnacle open (tie -> favorable to our side); bet if gap at S clears the L33 thresholds
  (spread 2.0, total 2.5); price = last 1-min candle ask <= tick + exact taker fee; graded on the final vs S; CLV =
  Pinnacle close vs S signed. Level bias check first: mean(model - open) per market; if |mean| > 1 pt the primary
  is de-biased (c = trailing mean of model - open over prior decisions, as L32b) -- the bias number is not an outcome.
- CFB (L36 frozen model, l36_cfb_sealed.build): K-CFB rule = S = Kalshi total strike nearest the DK open at Kalshi's
  first tight hour (>= 30 min before kickoff); bet if |pred - S| >= 3.5; price = ask + fee at that candle; graded on final
  vs S; CLV = DK close - S signed. Same bias check. Spreads: L36 spread model miscalibrated -> not used.
- Dev (2026-10-26..11-11) is descriptive + go/no-go: GO forward if dev CLV vs S > 0 (point estimate) and n >= 10;
  KILL (no forward) if dev CLV CI upper < 0. Forward judge per rule, same as K1: 100 settled, ROI@ask+fee > 0 one-sided
  p < .10 AND CLV > 0 -> sealed ask; kill if ROI CI upper < 0 after 60 or CLV <= 0 after 100.

### Tonight (11-12) forward logger (read-only, nohup, exits by 08:00Z 11-13)
results/kalshi_openers_xleague/tonight_logger.py: every 10 min, all open events in the 7 series + Pinnacle (nfl 889,
nhl 1456, cfb 880, cbb 493) -> per event: Kalshi first sighting, open_time, anchor bid/ask/strike, implied line, depth at the
anchor; Pinnacle first sighting + line. Rows feed the same Q1 metrics as forward data (not judged tonight).

## Results

### Registered before any Q2 outcome was seen (17:50Z)
edgelab knfl_l33_kalshi_taker-445b123e (nfl_lines, 130 trials, bar p < 3.85e-04) and kcfb_l36_kalshi_taker-613b7d62
(cfb_lines, 16 trials, bar p < 3.13e-03); rule forward:unwired (fails closed) like K1.
Bias check (pre-outcome, not an outcome): NFL logger model cache weeks 8-10 (43 games), model - market at cache time:
total -0.56 (sd 2.52), spread home margin +0.79 (sd 2.09, se 0.32) -> |mean| < 1, primary stays raw per the prereg.

### Q1 reference B: logger Pinnacle first sighting vs Kalshi listing (results/kalshi_openers_xleague/q1b_pinnacle.py ->
q1b_summary.json; games first seen by Pinnacle after the logger's first run; 15-min polling)
| lg mk | n | Pinnacle first sighting h before kick (med) | Kalshi open_time - Pin first (h, med [p10,p90]) | Kalshi first | first tight - Pin first (h, med) | depth at first tight, $ within 1c (YES ask / NO ask, med) |
|---|---|---|---|---|---|---|
| NFL spread | 23 | 160.7 | -2.2 [-40.2, +0.7] | 83% | -1.8 | 274 / 429 |
| NFL total | 17 | 160.7 | -2.2 [-28.8, +10.1] | 82% | -1.8 | 365 / 345 |
| NHL total | 52 | 27.7 | -42.3 [-49.7, -39.0] | 100% | -5.6 | 381 / 442 |
| NHL spread | 26 | 27.7 | -42.2 | 100% | -8.2 | 246 / 102 |
- NFL: Pinnacle posts next week's lines ~6.7 days out (Sunday night); Kalshi ladders appear ~2 h earlier and are tight
  (<= 4c) ~1.8 h before our first Pinnacle sighting. NHL: Kalshi lists ~42 h before Pinnacle's first number and is tight
  ~5.6 h before it. Depth at first tight is a few hundred $ within 1c (thinner than NBA's ~$4.7k at the L32 tick).

### Wrong turn: Kalshi occurrence_datetime is NOT kickoff
First NFL pass had |Kalshi last pre-kick - DK close| 2.4-4.3 pts (impossible for a liquid ladder). Cause: Kalshi
`occurrence_datetime` = kickoff + 3.0 h exactly (all 156 NFL events), so "last pre-kick" quotes were in-game. Fixed by
using the ESPN start; after the fix |K_last - DK close| = 0.38 (spread) / 0.39 (total) -> the implied-line method is sane.
Also: per-market candle GETs ran ~1 req/s; the batch endpoint GET /markets/candlesticks?market_tickers=a,b,.. (20/call) is ~20x faster.

### Q1 reference A (pre-registered gate): Kalshi first tight vs ESPN DraftKings open / close (q1_timing.py -> q1_summary.json,
q1_<lg>_rows.csv; hourly candles, implied = isotonic ladder 50% crossing; D = |K-open| - |K-close| on games whose line moved)
| lg mk | events | Kalshi lists h pre-kick (med) | first tight h pre-kick (med) | |K-open| | |K-close| | D [95%] | closer to open | |open-close| | GATE |
|---|---|---|---|---|---|---|---|---|---|
| NFL spread | 78 (68 moved) | 163 | 162 | 2.58 | 1.16 | +1.42 [+0.85, +1.97] | 29% | 3.10 | FAIL |
| NFL total | 78 (63 moved) | 163 | 162 | 1.93 | 1.56 | +0.38 [-0.17, +0.89] | 38% | 3.21 | FAIL |
| NHL total (goals, ladder median) | 290 (253) | 70 | 32 | 0.154 | 0.202 | -0.048 [-0.069, -0.026] | 65% | 0.22 | PASS |
| CFB spread | 285 (245) | 150 | 145 | 1.19 | 1.66 | -0.47 [-0.70, -0.24] | 59% | 2.12 | PASS |
| CFB total | 285 (225) | 150 | 139 | 1.22 | 1.89 | -0.67 [-0.90, -0.43] | 60% | 2.51 | PASS |
| CBB spread | 228 (177) | 23 | 20 | 1.05 | 1.46 | -0.42 [-0.65, -0.18] | 61% | 1.93 | PASS |
| CBB total | 228 (171) | 23 | 19.5 | 0.83 | 1.91 | -1.07 [-1.33, -0.81] | 72% | 2.28 | PASS |
(sanity: |Kalshi last pre-kick - DK close| = 0.36-0.41 pts in every football/basketball row, 0.10 goals NHL.
CFB/CBB = FBS/D-I games ESPN lists with a DK open+close that Kalshi also listed: CFB 285 of 360, CBB 228 of 474 (11-03..11-11).)
- NFL reading: ESPN's DK "open" is the LOOKAHEAD line (|open - close| 3.1-3.2 pts, ~2x a normal week's move), posted
  long before Kalshi lists. Kalshi lists Sunday night ~163 h pre-kick = the same time Pinnacle re-opens (ref B: Kalshi
  ~2 h before our first Pinnacle sighting). So vs DK's lookahead, Kalshi has already moved most of the way to the close.
  The gate as pre-registered FAILS for NFL. Kalshi first tight is still 1.2 (spread) / 1.6 (total) pts from the close,
  so there is room for opener CLV at Kalshi; it is just not the DK opener.
- NHL: Kalshi tight ~32 h pre-kick (ref B: ~5.6 h before Pinnacle's first number), and it sits on the DK open.

### Q1 reference B at the tick (q1c_pin_tick.py -> q1c_<lg>_summary.json; logger picks with a settled close only)
- NFL: Kalshi implied at our first Pinnacle sighting vs Pinnacle open / close: spread n 6 (3 moved) |K-open| 0.21 vs
  |K-close| 1.88; total n 3: 0.27 vs 1.06. Same direction as NBA, but n is tiny -- descriptive only.
- NHL totals (goals): n 54 (53 moved) |K-open| 0.073 vs |K-close| 0.133, D -0.059 [-0.091, -0.029], 66% closer to open.
  -> at the Pinnacle-open tick Kalshi IS the Pinnacle opener (as NBA).

### Q2 K-NFL dev (q2_nfl.py dev -> q2_nfl_rows_dev.csv, q2_nfl_summary_dev.json; registered before this was read)
19 L33 picks priced (2 no quote); 14 settled bets: 7-7, ROI @ ask+fee -4.7% [-57.7, +46.3]; CLV vs bought strike
+0.79 [+0.18, +1.43] pts (spread n 9: CLV +1.17, ROI -13.1%; total n 5: CLV +0.10, ROI +10.3%). Mean cost 0.544.
Rows inspected: quotes 0.7-56 min old (median ~4), spreads 1-7c, S within 0.5 of the Pinnacle open.
Pre-reg go/no-go: CLV > 0 and n >= 10 -> GO forward paper (CLV CI excludes 0; ROI not distinguishable from 0).
NOTE: the NFL timing gate (ref A) failed, but the logger decision tick IS the Pinnacle-open sighting (ref B), so the
K-NFL rule prices the right moment; the gate failure is about the DK lookahead line, not the logger's opener.

### Q2 K-CFB dev (q2_cfb.py -> q2_cfb_rows_dev.csv, q2_cfb_summary_dev.json; registered before this was read)
Frozen L36 preds (q2_cfb_preds.py: 691 2026 games, results-only walk-forward; 2022-25 games copied into cfb_espn/,
2026 refetched there, data/espn/cfb untouched). Decision = Kalshi first-tight hour, median ~134 h pre-kick.
| variant | n | W-L | ROI @ ask+fee [95%] | CLV vs S pts [95%] | mean cost | overs |
|---|---|---|---|---|---|---|
| **raw (primary)** | 63 | 35-28 | **-2.1% [-23.9, +20.3]** | **+1.49 [+0.86, +2.13]** | 0.577 | 42 |
| de-biased (c ~ +0.3, descriptive) | 52 | 31-21 | +3.5% [-19.2, +27.4] | +1.44 [+0.73, +2.17] | 0.591 | 30 |
- Pre-reg go/no-go: CLV > 0, n >= 10 -> GO forward. ROI not distinguishable from 0.
- Why cost 0.577 not ~0.52: S = strike nearest the DK OPEN, but Kalshi's first tight is ~1.2 pts off the DK open (see Q1),
  so the open-line strike is already priced away from 50c (and 32% of bought strikes had a > 4c spread; 5 asks > 0.70).
  POST HOC (not judged, forking path): bets whose bought strike was <= 4c wide n 43, ROI +11.4%, CLV +1.37. A cleaner
  rule for a FRESH prereg: measure the gap vs Kalshi's own anchor strike (the ~50c rung) at first tight, not the DK open.
- Bug caught: W counted 1 from an object-dtype bool sum (`won` column with NaN skips); fixed to (won == True).
- Model bias check (pre-outcome): trailing c = mean(pred - DK open) ~ +0.3 -> raw stays primary per the prereg.

## Verdict
| league x market | Kalshi tracks the opener? | model with opener CLV | K rule |
|---|---|---|---|
| NFL spread/total | vs Pinnacle's Sunday-night open YES (lists ~2 h before it; ref B tiny n); vs DK lookahead NO (gate FAIL) | L33 | K-NFL forward (dev CLV +0.79 [+0.18,+1.43], ROI -4.7% n14) |
| NHL total | YES (gate PASS; 32 h pre-kick, ~5.6 h before Pinnacle) | none (L34 killed) | none |
| CFB total | YES (gate PASS; tight ~139 h pre-kick) | L36 | K-CFB forward (dev CLV +1.49 [+0.86,+2.13], ROI -2.1% n63) |
| CFB spread | YES (PASS) | none (L36 spread model miscalibrated) | none |
| CBB total/spread | YES (PASS; strongest D -1.07 for totals; tight ~20 h pre-tip) | none | none -- the best place to point a NEW opener model |
Executable in mechanics everywhere tested; profit not shown anywhere (every ROI CI spans 0). Same story as K1 NBA:
the models carry CLV onto Kalshi strikes, and taker friction (~2.6c, plus off-50c strikes) eats the margin.

## Forward rules (paper, no orders)
- K-NFL (edgelab knfl_l33_kalshi_taker-445b123e): run `q2_nfl.py forward` after each NFL week settles (open_lines_logger
  picks with t >= 11-12; 1-min candles are historical, so pricing uses only pre-decision quotes). ~6-8 picks/week ->
  100 settled takes ~3 months (season ends first; judge at season end with whatever n exists, label underpowered).
- K-CFB (kcfb_l36_kalshi_taker-613b7d62): weekly `q2_cfb_preds.py --refetch` then
  `KX_LO=2026-11-12 KX_HI=<last settled date> KX_TAG=fwd q1_fetch.py cfb; KX_TAG=fwd q1_timing.py cfb; KX_TAG=fwd q2_cfb.py`.
  ~10-15 bets/week -> ~60 by the bowls. Both rules are `forward:unwired` in edgelab until an adapter reads the csvs.
- Proposal for the lead (needs a fresh prereg, not run): CBB totals opener model (KenPom-style ratings), since CBB is
  where Kalshi sits closest to the open (|K-open| 0.83 vs |K-close| 1.91) and tight ~20 h pre-tip.

### Tonight forward logger (running, read-only)
`results/kalshi_openers_xleague/tonight_logger.py --until 2026-11-13T07:55Z` (nohup, PID 86585 started 17:28Z; log tonight_logger.log
lines; exits by itself). Every ~10 min: 7 Kalshi series + Pinnacle nfl/nhl/cfb/cbb -> ticks.jsonl, first_seen.json.
First tick 17:28Z marks everything already listed as censored. Summarize after 08:00Z with `q3_tonight.py`
(q3_tonight_summary.json: new Kalshi events' first spread / hours before kick; new Pinnacle games' first sighting).

## Learnings

### Kalshi occurrence_datetime is kickoff + 3 h for NFL
All 156 NFL ladder events: occurrence_datetime - ESPN start = 3.0 h exactly. Using it as kickoff put "pre-game" quotes
in-game (|K_last - close| 2.4-4.3 pts; 0.38 after the fix). Source results/kalshi_openers_xleague/q1_nfl_rows.csv. Use
the ESPN/Pinnacle start, never occurrence_datetime, as the decision cutoff.

### Kalshi lists college ladders early and they sit on the opener
CFB totals/spreads list ~150 h pre-kick and are tight (<= 4c) ~140 h out; CBB ~23 h / ~20 h. At first tight, Kalshi is
nearer the DK open than the close: CFB total D -0.67 [-0.90,-0.43] (n 225), CBB total D -1.07 [-1.33,-0.81] (n 171).
Source q1_summary.json. Matters: opener models for college sports are executable on Kalshi.

### ESPN's DraftKings NFL "open" is the lookahead line
|DK open - close| averages 3.1-3.2 pts for NFL (vs ~1-2 for a week's move); Kalshi lists Sunday night with Pinnacle's
re-open, so vs DK open Kalshi looks "closer to the close" (spread D +1.42 [+0.85,+1.97]). Source q1_nfl_rows.csv.
Matters: never use ESPN's NFL open as "the opener" for timing questions; use a timestamped first sighting.

### NHL Kalshi totals list ~42 h before Pinnacle and match its open
Kalshi open_time is 42 h before our first Pinnacle sighting (n 52); at that tick Kalshi is 0.07 goals from the Pinnacle
open vs 0.13 from the close (D -0.059 [-0.091,-0.029], n 54). Source q1b_summary.json, q1c_nhl_summary.json.

### Opener models carry CLV onto Kalshi, ROI still ~0
K-NFL dev CLV +0.79 [+0.18,+1.43] ROI -4.7% (n 14); K-CFB dev CLV +1.49 [+0.86,+2.13] ROI -2.1% [-23.9,+20.3] (n 63).
Buying the strike at the book OPEN after Kalshi has moved costs ~0.58 per contract. Source q2_*_summary_dev.json.
Matters: measure the gap vs Kalshi's own 50c rung, not the book open, in the next prereg.

### Kalshi batch candlesticks endpoint
GET /trade-api/v2/markets/candlesticks?market_tickers=a,b,...&start_ts&end_ts&period_interval returns many markets in one
call; per-market GETs ran ~1 req/s with 429 back-offs. Source results/kalshi_openers_xleague/q1_fetch.py.


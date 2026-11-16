# e4_probabilistic -- calibrated probabilistic max-temp forecast as an E4 selector, for capacity (agent e4_probabilistic, started 2026-11-16)

## Question (from lead)
E4 buys NO on a KXHIGH bracket when ALL 5 deterministic models (fcsafe_*, previous_day1 runs, hours <= 18 local, >= 3
available) miss it by > 3F, decided 22:00 station-local D-1, maker at mid. It is depth-capped per bracket, so more
QUALIFYING brackets at the same safety = more capacity. Can a CALIBRATED P(bracket hits) from free archived sources that
exist at decision time select additional brackets (NO when P(hit) small) without raising the loss rate past break-even?
Sibling e4_timing changes the decision TIME; this study keeps 22:00 and changes the FORECAST.

## Do-not-redo (FUND_BOOK line ~310, notes/wx_timing.md)
- "Kalshi temp: forecast-error, ensemble, spread regime, fade longshots..." all died 2025-10 (L:kalshi_temp); those were
  stand-alone YES/NO-vs-fair-value trades on all brackets. This study only ADDS NO picks to E4 at the same price model and
  is scored on the NEW picks alone.

## Data sources -- feasibility (checked 2026-11-16 BEFORE any outcome)
- Open-Meteo ensemble API (GEFS/ECMWF-ENS members): allowed range today is 2026-08-15..2026-12-21 only, and it has no
  previous-runs (lead-fixed) archive for ensemble members -> every dev date is outside the range, and the dates it does
  cover sit inside the E1/X1 sealed windows. **Not usable on dev.** (curl ensemble-api.open-meteo.com ... start_date
  2024-07-01 -> "out of allowed range from 2026-08-15".) GEFS GRIB2 on AWS (noaa-gefs-pds) is possible but needs a GRIB
  pipeline for 31 members x 24 stations x ~530 days; not done here (disclosed; proposal if this study shows promise).
- **NBM text NBS via IEM MOS archive** (mesonet.agron.iastate.edu/cgi-bin/request/mos.py?model=NBS): runtime-stamped
  (leak-safe by construction), runs 01/07/13/19Z, columns txn (max/min) and **xnd (NBM's own std dev of max/min, integer F)**.
  NBM is itself the calibrated blend of GEFS/ECMWF-ENS/etc., so txn+xnd is the free archived probabilistic forecast.
  Covers 2024-03 and 2026-01 for KNYC/KSFO (checked). Day-D max = txn at ftime (D+1) 00Z.
  Run used: **19Z on D-1** (published ~20Z; the 22:00 local decision is >= 02Z D -> >= 5.5 h margin in every tz). The 01Z
  run is NOT used (publication ~02Z can be after a 22:00 EDT decision).
- 5 deterministic models = the frozen fcsafe_* columns already on data/wx_*.parquet (E4's own inputs).

## Pre-registration (written 2026-11-16 before any NBS value was joined to an outcome and before any P&L; script scripts/e4_probabilistic.py)
Universe: DEV only = scripts/wx_timing.universe(): E1 orig-7 2024-03-01..2025-06-30 + X1 17 cities 2026-01-14..2026-02-28;
  result_mismatch dropped; sealed rows never loaded. 16,620 brackets / 2,949 station-days (checked).
Split: CALIBRATION = E1 2024-03-01..2024-10-31. TEST = E1 2024-11-01..2025-06-30 + X1 2026-01-14..02-28.
  Walk-forward: for each test month the model is refit on ALL dev station-days dated before that month (expanding);
  X1 uses the fit on all E1 dev (through 2025-06-30). No test-period temperature enters its own fit.
Predictive model (EMOS, Student-t), per station-day with >= 3 fcsafe models:
  m = mean(fcsafe models), s2 = var(fcsafe models, ddof 0), x = NBS txn, sd = NBS xnd.
  WITH NBS: mu = b0 + b1 m + b2 x;  sigma^2 = exp(c0) + exp(c1) sd^2 + exp(c2) s2;  t with df = 2 + exp(d).
  NO NBS (missing run): mu = b0 + b1 m; sigma^2 = exp(c0) + exp(c2) s2; t df likewise (separate fit).
  Fit = maximum likelihood (scipy L-BFGS) on the training station-days' Kalshi expiration_value (one per event_ticker).
P(hit) with integer settlement: between [f, c]: F(c+.5) - F(f-.5); greater (hit iff y >= f+1): 1 - F(f+.5);
  less (hit iff y <= c-1): F(c-.5). (Bracket semantics = the frozen rule's own boundaries; checked against `result` on
  calibration rows only.)
Calibration diagnostics (reported, no trade outcomes): PIT histogram (randomised for integers) + tail frequencies
  P(PIT < .01/.025/.05), coverage of 90/98% intervals, on calibration (leave-one-month-out) and test.
Tradeable filter = E4's: yes_bid > 0, yes_ask < 1, spread <= 10c, yes_ask > .01; price = 22:00 candle on wx_*.parquet.
RULES (NO when):
  C  (constant): P(hit) <= p* = 0.02. p* = half the E4 dev picks' median maker break-even (median mid on 1,713 E4 dev picks
     = .955 -> b/e 4.5% -> /2 ~ 2%; computed from prices only, no outcomes).
  PA (price-aware): P(hit) <= 0.5 x (1 - cost), cost = taker ask + fee (taker) or q (maker): expected loss at most half
     of break-even.
Execution (= E4 / e4_timing): taker NO at 1 - yes_bid + 0.07 p(1-p) fee; maker M3 q = min(floor_cent(mid NO), ask - 1c),
  fee 0, filled iff any print at no_price < q in (22:00, close] (trade-through, PRIMARY); N10 secondary;
  filled-vs-unfilled loss rate (adverse selection).
TRIALS (4, edgelab family kalshi_weather_temp): e4p_c_taker, e4p_c_maker, e4p_pa_taker, e4p_pa_maker.
PRIMARY statistic = units on the NEW picks only (rule picks NOT selected by the E4 5-model rule at 22:00), TEST period,
  1 unit cost per bet, one-sided date-cluster bootstrap (4,000 draws, seed 0), p = P(units <= 0).
SUCCESS (-> forward-paper capacity candidate): NEW units > 0, p < 0.05/4 = 0.0125, NEW units > 0 in both halves of the
  test (split at median test date), AND NEW loss-rate CP95 upper bound < break-even OR maker adverse-selection check
  shows filled loss rate not > 2x unfilled.  KILL: NEW units <= 0 or p >= 0.0125 -> dead for that rule/execution.
  The edgelab family bar (~8e-4) is reported; a pass at the study bar = forward paper only.
Reported (not trials): picks/night (all, NEW), E4 picks the prob rule would DROP (not used: additive framing), loss
  rate + Clopper-Pearson 95% vs break-even, ROI, by cohort, capacity: NEW contracts printed through q after 22:00
  (maker) x price = $/night at tape depth.

## Log
### 2026-11-16 edgelab registration (before any NBS-outcome join and before any P&L)
- Registered (family kalshi_weather_temp, now 67 trials, bar 7.46e-4): e4p_c_maker-0c018912, e4p_c_taker-5c3c7dd1,
  e4p_pa_maker-610c14b9, e4p_pa_taker-5552e75c. Specs edgelab/specs/e4p_*.yaml; rule edgelab/rules/kalshi_wx_prob.py
  (selection by phit inside the rule); loader edgelab/loaders.py kalshi_temp_prob + vault.yaml kalshi_temp_prob
  (dev = TEST window 2024-11-01..2025-06-30, E1 cohort; X1 test rows scored only in scripts/e4_probabilistic.py).
- NBS pull: scripts/e4_probabilistic.py nbs -> results/e4_probabilistic/nbs_19z.parquet (log nbs.log). 0 FAIL;
  every test station-day (1,985) has an NBS 19Z txn/xnd.
### 2026-11-16 calibration (scripts/e4_probabilistic.py calib -> results/e4_probabilistic/{calib.json,fits.json,phit.parquet}) -- temperature only, no prices
- Skill: NBS txn MAE 1.85F vs mean of fcsafe models 2.30F (2,942 station-days with NBS, calib.json "mae").
- PIT (randomised, integer settlement). CAL (LOMO, 964 station-days): tails P(PIT<.01)=.0104, >.99=.0104; cover90 .895,
  cover98 .979 -> well calibrated. TEST (walk-forward, 1,985): low tail .0166 at .01 / .0353 at .025 (cold side
  under-dispersed, winter test vs warm-season early fits), high tail .0111 / .0186; cover98 .972.
- CAL bracket reliability (P(hit) bin: n, mean p, hit rate): [0,.005) 277 .0024 .0036; [.005,.01) 175 .0073 .0229;
  [.01,.02) 280 .0148 .0179; [.02,.05) 625 .0338 .0368; [.05,.1) 637 .0733 .0785 -> reliable at the tail.
- Pick counts on TEST tradeable brackets (selection only, before outcomes): 9,885 tradeable, E4 1,491.
  C: 1,147 picks, 544 NEW (E1 418, X1 126). PA taker: 1,340 / 843 NEW. PA maker: 2,074 / 1,349 NEW.
  E4 picks' own P(hit): median .029, p90 .118 -> the 5-model rule is NOT a "low model probability" rule; it picks many
  brackets this forecast thinks are live, and the prob rule finds as many brackets it thinks are safe that E4 skips.
- Tapes for test picks: scripts/e4_probabilistic.py tapes (cached E4 maker tapes + e4_timing tapes reused, rest pulled).
### 2026-11-16 TEST scoring (scripts/e4_probabilistic.py dev -> results/e4_probabilistic/{results.json,test_frame.parquet,edgelab_new.parquet})
TEST = 288 dates (E1 2024-11..2025-06 + X1 2026-01-14..02-28), 9,885 tradeable brackets. Tapes: trades.parquet 421,566
prints on 2,928 tickers (1,487 cached from kalshi_temp_maker/e4_timing, 1,449 pulled read-only; 8 tickers had no print).
Test reliability of P(hit) (bin: n, mean p, realised): [0,.005) 346 .0025 .0058; [.005,.01) 321 .0074 .0125;
  [.01,.02) 480 .0148 .0083; [.02,.05) 1125 .0339 .0293; [.05,.1) 1350 .0739 .0652 -> calibrated out of sample overall.
| rule | n | losses (rate, CP95) | b/e | ROI | units [95% date-cluster] | p | h1/h2 | /night |
|---|---|---|---|---|---|---|---|---|
| E4 ref taker | 1491 | 52 (3.49%, 2.62-4.55) | 5.15% | +1.96% | +29.25 [+14.2,+43.5] | .00025 | +14.0/+15.3 | 5.18 |
| E4 ref maker | 1057 fills/1491 | 52 (4.92%); unfilled 0/434 | 8.12% | +3.79% | +40.09 [+24.9,+55.0] | <.00025 | +19.8/+20.3 | 3.67 |
| **C NEW taker** | 544 | 5 (0.92%, 0.30-2.13) | 2.50% | +1.69% | **+9.17 [+3.78,+13.80]** | **.00075** | +4.01/+5.15 | 1.89 |
| **C NEW maker** | 331 fills/544 (61%) | 5 (1.51%, 0.49-3.49); unfilled 0/213 | 4.73% | +3.50% | **+11.58 [+6.01,+16.49]** | **<.00025** | +4.92/+6.66 | 1.15 |
| PA NEW taker | 843 | 160 (18.98%, 16.4-21.8) | 19.72% | -0.34% | -2.83 [-30.2,+24.6] | .576 | +2.28/-5.11 | 2.93 |
| PA NEW maker | 1088 fills | 210 (19.30%); unfilled 0/261 | 22.39% | +3.23% | +35.16 [+0.72,+68.7] | .0235 | +28.1/+7.0 | 3.78 |
By cohort (C NEW): taker E1 418 bets 4 L +5.48u, X1 126 bets 1 L +3.68u; maker E1 244 / 4 L +7.66u, X1 87 / 1 L +3.92u.
VERDICT vs the pre-registered study rule (bar .0125, both halves, CP upper < b/e):
- **C taker PASS, C maker PASS** (CP95 upper 2.13% < 2.50% b/e; 3.49% < 4.73%). Model expected 5.7 / 3.7 losses, saw 5 / 5.
- **PA taker DEAD** (p .58). **PA maker DEAD** (p .0235 >= .0125; h2 +7.0).
edgelab (E1 cohort only; family kalshi_weather_temp bar 7.35e-4 at evaluation): e4p_c_taker n418 ROI +1.31% p .0108;
  e4p_c_maker n244 ROI +3.14% p .0014 -> both DEAD on reason a_p_below_bar only (pass b halves, c concentration);
  e4p_pa_taker DEAD (a, b, c); e4p_pa_maker DEAD (a, b). As pre-registered: a study pass = forward-paper candidate only.
Row check (C NEW): between 266 / less 254 / greater 24; every city present; NO ask median .98 (p10 .95), maker q median .97;
  the extreme fcsafe model misses these brackets by only 0-2.7F (median 1.5F). These are the brackets E4's 3F margin excludes,
  where NBM txn (MAE 1.85F) + its xnd put the tail below 2%. All 5 losses were FILLED (adverse selection as in E4: 0/213 unfilled lost).
PA failure mechanism: PA buys cheap (~80c) NO where model P(hit) ~7% but realised 19%. The model is mis-calibrated
  exactly where it disagrees with the market (selection on disagreement). C only buys where both put the bracket near 1%.
Capacity: C NEW adds 1.89 taker picks/night (+36% vs E4's 5.18) and 1.15 maker fills/night (+31% vs 3.67).
  Through-q tape depth per fill: median 317 ct (p25 88) vs E4 456; median-capped $/night at q: C NEW $231 vs E4 $988
  (+23% tape capacity). At the live 6 ct/order cap: 1.15 x 6 x .97 = ~$6.7/night more staked, ~+$0.23/night EV at +3.5%.
Not done / caveats: GEFS/ECMWF-ENS members not used (no leak-safe free archive for dev); 288 test dates and 5 losses -> wide
  loss-rate CI; X1 test fitted on E1 only; no critic has reviewed this yet (lead: run /finding with a critic before acting).
Proposal (needs lead/user): forward-paper C (maker) alongside E4 in kalshi_paper.py using the live NBM NBS 19Z bulletin
  (txn/xnd) + the frozen EMOS fit (fits.json, last test month); no orders.

## Learnings
### NBM NBS txn/xnd is a free, runtime-stamped, leak-safe probabilistic max-temp forecast for 2024+ dev
IEM MOS archive model=NBS, 19Z run on D-1, day-D max = txn at ftime D+1 00Z; MAE 1.85F vs 2.30F for the mean of E4's 5
fcsafe models (results/e4_probabilistic/calib.json). Open-Meteo ensembles only go back ~3 months (to 2026-08-15) and have no
previous-runs archive, so they cannot be used on dev.
### A t-EMOS on NBM + model spread is calibrated in the 1% tail and adds ~31-36% E4 picks at the same safety
C rule (P(hit) <= 2%) NEW picks on test: taker 544, 5 losses (0.92%, CP 0.30-2.13% vs b/e 2.50%), +9.17u [+3.78,+13.80] p .00075;
maker 331 fills +11.58u [+6.0,+16.5]. Fails the edgelab family bar on E1-only (p .011/.0014 vs 7.35e-4) -> forward paper only.
### A price-aware "model prob < half break-even" rule selects on disagreement with the market and dies
The PA rule bought ~80c NO where model P(hit) averaged 7% but the realised rate was 19% (n 843, -0.34% taker). Where the model
disagrees most with the price, the market is right. Use a forecast probability as a filter only where it AGREES the bracket is remote.

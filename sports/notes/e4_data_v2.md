# e4_data_v2 -- improve E4 with BETTER DATA (headroom, CLI data-generating process, long point-in-time forecasts) (agent e4_data_v2, started 2026-11-21 UTC)

Research agent for the fund lead. AGENT_PROTOCOL: own notes, inbox rows as I go, NO git, NO orders, NO executor/.env/launchd
changes. Sealed stays sealed (E1 sealed 2025-07-01..2026-10-06, X1 sealed 2026-03-01+): no sealed row is loaded by anything here.
Scratch: results/e4_data_v2/. Script: scripts/e4_data_v2.py (sub-commands per step).

## Question (from lead)
E4 = Kalshi KXHIGH NO on brackets every one of 5 deterministic models misses by > 3F (fcsafe_*, >= 3 models), decided 22:00
station-local D-1, maker at M3 mid. Improve it with better DATA:
1. Headroom (C22): oracle bounds -- perfect forecast; best multi-model forecast; how much is capped by depth / our caps.
2. DGP of the NWS CLI daily max (ASOS sampling, F/C rounding, LST window, CLI vs METAR, late corrections) + per-station
   forecast-vs-CLI bias; is it already in the rule's inputs?
3. Longer point-in-time training set (IEM MOS/NBM archives, NOAA NBM, Open-Meteo -- verify issued, not reanalysis) + a
   station-level calibrated distribution, time-split only.
4. Pre-registered rule variant on dev only; delta vs frozen E4, paired date-clustered bootstrap + placebo; edgelab.

## Do-not-redo (read first)
- e4_probabilistic (11-16): pooled t-EMOS on NBS 19Z txn/xnd + 5-model mean/var; rule C (P(hit) <= .02, NEW picks) passed the
  study bar on TEST (2024-11..2025-06 + X1), failed the edgelab family bar -> forward paper E4P-C/CA (live since 11-17).
  PA (price-aware) died (selection on disagreement with the market).
- e4_timing: other decision times dead. wx_lockin: intraday lock-in dead (market ahead of hourly METAR). wx_timing: earlier D-1 dead.
- e4_live_audit: price floor .90 -> no rule. kalshi_temp L:2025-10: stand-alone fair-value trades on all brackets dead.
- My variants must differ from e4p C: the new ingredient is DATA (station-level bias + long pre-dev training archive), not a new selector.

## Plan / data sources (feasibility checked before outcomes)
- Dev universe = scripts/wx_timing.universe(): E1 orig-7 2024-03-01..2025-06-30 + X1 17 cities 2026-01-14..02-28, tradeable
  22:00 candle quote; result_mismatch dropped. E4 reference = WT.rejected(u, 22, 3.0) (reproduces 1,713 taker bets +30.46u).
- IEM MOS archive (mesonet.agron.iastate.edu/cgi-bin/request/mos.py): runtime-stamped (issued) MOS. Probe 11-21: GFS MAV
  exists 2018+ (older likely), NBS from at least 2021-07 (none 2018), NBE 2021-07. IEM rate-limits ("Too many requests") ->
  pace >= 5 s/request.
- E1 orig-7 price+outcome rows exist 2021-08..2024-02 (dev quarters) where E4 cannot fire (ECMWF/ICON/GEM previous-runs start
  2024-03). A MOS-based forecast would make that dev period usable (more dev data without touching sealed).

## Pre-registration step 1: HEADROOM (written 2026-11-21 BEFORE any oracle number was computed)
Universe = dev universe above, tradeable at the 22:00 quote. Primary metric = TAKER units (NO at 1 - yes_bid + 0.07p(1-p),
1 unit cost/bet) because taker needs no tape, so every oracle set is scorable; maker M3 trade-through reported where tapes exist.
Rows (one per lever, C22):
 R0 E4 frozen (WT.rejected 22h, margin 3): reference.
 R1 loss-veto oracle: E4 picks minus their losers (a forecast that only removes E4's losers). Headroom = what E4 loses to losers.
 R2 perfect forecast at margin 3: every model := realised CLI max y; = brackets whose nearest edge is > 3F from y.
 R3 perfect forecast, margin 0: every tradeable bracket that settles NO (NO-only ceiling, not reachable).
 R4 skill curve: model_i' = y + s (model_i - y), s in {1, .9, .8, .7, .5, .25, 0}, frozen rule at margin 3 (spread and error
    shrink together). s = .80 ~ the realistic gain already measured (NBS txn MAE 1.85F vs 5-model mean 2.30F,
    results/e4_probabilistic/calib.json). Also margin re-tuned to keep E4's dev tail rate (tail(M) <= tail_E4(3), as e4_timing).
 R5 best-of-available-inputs, IN-SAMPLE (optimistic): t-EMOS (e4p form + station intercepts) fit on ALL dev outcomes; NO when
    P(hit) <= p*, p* in {.005,.01,.02,.03,.05}; report each; the max is an over-fitted upper bound, not a result.
 R6 capacity: E4 dev maker picks with tapes; winners fill min(S, V_lt) (contracts printed at NO < q after T), losers fill
    min(S, V_lt) too (losers trade through by hundreds of contracts); $/night at S = 6, 24, 96, unlimited (tape).
Triggered levers + thresholds (fixed now):
 - BUILD-FORECAST lever: worth building (steps 3-4 as a real effort) iff R4 at s = .80 (fixed margin OR re-tuned margin) adds
   >= +25% taker units over R0, or R5 at p* = .02 adds >= +25% NEW-pick units. Otherwise steps 3-4 run in the cheap form
   (existing archives only) and the verdict says the forecast lever is capped.
 - SIZE lever: if R6 unlimited-tape $/night >= 3x the S = 24 $/night, depth/caps is a larger lever than the forecast.
 - LOSS-VETO lever: R1 - R0 is the most any veto filter can add; quoted, no threshold.
</content>
</invoke>

## Log
### 2026-11-21 step 1 HEADROOM (scripts/e4_data_v2.py headroom -> results/e4_data_v2/headroom.json; dev 529 dates, 10,454+ tradeable brackets)
Reproduction: R0 = 1,713 taker bets, 70 losses, +30.46u [+11.87,+48.06]; maker TT 1,163 fills +38.40u = e4_timing reference exactly.
| row (lever) | bets | losses | taker units [95% date-cluster] | note |
|---|---|---|---|---|
| R0 E4 frozen | 1713 | 70 | +30.46 [+11.87,+48.06] | maker TT 1163 fills +38.40 |
| R1 loss-veto oracle (E4 minus losers) | 1643 | 0 | +100.46 [+90.1,+111.9] | max any veto filter can add: +70u |
| R2 perfect forecast, margin 3 | 4131 | 0 | +391.05 [+359,+424] | 7.8 picks/night vs 3.2 |
| R3 perfect forecast, margin 0 (NO-only ceiling) | 10454 | 0 | +2207.65 | unreachable; check: margin-0 set == result 'no' (0 mismatches) |
| R4 skill s=.9, M3 | 1844 | 57 | +40.66 [+24.1,+56.3] | |
| R4 skill s=.8, M3 (NBM-level MAE gain) | 1939 | 32 | +67.22 [+52.4,+82.0] | +121% vs R0; retuned M2.5: 2548 bets 61 L +98.70 |
| R4 skill s=.5, M3 | 2571 | 1 | +139.30 | |
| R5 in-sample EMOS p*=.02 (optimistic) | 1414 | 14 | +21.78 [+12.4,+30.4] | NEW-only 695 bets 8 L +10.31 [+3.4,+16.3]; union w/ E4 2408 +40.78 (+34%) |
| R5 p*=.05 | 2805 | 55 | +52.80 | NEW 1638 38 L +29.25; union 3351 +59.72 |
Maker numbers for oracle sets are under-covered (tapes exist for only 29-44% of R2/R3 brackets) -> taker is the comparison.
R6 capacity (E4 dev maker picks, winners AND losers fill min(S, contracts printed NO < q after T)): $/night S=1 0.06, S=6 0.35,
S=24 1.22 (staked $44.7/night), S=96 3.58 (staked $159/night); "unlimited" = -$2,232/night (losers trade through by ~10^3-10^4
contracts, so at unbounded size losers dominate; not a capacity number). S 24->96 multiplies $/night by 2.9x.
Per-station headroom (perfect-forecast M3 minus E4, taker units) ranks KAUS +67.6, KNYC +62.9, KMDW +52.0, KDEN +42.1, KPHL +33.8 (all orig-7).
Lever verdicts vs pre-reg thresholds:
- BUILD-FORECAST: TRIGGERED (R4 s=.8 +121% fixed margin, R5 NEW +34% >= 25%). Caveat: uniform shrinkage also removes errors the
  models SHARE; R5 (a real calibrated model, in-sample) adds only +34% (union) -> realistic headroom is far below R4.
- SIZE: NOT triggered as pre-registered (unlimited row negative); S=96 gives 2.9x S=24 $/night (just under 3x), a size lever of
  similar order to the realistic forecast lever. Live orig7 is at 24 ct; wx_capacity found the same concavity.
- LOSS-VETO: +70u (taker) is the ceiling for any filter on E4's existing picks.

### 2026-11-21 E4 loser anatomy (descriptive, dev outcomes of E4 picks -> any rule it motivates is POST HOC, disclosed)
66 of 70 dev E4 losers are HOT misses (CLI max above EVERY model), 4 cold. Miss beyond the extreme model: median 4.25F
(3.1-7.9). Losers by month: Dec 15, Feb 15, Jan 5, Nov 5, Mar 7, Apr 7, May 6 -> cold-season heavy (40/70 Nov-Feb).
All five models are biased COLD vs the CLI max on dev (y - model, 2,949 station-days >= 3 models): GFS +1.43F, ECMWF +1.74,
ICON +1.39, GEM +0.66, NBM(OM) +1.52; 5-model mean +1.35F (MAE 2.30). By station x season the mean-model bias ranges -1.80
(KLAX JJA) to +3.20 (KATL DJF). The frozen rule uses raw models with a SYMMETRIC 3F margin -> the hot-side margin is
effectively ~1.6F and the cold side ~4.4F. Mechanism to test in step 2: CLI max = sub-hourly ASOS peak (hourly model values
and hourly METARs both miss it) + daytime 2m cold bias + hours<=18 window missing late/midnight maxima.

### 2026-11-21 step 2 DGP of the CLI daily max (scripts/e4_data_v2.py dgp -> results/e4_data_v2/{dgp.json,dgp_days.parquet}; dev METARs
from data/wx_lockin_obs.parquet, 5,365 station-days 2021-08..2026-02 dev; truth = Kalshi expiration_value)
- **CLI max = the ASOS 24-hour max group.** The METAR issued at midnight LST carries `4snTxTxTx...` (24 h max/min in tenths C).
  round_half_up(C*9/5+32) of that group equals the CLI/Kalshi value on 99.46% of 5,324 station-days (round-half-down: 98.70%,
  so the conversion is half-up). Mismatches: 14 at +1, a handful of large negatives (bad groups). So the settlement value is
  the ASOS continuous-sensor max over the LST day (00:00-24:00 local STANDARD time), converted from 0.1 C to whole F.
- **Hourly values under-state it.** CLI minus max of all hourly/special METARs (T-group, exact): mean +0.71F; 0F 2,221 days,
  +1F 2,578, +2F 487, +3..+6F 75. Adding the 6-hourly max groups (1snTTT) closes it: CLI == max(6h groups, obs) on 97.4%.
  Forecast models output hourly instantaneous 2m T, so a max of hourly model values targets the hourly max, not the CLI max.
- **Window.** CLI max falls after 18:30 local clock on 3.2% of days (Dec 10.4%, Jan 5.9%, Jul-Sep <= 0.6%); in the first LST
  hour (midnight carry-over) on 5.7%. The fcsafe window is hours 0..18 LOCAL CLOCK (in DST hour 0 = 23:00 LST of the previous
  climate day; hours 19-23 dropped). CLI minus obs max over local 00:00-18:30: mean +0.76F, >= 3F on 2.35% of days.
- **E4 hot losers explained** (66 of 70 dev losers; joined to METARs 70/70): only 6 had the max after 18:30 local, 3 in the first
  LST hour. 34 of 66 lost ONLY because of the sub-hourly peak: the hourly METAR max never reached the bracket, the ASOS continuous
  max did (CLI - hourly max on losers: 0F 20, +1F 34, +2F 9, +3..5F 3). On the losers the in-window hourly obs max was itself a
  median 3.35F above the warmest model (models cold that day). Many losers are "CLI just reached the bracket floor" (y == floor).
- **Model bias vs CLI (dev, 2,949 station-days >= 3 models):** GFS +1.43F, ECMWF +1.74, ICON +1.39, GEM +0.66, Open-Meteo NBM
  hourly +1.52; 5-model mean +1.35F. NBM NBS txn (the NBM max-T element, 19Z D-1, IEM) bias +0.13F (stations -0.04..+0.36),
  MAE 1.86F (n 2,547). Same NBM model: hourly 2m T max is 1.4F colder than its max-T element -> the max-T element (calibrated on
  observed max groups) has the CLI DGP built in; raw hourly model values do not.
- **Raw-model bias is not stable across years.** GFS-OM previous_day1 warm-season bias at KNYC: 2021 -3.00F, 2022 -1.81, 2023 -1.77,
  2024 +0.63, 2025 +0.65; KAUS warm 2023 -1.75 -> 2025 +2.07 (gfs_seamless composition/version changes). A bias table fitted on
  past years would mis-correct; MOS/NBM max-T (re-trained by NWS) is the stable way to get the DGP into the inputs.
- **Is it in the rule's inputs?** No. E4 = raw hourly-model maxima, symmetric 3F margin, no bias term. e4p EMOS has a pooled
  intercept (b0) and NBS txn, so it partly absorbs it (pooled, not by station).
- **Late corrections / truth source:** ACIS daily maxt (final climate record) vs Kalshi expiration_value on dev: 5,351 / 5,367
  equal (0.30% differ; e.g. KNYC 2025-06-02 Kalshi 66 vs ACIS 71 = the `$`-flagged day; HIGHCHI 2021-10-01 84 vs 86). Kalshi
  settles on the as-issued CLI; the record is later corrected on ~0.3% of days. No result_mismatch events in the dev universe.
- **Open-Meteo historical-forecast API is NOT a D-1 forecast** (results/e4_data_v2/openmeteo_histfc_check.json, KNYC JJA 2024
  n 92): GFS MAE 1.32F vs previous_day1 2.37F (errors uncorrelated, r .03); ECMWF 1.43 vs 1.93. It is stitched from the first
  hours of each run (near-analysis); using it as a forecast is lookahead. Only previous-runs (from 2021-03 GFS / 2024-03 others)
  or runtime-stamped MOS/NBS archives are point-in-time.
- Disclosure: in a truth-source check I filtered with a too-wide window and printed 5 orig-7 KAUS 2026-02 expiration values
  (E1 sealed calendar). Temperature only, no prices, no rule; nothing downstream uses them.

## Pre-registration steps 3-4 (written 2026-11-21 ~05:40Z, BEFORE any MOS value was joined to an outcome, before any MOS-based
## selection, and before any P&L; only MOS row counts have been seen. Disclosed: the E4 loser anatomy above (dev outcomes of E4
## picks) motivated T1, so T1 is post hoc in motivation; its rule below contains no tuned parameter.)
### Data (step 3)
Members (runtime-stamped IEM MOS archive, point-in-time by construction; pulled by scripts/e4_data_v2_pull.py):
  NBS = NBM text 19Z run of D-1, txn at ftime D+1 00Z (lead 29 h) + xnd; MAV = GFS MOS 12Z run of D-1, n_x at ftime D+1 00Z
  (lead 36 h); MET = NAM MOS 12Z run of D-1, n_x at ftime D+1 00Z. Typical issuance NBS ~20Z, MAV/MET ~16-17Z vs the earliest
  decision 22:00 EDT = 02Z D -> >= 6 h margin (issuance checked on a sample of IEM text products if reachable; else disclosed).
  12Z (not 18Z) MOS is used to keep a wide margin. Calendar: <= 2025-06-30 for all stations (+ X1 dev 2026-01-14..02-28 for
  scoring only). Nothing from 2025-07-01..2026-01-13 or after 2026-02-28 is requested or used.
Truth for fitting: ACIS maxt (official record; == Kalshi on 99.70% of dev station-days). Truth for scoring: Kalshi `result`.
### Model (step 3): station EMOS-L, Student-t, walk-forward
  mu = a_s + h_s(doy) + b1 txn + b2 mav + b3 met;  sigma^2 = exp(c0_s + c1 sin + c2 cos) + exp(k1) xnd^2 + exp(k2) var(members);
  df = 2 + exp(d). a_s, h_s (first harmonic, 2 coef) and c0_s per station; slopes/k/d pooled. Tiers by available members:
  A = NBS+MAV(+MET if present), B = MAV+MET (no NBS). MET missing in A -> its term is 0 and it leaves var(members) (fit with an
  indicator-free imputation: met := mav, disclosed). Fit = max likelihood (L-BFGS) on station-days strictly before the scoring
  month (expanding from 2018-01; tier A used only once >= 12 months of NBS training exist). X1 dev months use the fit through
  2025-06-30. Diagnostics BEFORE any price: PIT tails, 90/98% coverage, MAE vs NBS alone and vs the e4p pooled EMOS on the e4p
  TEST rows (same rows).
### Windows
  W1 = E4 dev = E1 orig-7 2024-03-01..2025-06-30 + X1 2026-01-14..02-28 (paired comparison with the frozen E4 rule).
  W2 = E1 orig-7 2021-08-06..2024-02-29 (dev quarters; E4 cannot fire there: < 3 Open-Meteo models). Fresh for this family.
### Trials (3, edgelab family kalshi_weather_temp; study bar p < 0.05/3 = 0.0167; PRIMARY execution = maker M3 trade-through,
### E4's live execution; taker reported, not a trial)
T1 e4m_mosvote (PRIMARY): frozen E4 rule with 3 extra voters. Members = 5 fcsafe models + {NBS txn, MAV n_x, MET n_x} where present;
  availability rule unchanged (>= 3 of the 5 fcsafe); reject iff EVERY available member misses the bracket by > 3F (same formula,
  same margin, same tradeable filter, same 22:00 quote). Picks are a subset of E4's -> a veto. Window W1.
  Statistic: paired per-date delta = units(T1) - units(E4) (maker TT; unfilled = 0), one-sided date-cluster bootstrap (4000, seed 0).
  PASS iff delta > 0, p < .0167, delta > 0 in both halves of W1 (median date), AND both placebos below the actual delta:
  (i) random veto: 2000 draws vetoing the same number of E4 picks per NO-ask bucket (<.90, .90-.95, .95-.97, .97-.98, >.98) ->
  actual delta > 95th pct; (ii) wrong-day voters: same rule with the MOS members of D-7 (same station, issued earlier, same
  season, wrong weather) -> its delta must be below the actual delta. KILL otherwise.
T2 e4m_mosconsensus: E4-style rule with long-archive members only: {NBS, MAV, MET}; requires NBS and MAV present; reject iff every
  present member misses by > M_wf, M_wf = smallest 0.5F step with tail(M) <= 0.0709 (E4's own 22:00 tail rate,
  results/e4_timing/margin.json) on all training station-days before the scoring month (ACIS truth, all 24 stations, expanding).
  PRIMARY window W2 (fresh): units > 0, p < .0167, both halves > 0, wrong-day placebo (D-7 members) units < actual. W1: paired delta
  vs E4 reported (T2 standalone units on W1 also must be > 0 for "holds").
T3 e4m_cl_new: additive NEW picks: NO on tradeable brackets NOT picked by E4 with EMOS-L P(hit) <= 0.02 (e4p's p*). W1 additive delta
  = units of the NEW picks; W2 = all picks with P_L(hit) <= .02 (no E4 there). PASS iff W1 units > 0 with p < .0167 AND W2 units > 0
  with p < .05 AND random-selection placebo (same count per date per NO-ask bucket from tradeable non-E4 brackets, 2000 draws):
  actual > 95th pct.
Maker tapes: pulled read-only (kx_throttle) for every pick lacking one; coverage must be >= 97% per trial-window, else taker is the
  primary for that window (decided now). Losers that lack a tape count as FILLED (losers always fill on dev), winners as unfilled.
Reported, not trials: taker versions, filled-vs-unfilled loss rates, picks/night, X1 vs E1 split, edgelab family-bar verdicts.
A study PASS = forward-paper / lead decision only (family bar ~7e-4 not expected on W1 alone). Sealed: not touched; any sealed look
  needs the lead + user.
### 2026-11-21 edgelab registration (after the prereg above, before any frame, MOS-outcome join or P&L)
Registered (family kalshi_weather_temp -> 76 trials, bar 6.58e-4): e4m_mosvote_maker-6618b84d (T1), e4m_mosconsensus_maker-92d2f343
(T2), e4m_cl_new_maker-bfb71381 (T3). Rule edgelab/rules/kalshi_wx_mos.py (selection inside the rule); specs
edgelab/specs/e4m_*_maker.yaml; loader edgelab/loaders.py kalshi_temp_mos; vault.yaml dataset kalshi_temp_mos (dev 2021-07-01..
2025-06-30 E1 orig-7; X1 W1 rows scored only in scripts/e4_data_v2.py).

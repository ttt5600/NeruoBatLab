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
  round_half_up(C*9/5+32) of that group equals the CLI/Kalshi value on 99.46% of 5,324 station-days. [critic correction: the
  earlier 'round-half-down 98.70% -> half-up' line was a denominator bug (index-aligned NaNs); no group converts to an exact .5F,
  so the rounding direction is NOT identified by these data.] Mismatches: 14 at +1, a handful of large negatives (bad groups). So the settlement value is
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
- **Bias decomposition** (results/e4_data_v2/bias_decomp.csv, bias_model_station.csv; 2,948 dev station-days with METARs):
  CLI - 5-model mean (+1.35F) = sampling premium [CLI - hourly obs max over the same local 00:00-18:30 window] +0.76F
  + model 2m bias at hourly resolution [hourly obs max - model mean] +0.59F. The sampling premium is stable (station x season
  0.24..1.17F, mostly 0.6-0.9F): it is the DGP (continuous sensor max vs hourly values). The model part is what varies: KLAX
  JJA -2.60F / MAM -2.50F (marine layer; ECMWF at KLAX -5.67F), KATL DJF +2.44F, KMDW DJF +1.94F, KNYC JJA -1.54F vs MAM +1.57F.

### 2026-11-21 lead input (winners_transfer: 66/70 losers above forecast; asymmetric spread?) -- DESCRIPTIVE SECONDARY ONLY
(prereg above was frozen before this message; T1 = MOS voters is the pre-registered asymmetric-information test; anything
below is descriptive, no trial.) results/e4_data_v2/side_asym.json, gap_vs_sky.json.
- E4 dev picks by side of the forecast: taker all picks ABOVE 1,418 / 66 losses (4.65%, mean NO ask .933) vs BELOW 295 / 4 (1.36%,
  ask .977), Fisher p .0058. Maker fills: above 1,049 / 66 (6.29%) vs below 114 / 4 (3.51%), p .30 (= winners_transfer's numbers;
  the below side is mostly 98c NO that rarely fills, which is why the filled comparison has no power).
- The warm tail IS heavier, by the DGP: P(CLI > warmest model + M) vs P(CLI < coldest model - M) on 2,949 dev station-days:
  M=1 23.7% vs 2.6%; M=2 12.7% vs 1.5%; **M=3 6.0% vs 1.09%**; M=4 2.8% vs 0.58%; M=5 0.92% vs 0.37%. Standardised error
  (CLI - model mean)/model spread: mean +1.11, P(z > 3) 12.5% vs P(z < -3) 1.15%. Cause = sampling premium (+0.76F) + model
  daytime cold bias (+0.59F), see bias decomposition. The symmetric 3F margin is ~1.6F effective on the warm side.
- The market partly prices it: above-side NO asks are cheaper (.933 vs .977); mean taker break-even 6.32% above vs 2.12% below;
  taker ROI above +2.00% (+28.30u) vs below +0.73% (+2.16u): the above side carries E4's profit despite carrying 66/70 losses.
- Sensor-in-sun check: days with CLI - hourly max >= 3F (n 75) had FEWER clear afternoons (12-17 LST obs CLR/FEW 18.5%) than days
  with 0F gap (31.2%, n 2,225); wind the same (~8.7 kt). Large sub-hourly gaps are not mainly sunny-sensor spikes on dev; they are
  transitional/cloudy days (brief warm peaks between hourly obs). n is small; descriptive.

### 2026-11-21 step 3 data built + EMOS-L (temperature only; no prices)
- Pulls (scripts/e4_data_v2_pull.py; read-only; IEM rate-limits ~1 req/5-10 s with 2-3 parallel pullers):
  results/e4_data_v2/mos_NBS.parquet (NBM text, 24 stations, 2020-11..2025-06 + X1 2026-01/02; runtimes 01/07/13/19Z),
  mos_GFS.parquet (MAV, 2018-01..2025-06, runs 00/06/12/18Z), mos_NAM.parquet (MET, 2018-01..2025-06, runs 00/12Z) -- max-T rows
  only; acis_maxt.parquet (24 stations 2015-01..2025-06 + X1 2026-01/02). Leads at the D+1 00Z max: NBS 19Z = 29 h, MAV/MET
  12Z = 36 h (checked from runtime/ftime). The station-day table = scripts/e4_data_v2.py station_days().
- Station-day coverage (orig-7, 2018-01..2025-06): MAV 99.8%, MET 99.9%, NBS 63% (NBS starts 2020-11), ACIS 100%.
- EMOS-L walk-forward, orig-7 (results/e4_data_v2/emos_orig7_partial.parquet, emos_fits_orig7.json; per-station monthly refits,
  ~22 min): W1 tier A 3,390 station-days: MAE(mu) 1.83F vs NBS txn 1.87, MAV 2.30, MET 2.73; member bias NBS +0.11, MAV +0.14,
  MET +1.26F (MOS max-T is near-unbiased for the CLI max, unlike raw models' +1.35F). PIT tails: cold P(PIT<.01) .0156,
  warm P(PIT>.99) .0050; .025 tails .0357 / .0174; cover98 .979. W2 tier A 6,368: MAE 1.86 (NBS 1.92, MAV 2.57, MET 2.81),
  tails .0110 / .0097, .0297 / .0210, cover98 .979. W2 tier B (2021-07..09, before 330 NBS training days) 450 days.
  -> the long station model removes the warm-tail problem; on W1 its COLD tail is too heavy (1.6% at the 1% level), the same
  sign e4p saw on its TEST (cold tail .0166). MAE gain over NBS alone is small (1.83 vs 1.87): the gain is calibration, not skill.

### 2026-11-21 frame (scripts/e4_data_v2.py frame -> frame_all.parquet, edgelab_frame.parquet) -- selections only, no P&L yet
Picks: E4 W1 1,713 (E1 1,269 + X1 444 = reproduction). T1 keeps 694 of 1,713 (vetoes 1,019 = 59%; E1 471, X1 223 kept).
T2 W2 163 (E1), W1 1,643. T3 NEW W1 821 (E1 680, X1 141), W2 167. Wrong-day placebo: T1_pl7 keeps 687; T2_pl7 selects 6,782 (W1)
/ 1,925 (W2) brackets. T2 margin m_wf by month: 4F in 2021-07..10, 2022-02..10, 2023-02..06, else 3F (with integer members and
integer truth 3.0 == 3.5, so the 0.5F grid only takes 3 or 4); member coverage W1 nbs 99.8% / mav 99.7% / met 100%, W2 99.8/100/100.
DEVIATION (decided now, before any P&L): the T2 wrong-day placebo would need 8,134 more tapes, so it is scored TAKER only (a
placebo only has to be worse than the rule; taker needs no tape). All other maker scoring as pre-registered.
Live-ops guard: tape pulls (kx_throttle, read-only) are paused 03-06:30Z (E4 22:00 decisions ET..PT); the live executor does not
use kx_throttle, so heavy pulls during its window could cost it rate limit.
- EMOS-L vs e4p pooled EMOS on the SAME e4p TEST rows (results/e4_data_v2/compare_e4p.json; 1,978 station-days, temperature only):
  PIT cold tail .0147 vs .0167, warm .0071 vs .0111, cover98 .978 vs .972; MAE(mu) 1.92 vs NBS txn 1.97; bracket Brier .11480 vs
  .11440 (e4p marginally better); tail reliability (p bin: n, mean p, realised) EMOS-L [0,.005) 677 .0021 .0074, [.005,.01) 396
  .0074 .0126, [.01,.02) 702 .0147 .0057 vs e4p 751 .0022 .0027, 481 .0074 .0083, 693 .0147 .0101. Verdict: the long, station-level
  MOS model is NOT a sharper probability than e4p's (which adds the 5 raw models); equal within noise in the <=2% region.

### 2026-11-21 TAKER results (descriptive; not the pre-registered primary, which is maker) -- results/e4_data_v2/score_taker_prelim.json
- T1 MOS voters: vetoes 1,019 of 1,713 E4 picks, including 60 of the 70 losers, yet delta = **-21.99u [-38.77, -4.68]** (p(delta<=0)
  .9935): the 959 vetoed winners earned more than the 60 avoided losers cost. Wrong-day voters (D-7 MOS) delta -25.81u
  [-38.72,-11.42] -> the real-day MOS veto is barely better than a wrong-day one. T1 kept set: 694 bets, 10 losses, +8.47u.
- T2 MOS consensus: W2 163 bets, 10 losses, +1.25u [-5.43,+7.11] p .336; W1 1,643 bets, 50 losses, +6.08u (ROI +0.37% vs E4 +1.78%).
  Wrong-day placebo W2 1,925 bets -84.95u (placebo behaves).
- T3 EMOS-L NEW: W1 821 bets, 14 losses, +12.10u [+3.78,+19.61] p .0015; W2 167 bets, 5 losses, +3.40u [-1.66,+7.47] p .080.

### Descriptive secondary for the lead's question: asymmetric margins on E4's own members (W1, taker; IN-SAMPLE 16-cell grid on
### already-used dev data -> forking paths; a lead, not a result). results/e4_data_v2/asym_margin_grid_taker.json
| warm margin Mw \ cold margin Mc | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 2 | 3384 b, 197 L, +61.59 | 2862, 187, +46.24 | 2568, 183, +40.25 | 2403, 183, +36.64 |
| 3 | 2529, 84, **+51.80** [+31.3,+71.7] | 2007, 74, +36.45 | **1713, 70, +30.46 (E4)** | 1548, 70, +26.85 |
| 4 | 1935, 41, +39.91 | 1413, 31, +24.56 | 1119, 27, +18.57 | 954, 27, +14.96 |
| 5 | 1525, 23, +32.16 | 1003, 13, +16.81 | 709, 9, +10.82 | 544, 9, +7.21 |
Reading: widening the WARM margin (what the DGP warm tail suggests) cuts losses 70 -> 27 -> 9 but cuts units 30.5 -> 18.6 -> 10.8:
the above-forecast NOs are priced for the warm tail (ask .933) and still pay. Narrowing the COLD margin (which the DGP supports:
P(CLI < coldest model - 1F) is only 2.6%) adds 816 bets / 14 losses / +21.3u at Mw 3, Mc 1. So the DGP asymmetry is real but the
market has priced the warm side; the unexploited part, if any, is the light cold tail. Needs a clean test (forward paper or a
fresh window) before anyone trusts it.

### 2026-11-21 why better forecasts do not convert (descriptive; results/e4_data_v2/market_calibration.json)
- Market YES mid vs realised hit rate, W1 tradeable brackets: mid (0,.02] 500 b: .0173 vs .0040; (.02,.05] 2,292: .0352 vs .0240;
  (.05,.10] 1,826: .0740 vs .0498; (.10,.20] 2,440: .1492 vs .1230; >= .20 calibrated (.2797/.2721, .4131/.4257). W2 same shape
  in (.02,.10]. -> the low-price YES side is overpriced by ~1-3c (longshot bias): this is what E4 harvests.
- On E4's picks: mean YES mid .0761, EMOS-L P(hit) .0954, realised .0409. On T1-vetoed picks: mid .0995, EMOS-L .1356, realised
  .0589; T1-kept: .0418 / .0365 / .0144; T3 NEW: .0498 / .0100 / .0171. The station MOS model is calibrated overall (PIT) but,
  conditional on "all 5 raw models far from the bracket", it OVER-states the hit rate 2.3x: the raw-model consensus carries
  information the MOS members lack, so MOS voters veto good picks (T1). And on its own NEW picks (T3) it UNDER-states (1.0% vs 1.7%).


### 2026-11-21 critic review (fresh critic subagent; scratch /tmp/e4rev/*.py; verdict ACCEPT-WITH-CHANGES)
Every number reproduced (own parser for the 24h group 5,324/5,324 identical, eq .99455; loser anatomy; bias decomposition exact;
MOS leads 29 h / 36 h; date-shift test MAE vs ACIS lowest at lag 0 (nbs 1.91 vs 4.80/4.84 at +-1 day); D-7 placebo column equals
the D-7 member on all 66,207 rows; T1 subset of E4 with 0 exceptions; frame_all 19,559 rows, 0 duplicate tickers; no sealed rows).
Corrections adopted:
- Rounding direction not identified (see DGP bullet). Every 24h group is issued at 23:5x LST, so it spans 23:5x-23:5x, not exactly
  00-24 LST.
- The bias decomposition is an accounting identity, not a causal split; the "hourly obs" include SPECIs (17% of obs). With routine
  reports only the sampling premium is 0.80F, so ~0.04F of sampling sits in the "model" term, which also absorbs grid-vs-station
  representativeness.
- Headroom mechanism reworded: the market does NOT price the brackets correctly (it OVER-prices YES on exactly E4's brackets: E4
  picks mid .0761 vs realised .0409). What is true: the MOS information is already in the price (T1-vetoed picks mid .0995 vs kept
  .0418; on E4 picks the market beats EMOS-L on Brier .0369 vs .0423, critic's computation), so the remaining edge is a longshot
  premium, not forecast error. The uniform-shrink oracle (R4) also overstates because it shrinks errors the models SHARE toward the
  truth, and s = .8 came from an MAE ratio (1.85/2.30) that is mostly bias removal (raw +1.35F vs NBS +0.13F), and the warm-side
  bias is already priced (above-side asks .933 vs .977).
- T1 extra (critic): real-day minus wrong-day veto = +3.82u [-7.74,+15.63], not distinguishable; a random veto with the same counts
  per NO-ask bucket is expected to give -23.25u -> the MOS veto carries about no information beyond the NO price. 294 of the 330 E4
  picks with ask < .90 are vetoed.
- Walk-forward fits/margins use truth through the day before the monthly cut, so the first scoring day of a month includes a D-1
  truth that is not final at decision time (negligible; disclosed).
Disclosures added:
- The IEM CLI pull requests whole years (year=2025 for orig-7, 2026 for X1), so sealed-period CLI values were fetched into memory
  and dropped before saving (filter in cmd_cli); nothing sealed was saved, printed or used. The notes' "nothing is requested" was
  too strong for that one pull.
- Before the step 3-4 prereg I had already seen NBS txn vs CLI bias/MAE (from e4p's existing nbs_19z archive, step 2), not only MOS
  row counts.
- Provenance: the descriptive numbers (taker prelim, market calibration, side asymmetry, asymmetric grid, bias decomposition, loser
  anatomy, Open-Meteo check) were ad-hoc snippets; they are being moved into `scripts/e4_data_v2.py descriptive` so each output has a
  generating command.
- **As-issued CLI vs Kalshi vs final record** (IEM CLI archive, orig-7 2021-2024 only; results/e4_data_v2/cli_sources.json,
  iem_cli.parquet; first pull attempt crashed on an 'M' value and was re-run for 2021-2024 only so no whole-year request touches
  the sealed calendar): 3,708 events; Kalshi == IEM CLI 99.89%, Kalshi == ACIS 99.87%, IEM CLI == ACIS 99.97%. 5 disagreements,
  e.g. HIGHCHI-23JAN04 Kalshi 46 vs corrected CLI (CCA issued 16:24Z next day) 45 -> Kalshi settles on the first-issued CLI, not on
  later corrections; HIGHCHI-21OCT01 84 vs 86; KXHIGHMIA-24NOV27 84 vs 81 (later product). Settlement risk from corrections ~0.1%.
  CLI-reported time of max at or after 18:00 LST on 10.8% of 2021-2024 days (Dec 14.5%, Jan 15.0%, Jun-Aug ~5%; KNYC 14.7%, KAUS 2.8%),
  before 01:00 LST 3.3%. (Different definition from the METAR-based 3.2% "after 18:30 local clock" on dev days; in DST the fcsafe
  window hour 18 local is 17:00 LST, so it can end an hour before an LST-18:00 max.)

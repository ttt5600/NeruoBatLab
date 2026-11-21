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

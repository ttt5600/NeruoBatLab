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

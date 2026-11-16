# tropical_nhc -- Kalshi tropical/hurricane markets vs official NHC products (agent tropical_nhc, started 2026-11-16)

Question: does the E4 pattern (official forecaster vs market) work on Kalshi tropical markets -- i.e. buying NO at the
ask + taker fee on outcomes the latest NHC product puts out of reach -- earn a positive return?
Related, not redone: LX2 (notes/lx2_tail_no_carry.md; climate/hurricane long-dated NO +3.0%, 36 ev, 0 losses, post hoc;
its sealed window = settled 2025-10-01..2026-09-12, i.e. the 2025 season); phys_ladders.md (KXHURCAT 95 storms listed,
"counterparty reads the same NHC forecast").

## Sealed / excluded (fixed before any price or outcome was read)
- 2025 season (every tropical market with open_time in 2025): EXCLUDED entirely -- LX2's sealed climate lead. Never priced
  here.
- 2026 season (open_time in 2026): this agent's own SEALED holdout. Metadata only; not priced, not scored without the
  user's yes.
- DEV = 2021-2024 seasons.

## Census (2026-11-16, metadata only; scripts/tropical_nhc.py census -> results/tropical_nhc/markets.parquet,
## results column split off to results_SEPARATE.parquet and not read before the prereg)
68 Climate series match hurricane/tropical/landfall (results/tropical_nhc/series.json); 1,055 markets (historical +
live endpoints). Volume = sum volume_fp (contracts):
| series | what | markets / events | volume | seasons | settlement source |
|---|---|---|---|---|---|
| KXHURCAT (HURCAT) | per-storm "reaches >= 74/96/111/130/157 mph" (T1..T5) | 564 / 118 storms | 3.80M | 2021-26 | NHC (advisory max sustained wind) |
| KXFIRSTHURRICANE | name of first hurricane (basin) | 77 / 3 | 1.08M | 2026 | NWS/NHC |
| KXHURCTOTMAJ / KXHURCTOT | season count of major / all hurricanes, ladder | 43/5, 37/5 | 0.97M / 0.96M | 2022-26 | NHC |
| KXTROPSTORM | season named-storm count ladder | 32 / 4 | 0.62M | 2023-26 | NHC |
| KXNEXTHURDATE | "a hurricane forms in the Atlantic before <date>" | 14 / 2 | 0.45M | 2025-26 | NHC |
| KXHURRICANENAMES | name list reached | 58 / 3 | 0.40M | 2026 | NHC |
| KXHURTB/MIA/ORL/NO/NYC/NJ... | city wind >= category in season (NWS PSH/CLI) | 5 rungs x 1/season | 0.3M .. <1k each | 2022-24 | NWS |
| KXHURPATH*/KXHURCITY*/KXHURRLAND* | state/city landfall, season | 1-6 each | 5k-250k | 2025-27 | AccuWeather/AP or NHC |
| KXNEXTCAT5HURDATE, KXNAMEDSTORM, KXHURRICANE (basin counts) | | 7-35 | 23k-106k | 2025-26 | NHC/NWS |
KXHURCAT by open year: 2021 6 storms / 2022 7 / 2023 19 / 2024 22 / 2025 20 / 2026 44. DEV storms = 54.
Only KXHURCAT has a per-storm unit (many independent events) AND a direct NHC forecast input (the advisory's 5-day
intensity forecast). Season counts / city-season landfall have 1-5 events per season and the NHC products (2/7-day
genesis, 5-day wind probabilities) cover days, not the season -> at most a handful of events per year; a real test of
those is impossible in dev (one event per season). Wind-speed-probability landfall rules would need city/state markets
with a window of days; Kalshi's are season-long (KXHURCITYTPA etc. are 2026+, sealed/forward only).

## NHC input
ATCF A-decks, ftp.nhc.noaa.gov/atcf/archive/<YYYY>/a{al,ep,cp}<NN><YYYY>.dat.gz, rows with tech OFCL = the official
forecast at synoptic time t0 (tau 0..120 h, Vmax kt). The public advisory for t0 is issued at t0 + 3 h (03/09/15/21Z).
Storm name -> ATCF id via ftp.nhc.noaa.gov/atcf/archive/storm.table.
Kalshi thresholds mph -> kt (Saffir-Simpson): T1 74 mph = 64 kt, T2 96 = 83 kt, T3 111 = 96 kt, T4 130 = 113 kt,
T5 157 = 137 kt (NHC advisory winds are kt in 5-kt steps; 65 kt = 75 mph).

## PRE-REGISTRATION (written 2026-11-16, before any Kalshi candle, NHC intensity or Kalshi result was loaded)
Universe: KXHURCAT/HURCAT markets T1..T5 (one-off letter thresholds dropped), DEV storms (event first open_time
2021-01-01..2024-12-31). Storm matched to ATCF by name + year (+ basin if ambiguous); unmatched storms dropped and counted.

GATE (data validity, before any P&L): Kalshi result vs "max OFCL tau-0 Vmax over all advisories >= H_kt". Require
agreement on >= 95% of DEV markets, else no evaluation (settlement source is not what the rule reads).

HC1 (hc1_hurcat_nhc_gap): for each OFCL advisory t0 of the storm, V = max OFCL Vmax over tau 0..120 h; gap = H_kt - V.
  Input available at r = t0 + 3 h. Quote = the first hourly candle (period 60) with end_period_ts in [r + 1 h, r + 3 h]
  (strictly after the advisory release); need yes_bid >= 0.01 and yes_ask > yes_bid. If gap >= 20 kt -> buy 1 NO at the
  NO ask = 1 - yes_bid, taker, Kalshi quadratic fee 0.07 x mult x p(1-p) exact (1-ct ceil-to-cent as sensitivity).
  First qualifying advisory per ticker only; hold to settlement. Cluster = storm.
HC2 (hc2_hurcat_post_advisory): after the storm's LAST OFCL advisory t_last, at r = t_last + 27 h (>= 24 h with no new
  advisory, as known at r only if no OFCL t0 in (t_last, r - 3h]), for thresholds with H_kt > max OFCL tau-0 Vmax so far:
  quote = first hourly candle ending in [r + 1 h, r + 3 h], same quote filter -> NO at ask + fee, hold.
  NB lookahead check: "last advisory" is only known after the fact; the rule as run uses "no advisory for 24 h at r",
  which is observable at r. A storm that regenerates later is a loss the rule eats.
SUCCESS (each trial): storm-cluster bootstrap (10,000) ROI CI lower > 0 (exact fee) AND >= 30 bets AND >= 15 storms AND
  no storm > 30% of total P&L (if total > 0) AND edgelab evaluate passes the family bar. Report exact Clopper-Pearson 95%
  CI on the loss rate vs break-even loss rate (= 1 - mean cost).
KILL: any success condition fails -> trial dead in dev. Sensitivities (gap 10 / 30 kt) are reported, not trials.
Registered in edgelab family kalshi_data_ladders (same family as the KXTORNADO / SpaceX physical ladders; no new family,
so the multiple-testing bar is not reset).

AMENDMENT (2026-11-16, before any frame/gap/result was computed; only 2 candle series were looked at for format,
HURCAT-NICOLE-T3 and HURCAT-IAN-T5): Kalshi /historical candlesticks emit an hourly candle only when the book or trades
change (Nicole T3: 15 candles in 2 weeks), so "first candle ending in a window" would drop most decisions. Quote is now
the standing book at decision time d: last hourly candle with end_period_ts <= d and >= d - 7 d. HC1 d = t0 + 5 h
(advisory released t0 + 3 h); HC2 d = r + 2 h. edgelab specs amended (--amend) before touching data.

## Log
- 2026-11-16: census done (above). NHC storm.table has 2021-2025 entries (40/36/42/34/33 storms).
- 2026-11-16: edgelab registered BEFORE data: hc1_hurcat_nhc_gap-e947ea67, hc2_hurcat_post_advisory-299e3ecd (family
  kalshi_data_ladders, 52 trials all-time -> bar p < 9.6e-4); amended once for the carry-forward quote (above).
  Dataset kalshi_hurcat_nhc (vault.yaml: dev 2021-24, 2025 in no split, sealed 2026, forward 2027+), loader in
  edgelab/loaders.py, rule edgelab/rules/hurcat_nhc.py, specs edgelab/specs/hc{1,2}_*.yaml.
- DEV universe: 54 KXHURCAT events (236 T1-T5 markets). Kalshi pre-lists the whole name list (2024: Alberto..William
  listed in June) -> storms that never formed (no ATCF id) have no NHC input and are outside HC1/HC2 by construction.
- GATE PASSED: Kalshi result vs (max OFCL tau<=3 Vmax >= H_kt) agree 236/236 = 100% (results/tropical_nhc/gate.csv;
  `scripts/tropical_nhc.py gate`); 66 yes / 170 no (never-formed storms count as pred no). Kalshi settles on the
  operational NHC advisory intensity, which is what the rule reads.
- Build (`scripts/tropical_nhc.py build` -> data/tropical_nhc/frame.parquet, candles.parquet): 47 storm keys in DEV
  with T1-T5 rungs (2021 used letter rungs I5/N1.. -> mostly dropped), 42 matched to ATCF, 5 never formed (TONY-2024,
  VALERIE-2024, VINCE-2023, WHITNEY-2023, WILLIAM-2024). 206 tickers; 3,727 HC1 advisory rows, 147 HC2 rows.
- Rule-module edits after registration (disclosed): decided_at = d (pre-data) and close_time ISO8601 parsing (after the
  first script scoring crashed on a fractional-second timestamp; no selection logic changed). Trials re-amended so
  edgelab's rule hash matches; edgelab had never loaded the dataset before.

## RESULT (DEV 2021-24) -- HC1 and HC2 both FAIL -> dead
Source: `scripts/tropical_nhc.py dev` -> results/tropical_nhc/dev_summary.json, bets_dev.csv; edgelab evaluate
reproduces HC1 exactly (independent path): hc1 n106 -4.48% [-13.79, +3.21] p .86 dead; hc2 n15 +1.32% [+0.94, +1.71]
dead (n < 30, concentration, no CLV).
| trial | n / storms | losses (CP 95%) | breakeven loss | mean NO px | ROI exact [storm CI] | ROI 1-ct |
|---|---|---|---|---|---|---|
| HC1 gap>=20 kt (prereg) | 106 / 36 | 18 = 17% [10.4, 25.5] | 13.1% | 86.6c | -4.48% [-13.8, +3.3] | -5.3% |
| HC2 post-advisory | 15 / 9 | 0 [0, 21.8] | 1.3% | 98.6c | +1.32% [+0.94, +1.72] | +0.41% [0.0, +0.8] |
| sens HC1 gap>=10 | 120 / 38 | 27 | 15.5% | 84.1c | -8.3% [-18.3, +0.5] | |
| sens HC1 gap>=30 | 86 / 34 | 11 | 12.7% | 87.0c | -0.05% [-7.0, +5.1] | |
Per season HC1: 2022 8 bets +0.18 $/ct, 2023 38 -1.12, 2024 60 -3.18. Traded volume in the 206 dev tickers' hourly
candles: 85k (2022) / 71k (2023) / 202k (2024) contracts -- whole-market, so capacity at the edge is a few hundred $/season.
WRONG TURN in the prereg rule: HC1 did not require "threshold not yet reached". 8 of the 18 losses are 1c NO buys on
rungs the storm had already hit (Kirk T4, Helene T4, Ernesto T2...: yes_bid 99c, a weakening storm's forecast max is
below a rung it already passed). Post hoc fix (not a trial; results/tropical_nhc/posthoc_notreached.json): gap>=20 and
not reached: n98 / 35 storms, 10 losses vs breakeven 6.1%, -4.4% [-13.7, +3.4]; gap>=30: n79, 4 losses, +0.06%
[-6.9, +5.3]. Still nothing. The real losses are rapid intensification the OFCL forecast missed by 20-47 kt: Beryl T4/T5
(06-28-2024, gap 23/47, NO at 97c/99c), Milton T5 (gap 42, NO 91c), Idalia T3/T4 (gap 31/48, NO 67c/83c), Isaac T2,
Leslie T2, Franklin T3/T4. The market priced those YES at 10-50c while NHC's deterministic peak said 20-48 kt short --
the market was right. The NHC intensity forecast is a central estimate, not a bound; its RI tail is fat.
Sealed 2026 holdout (44 storms) NOT opened; with dev dead there is no reason to ask for it. 2025 never touched.

## Forward logging (proposal only; nothing deployed)
Not worth a strategy logger. If the lead wants the cheap option for June 2027: a read-only launchd job at t0+5h
(03/09/15/21Z + 2h) that saves every open KXHURCAT book (kx_throttle) plus the NHC real-time OFCL
(ftp.nhc.noaa.gov/atcf/aid_public/a<bb><nn><yyyy>.dat.gz) and the 5-day wind-speed probabilities (nhc.noaa.gov GIS
wsp_120hr). That would allow (a) a YES-side test (does the market UNDER-price RI when SHIPS/RI-index guidance is high?)
and (b) state/city landfall markets with day-scale windows if Kalshi lists them (KXHURCITYTPA-type, 2026+ only).
Season-count ladders (KXHURCTOT, KXTROPSTORM, KXHURCTOTMAJ) have 1 event/season -> never testable at any useful power.

## Learnings
### NHC deterministic intensity forecast is not a bound -- Kalshi prices the RI tail correctly
HC1 (NO when every OFCL tau 0-120 h Vmax is >= 20 kt below the rung) lost 18/106 (17%) vs 13% breakeven, ROI -4.5%
[-13.8, +3.3] over 36 storms 2022-24 (results/tropical_nhc/dev_summary.json). Beryl, Milton, Idalia beat the forecast
peak by 23-48 kt. An official central forecast is not an E4-style input for tail rungs.
### Kalshi hurricane-category markets settle on the operational NHC advisory intensity
236/236 dev KXHURCAT rungs match "max OFCL tau-0 Vmax >= threshold kt" (results/tropical_nhc/gate.csv).
Thresholds: 74/96/111/130/157 mph = 64/83/96/113/137 kt.
### Kalshi historical candles are emitted only on change -- use the standing quote
HURCAT-NICOLE-T3 has 15 hourly candles in 2 weeks. "First candle in a window" drops most decisions; take the last
candle ending <= decision time (cap its age).
### A NO rule on per-storm rungs must exclude rungs already hit
A weakening storm's forecast max sits below a rung it already reached; 8 of HC1's 18 losses were 1c NO buys at YES 99c.
### Kalshi pre-lists the whole storm-name list
2024 KXHURCAT listed Alberto..William in June; 5 dev names never formed. Those are a climatology bet, not an NHC one.
### Post-dissipation hurricane NO is ~1c and tiny
HC2 n15 / 9 storms, 0 losses, +1.3% exact but +0.4% with 1-ct fee rounding; dev 2022-24 gives ~5 bets a season.

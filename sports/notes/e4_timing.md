# e4_timing -- E4 consensus-reject at OTHER decision times, for capacity (agent e4_timing, started 2026-11-16)

## Question (from lead)
E4 (Kalshi daily-high consensus-reject NO, all 5 models miss by > 3F, decided 22:00 station-local on D-1, maker at mid)
is depth-capped (notes/wx_capacity.md). Do OTHER decision times add capacity on the same markets with the same prior?
(a) D-2 22:00; (b) D-1 10:00 local; (c) D 06:00 local. Per time: picks/night, loss rate vs break-even (exact binomial),
ROI taker (ask+fee) and maker (trade-through fill + adverse selection), overlap with the 22:00 picks, fills/month.

## What already exists (do-not-redo)
- notes/wx_timing.md (10-27): D-1 11/14/18:00 at margin 3/5 as a SUBSTITUTE for 22:00 -> 0/6 taker pass (all deltas
  vs 22:00 negative); standalone T11 m3 taker +1.64%/1,373, loss 7.6% vs b/e 8.8%. 6 maker trials NOT SCORED (no tapes).
- N1 (scripts/wx_dayof.py, notes/results.md 10-21): D 06:00 with 00Z single runs, 2026-04..07 only (single runs start
  2026-04): taker D1 +0.19%/1,279 p .34; D2 new brackets -1.31%/364. Dead as a taker. That window is inside the E1/X1
  sealed calendar -> NOT reused here (brief: dev only).
- This study differs: ADDITIVE framing (standalone P&L of the extra orders, plus overlap/crowding with the 22:00 order),
  margin set from forecast-error climatology at the lead, M3 maker with tapes + adverse-selection check.

## Feasibility finding for (a), checked BEFORE any price/outcome (2026-11-16)
KXHIGH* markets list at 10:00 ET on D-1. data/wx_*.parquet open_time vs the event date (one row per event,
2021-2026): median 10.0 h before 00:00 UTC of D in every year (2021 n 266 ... 2026 n 1,874); open_time hour
14:00Z (3,124) / 15:00Z (1,732) = 10:00 EDT/EST. **At D-2 22:00 local the market does not exist** -> no order can be
placed; (a) has 0 picks by construction. Not a trial. (The earliest possible order is the listing itself = (b).)

## Pre-registration (written 2026-11-16 before any new price, tape or P&L was computed; script scripts/e4_timing.py)
Universe (DEV only, = wx_timing / kalshi_temp_maker dev universe): E1 orig7 data/wx_<CITY>.parquet 2024-03-01..2025-06-30
  (inside kalshi_temp dev 2021Q3..2025Q2); X1 data/wx_expand.parquet kind high, 2026-01-14..2026-02-28 (< X1 sealed
  2026-03-01). result_mismatch dropped. Sealed rows dropped on load.
Times:
  T10  = 10:00 station-local on D-1. Forecast per model: hour h of D uses previous_day1 if h <= 10 - 4 = 6, else
         previous_day2 (wx_timing.fc_at, T=10); daily = max over hours 0..18, >= 15 hours, >= 3 models.
         (ET cities: book opens exactly 10:00 ET -> usually no candle ending <= T -> no pick. Accepted, reported.)
  T06D = 06:00 station-local on D. Leak-free forecast available on 2024-25 dev = the same previous_day1 archive as 22:00
         (the archive's freshest hours for the peak were issued on D-1); fresher 00Z runs exist only from 2026-04
         (sealed calendar). So T06D = the frozen fcsafe forecast at margin 3, priced at 06:00 D. Morning obs NOT used
         (disclosed limitation). The question it answers: are the 22:00 rejects still worth buying 8 h later, once
         the market has seen the overnight runs (adverse) -- i.e. size added on the same brackets.
  T22  = reference (frozen E4: fcsafe, margin 3).
Margin (fixed from ERROR CLIMATOLOGY before any price/P&L): per station-day with >= 3 models, e_up = actual - max(models),
  e_dn = min(models) - actual; tail(M) = P(e_up > M) + P(e_dn > M) over the dev station-days. Target = tail_22(3).
  M_T = smallest M on a 0.5F grid with tail_T(M) <= target. T06D uses the 22:00 forecast -> M = 3 by construction.
  (Uses temperature outcomes on dev days, never trade outcomes; disclosed.)
Price at T: last hourly candle ending <= T (wx._row); tradeable: yes_bid > 0, yes_ask < 1, spread <= 10c, yes_ask > 0.01.
Execution:
  taker : NO at 1 - yes_bid, fee 0.07 p (1-p) unrounded.
  maker : M3 q = min(floor_cent(1 - mid), NO ask - 1c), fee 0, rest (T, close].
          PRIMARY fill = TRADE-THROUGH: any print at no_price < q (level cleared; queue irrelevant).
          Secondary = E4 N10: >= 10 contracts at no_price <= q.
          Adverse selection: loss rate filled vs unfilled (unfilled outcome still observed).
Stake 1 unit of cost per bet; pnl = 1/cost - 1 or -1.
TRIALS (4, edgelab family kalshi_weather_temp): e4t_t10_taker, e4t_t10_maker, e4t_t06d_taker, e4t_t06d_maker
  (maker = trade-through primary). Primary statistic = the trial's own (standalone, additive) units over pooled dev,
  one-sided date-cluster bootstrap (4,000 draws, seed 0), p = P(units <= 0).
SUCCESS (-> capacity candidate for a forward-paper ask): units > 0, p < 0.05/4 = 0.0125, units > 0 in both halves
  (split at median dev date), AND the NEW-capacity subset (T10: tickers not picked at 22:00; T06D: filled where the 22:00
  maker order had already filled or was never placed -> no self-crowding) has units > 0.
KILL: units <= 0, or p >= 0.0125 -> dead at that time. The edgelab family bar (~8e-4) is reported; near-certain contracts
  are not expected to clear it on this window (notes/bar_calibration_v2.md) -- a pass at the study bar = forward paper only.
Reported (not trials): picks/night, losses with exact Clopper-Pearson 95% CI vs break-even, ROI, fills/month, overlap
  with 22:00 picks (same ticker), crowding (our own 22:00 order still resting at T or our T10 order still resting at 22:00),
  per cohort.

## Log
### 2026-11-16 forecasts + margin (scripts/e4_timing.py forecasts; results/e4_timing/{fc.parquet,margin.json}) -- no prices yet
- Reproduction: fc22 == frozen fcsafe_* on 16,620 rows (NBM 7,379), 0 diffs > 0.05F.
- Error climatology, 2,949 dev station-days with >= 3 models: tail(M) = P(actual > max model + M) + P(actual < min - M).
  22:00 forecast: tail(3) = .0709 (target). 10:00 forecast (peak hours lead-2): tail 3.0 .099, 3.5 .0732, **4.0 .0512**.
  -> **M10 = 4.0F** (first 0.5F step at or below target), FIXED before any price was pulled. M06D = 3 (same forecast).
### 2026-11-16 edgelab registration (before any P&L)
- Registered (family kalshi_weather_temp, now 63 trials, bar 7.94e-4): e4t_t10_taker-f7f7d227, e4t_t10_maker-f6561b9c,
  e4t_t06d_taker-b5ed15d1, e4t_t06d_maker-376896ee. Specs edgelab/specs/e4t_*.yaml; rule edgelab/rules/kalshi_wx_timing.py
  (maker_tt = trade-through); loader edgelab/loaders.py kalshi_temp_t10 + vault.yaml kalshi_temp_t10 (dev = E1 dev window;
  E1 cohort only -- X1 dev dates sit inside the E1 sealed calendar, so X1 is scored only in scripts/e4_timing.py).
- T06D has no vault dataset: edgelab PIT check requires decided_at < event_start (local midnight D); a 06:00-D decision
  is in-play by that definition. Scored in scripts/e4_timing.py only; lead may add an in-play exception if wanted.
- Candidate tickers for prices: 3,013 (rejected at T10 M4 or at 22:00 M3). Pull via kx_throttle (read-only).

### 2026-11-16 RESULTS (scripts/e4_timing.py dev; results/e4_timing/results.json, picks_T{22,10,06D}.parquet)
Pull: quotes T10 1,817 / T22 3,013 / T06D 3,013 tickers; tapes 254,998 trades on 2,245 tickers (results/e4_timing/prices.log).
**Reproduction:** T22 taker 1,713 bets +30.46u and T22 M3_N10 1,239 fills +39.91u = the E4 dev study exactly.
Pooled E1+X1 dev, 529 dates, halves split 2024-11-24, date-cluster bootstrap 4,000; losses with Clopper-Pearson 95%;
/mo = per 17.5 cohort-months (E1 16 + X1 1.5).

| time | exec | n | losses (rate, CP95) | b/e | ROI | units [95% CI] | p | h1 / h2 | /mo | verdict (bar .0125) |
|---|---|---|---|---|---|---|---|---|---|---|
| 22 ref | taker | 1713 | 70 (4.1%, 3.2-5.1) | 5.6% | +1.78% | +30.5 [+11.9,+48.1] | .0003 | +2.0/+28.5 | 98 | reference |
| 22 ref | maker TT | 1163 (fill 68%) | 70 (6.0%, 4.7-7.5) | 8.8% | +3.30% | +38.4 [+19.8,+56.5] | <.001 | +0.6/+37.9 | 67 | reference |
| D-1 10:00 (M4) | taker | 536 | 31 (5.8%, 4.0-8.1) | 7.6% | +2.08% | +11.1 [-2.0,+23.3] | .047 | +3.0/+8.2 | 31 | FAIL |
| D-1 10:00 | maker TT | 405 (fill 76%) | 31 (7.7%, 5.3-10.7) | 10.8% | +3.65% | +14.8 [+2.0,+27.1] | .0125 | +1.9/+12.9 | 23 | FAIL (p not < bar; new < 0) |
| D-1 10:00 NEW brackets | taker | 206 | 21 (10.2%, 6.4-15.2) | 9.1% | -0.81% | -1.7 [-11.1,+7.3] | .64 | +3.5/-5.1 | 12 | no new capacity |
| D-1 10:00 NEW, uncrowded | maker TT | 153 | 21 (13.7%, 8.7-20.2) | 13.0% | -0.23% | -0.4 [-9.7,+8.8] | .53 | +2.0/-2.4 | 9 | no new capacity |
| D 06:00 (same fc, M3) | taker | 1123 | 33 (2.9%, 2.0-4.1) | 3.9% | +1.09% | +12.3 [+0.1,+23.7] | .024 | -2.1/+14.4 | 64 | FAIL |
| D 06:00 | maker TT | 644 (fill 57%) | 33 (5.1%, 3.6-7.1) | 7.6% | +2.82% | +18.2 [+5.4,+30.0] | .003 | -1.2/+19.4 | 37 | FAIL (half 1 < 0) |
| D 06:00 add-on (22:00 order filled / not placed) | maker TT | 404 | 26 (6.4%, 4.2-9.3) | 8.7% | +2.76% | +11.2 [-0.5,+22.4] | .031 | -1.1/+12.3 | 23 | fail |
Secondary N10 fill model: T10 421 fills +15.5u p .010; T06D 711 +21.0u p <.001 (h1 -1.3) -- not the primary.

Adverse selection: unfilled losers = 0 at every time (T10 0/131, T06D 0/479, T22 0/550) -- every loser fills, as in E4.
Overlap with 22:00 picks: T10 330/536 same ticker (122 with our T10 bid still resting at 22:00 = self-crowding);
T06D 1,012/1,123 same ticker by construction, 507 crowded (22:00 order still resting at 06:00).
T10: no ET city has a quote at 10:00 local (book opens 10:00 ET) -> only CT/MT/PT cities trade at T10.
edgelab (E1 cohort only; family 67 trials, bar 7.5e-4): e4t_t10_taker 315 bets +0.94% [-2.83,+4.71] p .31 DEAD;
e4t_t10_maker 235 fills +2.23% [-2.85,+7.55] p .18 DEAD. e4t_t06d_* retired DEAD on the prereg kill.
E1-only T10 is much weaker than pooled -> X1's 6 dev weeks carry the pooled T10 number.

### Verdict
- (a) D-2 22:00: IMPOSSIBLE -- the markets list at 10:00 ET on D-1. 0 picks.
- (b) D-1 10:00: FAIL. The profit comes from brackets the 22:00 rule also picks (the same rejects bought earlier).
  The genuinely NEW brackets lose at about break-even: 10.2% vs 9.1% taker, 13.7% vs 13.0% maker.
- (c) D 06:00 (stale fcsafe; no 00Z runs, no obs on 2024-25 dev): FAIL on the halves rule. Extra maker size on brackets
  whose 22:00 order already filled: +11.2u [-0.5,+22.4] over 17.5 cohort-months, ~23 fills/month, same risk as the
  22:00 position (correlated), so it adds size, not diversified capacity. Forward-only lead at most; not a trial pass.
- Capacity answer: no other decision time adds independent capacity on dev. Lever stays size at 22:00 + more series.

## Learnings
### Kalshi daily-high markets cannot be traded on D-2
Every KXHIGH event lists at 10:00 ET on D-1 (data/wx_*.parquet open_time, 7,750 events 2021-26, median 10.0 h before
00Z of D). A "decide two nights early" temperature idea is infeasible on Kalshi; earliest order = the listing.
### At 10:00 local D-1, only the brackets the 22:00 rule also rejects make money
Rejects at 10:00 (margin 4F) that the 22:00 forecast no longer rejects lose 10.2% vs 9.1% break-even (206 bets, -1.7u;
results/e4_timing/results.json). Earlier is cheaper only on the same brackets; new early rejects are priced correctly.
### Set margins across decision times by matching the forecast-error tail rate
tail(M) = P(actual beyond the extreme model by > M): 22:00 tail(3) .071; 10:00 tail(3) .099, tail(4) .051
(results/e4_timing/margin.json, 2,949 station-days). Fixing M from weather errors, not P&L, avoids tuning on trades.
### Adding size at 06:00 D on the same E4 brackets is positive but not robust
Maker trade-through at 06:00 D: 644 fills +18.2u p .003, first half -1.2u; add-on to already-filled 22:00 brackets
+11.2u [-0.5,+22.4]. Correlated with the 22:00 position: more size, not new capacity.
### Pooled E1+X1 timing results are carried by X1's 6 dev weeks
T10 maker pooled +14.8u p .0125 vs E1-only (edgelab) +2.23% p .18. Check the cohort split before reading a pooled gain.

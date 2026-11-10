# raind_daily_rain -- daily rain NO when the forecasts rule rain out (2026-11-09, subagent of quant-trader lead)

## Question
Kalshi daily rain markets ("precip > 0 in at station X on day D"): when every NWS MOS guidance model available at
the decision says measurable rain is very unlikely, is NO underpriced at real executable prices (taker at ask + fee;
maker with a conservative tape fill)? Physical outcome, no informed counterparty -> the E1/E4/L39 pattern.

## What exists already (read before starting)
- **R1** (scripts/kalshi_rain.py): KXRAIN multi-city, Open-Meteo 5-model lead-2 all < 0.01 in -> NO taker. Dev
  (07-15..09-14) +3.42%/361; **sealed FAIL** (09-15..10-18) +0.49%/274 p .28. Dead. Both its windows are seen data.
- **KXRAIND\* per-city daily series: NO HISTORY.** Census today (results/raind_daily_rain/census.py): every KXRAIND*
  series has 0 settled markets except KXRAINDNYC (3 settled, $54k lifetime) and KXRAINORL (1). The 11-06 lead's "~60
  days/city" is wrong -- these are newly listed (open markets dated through Dec 31). Nothing to backtest there.
- Untested daily-rain history that DOES exist: **KXRAINNYC** (legacy NYC daily rain, 869 events 2021-09-09..2026-07-15,
  $30.5M lifetime) and **KXRAINSEA** (339 events 2021-08..2022-08, $0.5M). After 07-16 NYC rain lives in KXRAIN
  (KXRAIN-<date>-NYC). So this test = NYC 2021-26 (+ SEA 2021-22), a different series and forecast source from R1.

## Settlement (rules text read 2026-11-09, results/raind_daily_rain/probe_rules.py)
- KXRAINNYC: "precipitation recorded at Central Park ... strictly greater than 0" (NWS CLI NYC). **Trace (T) resolves
  YES.** So a dry MOS forecast (P06 = chance of >= 0.01 in) does NOT rule out a YES: trace days lose. Physical mismatch
  the market can price; the test decides whether the remaining edge survives it.
- KXRAINSEA (tickers RAINSEA-*): NWS Daily Climate Report WFO Seattle (CLI SEA); T counts as > 0 -> YES on the T0 strike.
  Only the T0 strike is used (some days also list 0.08 etc.).
- KXRAIN (NYC row, 07-16+): CLINYC via The Weather Company; **trace and missing count as 0 -> NO**. Rule change at 07-16.
- Fee type quadratic (taker 0.07 p(1-p); maker 0).

## PRE-REGISTRATION (written before any price, forecast or outcome was pulled)
Dataset `kalshi_rain_daily_nyc` (edgelab family kalshi_weather_precip; bar = 0.05 / family trials incl. these; was
p < 6.25e-3 with 8 trials -> 0.05/10 = **5.0e-3** after registering the 2 below).
Rows: KXRAINNYC (T0 / greater 0), KXRAINSEA T0 tickers, KXRAIN *-NYC tickers. date = event date D.
Split by date: sealed = the most recent 30% of unique dates (cutoff computed from the market list only, recorded below
before any pull); dev = the rest. Sealed is NOT scored here.
- **Decision T** = 22:00 station-local on D-1 (NYC America/New_York, SEA America/Los_Angeles).
- **Forecast** = NWS MOS from the IEM archive, 18Z run of D-1 (UTC), models GFS (MAV) and NAM (MET). 18Z MOS is
  issued ~21-22Z; T is >= 02Z D -- >= 4 h margin; inputs_as_of = runtime + 4 h (conservative issue time) <= T.
  Signal = max P06 over the five 6-h periods ending 06Z, 12Z, 18Z on D and 00Z, 06Z on D+1 (covers 00Z D..06Z D+1,
  a superset of the LST climate day 05Z..05Z NYC / 08Z..08Z SEA is NOT fully covered for SEA -> SEA uses periods
  ending 12Z,18Z D, 00Z,06Z,12Z D+1 = 06Z D..12Z D+1, superset of 08Z..08Z). GFS required; NAM used when present.
  **Input true iff every available model's max P06 <= 5 (%).**
- **Quote at T** = last hourly candle ending <= T: require 0.01 <= yes_bid < yes_ask <= 1, spread <= 10c.
- **RD1 taker**: buy NO at NO ask = 1 - yes_bid, require NO ask in [0.50, 0.99]; fee 0.07 p(1-p) exact; stake 1.
- **RD2 maker**: post-only NO bid at q = min(floor_cent(NO mid), NO ask - 1c), q in [0.50, 0.99], never cancelled.
  **Filled iff >= 10 contracts print at no_price STRICTLY < q** ("traded through", stricter than E4's <=) in
  (T, close_time]; fill time = when the 10th such contract prints. Unfilled -> no bet. Maker fee 0. The window runs
  through the event day and the settlement morning, so fills after rain starts are counted (adverse selection kept).
- One bet per ticker. cluster = date (cities on one date share weather). CLV mark = NO ask on last candle ending <=
  local midnight starting D (event_start).
- **Success** = edgelab dev verdict pass (p < bar, both halves > 0, concentration ok, n >= min_n = 100, CLV >= 0).
- **Kill** = dev CI upper < 0, or n < 100 in dev.
- Verdict mapping: pass -> ready-for-sealed-ask; fail but CI crosses 0 with ROI > 0 -> lead (forward log); kill -> dead.

## Sealed cutoff
`python scripts/raind_daily_rain.py markets` (market list only, before any price/forecast pull; outcomes not looked at):
KXRAINNYC T0 769 markets 2021-09-08..2026-07-15; KXRAINSEA T0 176 (2021-08-19..2022-08-12); KXRAIN-NYC 97
(2026-07-17..11-08); 2026-07-15 listed in both -> legacy kept. 926 unique dates.
**SEALED = dates >= 2026-01-16 (278 dates / 278 markets, all NYC). DEV = dates < 2026-01-16 (764 markets).**
Note: the whole KXRAIN-NYC (TWC, trace = NO) era falls in sealed; dev is entirely the trace = YES rule.

## LEAD CONSTRAINT (2026-11-09, received mid-build, before any price/MOS/outcome was looked at)
R1 sealed (lab/lockbox.jsonl line 3, KXRAIN 2026-09-15..10-18) must never be re-scored. Implemented: the frame builder
drops EVERY row dated 2026-07-15..10-18 (R1's dev AND sealed window; conservative) so no split can load them. My dev
(< 2026-01-16) never overlapped R1 (R1 data start 07-15). Sealed after the drop = 2026-01-16..07-14 + 10-19..11-08 (NYC).
Only legitimate new mechanisms: maker execution and series R1 didn't cover. KXRAINNYC 2021-26 / KXRAINSEA 2021-22 were
not in R1 (R1 = KXRAIN only, which starts 07-15). KXRAIND* has no history (census). E5 lesson
(notes/e5_forecastex_audit.md): resting bids fill on losers -> report maker loss rate filled vs unfilled picks.
Reuse: price machinery = sports/wx.py (same as kalshi_rain.py) + _tape from kalshi_temp_maker.py; kalshi_rain.py itself
is KXRAIN/Open-Meteo-specific (Open-Meteo previous-runs has no data before 2024 -> probe_om.py), so MOS from IEM instead.

## Log
- 2026-11-09 edgelab registered BEFORE any data pull: rd1_rain_dry_taker-2a054388, rd2_rain_dry_maker-7a33c03d
  (family kalshi_weather_precip now 10 trials -> bar p < 5.00e-3). Rule edgelab/rules/rain_daily.py, loader
  kalshi_rain_daily_nyc, vault entry, specs edgelab/specs/rd{1,2}_*.yaml.
- MOS (`raind_daily_rain.py mos`, IEM yearly CSVs): GFS 18Z runs NYC 2,137, SEA 728. **IEM keeps NAM MOS only at
  00Z/12Z -> no 18Z NAM; per the frozen rule ("NAM when present") the input is GFS MAV alone.** Not changed to 12Z NAM
  (that would be a post-hoc rule edit).
- Open-Meteo previous-runs has no data before 2024 (probe_om.py: 2021-2023 all null) -> why MOS, not R1's forecasts.
- Frame build drops 77 rows dated 2026-07-15..10-18 (R1 window).
- Frame: 965 rows (after R1 drop), GFS signal 100%, quote at T 91%, input-true 361 (all splits); tapes 26,646 trades on
  325/361 input-true tickers. traps checks (epoch units, inputs<=T<event_start, quote<=T, local-22:00 hours) pass.
- Dev (pre-score look, no outcomes): 764 markets (NYC 588, SEA 176); input-true 291; quote passes 140 (71 had
  yes_bid 0 = NO already $1; 62 spread > 10c). Median NO ask .96, median volume to T 52 contracts.

## RESULT (edgelab dev, 2026-11-09) -> DEAD, both trials
| trial | n | net ROI [95% CI] | p (bar 5.0e-3) | halves | CLV | verdict |
|---|---|---|---|---|---|---|
| rd1_rain_dry_taker-2a054388 | 140 | +0.07% [-4.53%, +4.28%] | .48 | +2.1% / -2.0% | -0.69c (p_neg .0055) | dead |
| rd2_rain_dry_maker-7a33c03d | 69 (< min_n 100 = KILL) | -4.42% [-13.15%, +3.53%] | .85 | -10.2% / +1.2% | +1.06c | dead |
(`python -m edgelab status`; runs lab/edgelab_runs/rd{1,2}_*_dev.parquet; top cluster 2021-12-05.)
- **The market prices it right.** Taker loss rate 9/140 = 6.4% vs break-even ~6.1% at mean NO ask .936. Losses came on
  days the market priced YES 7-16c even though GFS P06 was <= 5% (the market knew; trace also counts as YES under legacy rules).
- **The E5 adverse selection happens again:** of 139 maker picks, 69 filled, and ALL 9 losses are in the filled set (9/69 =
  13.0%) vs 0/70 unfilled; Fisher p .0014. Resting NO bids get hit when rain is coming.
- By city: NYC 6/96 losses, SEA 3/44. Dev years 2021 23, 2022 74, 2025 38, 2026 5 bets (KXRAINNYC not listed 2023-24).
- Capacity (moot): median 52 contracts traded before T on picked days -> <$50/day at a ~1c margin.
- Sealed (not scored, not worth asking for): 201 markets 2026-01-16..11-08 excl. R1 window, 70 input-true.
- KXRAIND* per-city: no history to test (census). A forward logger isn't worth it: same physics and same market
  as KXRAIN/KXRAINNYC, where taker is priced at break-even and maker is adversely selected.
- NOT done (post-hoc): a cheap-YES filter (yes_bid <= 5c) or 12Z NAM. Any such variant would be a new trial.

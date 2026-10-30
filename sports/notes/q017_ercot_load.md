# Q017 -- Kalshi ERCOT daily peak load (KXTXERCOTPEAKD) vs ERCOT's own published load forecast

PRE-REGISTERED 2026-10-29, written BEFORE any Kalshi quote was joined to any forecast or outcome (no candles fetched).
Script: `scripts/q017_ercot_load.py` (`markets` -> `eia` -> `audit` -> `candles` -> `build` -> `score`).
Ledger dataset: `q017_ercot_load`, one row per trial. Cache: `data/q017/`.
Forward logger (built regardless, see Power): `scripts/q017_logger.py` (tick/report).

What I had already seen before writing this (disclosed): the market list with the highest YES strike per event (i.e. a
lower bound on each day's outcome), one event's full ladder with `expiration_value` (26OCT29: 63,406.90 MW), and
EIA-930 ERCO daily peaks of DF vs D for 2026-10-17..10-29 (to confirm the EIA API works keyless with DEMO_KEY and to
pin the date mapping). No price was looked at against any forecast or outcome.

## Contract (read from the API 2026-10-29: /series/KXTXERCOTPEAKD, /markets?series_ticker=KXTXERCOTPEAKD)
- The only Kalshi load series. (KXTXERCOTPEAK = annual 2026 peak, one event; KXERCOT*W/M = hub PRICE series, not load.)
- Rule: "If the highest hourly total system load reported for the ERCOT grid on <operating day D> is above K MW, YES."
  Secondary: Expiration Value = highest value in the TOTAL column of the FIRST complete "Actual System Load by Forecast
  Zone" CSV (NP6-346-CD) for D; hourly (hour-ending) values; no rounding; later revisions ignored. Instantaneous
  readings excluded.
- `strike_type: greater`, `floor_strike` K, ladders of 10-31 strikes 500 MW apart (19 typical since late Sep).
- **Event ticker date = D + 1** (e.g. KXTXERCOTPEAKD-26OCT29 is operating day 2026-10-28; its expiration_value 63,406.9
  vs EIA D peak 63,410 for CT day 10-28). The operating day is parsed from `rules_primary` per event, not the ticker.
- Trading window: open ~23:55 CT on D-1 (04:55Z), close ~23:55 CT on D. Some early events opened later on D-1.
  can_close_early when the data is released.
- fee_type `quadratic`, fee_multiplier 1 -> taker fee 0.07 p (1-p) per contract, unrounded (house convention);
  sensitivity: fee rounded up to the cent for a 1-contract order.
- History: events 26AUG29 (op day 08-28) .. 26OCT30 (op day 10-29) -> ~61 settled operating days. 26AUG29 is on
  /historical (cutoff 08-30); the rest on live endpoints.

## Forecast (the "public, better-calibrated forecast" leg)
- ERCOT's hourly 7-day forecast (NP3-560-CD / NP3-565-CD) is public but the MIS archive keeps only ~8 days (vintages
  10-22..10-29 on 2026-10-29); the public ERCOT API needs a registered key. So the historical leg uses
  **EIA-930 ERCO `DF` (day-ahead demand forecast, hourly, UTC hour-ending)**, keyless via `api_key=DEMO_KEY`.
- F_D = max over the CT operating day D of DF (UTC period p -> interval start p - 1h -> CT date).
- **Vintage / lookahead audit (`audit`, must PASS before `build`):** EIA DF is the BA's day-ahead forecast for D, so it
  must have been made on D-1. Check: for every D in the ERCOT MIS archive window, compare F_D(EIA) with the D peak of
  each ERCOT NP3-560 vintage. PASS if the best-matching vintage (min |diff|) was published before 00:00 CT on D, OR
  |F_D(EIA) - peak of the last vintage published before 00:00 CT on D| <= 1.0% on every audit day. FAIL -> the
  historical backtest is void (lookahead risk), forward logger only.
- Decisions on D at 06:00 and 10:00 CT use only F_D. ERCOT's later intraday updates and the morning's realised load
  are NOT used (that is conservative: the market may know more than we do).

## Error model (frozen now; fit only on pre-Kalshi history)
- r = ln(A_D / F_D), A_D = max over CT day D of EIA `D` (actual demand), over CT days in months Aug-Nov of 2023, 2024,
  2025 (all strictly before the first Kalshi event). Drop days with < 23 hours of either series.
- Settlement basis c = median over DEV days of expiration_value / A_D(EIA) (actual vs actual; fit on DEV only, frozen
  for sealed). Reported descriptively; if |ln c| > 0.01 it is applied, else c = 1.
- p_yes(K) = (#{r_i : c F_D e^{r_i} > K} + 0.5) / (N + 1).

## Quotes and entry
- Kalshi candlesticks, period_interval = 60 (live endpoint `/series/KXTXERCOTPEAKD/markets/{tk}/candlesticks`,
  `/historical/markets/{tk}/candlesticks` for 26AUG29). Quote at T = last candle with end_period_ts <= T and
  >= market open; yes_bid/yes_ask `close`. YES ask = yes_ask; NO ask = 1 - yes_bid. A side is buyable only if its
  ask is in [0.01, 0.99]. No depth is known (census: median YES-bid size 0) -> 1 contract per bet; fill at the
  candle ask is an assumption, stated as such.
- T in {06:00, 10:00} CT on operating day D (UTC via America/Chicago; DST handled by zoneinfo).

## Trials (k = 4; one contract per market per trial; at most one side per market)
edge(s) = p_model(s) - ask(s) - 0.07 ask(s)(1 - ask(s)).
- A06 EDGE: at 06:00, buy the side with edge >= 0.05.
- A10 EDGE: same at 10:00.
- B06 FAR (the E1 / weather pattern): at 06:00, on strikes the forecast puts far away (p_model of the favoured side
  >= 0.97), buy the favoured side if its ask <= 0.96 (i.e. NO on far-above strikes, YES on far-below strikes).
- B10 FAR: same at 10:00.
pnl = 1{side wins} - ask - fee; ROI = sum pnl / sum (ask + fee).

## Split
Settled operating days sorted; DEV = first floor(0.7 n) days, SEALED = the rest. SEALED is scored once, only for
trials that pass DEV (user pre-authorised sealed runs for this study), with bar one-sided p < 0.05 / (#passing),
ROI > 0. SEALED rows are not loaded by `score --split dev`.

## Bar (DEV; all must hold)
- ROI > 0 and one-sided p < 0.05/4 = 0.0125; p = share of 10,000 day-clustered bootstrap resamples (seed 17017) with
  sum pnl <= 0. Quote the OBSERVED ROI with the 2.5-97.5 percentile interval (not the resample mean).
- Both halves positive (DEV days split chronologically in two; bets follow their day).
- >= 30 bets and >= 10 day clusters, else "underpowered, no verdict".

## Kill
- No trial passes DEV -> historical leg DEAD. If additionally all four DEV ROI point estimates are <= 0, the idea is
  DEAD outright (no logger deployment recommended). If some are positive but not significant -> LEAD, forward logger.
- A pass with model Brier worse than the Kalshi mid's Brier on the same quotes = bug until rows are inspected.

## Power (written now)
~61 days -> ~42 DEV days, ~19 SEALED. Daily outcomes are one draw shared by all strikes of the day, so the effective
n is the number of days. A far-tail rule like E1 earns ~1-3% ROI; detecting that needs >1,000 independent bets --
impossible here. Only a gross lag (ROI of order +15% or more, sustained across ~40 days) can pass. The historical test
is therefore a screen; a non-pass is "not shown", not "proven priced", unless the point estimates are negative.
The forward logger (ERCOT's own hourly forecast vintage, captured live, + Kalshi book) accrues ~1 day/day.

## Descriptive (not trials)
Model calibration vs Kalshi mid at 06:00/10:00 (Brier, by bucket); EIA-vs-settlement basis; forecast MAPE of F_D on
pre-Kalshi and DEV days; volume and quote availability.

## Results
(filled in after `score`; nothing below this line existed when the above was written)

### Run 2026-10-29 (DEV only; sealed NOT scored because no trial passed DEV)
Deviations from the text above, all made before `score` and disclosed:
- EIA `api_key=DEMO_KEY` allows 10 requests/day/IP (HTTP 429, Retry-After ~20 h); it ran out after the DF pull. `D`
  came from the keyless EIA-930 six-month BALANCE files (`eia930` stage, 2023..2026 Jul-Dec). DF in those files equals
  the API DF on all 16,011 overlapping hours (max |diff| 0).
- Audit filter bug: the first audit run counted 10-22 (archive's first day, no vintage before D exists) as an audit day
  and printed FAIL. Fixed to "days with >= 1 vintage before D" (what the text says). Result: on all 7 evaluable days
  (10-23..10-29) EIA DF's daily peak matches the ERCOT NP3-560 vintage published 13:30-14:30 CT on D-1 to < 1 MW
  (data/q017/audit.csv) -> **AUDIT PASS: DF is a D-1 afternoon vintage, no lookahead** (verified on 7 recent days
  only; assumed for Aug-Sep).
- EIA has no DF for op days 2026-09-04 and 2026-10-05 -> dropped by the < 23 h rule. 60 op days: DEV 42
  (08-28..10-10), SEALED 18 (10-11..10-28).

Model inputs: pre-Kalshi error sample n 366 (Aug-Nov 2023-25), mean ln(A/F) -0.0107, sd 0.0274. Basis
settlement / EIA-D peak on DEV: median 1.0000 (range 0.9997-1.0003) -> c = 1. On DEV days DF's peak MAPE vs settlement
1.63%, mean ln(A/F) -0.0031 (the forecast's high bias shrank vs 2023-25; the frozen model shifts ~0.8% too low).

Calibration (two-sided quotes, spread <= 10c): 06:00 n 178 / 37 days, Brier model 0.0477 vs Kalshi mid 0.0535;
10:00 n 238 / 40 days, model 0.0440 vs mid 0.0268 (by 10:00 the market has the morning's information; we do not).

| trial | bets | days | win | avg ask | units | ROI [95% day-clustered CI] | p | halves | verdict |
|---|---|---|---|---|---|---|---|---|---|
| A06 EDGE | 128 | 32 | .352 | .376 | -4.68 | -9.42% [-39.24, +27.47] | .72 | -12.76 / -8.14 | fail |
| A10 EDGE | 143 | 36 | .273 | .294 | -4.52 | -10.39% [-48.47, +26.71] | .71 | -6.19 / -12.35 | fail |
| B06 FAR | 12 | 7 | 1.000 | .943 | +0.64 | +5.59% [+4.64, +6.71] | .0000 | +5.42 / +5.62 | underpowered (< 30 bets, < 10 days) |
| B10 FAR | 7 | 6 | 1.000 | .864 | +0.90 | +14.84% [+4.72, +30.39] | .0000 | +22.22 / +6.29 | underpowered |
Rounded-fee ROI: A06 -10.63%, A10 -11.80%, B06 +4.90%, B10 +14.19%. All B bets were NO on strikes 3.3-5.9% above F;
B10's +14.8% leans on two 08-28 (first-ever event) fills at NO asks .84 and .49. The CI/p for B are meaningless with
6-7 clusters and zero losses (bootstrap cannot resample a loss that never occurred).

**Verdict: no DEV pass -> historical leg DEAD for the edge rule (A06/A10 both negative, mostly NO buys at ~.37 where
the model's 1%-low bias made it too bearish). The E1-style far-tail rule (B) is a LEAD: 19 fills, all won, but on
7 days -- not evidence.** Sealed not scored. Forward logger `scripts/q017_logger.py` built and run once (tick logged
191 ERCOT vintages + 19 open markets; report path exercised with synthetic quotes); not installed in launchd.
Thinness: the 10-30 tick shows 5,000-lot YES bids at .99 on settled-looking strikes but yes_ask_size 0; census median
YES-bid size 0 -> fills at the candle ask are an assumption. Not critic-reviewed.

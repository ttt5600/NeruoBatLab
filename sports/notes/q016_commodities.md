# Q016 -- Kalshi gas (AAA) and commodity ladders vs a public futures-based fair value

PRE-REGISTERED 2026-10-29, written BEFORE any Kalshi quote was joined to any outcome. Script:
`scripts/q016_commodities.py` (`markets` -> `ref` -> `fitgas` -> `decide` -> `quotes` -> `avail` -> `score dev` -> `score sealed`).
Ledger dataset `q016_commodities`, one row per trial. Cache (gitignored): `data/q016/`.

What had been looked at before this note was written (all outcome-free for the purpose of a price-vs-outcome test):
series metadata + rules (below), settled-market lists (strike grids, open/close times), a 3-market candle format check,
and one structural check that each single-rung KXAAAGASW strike equals the previous event's settlement value
(97 single-rung events; 87 of 97 exact to 0.0015). No Kalshi price was joined to any result.

## Contracts (read from /series and market rules, 2026-10-29)
| series | settles on | close (decision horizon) | strikes | fee_type |
|---|---|---|---|---|
| KXAAAGASW | AAA US regular average on Monday M ("strictly greater than X") | Sun 23:59 ET before M | 2023-10..2025-11: ONE rung = previous Monday's AAA (an "up this week?" binary); 2025-12+: ladder of 3-35 rungs | quadratic |
| KXAAAGASM | AAA on last day of month | -- | ladder | quadratic_with_maker_fees (not tested: 34 events) |
| KXWTI | NYMEX/ICE WTI front-month daily settlement (Kalshi rolls 2 business days before LTD) | 14:30 ET D | between / greater / less, 10-30 rungs | quadratic |
| KXWTIW | same, Friday settlement of the named contract | Fri 14:30 ET | between (+ tails) | quadratic |
| KXGOLDD/W, KXNATGASD/W, KXBRENTD/W | Pyth index 1-min candle close at 17:00 ET (gold index is SPOT, not futures) | 17:00 ET | greater, 20-80 rungs | quadratic |

Taker fee 0.07 p (1-p) per contract, unrounded (house convention). Maker side not tested.

## Point-in-time reference data (all free; frozen to data/q016/ on the day of the run)
- Yahoo Finance via yfinance: CL=F, RB=F, GC=F, NG=F, BZ=F daily (2000+) and hourly (available 2024-06-07+ only).
  Hourly bar index = bar START (ET); a bar is usable at t only if start + 1 h <= t. Daily bar of date d usable only on
  dates > d.
- EIA weekly US regular retail gasoline (Mondays, 1990+), `EMM_EPMR_PTE_NUS_DPGw.xls`. A Monday's EIA value is
  treated as known only from the following day (EIA publishes Monday afternoon).
- Kalshi's own past settlement values (`expiration_value`) of the same series are used as INPUTS only for events whose
  close is >= 1 h before the decision (AAA level; futures/Pyth basis).

## Models (frozen here)
**Gas pass-through (T1).** Weekly OLS fit ONLY on EIA Mondays 2006-01-02..2022-12-26 (before any Kalshi gas market):
y = R(w+1) - R(w) on x1 = RB(F_w) - RB(F_w-1), x2 = RB(F_w-1) - RB(F_w-2), x3 = margin(w) - mean(margin over the
previous 52 weeks), margin = R(w) - RB(F_w); F_w = last RB=F daily close strictly before Monday w. Predictive sd s(w) =
sd of the model's EIA residuals over the 26 most recent weeks whose outcome Monday is strictly before w (floor 0.01).
Application at Kalshi decision t (Monday): A = previous KXAAAGASW settlement value (= AAA that Monday); the AAA-EIA
offset = median(AAA_Mon - EIA_Mon) over the last 8 earlier Mondays where both are known; R(w) := A - offset.
AAA(next Monday) ~ Normal(A + yhat, s). P(YES at X) = 1 - Phi((X - A - yhat)/s).
**Futures/Pyth (T2-T4).** log S_settle ~ log(C_t) + basis + sigma_h sqrt(h) Z, Z = Student-t(4) scaled to unit variance.
C_t = close of the last hourly bar (of the mapped Yahoo future) usable at t. sigma_h^2 = mean of squared hourly log
returns over the last 300 usable bars with volume > 0. h = calendar hours from t to close minus 1 h per 17:00-18:00 ET
break crossed. basis = median over the 5 most recent same-series events closing >= 1 h before t of
log(expiration_value) - log(Yahoo close of the hourly bar starting at the hour containing that event's close).
Guard (skip the event): fewer than 3 basis points, or |last basis - median basis| > 0.005 (contract-roll mismatch).
Mapping: WTI -> CL=F, BRENT -> BZ=F, GOLD -> GC=F, NATGAS -> NG=F.
P(YES): greater: P(S > floor); less: P(S < cap); between: P(floor <= S < cap + 0.01).

## Decision times and quotes
- T1 gas weekly: t = the first full hour >= max(listing + 60 min, Monday 12:00 ET) on the listing day.
- T2 KXWTI daily: t = 10:00 ET on the settlement day. Only events with hourly Yahoo data (close >= 2024-06-10).
- T3 Pyth dailies (KXGOLDD, KXNATGASD, KXBRENTD): t = 10:00 ET on the settlement day.
- T4 weeklies (KXWTIW, KXGOLDW, KXNATGASW, KXBRENTW): t = Monday 10:00 ET of the settlement week (close >= 2024-06-10).
- Quote: last hourly candle (historical/markets or series/markets candlesticks, period 60) with end_period_ts <= t,
  window [max(open, t - 72 h), t]; yes_bid/yes_ask `close`. No candle -> no quote. YES ask = yes_ask; NO ask = 1 - yes_bid.
  A side is buyable only if its ask is in [0.01, 0.99]. One contract-dollar per bet, one side per market.

## Trials (k = 4; bar one-sided p < 0.05/4 = 0.0125)
edge(side) = p_model(side) - ask(side) - 0.07 ask (1-ask).
- T1 GAS_W: KXAAAGASW all events (single-rung and ladder); buy the side with edge >= 0.05 and yes spread <= 0.05.
- T2 WTI_D_FADE: KXWTI; longshot L = side with p_model <= 0.10; buy the other side F if edge(F) >= 0.03 and yes spread <= 0.05.
- T3 PYTH_D_FADE: same rule on KXGOLDD + KXNATGASD + KXBRENTD.
- T4 WEEKLY_FADE: same rule on KXWTIW + KXGOLDW + KXNATGASW + KXBRENTW.

## Split, statistics, pass/kill
- Per trial, DEV = earliest 70% of the trial's distinct settlement dates (from the event list, outcome-free);
  SEALED = the rest. The split date is printed by `decide` and recorded below before `score dev` runs.
- Metric: ROI per $ at ask + fee. Cluster = settlement date (T1/T2/T3) or settlement week (T4). 10,000 cluster
  bootstrap resamples (seed 16016); report observed ROI, percentile 95% CI, one-sided p = share of resampled total
  P&L <= 0.
- PASS on DEV: p < 0.0125 AND both date-halves of DEV have ROI > 0 AND >= 100 bets. Then SEALED is scored once
  (user authorised sealed tests); sealed pass = ROI > 0 with one-sided p < 0.05.
- KILL: DEV ROI <= 0, or p >= 0.0125, or < 100 DEV bets (= untestable, the N6 failure mode).
- Power (house `edge_math power`): +3% at ~90c -> ~574 bets; +5% at 50c -> ~2,467. Expected DEV bet counts come from
  `avail` (outcome-free) and are recorded below before scoring.
- Sensitivities (not trials, cannot rescue a fail): Normal instead of t4; fee rounded up to the cent.
- Descriptive, not a trial (added before scoring): Brier(model) - Brier(Kalshi mid) on every two-sided quoted market,
  cluster-bootstrap CI. For T1 this is the main readout since T1 cannot reach 100 DEV bets.
- Too-good guard: any DEV ROI > +15% on > 100 bets is treated as a bug until 20 rows are inspected by hand.

## Recorded before scoring (filled from `decide` / `avail`, no outcomes)
Clarification made before any quote was fetched: the basis bar is "the hourly bar containing the last minute before
the event's close" (14:00-15:00 bar for a 14:30 close; 16:00-17:00 bar for a 17:00 close, since the CME 17:00-18:00
break has no bar). Basis uses past events only, so this choice cannot leak.

Gas OLS (`fitgas`, EIA 2006-2022): n = 887, beta (const, dRB 1w, dRB 2w, margin dev) = [0.0026, 0.1575, 0.1046, -0.1692],
R2 = 0.432, residual sd 0.0469, in-sample sign agreement 0.749.

`decide` (events kept / skips):
- T1_GAS_W 132 events, 1,036 markets, 2023-11-27..2026-10-26; SEALED starts 2026-01-19. Skips: 7 no 7-day-prior
  settlement, 3 < 4 offset points. NOTE: DEV is the single-rung era only (one market per event), so DEV has at most
  ~92 bets -> below the 100-bet floor -> T1 is untestable as pre-registered whatever the ROI. Reported descriptively.
- T2_WTI_D_FADE 440 events, 7,768 markets, 2024-06-24..2026-10-29; SEALED starts 2025-11-19 (67 events skipped by the
  roll guard, 9 no reference bar).
- T3_PYTH_D_FADE 284 events, 14,142 markets, 2026-03-30..2026-10-29; SEALED starts 2026-08-31 (guard skips: gold 10,
  natgas 26, brent 36).
- T4_WEEKLY_FADE 100 events, 2,545 markets, 2024-07-05..2026-10-23; SEALED starts 2026-06-18.
Observed basis (descriptive): Pyth NATGAS and BRENT settle equal to the Yahoo front-future bar close (median basis ~0);
Pyth gold sits ~0.56% below GC=F (spot vs future); KXWTI vs CL=F -0.14%.

## AMENDMENT 2026-10-29 (before any candle was read): rate-limit sample
The coordinator stopped the full 25,491-market candle pull (sustained HTTP 429s on our IP were stalling live paper
loggers). About 1,000 candle responses had been cached, none read. Replacement, fixed before reading any candle or outcome:
- `sample` step: `numpy.random.default_rng(1616)`, 750 markets per trial (3,000 total), stratified by series by largest
  remainder of eligible counts. Eligible = every T1 market; for T2-T4 only markets the pre-registered rule could ever bet
  (model longshot prob min(p, 1-p) <= 0.10, a model-only filter). Draw: T1 750 of 1,036; T2 750 of 6,472 (KXWTI);
  T3 750 of 10,742 (KXGOLDD 248, KXNATGASD 413, KXBRENTD 89); T4 750 of 1,938 (KXWTIW 311, KXGOLDW 202,
  KXNATGASW 176, KXBRENTW 61). File `data/q016/sample.parquet`.
- Fetch: single thread, >= 0.34 s between requests (<= 3 req/s; lowered to 1 req/s after four 429s in the first
  minutes), rungs listed after the decision time get no quote (they did not exist), 429/5xx backoff 5, 10, 20, 40, 60 s; cached responses reused.
- Consequences: quote-availability tables now cover only longshot-eligible rungs for T2-T4 (that is the question asked).
  Bet counts shrink about 8.6x (T2), 14x (T3), 2.6x (T4) and 1.4x (T1) versus the full universe; the 100-bet DEV floor
  and every other criterion stay unchanged.

## RESULTS 2026-10-29 (DEV only; SEALED not scored because no trial passed DEV)
Quote availability at the decision time, over the 3,000 sampled markets (`data/q016/avail_sampled.py`). "candle" means
any standing quote; "two-sided" means yes_bid > 0 and yes_ask < 1. "fav edge" is the median model edge of the favourite side on
two-sided rungs with spread <= 5c.

| trial | longshot band | n | candle | two-sided | spread<=5c | fav edge |
|---|---|---|---|---|---|---|
| T2 KXWTI | <.001 / .001-.01 / .01-.05 / .05-.10 | 402/195/105/48 | .55/.72/.75/.77 | .09/.27/.65/.67 | .04/.21/.56/.56 | +.009/+.008/.000/+.033 |
| T3 Pyth D | same | 215/280/174/81 | .99/.98/.96/.99 | .04/.36/.82/.96 | .02/.21/.56/.69 | +.009/+.013/+.004/.000 |
| T4 weekly | same | 109/184/304/153 | .97/.96/.97/.96 | .14/.36/.80/.88 | .06/.26/.68/.75 | +.009/+.006/-.004/-.018 |
| T1 gas W | <.001 .. >.10 | 37/25/99/89/497 | .49-.70 | .24-.65 | .05-.29 | -- |

Are far brackets quoted at all? The rungs exist and carry a standing quote. But when the model puts the longshot below 1%,
there is almost never a YES bid (two-sided 4-36%), so the favourite costs $1.00 and cannot be bought. Where both sides
are quoted, the favourite is priced within ~1c of the t4 futures model. This is the same structure that killed N2 and N6.

| trial | DEV bets | ROI | 95% CI | one-sided p | verdict |
|---|---|---|---|---|---|
| T1_GAS_W | 4 | -57.4% | [-100%, +27.7%] | .95 | KILL: untestable (< 100 bets; DEV = single-rung era) |
| T2_WTI_D_FADE | 2 | +11.3% | degenerate (2 clusters; bootstrap p 0 is meaningless) | -- | KILL: untestable |
| T3_PYTH_D_FADE | 27 | -4.9% | [-20.8%, +8.1%] | .74 | KILL: untestable, and negative |
| T4_WEEKLY_FADE | 5 | -11.7% | [-55.9%, +11.0%] | .68 | KILL: untestable |

Descriptive Brier (model - mid, two-sided sampled DEV markets; negative = model better):
- T1: +0.035 [-0.017, +0.084], n 60. The RBOB pass-through is NOT better than the Kalshi price on "up this week?" (not distinguishable).
- T2: -0.072 [-0.120, -0.026], n 62.
- T3: +0.010 [-0.006, +0.031], n 271.
- T4: -0.027 [-0.057, -0.007], n 196.

The T2/T4 "model better" readings come from longshot rungs where the mid of a nearly one-sided book (e.g. 0.01/0.20) is a
poor price proxy. The buyable ask is what matters, and it gave 2-5 bets. Not actionable.

Power: scaling the sample counts by eligible/sampled, the FULL universe would give roughly T2 ~17, T3 ~380, T4 ~13 DEV bets.
Only T3 could clear the 100-bet floor with a full pull, and its sampled ROI is negative (-4.9%, CI crosses zero). That is
not worth another ~10k requests on a shared, rate-limited IP.

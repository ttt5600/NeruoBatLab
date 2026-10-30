# Q012 -- Kalshi hourly crypto above/below (KXBTCD, KXETHD) vs a spot + realised-vol fair value

PRE-REGISTERED 2026-10-29, written BEFORE any Kalshi quote was joined to any outcome (no candles fetched yet; the only
outcome-bearing data looked at so far is one illustrative archived event, KXBTCD-26AUG2919, used to read the rules and
the strike grid). Script: `scripts/q012_crypto_hourly.py` (`draw` -> `build` -> `counts` -> `score`).
Ledger dataset: `q012_crypto_hourly`, one row per trial. Cache: `data/q012/`.

## Contract (read from the API 2026-10-29, /series/KXBTCD, /series/KXETHD, archived market rules)
- "If the simple average of the sixty seconds of CF Benchmarks' BRTI (ETH: ETHUSD_RTI) before H is above K at H, YES."
  Strikes are `floor_strike` = round number - 0.01 (`strike_type: greater`). Ladders of 40-188 strikes, $100-$250 apart
  (BTC) / $20-$40 (ETH). Each hourly event opens 1 h before its close (some ET-17:00 events live longer).
- fee_type `quadratic`, fee_multiplier 1 -> taker fee 0.07 p (1-p) per contract (primary, unrounded, as in every other
  Kalshi study here); sensitivity: 1-contract fee rounded up to the cent.
- Settlement value (`expiration_value`) is published per market; used ONLY for outcomes and a descriptive basis check.
- Historical cutoff 2026-08-30: earlier markets are served by /historical/*, later ones by the live endpoints.

## Not a redo of L19
L19 K-FAV (DEAD) bought the 88-97c favourite of the 17:00 ET daily close at T-3h, with no fair-value model, and as a
maker. Q012 is a model-vs-price test on the hourly events, in the last 10-30 minutes, taker only, conditioned on a
model gap. If the trials below fail, the hourly crypto ladder is dead for model-gap taking too.

## Sampling frame and split (outcome-free)
- Frame: slots (series in {KXBTCD, KXETHD}) x (ET date 2025-01-01 .. 2026-10-27) x (ET hour 00..23) -> event ticker
  `KX{BTC,ETH}D-{YY}{MON}{DD}{HH}`. A slot whose event does not exist is dropped (existence is not outcome-related).
- Split by ET date, 70/30 of the 665 frame days: DEV = ET date < 2026-04-11; SEALED = 2026-04-11 .. 2026-10-27.
  SEALED is not drawn, fetched or scored here; scoring it needs the user's explicit yes.
- Draw 1: `numpy.random.default_rng(12012).choice(all DEV slots, 1600, replace=False)`.
- Pre-registered extension (decided on COUNTS only, before any result is read): if any trial has < 1,000 bets or
  < 300 event clusters in the outcome-free decision file, add draw 2 = `default_rng(12013)` 1,600 more DEV slots not in
  draw 1, once. No further extension.
- Outcomes (`result`, `expiration_value`) arrive in the same API response as the strike list; the build step writes them
  to a separate file (`data/q012/dev_results.parquet`) that only `score` reads.

## Decision times and quotes
- t = close - m minutes, m in {30, 15, 10}.
- Kalshi quote at t: last 1-min candle (historical/markets/{tk}/candlesticks, period_interval=1, window
  [close-4h, close]) with end_period_ts <= t; yes_bid/yes_ask `close`. Candles are emitted only when something changes,
  so the last one at or before t is the standing quote. No candle at or before t -> no quote.
  YES ask = yes_ask; NO ask = 1 - yes_bid. A side is buyable only if its ask is in [0.01, 0.99].
- Strike selection (spot only, no prices, no outcomes): markets with |z_gauss| <= 3.5 at any of the three decision
  times, z_gauss = ln(K/S_t) / (sigma_t sqrt(m_eff)).

## Fair-value model (frozen now; only its empirical CDF is fit, on data strictly before the decision day)
- Spot: Coinbase Exchange public 1-min candles BTC-USD / ETH-USD (BRTI constituent; BRTI itself is not free).
  S_t = close of the candle that STARTS at t-60 s (it closes at t). Missing -> last candle starting in [t-5 min, t-60 s];
  else no decision. Lookahead guard: a unit test asserts every spot candle used has start + 60 <= t.
- sigma_t^2 (per-minute variance) = 0.5 * mean(r^2 over the 60 one-min log returns ending at t)
  + 0.5 * mean(r^2 over the 1,440 ending at t); missing minutes forward-filled (zero return).
- m_eff = m - 1 + 1/3 (diffusion to H-60 s, then the 60-s average adds one third of a minute of variance).
- z_K = ln(K / S_t) / (sigma_t sqrt(m_eff)); p_yes = (#{Z > z_K} + 0.5) / (N + 1), where Z = empirical standardised
  m-minute returns ln(S_{t'+m} / S_{t'}) / (sigma_{t'} sqrt(m)) over t' on a 5-minute grid in the 60 days ending at
  00:00 UTC of the decision day (all t'+m before that instant). One CDF per (asset, m, decision day).
- Sensitivities (not trials): Gaussian Z (lognormal); none of these can rescue a failed trial.

## Trials (k = 4, one contract per market per trial, one side per market)
Far strike = model prob of one side (the longshot L) <= 0.10; F = the other side. edge(side) = p_model(side) - ask(side) - 0.07 ask (1-ask).
- T1 FADE_T15: m = 15, far strike, buy F at its ask if edge(F) >= 0.03 and yes spread <= 0.02.
- T2 FADE_T30: same at m = 30.
- T3 FADE_T10: same at m = 10.
- T4 ANY_T15: m = 15, any selected strike, buy the side with edge >= 0.05 (yes spread <= 0.03), whichever side.
pnl = 1{side wins} - (ask + fee); ROI = sum pnl / sum (ask + fee).

## Bar (all must hold for a PASS -> sealed ask to the user)
- ROI > 0 and one-sided p < 0.05/4 = 0.0125, p = share of 5,000 event-clustered bootstrap sums <= 0 (seed 12012).
- Both halves positive: events sorted by close time, first half vs second half of events (bets follow their event).
- >= 300 bets and >= 100 event clusters.
- Reported: ROI with 95% event-clustered percentile CI, n bets, n events, halves, by asset, rounded-fee sensitivity.

## Kill
- All four trials fail -> Q012 DEAD for taker-at-ask with this model family (write it into FUND_BOOK 3d).
- A trial that passes while the model's Brier on the T-15 quotes is worse than the Kalshi mid's is treated as a bug
  until the rows are inspected (too good = bug).

## Power (written now)
A ~95c buy has per-bet sd ~0.22; detecting +2% ROI at one-sided alpha 0.0125 with 80% power needs ~1,300 independent
bets, more with strike clustering inside an event. 1,600 slots -> ~1,000-1,500 existing events (BTC is not hourly
24/7 in early 2025; ETH similar). Bets per event are unknown until the decision file exists; the extension rule above
handles a shortfall. If still < 300 bets, the trial is reported as underpowered (no verdict), not as a pass or a fail.

## Descriptive (not trials)
Calibration of the Kalshi mid at T-15 (quotes with spread <= 0.10) by mid bucket: n, mean mid, realised YES rate,
mean model p. Model vs mid Brier. Coinbase-vs-settlement basis.

## Results
(filled in after `score`; nothing below this line existed when the above was written)

### Pre-score log (2026-10-29, outcome-free; written before `score` ran)
- Draw 1: 1,600 slots -> 1,122 events with markets (478 slots had no event), 1,120 with decisions, 7,736 spot-selected
  markets, 23,158 decision rows. Counts: T1 62 bets/50 events, T2 103/67, T3 41/35, T4 125/94 -> all < 1,000 -> the
  extension rule fired.
- Draw 2 (seed 12013, +1,600 slots): total 2,285 events with markets, 2,278 with decisions, 47,183 decision rows.
  Counts: T1 113 bets/91 events, T2 222/145, T3 83/71, T4 255/185. **All four are < 300 bets -> UNDERPOWERED by the
  pre-registered rule, no verdict possible on DEV.** No further extension (the rule allows one). Score is run once anyway,
  for the record and the descriptive calibration.
- Outcome-free observation: few bets because the Kalshi quote rarely sits >= 3c (after fee) outside the model with a
  <= 2c spread. Median quote age at decision 0 min (mean 1.4), so quotes are not stale.

### DEV score (2026-10-29, the one run; results/q012_crypto_hourly/dev_score.json, ledger dataset q012_crypto_hourly rows 1-4)
DEV = 2,278 events (BTC ~1,120, ETH ~1,160 by the basis check), 2025-01..2026-04-10. Bar p < 0.0125, >= 300 bets.
| trial | bets / events | ROI | 95% CI (event-clustered) | p | halves | verdict |
|---|---|---|---|---|---|---|
| T1 FADE T-15 | 113 / 91 | +3.72% | [-1.98%, +8.42%] | 0.090 | +1.89% / +5.08% | UNDERPOWERED |
| T2 FADE T-30 | 222 / 145 | +1.74% | [-2.86%, +5.91%] | 0.216 | +3.09% / +0.88% | UNDERPOWERED |
| T3 FADE T-10 | 83 / 71 | -1.10% | [-9.17%, +5.52%] | 0.605 | -4.91% / +2.01% | UNDERPOWERED |
| T4 ANY T-15 | 255 / 185 | -1.76% | [-8.94%, +5.10%] | 0.691 | +1.11% / -4.54% | UNDERPOWERED |
All four CIs cross zero ("not distinguishable"); none would have passed even with n waived (T1 p 0.090 > 0.0125).
Bets are ~90% BTC. Sensitivities: rounded 1-contract fee T1 +2.47% (91 bets, p .22); Gaussian model T1 +1.03% (p .36),
T4 -8.92% (391 bets, p .91) -- the thin-tailed lognormal "finds" more longshot value and loses on it.

Calibration of the T-15 mid (12,816 quotes, spread <= 10c; rows cluster within events):
| mid bucket | n | mean mid | YES rate | mean p_emp |
|---|---|---|---|---|
| [0, .02) | 2,227 | 0.0089 | 0.0018 | 0.0054 |
| [.02, .05) | 1,849 | 0.0290 | 0.0135 | 0.0148 |
| [.05, .10) | 791 | 0.0662 | 0.0442 | 0.0450 |
| [.10, .20) | 632 | 0.1426 | 0.1266 | 0.1147 |
| [.20, .35) | 568 | 0.2698 | 0.2623 | 0.2448 |
| [.35, .50) | 458 | 0.4263 | 0.4432 | 0.4156 |
| [.50, .65) | 405 | 0.5742 | 0.5926 | 0.5833 |
| [.65, .80) | 513 | 0.7253 | 0.7173 | 0.7484 |
| [.80, .90) | 576 | 0.8541 | 0.8594 | 0.8717 |
| [.90, .95) | 600 | 0.9264 | 0.9400 | 0.9337 |
| [.95, .98) | 1,546 | 0.9654 | 0.9793 | 0.9807 |
| [.98, 1) | 2,651 | 0.9889 | 0.9962 | 0.9933 |
Brier: mid 0.05445, p_emp 0.05496, p_gauss 0.05524 -> the model does NOT beat the market mid overall.
The mid shows a longshot premium on both tails (1-2c in the far buckets), but the post-hoc check (NOT a trial) of
buying the far favourite (mid in [0, .10) or (.90, 1]) at the ask + fee gives -0.82% [-1.64%, -0.09%] at T-30
(4,479 bets), -0.48% [-1.20%, +0.20%] at T-15 (3,338), -0.35% [-1.04%, +0.34%] at T-10 (2,692): the half-spread
(mean spread ~3c) plus fee absorbs the premium. This is the L19 result again, at hourly horizons.
Basis: Coinbase close at H vs the settlement value, median |diff| 1.3 bp BTC / 2.2 bp ETH (p95 6.5 / 9.5 bp).

### Verdict
UNDERPOWERED on all four pre-registered trials (113-255 bets after the single allowed extension; bar >= 300).
Point estimates: two FADE trials slightly positive, not distinguishable from zero; T3 and T4 negative. The model loses
to the mid on Brier, and the visible tail premium does not survive the ask. Recommendation: no sealed ask. Treat Q012 as
DEAD-for-taking unless a new mechanism appears (e.g. a maker quote inside the spread on far strikes, which would run into
the L19 adverse-selection lesson). The sealed window 2026-04-11..10-27 stays untouched.

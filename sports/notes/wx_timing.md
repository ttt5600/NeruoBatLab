# E1/E2 at earlier decision times on the day before (DEV only), 2026-10-27

## Verdict: NO. Deciding earlier buys NO cheaper, but the extra losses cost more than the discount. 0 of 6 taker trials pass.
- Earlier on D-1, the NO price is lower: at margin 3 the mean NO ask is 90.7c at 11:00, 91.1c at 14:00, 92.7c at 18:00
  and 94.1c at 22:00. Each win pays 8.2c, 7.7c, 6.3c and 5.1c respectively.
- The forecast the rule can use earlier is staler (the peak hours come from the lead-2 archive). The rule therefore
  rejects brackets that later come in: the loss rate is 7.6% at 11:00, 7.7% at 14:00, 5.2% at 18:00 and 4.1% at 22:00.
  The price was cheap because the market priced that risk correctly.
- All 6 taker deltas against the 22:00 rule on the same dates are negative, from -2.1u to -15.2u. No p value comes
  near the bar (best p .59 against a bar of .0042).
- The 22:00 rule stays the best time to decide. Margin 5 does not help either. At 22:00 it makes +8.7u on 472 bets,
  against +30.5u on 1,713 bets at margin 3.
- The 6 MAKER trials are **NOT SCORED (data incomplete)**. On the lead's instruction the trade-tape pull was stopped at
  0 of 3,562 tickers, before any P&L was computed. They stay in the trial count.

## Pre-registration (scripts/wx_timing.py docstring, written before any price was pulled)
- Universe (dev only): E1 orig7 2024-03-01..2025-06-30 (kalshi_temp dev quarters) and X1 highs before 2026-03-01.
  That is 20 series, 2,949 station-days and 529 dates. Sealed rows are dropped on load.
- Decision times are 11:00, 14:00 and 18:00 station-local on D-1. The reference is 22:00. 10:00 was dropped because
  the book opens at 10:00 ET.
- Forecast available at T: for each of the 5 models, hour h of D uses `previous_day1` if h <= T-4 and `previous_day2`
  otherwise. The value is the max over hours 0..18 of D, with at least 15 hours present.
  **At T=22 this reproduces the frozen fcsafe_* exactly: 16,620 rows with 0 differences.**
- The rule is the frozen consensus-reject (at least 3 models) at margin 3 or 5. Quotes must be tradeable
  (yes_bid > 0, yes_ask < 1, spread <= 10c, yes_ask > 0.01).
- Taker buys at the NO ask and pays the unrounded fee 0.07p(1-p). Maker posts at the NO bid and is filled when 10 or
  more contracts print at or below that bid after T.
- Primary statistic: delta units against the same margin and execution at 22:00, by date-cluster bootstrap
  (4000 draws). The pass bar is p < .05/12, and delta must be > 0 in both halves.
- **Sanity check:** the 22:00 margin-3 taker reference reproduces the E4 study's taker baseline exactly:
  1,713 bets, +30.46u.

## Results (taker; stake 1 unit of cost per bet)
| T | M | bets | loss rate [Wilson 95%] | break-even | mean NO | c/contract | c/win | units | ROI | halves | delta vs 22:00 [95% CI] | p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 11 | 3 | 1373 | 7.6% [6.3, 9.1] | 8.8% | .907 | +1.19 | 8.15 | +22.6 | +1.64% | +7.1 / +15.4 | -7.9 [-31.7, +15.5] | .74 | fail |
| 14 | 3 | 1462 | 7.7% [6.4, 9.1] | 8.4% | .911 | +0.73 | 7.73 | +15.3 | +1.05% | +1.8 / +13.5 | -15.2 [-39.8, +8.6] | .89 | fail |
| 18 | 3 | 1429 | 5.2% [4.2, 6.5] | 6.9% | .927 | +1.66 | 6.26 | +27.8 | +1.95% | +2.8 / +25.0 | -2.6 [-23.3, +17.2] | .59 | fail |
| **22** | 3 | 1713 | 4.1% [3.2, 5.1] | 5.6% | .941 | +1.51 | 5.12 | +30.5 | +1.78% | +2.0 / +28.5 | reference | | |
| 11 | 5 | 368 | 5.2% [3.3, 7.9] | 6.0% | .936 | +0.85 | 5.43 | +2.9 | +0.79% | +3.6 / -0.6 | -5.8 [-17.3, +4.7] | .86 | fail |
| 14 | 5 | 402 | 4.2% [2.7, 6.7] | 5.8% | .939 | +1.54 | 5.27 | +6.6 | +1.63% | +4.5 / +2.1 | -2.1 [-12.3, +7.6] | .66 | fail |
| 18 | 5 | 346 | 3.2% [1.8, 5.6] | 4.5% | .952 | +1.32 | 4.00 | +4.3 | +1.24% | -0.7 / +5.0 | -4.4 [-12.4, +3.2] | .86 | fail |
| **22** | 5 | 472 | 1.3% [0.6, 2.7] | 3.1% | .967 | +1.80 | 2.92 | +8.7 | +1.85% | +2.8 / +5.9 | reference | | |

The halves split at 2024-11-24 (the median dev date). About 58-63 bets/month at margin 3, against 71/month at 22:00.
Overlap with the 22:00 picks: 713 of 1,373 at 11:00 and 944 of 1,429 at 18:00. Earlier decisions pick many brackets
that the 22:00 forecast would no longer reject.

By cohort at margin 3: E1 gets steadily worse the earlier it decides (22:00 +25.1u, 18:00 +16.7, 14:00 +1.8,
11:00 +8.9). X1 shows the reverse in its point estimates (22:00 +5.4u, 11:00 +13.6u). But X1 has only ~1.5 months of
dev data (~420 bets), it is descriptive only, and its edge is not a trial here. Do not mine it. A forward look would
need its own pre-registration.

## Maker (descriptive only; no fill model, so no P&L)
The NO bid at T on the same rejects is 86.6c at 11:00, 87.3c at 14:00, 89.1c at 18:00 and 90.7c at 22:00 (margin 3).
The spread is 3.3-4.1c. A resting bid is ~4c cheaper than the ask at every time. But E4 showed that losers always fill
(adverse selection), and the earlier loss rate is ~2x higher. Without tapes, nothing can be said about whether it nets
positive.

## Capacity
Historical displayed depth is not available. The proxy used is volume printed in the decision-hour candle. Its median
is **0 contracts at 11:00 and 14:00**, 2 at 18:00 and 4 at 22:00. Early books are quoted but barely trade, so even a
positive edge there would have little size. Summed decision-hour volume x price over all dev bets:
$61k (11:00) vs $175k (22:00).

## Caveats
- `previous_day1/2` are Open-Meteo's archives shifted by 24 h and 48 h, not run-by-run data. The 4 h publication margin
  is inherited from the frozen rule. With true intraday runs (e.g. the 12Z run at 14:00) an early decision could see a
  fresher forecast than lead-2. That would be a different test, and Open-Meteo single-runs only exist from 2026-04,
  which is inside the sealed windows.
- Quotes are hourly candle closes, not book snapshots.

Files: scripts/wx_timing.py (`forecasts`, `prices`, `dev`); data/wx_timing_{fc,quotes}.parquet;
data/wx_timing_results.json. Ledger: lab/ledger.jsonl dataset `wx_timing` (6 ok + 6 not_scored_data_incomplete).

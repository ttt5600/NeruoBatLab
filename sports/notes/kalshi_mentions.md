# Kalshi MENTION markets: buy NO (leads L10 / L23) -- 2026-10-22

Script: `scripts/kalshi_mentions.py` (pre-registration is in the docstring, written before any decision-time price or P&L).
Ledger: 6 trials, dataset `kalshi_mentions`, appended to `lab/ledger.jsonl` 2026-10-22.

## Data
- Universe: the 464 series with category `Mentions` in the full /series list (`data/earnings/series.json`, 2026-10-21).
  All 464 report `fee_type quadratic, fee_multiplier 1` (series API), i.e. the general taker fee.
  (The Kalshi fee-schedule PDF returned 429 to WebFetch, so the fee formula comes from the series API plus the formula
  already used in kalshi_longshot.py. It was not re-read from the PDF.)
- Settled markets from `/historical/markets` + `/markets?status=settled`: 83,960 markets / 4,165 events
  -> `data/kalshi_mentions_markets.parquet`.
- Eligible (binary word-mention contracts; excludes KXMVE* multivariate combos, "stated by Trump before <date>" week
  windows, "late to briefing", "speaks for N minutes", DURATION series): **65,574 markets / 3,806 events**.
- Growth (outcome-free describe): 2-44 events/month through 2025-09, then 126-444/month; monthly volume went from
  $3-19M (2025) to $100-170M (2026-01..07). Median market volume is ~3-15k contracts.
  Groups by events: other 1,841, sports announcers 1,054, earnings 464, trump 401, fed 68.
- DEV (event anchor < 2025-10-01 ET): **4,680 markets / 214 events**. SEALED (>= 2025-10-01): ~61k markets /
  ~3,600 events. Sealed has NOT been priced or scored.
- Hourly candles: markets settled before 2026-08-23 are served only by `/historical/markets/{t}/candlesticks`
  (the event-level endpoint returns nothing for them). 81 dev markets have no candles anywhere and stay unpriced.

## Design (pre-registered)
- **No start time exists in the API**, and ticker dates are unreliable: in about 20% of events the ticker date is a
  deadline or listing date, e.g. KXSECPRESSMENTION-26JAN29 actually took place 2025-07-31. So the timing is built
  from data a trader also sees:
  A = earliest close_time in the event (markets close at the event's end or hours later, after review);
  V = end of the busiest event-level volume hour in [A-96h, A]; anchor = min(A, V).
  D24 = anchor - 24h, D6 = anchor - 6h; quote = yes_bid/yes_ask close of the last hourly candle ending <= D.
- Timing check against known schedules: every 2025 FOMC event has V = 15:00 ET, the hour of the 14:30 presser.
  NVDA and TSLA calls (17:00 / 17:30 ET) have V at 18:00-19:00 ET. One FOMC event (25SEP) has V two days early, which
  only moves the decision earlier, so it is safe.
- Guard (prices only): drop an event at a decision time if >= 50% of its quoted markets already sit at an extreme
  (ask <= 3c or bid >= 97c).
  Quote filter: bid > 0, ask < 1, spread <= 10c.
- Bet: taker, buy NO at 1 - yes_bid, 1 contract.
  Fee: ceil_to_cent(0.07*C*P*(1-P)) per order, primary C = 10; C = 1 and C = 100 are reported as robustness only.
- Trials: M1 buy NO on every market; M2 YES mid in [0.15, 0.60]; M3 YES mid <= 0.15; each at D24 and D6.
  6 trials -> bar p < 0.0083, one-sided, 3000x bootstrap over EVENTS. Pass also requires both dev halves positive.

## Results (DEV, 2025-01-30..2025-09-30)
Calibration (hit rate minus mid): YES wins less often than its price in every bucket from 5c to 80c:
-4 to -10 pp at D24, and -4 to -10 pp at D6. Above 80c it is about calibrated (+0.0 to +1.3 pp). So YES is overpaid,
as the paper claims.

| trial | bets | events | units | ROI | 95% CI (units) | p | halves | ROI C=1 / C=100 | verdict |
|---|---|---|---|---|---|---|---|---|---|
| M1_d24 all | 2,098 | 113 | +24.1 | +2.18% | [-34.8, +86.8] | .225 | +5.8/+18.4 | +1.35/+2.26% | fail |
| M2_d24 15-60c | 924 | 110 | +17.4 | +2.70% | [-18.8, +52.5] | .170 | +1.4/+16.0 | +2.13/+2.77% | fail |
| **M3_d24 <=15c** | 293 | 79 | +11.9 | **+4.32%** | [+5.5, +18.5] | **<.0003** | +4.5/+7.4 | +3.74/+4.38% | **PASS** |
| M1_d6 all | 2,685 | 159 | +36.1 | +2.47% | [-32.1, +104.0] | .161 | +33.9/+2.2 | +1.66/+2.56% | fail |
| M2_d6 15-60c | 1,217 | 148 | +47.2 | +5.53% | [+4.3, +91.9] | .018 | +22.7/+24.6 | +4.96/+5.60% | fail (bar) |
| **M3_d6 <=15c** | 394 | 106 | +11.1 | **+2.99%** | [+3.2, +18.1] | **.0027** | +4.8/+6.2 | +2.41/+3.03% | **PASS** |

- M3 by group at D24: earnings +2.6%, fed +3.6%, other +5.5%, trump +7.4%. At D6: +3.0 / +2.1 / +4.7 / +0.3%.
  Positive in every group, but each group alone is small.
- M3 by month (D24): 7 of 8 months positive (2025-04 -0.7u). Dropping the top 5 events leaves +7.0u (D24) and +6.5u (D6).
- M2/M1 are dragged down by trump events (-11 to -13% at D24). That is a group cut after the fact, so it is not a rule.

## Traps / caveats
- **M3 is a sell-insurance payoff.** At D24 it wins about +6c on a 94c stake and lost only 6 of 293 bets (13/394 at D6);
  break-even needs a loss rate under ~6%. The dev edge rests on roughly 6-13 rare losses, all in different events.
  A bad month (several surprise mentions) can wipe out many months of gains. The event bootstrap is honest about
  clustering, but it cannot show tail risk the sample never contained.
- Small dev sample: 79-106 events, all in 2025, when these markets had 1/10 of today's volume. 2026 traders may have
  arbitraged the bias away. That is exactly what sealed tests.
- M3_d24 and M3_d6 overlap heavily (same longshots at two times), so they are not two independent confirmations.
- Fills: the price is the hourly candle's closing yes_bid; size at the bid is unknown. At $10-50 per market this is
  probably fine (median market volume of thousands of contracts), but it is unverified.
  Quotes can be several hours stale (the last candle at or before D). Stale quotes are still leak-free, just earlier.
- The fee depends on order size: at 1 contract per order the ceil-to-cent fee costs about 0.6 pts of ROI on M3
  (+3.74% vs +4.38%).
- Leak risk, D6 only: D6 is pre-event only if the busiest hour falls < ~5 h after the event starts. That holds for
  calls, pressers, rallies and shows. It can fail for all-day events, where the extreme-price guard is the backstop.
  D24 has much more slack.
- Rules risk: settlement depends on Kalshi's transcript reading (exact word, plurals/possessives only). Ambiguous
  resolutions were not modelled.
- Not tested: maker (resting) NO orders and depth-aware sizing.

## SEALED ASK (needs the user's explicit "yes M3")
- Rules frozen exactly as in the script: buy NO (taker, 1 - yes_bid) on every eligible mention market with
  YES mid <= 0.15 at **D24** (primary) and **D6** (secondary). Same filters, guard, fee (C = 10) and event bootstrap.
- Sealed set: events with anchor >= 2025-10-01 (~3,600 events, ~61k markets, through 2026-10).
- Pass = p < 0.05 / 2 = 0.025 (event bootstrap) AND units > 0, scored once.
- Steps:
  1. `../.venv_quant/bin/python scripts/kalshi_mentions.py prices --sealed` pulls prices only
     (~61k candle requests, ~4-5 h at 4 req/s, cached).
  2. `scripts/kalshi_mentions.py sealed --confirm M3_d24 M3_d6`.
  Step 1 can run before the yes, because it writes no P&L.
- If it passes, the live form is: at D24/D6 before a scheduled mention event, rest or take NO on words <= 15c,
  1-contract units, with a per-event cap because losses cluster within an event.

## Sealed prices frozen 2026-10-22 (no P&L computed)
`data/kalshi_mentions_prices_sealed.parquet` sha256 a25b80f2341b: 60,894 markets; candle at decision d24 35,188, d6 51,745
(6,766 recent markets via the live endpoint, 6,693 filled). Score only on "yes M3": `sealed --confirm M3_d24 M3_d6`.

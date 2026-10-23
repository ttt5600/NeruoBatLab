# L19 K-FAV: Kalshi favourite side (88-97c), non-sports daily markets. DEV: both trials FAIL

Script `scripts/kalshi_fav_maker.py` (pre-registration and two amendments in the docstring, all written before any
decision-time price or P&L was looked at). Data: `data/kalshi_fav_markets.parquet` (census of settled markets),
`data/kalshi_fav_candles_dev.parquet` (DEV decision quotes + fill-window trade range). Ledger dataset
`kalshi_fav_maker`, 2 trials, bar p < 0.025. Scored 2026-10-22. SEALED (close >= 2025-07-01) never built, never scored.

## Design (short)
- Universe: Kalshi `frequency == daily` series outside Sports, minus every daily-HIGH temperature series (we already
  trade those) and test series. Amendment 1 added the daily-close event of the hourly-listed KXBTCD/KXETHD/KXBTC/KXETH
  (17:00 ET) and KXINXU/KXNASDAQ100U (16:00 ET). One event per series per day, at the series' scheduled close time.
  This also drops markets that closed early.
- Decision at close - 3 h, using the last hourly candle at or before then; spread <= 5c. Favoured side = the side
  whose ask is 88-97c.
- F1 taker: buy at the ask plus 0.07p(1-p). F2 maker: rest a bid 1c below the ask. It counts as filled if any trade in
  (T, close] printed at or below the bid. Maker fee 0.0175p(1-p) is charged on all dates.
- Cluster bootstrap by ET calendar day. A pass needed all of: ROI > 0, p < 0.025, worst quarter >= -3x the mean
  quarter, and >= 60% of quarters positive.
- Amendment 2 (build cost only): dropped the crypto RANGE ladders, which duplicate the above/below ladders on the same
  underlying and close. Above/below ladders with > 16 strikes are bisected on decision-time quotes. Checked on 3
  events: it found the same in-band rungs as a full fetch.

## Census (no outcomes)
Most non-sports daily series only exist after 2025-07, so they fall in the sealed period. DEV has 8,392 daily events
and 136k markets, mostly the S&P/Nasdaq range ladders (2022-25), FX ranges, WTI, the 10Y note, crypto (2024-11 on),
Spotify/app charts (2025) and rain. Economics (gas, TSA, etc.) is effectively empty in DEV (1 market).
A decision-time quote needs a candle in (close-6h, close-3h]. The archive emits candles only in periods with activity,
so illiquid series mostly drop out: EURUSD has 158 of 7,931, USDJPY 256 of 7,960.
Eligible markets: 4,419 across 919 days. By category: Financials 2,970 (91% on the NO side), Crypto 695,
Commodities 358, Other 282, Weather 114.

## Results (DEV, 2021-08..2025-06)
| trial | bets | ROI | units (1 contract) | 95% CI units | p | quarters + | worst Q / mean Q | verdict |
|---|---|---|---|---|---|---|---|---|
| F1 taker 88-97 | 4,419 | -0.32% | -13.1 | [-53.8, +26.7] | 0.716 | 7/16 | -7.97 / -0.82 | FAIL |
| F2 maker 1c below ask | 2,829 filled (64%) | -2.19% | -57.4 | [-99.1, -17.2] | 0.999 | 3/16 | -17.07 / -3.58 | FAIL |

By category, F1: Weather +1.51% on 114 bets (noise, n is tiny), Crypto 0.00%, Commodities -0.34%, Financials -0.32%,
Other -1.75%. F2: every category is negative (Other -0.30%, Crypto -1.45%, Financials -2.16%, Commodities -5.5%,
Weather -8.1%).
Sensitivities (these are not trials): F1 with the 1-contract fee rounded up to the cent loses -0.93%. F2 with zero
maker fee loses -2.06%. F2 with a strict trade-through fill loses -3.04%. F2 with the 1-contract rounded fee loses
-3.11%.

**Taker:** the favoured side wins 93.7% against a 94.0c average all-in cost. Before fees, the ask is about the true
probability, so there is no favourite bias left to buy at T-3h in these markets. The paper's "> 70c earns a small
positive return" does not show up here at the ask.

**Maker: this is adverse selection, the same lesson as the dead ladder-maker test.** Filled bids won 90.5%.
Unfilled ones won 99.3%. The breakeven win rate at the filled bids is 92.5%. A bid 1c under the ask gets filled when
the price is falling toward it, so it fills mostly on the days the favourite is about to lose. Capturing the spread
(~1c) is far smaller than the ~9-point drop in win rate. The same gap appears in every category with n > 100:
Financials 90.5 vs 99.6, Crypto 91.3 vs 100, Commodities 87.1 vs 100.

## Verdict
DEAD as specified. **There is no sealed ask.** The maker variant is clearly negative, not borderline. Do not re-run
with a different band, decision time or category without a new mechanism, such as a fill model with queue position
from real order-book data. Choosing the Weather F1 cell (+1.5% on 114) after seeing it would be a post-hoc trial on
n = 114, not an edge.

Caveats: hourly candles only give trade min/max, not the order of events, so "a print at or below the bid" is an
approximation and is optimistic about queue position. Even so, F2 loses. DEV crypto and index above/below data only
starts at 2024-11 and 2025-03. Most non-sports daily series (gas, low temperatures, FX "AD" series) only exist in the
sealed period, so this test says nothing about them.

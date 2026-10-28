# Are small / long-tail prediction markets less efficient? (exploratory scan, 2026-10-27)

EXPLORATORY ONLY. This is a hypothesis generator, not a backtest. No orders were placed; all data is read-only public
data (Polymarket gamma + CLOB `prices-history`, Kalshi public trade-api). Many strata were examined, roughly 60 per venue
and horizon times 3 rules, so several "CI excludes 0" cells are expected by chance alone. Nothing below is an edge.

Code: `scripts/smallmarket_calibration.py` (pull + reprice), `scripts/smallmarket_analyze.py` (tables),
`scripts/smallmarket_exante.py` (ex-ante activity check), `scripts/smallmarket_live_overround.py` (live-book check).
Outputs: `data/smallmarket/` (tables.json, exante.json, obs.parquet, live_overround.parquet; gitignored).

## Bottom line

1. **Q1 (small = less efficient?): mostly no, once liquidity is measured before the decision.** Polymarket looks
   strongly graded by lifetime volume. Longshots are overpriced by +2.6 to +3.5 pp in markets with $0 to 10k of
   volume, and *under*priced (-3 to -9 pp) above $100k. That gradient is look-ahead: lifetime volume is measured after
   the fact, and upsets attract volume after they happen. In the >$100k bins, 5-15c longshots won 15-19% at 1d/7d against
   ~9% priced. The pre-decision measures are flat:
   - Kalshi contracts traded in the 24 h before t (exact, ex-ante): no monotone pattern.
   - The Polymarket activity proxy (price changes in the prior 72 h): about +1.4 to +2.0 pp in every bin.

   What does get worse as liquidity falls is the **spread**. Kalshi median quoted spread is 3-5c at >1k contracts/24h
   and 12-14c at zero volume. So small markets are net *worse* to take, not better.
2. **The favourite-longshot bias is real and roughly venue-wide, and it is about the size of the cost to cross.** At
   the mid, buying the >=80c side earns +1.8 to +4.5% (Polymarket +1.8-2.3%, Kalshi +2.3-4.5%). After the quoted spread
   and fees it nets about 0: Polymarket -0.7..0.0%, Kalshi -1.1..-0.1%, every 95% CI straddles 0. Buying the
   longshot side loses 34-59% net.
3. **Q2 (multi-outcome sums):** Polymarket mutually exclusive (negRisk) event prices sum to a median of 1.01-1.02.
   38-47% of events sum above 1.03, and the top decile sums to 1.26-1.43. In historical display prices, the excess
   sits in **20-50c legs of events with sum > 1.10**. Those legs priced 0.33-0.40 settled YES 19-25% of the time
   (non-weather), which backtests at +10 to +14% net with a proxy spread. **This is a display-price artifact.** On
   today's live books, 65% of such legs would cost more than 5c above the assumed price to buy NO. Their median real
   spread is 63c and the median real NO cost is $0.989. Basket arbitrage: Kalshi 0 events after fees (124 checked at
   1d); Polymarket live, 2 of 237 events have bids summing above 1 before fees.
4. **Q3 (worst-calibrated categories):** **Mentions** on both venues: YES is overpriced 4-7 pp and calibration error
   above the sampling-noise level is +0.02 to +0.05. Next is **Polymarket culture/awards/app-chart** markets (YES +4-8 pp,
   excess +0.035 to +0.062), and Polymarket sports at 7d (+0.064, mostly futures-style legs). Best calibrated:
   Polymarket weather at 1h, sports at 1h, and politics at 7d (excess about 0). This repeats the dead Kalshi
   mention-NO result (S4/M3: the dev edge shrank to nothing as volume grew 10x).

## Data and design (fixed before any outcome was tabulated)

| | Polymarket | Kalshi |
|---|---|---|
| Universe | 92 random days, 2025-05-01..2026-04-30; all closed events with endDate that day (198k markets) | 1,796 random series (up to 220 per category, hourly/15-min series excluded), all settled markets with close in the same window (107k markets) |
| Priced sample | 26,603 markets / 4,601 events. Whole events are kept (needed for sums), events of up to 40 legs, about 4,000 markets per category. Crypto up/down capped at 400 | 8,133 markets / 1,401 events. Up to 6 events per series; whole event if 15 legs or fewer, else 3 legs; up to 3,500 per category |
| Anchor S | min(endDate, gameStartTime). Without the game start, "1 h before endDate" was often after the game | latest close_time among the event's settled markets (the scheduled close). A leg that closed early (mention said) only counts if still open at t |
| Price at t = S - h, h = 7d / 1d / 1h | last CLOB display price within 6 h before t (the mid, or the last trade if the spread is over 10c) | mid of the last hourly candle's yes_bid/yes_ask within 6 h, two-sided only |
| Spread | **proxy**: median live spread today by volume bin and extremeness (2c at $1-10k; 28c under $1k) | **actual** quoted bid/ask |
| Fees | current market flags: crypto 0.07 x p(1-p), sports 0.05 x p(1-p) (docs.polymarket.com/trading/fees 10-27); other categories 0 | 0.07 x fee_multiplier x p(1-p) |
| Liquidity | lifetime volumeNum (**ex-post**) plus an ex-ante proxy: price changes in the prior 72 h | lifetime volume plus **contracts traded in the 24 h before t** (ex-ante) |

- Observations: 48,977 market-horizon rows in total (Polymarket 6,101 / 16,264 / 19,225 at 7d / 1d / 1h; Kalshi
  1,908 / 3,766 / 1,713).
- CIs: 95%, bootstrap over events (300 draws). They do not cluster by day or series, so they are too narrow for
  weather and mention days.
- `ls_gap` = mean longshot-side price minus longshot win rate, over longshot prices 1-20c. Positive means longshots
  are overpriced.
- Rules (one bet per market-horizon, taker, 1 unit of cost):
  - **FAV**: buy the side with mid in [0.80, 0.98].
  - **LONG**: buy the side with mid in [0.02, 0.20].
  - **NO_LONG**: YES/NO markets only; buy NO when YES is in [0.02, 0.20].
  - `net` = at the ask plus fee; `mid` = at the mid with no costs.
- **HOLDOUT**: 2026-05-01..2026-10-20 was never priced. Only counts were taken, with the outcome column dropped. It is
  reserved for pre-registered tests.

## Q1: liquidity

Polymarket, 1d horizon, binned by lifetime volume (ex-post):

| volume | n / events | ls_gap | FAV net [CI] | FAV mid | NO_LONG net [CI] |
|---|---|---|---|---|---|
| < $1k (median $0) | 4,941 / 1,201 | +0.026 [+0.011, +0.041] | -1.5% [-3.6, +0.2] | +3.1% | +0.4% [-1.0, +1.7] |
| $1k-10k | 6,036 / 1,615 | +0.035 [+0.026, +0.043] | +2.6% [+1.4, +3.6] | +4.3% | +2.7% [+1.6, +3.8] |
| $10k-100k | 4,131 / 1,584 | +0.003 [-0.009, +0.014] | -1.0% [-2.8, +0.6] | +0.1% | -0.2% |
| $100k-1M | 944 / 493 | -0.027 [-0.058, +0.006] | -3.9% | -3.3% | -5.1% [-11.0, -0.4] |
| > $1M | 212 / 105 | -0.088 [-0.192, +0.004] | -13.1% | -12.9% | - |

The same pattern holds at 7d and 1h. The ex-ante replacement, Polymarket prior-72h price-change bins at 1d (0 / 1-3 /
4-10 / 11-30 / >30 changes), gives ls_gap +0.046 [-0.001, +0.087] / +0.023 / +0.020 / +0.019 / +0.018. Every FAV net
CI includes 0 (+2.0%, +0.6%, 0.0%, +0.2%, -0.4%). The proxy is uncorrelated with log lifetime volume (r = 0.005), so it
measures activity, not size. That makes it a weak test. **The clean ex-ante test is Kalshi**:

| Kalshi 24h-before-t volume (1d horizon) | n | ls_gap | median spread | FAV net | FAV mid |
|---|---|---|---|---|---|
| 0 | 690 | +0.015 [-0.022, +0.049] | 12c | -3.5% [-7.6, +0.6] | +1.7% |
| 1-10 | 243 | +0.042 [-0.000, +0.084] | 8c | +0.2% | +4.8% |
| 11-100 | 677 | +0.025 [-0.008, +0.056] | 7c | -1.3% | +2.9% |
| 101-1k | 1,406 | +0.034 [+0.014, +0.051] | 4c | +0.6% | +3.9% |
| > 1k | 750 | +0.014 [-0.015, +0.038] | 3c | -1.3% | +1.5% |

At 7d the same bins read +0.031 / +0.007 / +0.009 / -0.003 / +0.033, so there is no gradient. Binned by Kalshi
*lifetime* volume, the 101-1k-contract bin shows ls_gap +0.048..+0.073 with a 9-18c spread, again with net CIs
including 0.

**Lesson:** any "small markets are mispriced" study that stratifies on total or lifetime volume is contaminated. It
needs volume up to the decision time (Polymarket `data-api` trades, Kalshi candles).

## Q2: multi-outcome sums

| | events | median S | S 10th / 90th pct | share S > 1.03 | Brier raw -> normalised (S > 1.10 events) |
|---|---|---|---|---|---|
| Polymarket 7d | 329 | 1.025 | 0.985 / 1.425 | 47% | 0.164 -> 0.149 |
| Polymarket 1d | 844 | 1.020 | 0.987 / 1.272 | 42% | 0.097 -> 0.091 |
| Polymarket 1h | 787 | 1.012 | 0.980 / 1.261 | 38% | 0.104 -> 0.094 |
| Kalshi 1d (mids) | 110 | 1.035 | 0.98 / 1.41 | 50% | 0.144 -> 0.135 |

- Normalising by S does not help events with S <= 1.10.
- S rises with the number of legs: Polymarket 11+ legs 1.03-1.09 vs 2-4 legs ~1.00.
- By category: Polymarket weather 1.03-1.05, finance 1.02-1.05, sports and politics ~1.00-1.01 (at 7d, sports
  futures 1.09).

Leg-level, Polymarket, buying NO on legs with YES at 20-50c:

| horizon | S <= 1.03 | S > 1.10, weather | S > 1.10, non-weather |
|---|---|---|---|
| 1d | -7.9% [-10.8, -5.1] | +3.7% [-0.3, +8.0] (346 legs) | +10.1% [+3.0, +17.5] (157 legs / 60 events; priced .356, won .185) |
| 1h | -10.2% | +2.5% | +13.8% [+7.8, +19.2] (203 / 76) |
| 7d | -8.0% | - | +11.2% [+7.1, +15.3] (305 / 106) |

**Live tradeability check (10-27 books, 237 active negRisk events, 129 target legs in 40 events):**

- median spread 0.63; median real NO cost 0.989 against an assumed 0.634;
- 65% cost more than 5c above the assumption;
- only **7 legs** (6 events) had a spread of 5c or less.

The historical "edge" is the display price of empty or wide books. Kalshi asks summed to a median of 1.15 and bids
to 0.89 at 1d, and there was 0 basket arbitrage after fees.

## Q3: categories (ECE minus the ECE expected from sampling noise when prices are exactly right; mean p - y)

| | Polymarket 1d | Polymarket 1h | Polymarket 7d | Kalshi 1d | Kalshi 1h |
|---|---|---|---|---|---|
| mentions | +0.043, YES +6.8 pp | +0.053, +7.2 pp | +0.013, +4.8 pp | +0.044, +5.7 pp | +0.022, +3.9 pp |
| culture/tech (PM) or entertainment (K) | +0.035, +4.3 pp | +0.040, +5.2 pp | +0.062, +7.7 pp | 0.000, +1.1 pp | +0.031 (n 106) |
| sports | +0.014 | -0.003 | +0.064, +7.5 pp | +0.025 | +0.003 |
| finance/econ | +0.013 | +0.018 | +0.009 | +0.007 | +0.037 (n 139) |
| politics/world | +0.011 | +0.007 | +0.002 | n 46 | n 32 |
| weather | +0.007 | 0.000 | (n 80) | 0.000 | +0.003 |
| esports (PM) | +0.010 | +0.022 | +0.008 | - | - |

Buying NO in the two worst Polymarket categories, YES at 2-35c, zero-fee, proxy spread:

- mentions 1h: +3.7% [+0.8, +6.7], 654 legs / 124 events; 1d: +2.6% [-1.5, +6.9].
- culture 1d: +1.9% [+0.2, +3.8], 595 / 192; 1h: +2.5% [-0.2, +4.8].
- The 35-65c culture band shows mid +27 to +39% with a 28c spread. These are thin books and almost certainly the same
  display artifact as in Q2.
- The culture sample is concentrated: about 30% "what will be #1" app/song charts and about 25% award shows (Critics
  Choice, Spirit Awards). Mentions are about 45% "what will Trump say", plus Powell and Starmer.

Kalshi mentions, NO on YES at 2-35c: +3.2% net at 1d and 1h (CIs touch 0), against mid +10%. The 6-7c spread eats
two thirds. M3 sealed already showed this family shrinking: d24 +0.45%, p .22.

## Hypotheses for pre-registration (ranked)

**H1. Polymarket mention + culture/award/chart YES-overpricing (zero-fee categories).** Buy NO at the *real* ask when
YES display is 2-35c at T-1h (mentions, event anchor) or T-1d (culture). Pass: ROI > 0 at the real ask, one-sided
p < .05, events clustered.
- Dev: +2 to +4% net at a proxy 2-3.5c spread.
- Per-bet SD about 0.38 per unit. For a +2% edge: about 2,200 legs times a design effect of about 2, so **about 4,500
  legs**.
- Supply: about 55 candidate legs/day (holdout census: mentions 62 + culture 102 markets/day, ~1/3 in band), about
  14 events/day, so roughly 80-90 days forward.
- The untouched holdout (2026-05..10, about 9k legs) can settle the *calibration* claim at the mid. It cannot settle
  tradeability, which needs forward logging of real books.
- Capacity: median lifetime volume is ~$4k (mentions) and ~$0 (culture). Realistically $20-200 per leg at the touch,
  so about $1-3k/day turnover and **tens of $/day at best**.
- Prior: weak. The Kalshi twin (M3) died as volume grew.

**H2. Venue-wide longshot bias harvested as MAKER (not taker).** The bias of +1.5 to +4 pp is consistently present
(Polymarket ls_gap 1d +0.019 [+0.012, +0.025], Kalshi +0.026 [+0.013, +0.038]) and about equals the crossing cost. Rule:
in Polymarket zero-fee categories, rest a NO bid 1c inside the touch on YES at 2-20c from T-1d to close.
- Test by replaying the public trade tape (`data-api.polymarket.com/trades`) on the HOLDOUT window, pre-registered.
  Fill only if a trade prints through the bid. Report adverse selection (win rate when filled vs when unfilled).
- Needs about 11k fills for a +1% edge (SD 0.30, design effect 2). Supply is about 150 candidates/day, so 2-3 months of
  tape.
- Prior: low. Kalshi K-FAV maker died to adverse selection: filled win rate 90.5% vs 99.3% unfilled.
- Capacity: bounded by taker flow into longshot YES, perhaps $100s/day.

**H3. Overround legs with a tight real book.** In negRisk events with display sum > 1.10, a 20-50c leg whose *live*
spread is 5c or less is overpriced. Buy NO at the real ask.
- Dev effect (+10-14% net) is mostly an artifact. The residual claim is untested.
- Supply today: **7 legs in 6 events**. At a +5% edge, about 1,600 legs are needed, which is more than a year of
  forward logging.
- Capacity: tiny. Log-only; do not prioritise.

**H0 (kill or null, worth recording).** Ex-ante liquidity does not predict calibration error. Kalshi 24h-volume bins
show no gradient. Re-test on the Polymarket holdout with pre-decision traded volume from data-api. If the
lifetime-volume gradient disappears there too, close "small markets are mispriced" as a theme. What remains is the
spread, and it scales with smallness.

## Caveats

- Polymarket prices are CLOB display prices, not trades or books. The proxy spread is a median across *today's* books,
  and the live check shows it is far too optimistic for thin legs. All Polymarket "net" numbers are **upper bounds**.
- Fee flags are today's. In the dev window, crypto and sports fees were mostly not yet live, so those numbers are
  conservative.
- The Kalshi sample is mention-heavy: 3,440 of 8,133 markets, because most sampled sports and crypto series fell
  outside the window or were capped. Kalshi politics, sci/tech and crypto strata have n < 250.
- `can_close_early` is true for about 100% of Kalshi markets, so actual early closes were flagged instead. Results
  excluding them ("no_early") are about the same.
- Polymarket endDate can be edited by the operator. Early-YES "by date" markets drop out of a horizon once closed. That
  is correct ex-ante behaviour, but it changes the mix by horizon.

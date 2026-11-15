# options_implied_tails -- Kalshi DAILY crypto / index ladders vs the options market's distribution (FUND_BOOK 3c lead)

Agent: options_implied_tails (research subagent, 2026-11-15). Dev only, no orders, no git.
Script: `scripts/options_implied_tails.py` (build -> frame). edgelab dataset `kalshi_opt_tails`
(rule module `edgelab/rules/opt_tails.py`), family `kalshi_data_ladders`.

## Why this is not a redo
- Q012 (DEAD-for-taking): HOURLY KXBTCD/ETHD in the last 10-30 min vs a spot + realised-vol model. Model lost to the mid
  on Brier; mid shows a 1-2c longshot premium on both tails that spread + fee eat. Far rungs often lack a bid.
- kalshi_fav_maker F1/F2 (DEAD): unconditional 88-97c favourite at close-3h on the same daily ladders, taker ~0, maker
  adverse-selected (filled 90.5% vs unfilled 99.3% win).
- Q016 commodities, Q009 SPY VRP: dead (different instruments).
- New here: (a) a DAILY horizon (6 h / 5 h to settle) where tails can be stale for hours; (b) the forecaster is the
  options market's risk-neutral distribution, not a realised-vol model; (c) conditioning on a gap between the two,
  not an unconditional favourite band. If this fails, "options-implied vs Kalshi daily tails" goes into FUND_BOOK 3d.

## Data feasibility (checked 2026-11-15, outcome-free)
- Deribit: `history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time` serves expired-instrument option
  trades with `iv`, `mark_price` (Deribit's smile-fitted mark, BTC units), `index_price`, timestamp (ms). The main host
  returns nothing for 2025 dates; the history host works (582 BTC option trades on 2025-01-01 12:00-14:00Z).
- SPX/NDX: NO free historical intraday option chains (CBOE delayed = current only; DoltHub SPY = MWF, 3 expiries >= 2 wk,
  see q009). Free proxy: Yahoo hourly ^VIX1D (1-day model-free implied vol of SPX, a variance-swap-type level that
  includes the put wing), ^VIX, ^VXN, ^SPX, ^NDX, available 2024-01-29 .. now. This gives the options-implied LEVEL of
  1-day variance, not the smile shape. Stated limitation.
- Kalshi: fee_type quadratic on KXBTCD, KXETHD, KXINX, KXNASDAQ100, KXINXU, KXNASDAQ100U (taker 0.07p(1-p), maker 0).
  Historical cutoff (/historical/cutoff) market_settled_ts 2026-09-16.

## PRE-REGISTRATION (written 2026-11-15 BEFORE any Kalshi quote, candle or outcome for these markets was fetched or
## joined; the only outcome-bearing file that exists is data/kalshi_fav_markets.parquet from the F1/F2 study, of which
## only series/close-time counts were read, never `result`)

### Universe and split
- Crypto: KXBTCD, KXETHD, the 17:00 ET daily-close event of each day (event ticker suffix `17`).
- Index: KXINX, KXNASDAQ100 (16:00 ET range ladders incl. their greater/less tail rungs) and KXINXU, KXNASDAQ100U
  (16:00 ET above/below), the event settling at 16:00 ET on its date. Index events need VIX1D -> close date >= 2024-02-01.
- Split by ET close date (vault `kalshi_opt_tails`): DEV <= 2026-04-10 (same cut as Q012); SEALED 2026-04-11..2026-10-31
  (never fetched here); FORWARD >= 2026-11-16. DEV crypto starts with the first daily events (2024-10-24).
- Outcomes (`result`) come in the same market-list response; the builder writes them to a hidden column only the vault
  resolver reads.

### Decision time and Kalshi quote
- T = 11:00 ET on the close date (crypto: close - 6 h; index: close - 5 h).
- Quote = last HOURLY candle (historical or live markets candlesticks, period 60) with end_period_ts <= T, window
  [open, T]; yes_bid / yes_ask = `close`. No candle -> no quote. Spread filter yes_ask - yes_bid <= 0.05.
- CLV mark = same side's ask on the last candle ending <= close - 1 h.

### Options-implied P(YES) (frozen now)
Crypto (Deribit, risk-neutral, smile):
- Trades of that currency's options in [T - 60 min, T) from the history host; target expiry = the first Deribit expiry
  (08:00 UTC) at or after the Kalshi close (normally next day 08:00 UTC, tau_D ~ 15-16 h at T).
- For each instrument of the target expiry, its LAST trade in the window: F = index_price of that trade, mark USD =
  mark_price x index_price; invert Black-76 (r = 0) for the mark IV; keep OTM only (calls K >= F, puts K < F).
  Smile points (x = ln(K/F_T), sigma), F_T = index_price of the latest trade (any instrument) before T. Need >= 5 points
  with >= 1 on each side of x = 0, else no decision for that day.
- Kalshi horizon tau_K = close - T (calendar years, 365 d). A Kalshi strike K maps to x_D = ln(K/F_T) sqrt(tau_D/tau_K)
  (smile is a function of standardised moneyness). sigma_K(K) = linear interpolation of the smile at x_D; x_D outside
  [min x, max x] of the observed points -> NO decision on that rung (no extrapolation: wing vol is unknown).
- P(S_close > K) = -dC/dK by central difference (h = 0.1% K) of Black-76 calls with sigma_K(K), tau_K, F_T.
  between [a, b]: P(> a) - P(> b) (needs both ends mapped); less (< cap): 1 - P(> cap).
Index (VIX1D level, no smile):
- sigma_1d = VIX1D / 100 / sqrt(252) (one full session); NDX: sigma_1d x VXN / VIX. Inputs = close of the Yahoo hourly
  bar starting 09:30 ET (ends 10:30 ET <= T). S = ^SPX / ^NDX close of the same bar.
- Remaining variance over 11:00-16:00 = sigma_1d^2 x 5 / 6.5 (uniform intraday; stated). log(S_close / S) =
  sd x Z, Z = Student-t(4) scaled to unit variance (fat-tail shape, since VIX1D gives only the level), zero drift.
Risk-neutral vs physical: NO adjustment. Option-implied variance carries a variance risk premium (implied > realised on
average), so RN tails are wider than physical ones -> P_RN(tail) is treated as an UPPER bound for the tail side; this is
conservative for a rule that only bets AGAINST tails. Counter-risk, stated: crypto intraday seasonality (US hours are
more volatile than the overnight part of tau_D) makes sqrt-time scaling understate 6-h variance; the VRP is assumed to
cover it. Sensitivity (not a trial): sigma x 1.25.

### Trials (k = 4; one side per market, 1 contract)
Tail side = the side (YES or NO) with options-implied probability p_T <= 0.10; F = the other side.
Range brackets: the tail side is YES when p_YES <= 0.10 (the task's "NO on brackets with P(YES) < k").
- OT1 CRYPTO_TAKER: buy F at its ask if edge = (1 - p_T) - ask_F - 0.07 ask_F (1 - ask_F) >= 0.03 and ask_F in [0.50, 0.99].
- OT2 CRYPTO_MAKER: post F at q = min(floor_cent(mid_F), ask_F - 0.01), q in [0.50, 0.99], if (1 - p_T) - q >= 0.03.
  Filled iff some hourly candle in (T, close] shows an F-side TRADE price <= q - 0.01 (traded through by >= 1c; F = NO
  -> yes trade high >= 1 - q + 0.01). Maker fee 0. Unfilled -> no bet. Reported: filled vs unfilled win rates.
- OT3 INDEX_TAKER: OT1 rule on the index universe.
- OT4 INDEX_MAKER: OT2 rule on the index universe.
Cluster = (asset class, ET close date) -> BTC and ETH on a day share one cluster; SPX and NDX on a day share one.

### Bar (edgelab family kalshi_data_ladders, 44 trials before this -> 48 after -> p < ~1.04e-3)
PASS on DEV -> sealed ask to the user (nothing more): edgelab dev verdict pass (ROI > 0 at the family bar, halves,
concentration, CLV) and n >= min_n = 150 bets. Underpowered (< 150 bets) -> no verdict.
Kill: all four fail or are underpowered with ROI CI upper < +2% -> DEAD (FUND_BOOK 3d).
Too-good guard: ROI > +10% on > 150 bets = bug until rows are inspected.

### Power (written now)
A ~93c buy has per-bet sd ~0.25; detecting +2% ROI at one-sided p ~1e-3 (z 3.1) needs ~(3.1 x 0.25 / 0.0186)^2 ~ 1,700
independent bets; with clustering more. So only a large effect (>= +4-5%) can pass on DEV. Written down so a
null is read as "no large edge", not "no edge".

## Log
- 11-15 Trials registered in edgelab BEFORE any Kalshi quote/outcome fetch: ot1_crypto_taker-18d38e5c,
  ot2_crypto_maker-6a2e6488, ot3_index_taker-58e4b942, ot4_index_maker-eeac4f27 (family kalshi_data_ladders now 48 ->
  bar p < 1.04e-3). Vault dataset kalshi_opt_tails added (dev 2024-02-01..2026-04-10, sealed 04-11..10-31, fwd 11-16+).
- 11-15 Implementation amendment (outcome-free, input plumbing): Yahoo ^VIX hourly bars start on the hour (CT feed), so
  the NDX VXN/VIX ratio uses the ^VIX bar starting 09:00 ET (ends 10:00 <= T); VIX1D/VXN/SPX/NDX use the 09:30 bar.
  Yahoo 1h history via period=730d reaches 2024-01-29 (start/end form refuses > 730 days).
- 11-15 Market listing: series-wide /historical/markets for hourly KXBTCD was too slow (millions of hourly markets);
  switched to one /historical/markets?event_ticker= lookup per (series, date), legacy INX-/NASDAQ100- names as fallback.
- 11-15 Deribit sanity (2025-03-12, T=11:00 ET): 1,378 BTC option trades in [T-60m,T), 22 OTM smile points for the
  13-MAR 08:00Z expiry (tau_D 17 h), ATM mark IV ~0.72, put wing 1.27 at 72k; implied P(above +5%) = 0.0025, P(above
  F) = 0.566 (put skew pushes P(>F) above 0.5 -- expected sign).
- 11-15 AMENDMENT 2 (request budget; written after market LISTS only, before any candle/quote is fetched): KXBTCD daily
  events carry ~43 rungs each, so "every tail candidate" would be ~60k per-market candle requests (~5+ h on the shared
  3.3 req/s throttle the live bots also use). Candles are fetched only for the 3 tail candidates CLOSEST to the money on
  each side of every event (rank by p_T descending within (event, tail side); between-brackets: p_YES descending).
  Consequence, stated: the deepest rungs (smallest p_T) are not tested; E4-style stale quotes on deep rungs are out of
  scope here. Rule text and trials unchanged (the frame simply holds fewer rows).
- 11-15 Lead request (literature scout, pm_literature_scout.md #3: mispricing largest AT LISTING and on WEEKENDS): the
  prereg was already FROZEN (trials registered) when this arrived, so listing age (hours from market open to T) and
  weekend (Sat/Sun close date) are reported as DESCRIPTIVE strata only, not trials. Note the design limits them: the
  decision is fixed at 11:00 ET, so crypto listing age at T is near-constant (daily events open the prior evening) and
  "at listing" is NOT tested here; a listing-time test needs its own pre-registered trial (decision = open + m minutes).
- 11-15 Outcome-free counts (2,255 BTC markets fetched at the time): 199 OT1 bets, 732 OT2 posts -> crypto projected ~450.
- 11-15 Candles: 13,717 tail-candidate markets (crypto 5,066; index 8,651); quote at T for 5,147 of 5,066+ crypto
  (all crypto fetched before the crypto evaluations below; index still fetching).

## Results -- crypto (edgelab dev, the one run each; lab/edgelab_registry.jsonl)
| trial | n | net ROI | 95% CI (cluster = crypto day) | p | bar | halves | CLV | verdict |
|---|---|---|---|---|---|---|---|---|
| OT1 crypto taker | 387 | +4.60% | [+1.07%, +7.62%] | 0.0058 | 1.04e-3 | +7.35% / +1.86% | +2.94c (p_neg .99) | FAIL (p > bar) |
| OT2 crypto maker | 830 filled | -1.64% | [-5.20%, +1.68%] | 0.83 | 1.04e-3 | -4.25% / +0.97% | -- | FAIL |
OT1 rows inspected (scripts inline, data/opt_tails/frame.parquet): 387 bets = 295 BTC (+3.95%) / 92 ETH (+6.27%);
NO-side (fade an up-tail) 250 bets +3.94%, YES-side (fade a down-tail) 137 bets +5.54%. Mean cost 0.898, win 93.8%,
mean implied tail prob 5.4% vs realised tail 6.2% vs Kalshi-implied ~10% -> the options smile is close to calibrated
(slightly thin, as the seasonality caveat predicted) while Kalshi's tail bid is rich. Quote age at T: 0 h (median and max).
By quarter (noisy, not a test): 24Q4 +4.7% (79), 25Q1 +8.3% (95), 25Q2 +11.8% (23), 25Q3 -9.3% (20), 25Q4 -3.4% (49),
26Q1 +7.2% (114), 26Q2 -21.5% (7). Median crypto quoted spread fell 5c (2025) -> 3c (26Q1) -> 2c (26Q2).
DESCRIPTIVE dose-response (not a trial; all quoted crypto frame rows, spread <= 5c, F ask in [.50,.99]), ROI by
options edge = pF - ask - fee: <= -3c: -6.9% (59) | -3..0c: -1.8% (978) | 0..3c: +0.2% (1,456) | 3..6c: +2.8% (297) |
> 6c: +11.5% (68). Monotone: the options-implied gap ranks Kalshi's daily tail mispricing.
Depth (LIVE book today, 2026-11-15, KXBTCD-26NOV1517, read-only GET /orderbook): rungs with NO ask 0.85-0.97 show 1c
spreads and 300-10,000 contracts at the top level (e.g. T67249.99 yes bid .05 x 1,423 -> NO ask .95). Historical
depth is not available; today's book is tighter (1c) than the DEV median (3-5c), i.e. the market has gotten more
efficient since DEV, which cuts against the edge persisting.

## Results -- index (edgelab dev, one run each, after all 13,717 candle files existed; frame 13,659 quoted)
| trial | n | net ROI | 95% CI (cluster = index day) | p | halves | verdict |
|---|---|---|---|---|---|---|
| OT3 index taker | 541 | -4.54% | [-10.16%, +0.74%] | 0.955 | -6.38% / -2.71% | FAIL |
| OT4 index maker | 621 filled | -9.27% | [-14.60%, -4.47%] | 1.00 | -10.63% / -7.91% | FAIL |
Why the index arm fails although its implied p is calibrated in aggregate (table below): the VIX1D model reads S at
10:30 ET (Yahoo hourly bar close) while the Kalshi quote is at 11:00 ET, so up to 30 min of index movement is known to
Kalshi and not to the model; selecting on "model disagrees with Kalshi" then selects the model's staleness. The crypto
arm used F_T = the Deribit index at the last option trade before T (seconds old). Pre-registered design flaw; the fail
stands.

## Descriptives (scripts/options_implied_tails_desc.py -> results/options_implied_tails/desc.json)
- Maker adverse selection again: crypto filled win 88.4% vs unfilled 99.8% (830 filled of 1,450 posted); index 79.1%
  vs 99.8% (621 of 1,030).
- sigma x 1.25 sensitivity: crypto taker 89 bets +8.75% [+3.86%, +12.49%] (fewer, bigger gaps); index 282 bets -4.26%.
- Tail calibration (quoted tail candidates, spread <= 5c; implied pT / Kalshi mid / realised, n):
  crypto (0,.005]: .0032 / .0246 / .0060 (168) | (.005,.02]: .0124 / .0341 / .0145 (963) | (.02,.05]: .0336 / .0590 /
  .0459 (1,350) | (.05,.10]: .0731 / .1063 / .0919 (1,056). Implied is a bit thin, Kalshi mid clearly rich; the truth
  sits between, nearer the options. Brier implied .04568 vs mid .04617 (n 3,549).
  index (0,.005]: .0034 / .0105 / .0013 (746) | (.05,.10]: .0718 / .0826 / .0708 (1,836); Brier implied .02712 vs mid
  .02573 -> on index the mid wins (staleness above).
- Lead-requested strata (DESCRIPTIVE ONLY; prereg was frozen before the request): crypto taker weekend 25 bets +6.97%
  [-2.5, +13.3] vs weekday 362 +4.33% [+0.7, +7.5]. Listing age at T is 19 h for most crypto events (quartiles
  19/19/19 h), so "at listing" is untestable with a fixed 11:00 ET decision; age > 19 h 74 bets +8.6% [+4.2, +11.8] vs
  <= 19 h 313 +3.5% [-0.5, +7.2]. Crypto maker weekend 41 bets +8.1% vs weekday -2.2%. All small n, not trials.

## Verdict
OT1 crypto taker = LEAD, not a pass: +4.60% [+1.07%, +7.62%] on 387 bets / 195 crypto-days, p 0.0058 vs family bar
1.04e-3; both halves positive, CLV +2.9c, rows inspected (not too good), dose-response monotone, sigma x 1.25 stronger.
Under the prereg there is NO sealed ask (dev did not pass the bar). OT2, OT3, OT4 FAIL (OT2/OT4 maker adverse
selection). Index: dead for this VIX1D-level model; a fair index test needs a fresh spot at T and a real 0DTE smile
(no free historical source).
PROPOSAL (lead / user decides; nothing done): pre-register OT1 as a FORWARD paper trial (vault forward window from
2026-11-16 already set): daily 11:00 ET, Deribit live smile (public API), KXBTCD/KXETHD 17:00 event, same rule, logger
only. Against it: live spreads are now 1c (DEV median 3-5c) and 2025Q3-Q4 were negative, so the edge may have been
competed away; forward data is the honest arbiter. The scout's "at listing" claim needs its own pre-registered trial
(decision = event open + 15 min).

## Learnings
### Options-implied tail probs beat the Kalshi mid on daily crypto ladders
Deribit smile (first expiry after the Kalshi close, sqrt-time standardised-moneyness map): realised tail .0919 vs
implied .0731 vs Kalshi mid .1063 (1,056 quotes, p in (.05,.10]); Brier .04568 vs .04617; taker fade of gaps >= 3c
+4.60% [+1.07, +7.62] on 387 DEV bets (p .0058, fails the 48-trial family bar 1.04e-3). Source: this file,
results/options_implied_tails/desc.json. First Kalshi crypto model here that beats the mid (Q012's realised-vol model did not).

### A model input older than the market quote turns a gap rule into a staleness selector
OT3 used Yahoo's 10:30 ET index close against an 11:00 ET Kalshi quote: aggregate calibration fine, but the selected
gaps lost -4.54% [-10.2, +0.7] (541 bets). Any model-vs-market gap rule needs inputs at least as fresh as the quote.

### Deribit expired-option history is on history.deribit.com, not www
www.deribit.com get_last_trades_by_currency_and_time returns empty for 2025; history.deribit.com returns full trades
with mark_price, iv and index_price (582 BTC option trades in 2 h on 2025-01-01). Free, no key.

### Maker bids on Kalshi tail ladders are adverse-selected again
Crypto filled 88.4% vs unfilled 99.8% win; index 79.1% vs 99.8%: the kalshi_fav_maker F2 lesson repeats.

# altdata_kpi -- do the new alt-data KPI Kalshi series have the E4 shape? (agent altdata_kpi, 2026-11-18)

Question: KXJBLU, Carbon Arc retailer foot traffic / card spend / TikTok Shop series, KXSPRLVLDATE (+ sibling families):
does each settle on a published number that a FREE public source nowcasts better than the Kalshi price, with enough
events to test? Rank; pre-register + dev (or a forward spec) for the best 1-2.

## Census (read-only Kalshi public API, 2026-11-18)
Scripts: results/altdata_kpi/pull_series.py (16,702 series -> series_all.json), pull_markets.py (families.json ->
markets_<fam>.json). Families by settlement source:
- **Carbon Arc** (168 series; source "Carbon Arc ... carbonarc.co/prisms"): every target series is here
  (DGFT/DDSFT/FIVEFT/BJFT foot traffic, ALDI/APPLEBEES/CHILIS CC, RHODE/RARE/REM TTS, CELEBTTSARRAY).
  Settled events by close month: 2026-08 1, 09 3, 10 88, 11 104 (= 196); open events close 2026-12-03..08 (172).
  By kind (series, events, settled, markets, lifetime vol): CC 50/114/64/1271/3.69M; APP 23/63/41/648/5.25M;
  POS 23/59/36/603/5.61M; FT 24/47/23/529/1.64M; ADS 11/33/22/438/1.78M; ARRAY 14/23/9/266/1.03M; TTS 5/5/0/62/15k;
  daily CCD/FTD 15 series, 0 settled, vol ~180 total. Max 2 settled events per series.
  The ten named targets: each 1 open event, 0 settled; lifetime vol KXFIVEFT 14,829, KXRHODETTS 9,183, KXDDSFT 3,873,
  KXCELEBTTSARRAY 3,848, KXDGFT 2,674, KXBJFT 1,399, KXREMTTS 533, KXRARETTS 200, KXCHILISCC 25, KXAPPLEBEESCC 16, KXALDICC 5.
- **Fiscal.ai** company-KPI family (283 series, 463 events, 130 settled: 09 2, 10 58, 11 70). Includes KXJBLU (4 events:
  Q4-26 and Q1-27 load factor PLF + avg FARE, close 2027-05-26 / 2027-08-25; 0 settled). Biggest: KXBAA 566k, KXHOODA
  420k, KXBA 352k, KXCAKE 334k, KXLUV 307k (markets_fiscalai.json).
- **SPR** (5 series): KXSPRLVL weekly (vol 923k) already tested = SP1 DEAD (notes/phys_ladders.md, mechanical-cap rule);
  KXSPRLVLDATE 1 event / 3 markets (above 300 Mbbl before 2027-07-01) = a refill-policy forecast, n=1. Not E4 shape.

## Contract terms (rules_secondary, markets_carbonarc.json)
Carbon Arc FT/CC/APP: "finalized Carbon Arc values recorded at 11:59 p.m. ET on the 7th calendar day after the end of
the month"; YoY index, 100 = same month prior year; FT "200M+ devices ... The panel updates daily as new data arrives".
TTS: "published full-month ... revenue index ... recorded ... on the 5th calendar day after the end of that month. Later
corrections or revisions will not affect the recorded value. Month-to-date readings do not determine the full-month
outcome." => Carbon Arc subscribers see month-to-date readings; we do not. The prisms page is JS-only (WebFetch shows
only the title); Kalshi's launch post (news.kalshi.com/p/announcing-sector-kpi-markets-with-carbon-arc) says nothing on
free access; info_gap_census_2026-11-07.md row 12 already called it "paywalled ... none free". Treat as NOT free.
JetBlue (Fiscal.ai): settles on reported Q load factor / fare at earnings. Free nowcasts: none timely (BTS T-100 lags
~3 months; JetBlue publishes no monthly traffic). 2 events per KPI per quarter.

## The only free signal for Carbon Arc: last month's settled value
Kalshi publishes each settled event's expiration_value. YoY indices are persistent, so v(m-1) is a free nowcast of v(m)
that any trader can see -- the question is whether far rungs are still mispriced relative to it.

## PRE-REGISTRATION CA1 (written 2026-11-18 BEFORE reading any expiration_value, candle or result)
- Universe: Carbon Arc monthly YoY-index series (kinds CC, FT, APP, POS, ADS, ARRAY, TTS; not daily CCD/FTD), 'greater'
  strike rungs, events whose same-series previous-month event settled with a numeric expiration_value.
- Nowcast: mu = v_prev. Scale s = 1.4826 * MAD of r = ln(v_m / v_prev) over DEV pairs (frozen for sealed).
- Decision time T = max(event open_time, prev event close_time) + 24 h; price = last hourly candle <= T, side ask
  (YES ask; NO ask = 1 - YES bid). No candle with a two-sided quote -> no bet.
- Rule (taker): z = ln(K / mu) / s. If z >= 2: buy NO; if z <= -2: buy YES. Bet iff side ask in [0.50, 0.97]. 1 contract
  per rung; cluster = event.
- Fee: edgelab.fees kalshi_taker 0.07 P(1-P), rounding exact (check series fee_type; if 'quadratic_with_maker_fees' same
  taker rate).
- Split (time-based): DEV = events closing 2026-11-01..2026-11-30 (October data). SEALED = events closing 2026-12-01..
  2027-02-28 (Nov-Jan data; not yet settled -> forward only).
- Data gate (before any ROI): >= 40 dev pairs with v_prev; if every z>=2 rung's side ask at T is > 0.97 (mortgage
  pattern) -> DEAD (priced), no ROI computed.
- Success (dev -> propose forward paper logger): n >= 30 bets AND mean net ROI > 0 AND event-cluster bootstrap 95% CI
  lower > 0 AND passes edgelab family bar. Kill: n < 30, or CI crosses 0 ("not distinguishable"), or >= 3 losses in the
  first 20 bets by T order.
- Caveat written now: all dev months share one calendar month (Oct 2026) -> a common shock (e.g. holiday timing) hits
  every series at once; even a pass is one macro draw. s is estimated in-sample on dev.

## Log
- 2026-11-18 frame (results/altdata_kpi/ca1_frame.py): 192 consecutive-month pairs, 88 dev (close Nov 2026, Oct data;
  prior = Sep-data event). s = 0.0764 (MAD), but sd of r = 0.1937 and median r -0.0105 (ca1_scale.json) -> very fat
  tails: GEMINIAPP ln +1.03 (75 -> 211), GROKAPP -0.71, ESPNAPP +0.58, DKNGAPP +0.51 (ca1_pairs_dev.csv). Data gate
  >= 40 pairs: PASS (88). 215 far rungs (|z| >= 2) in 52 events -> ca1_far_dev.csv.
- T sanity: dev events open 2026-10-01 (start of the data month), prev event closes 10-05..10-08, so T ~ 10-06..10-09,
  ~3 weeks before the data month ends: a real forecast, not a read of a finished number.
- Candles (ca1_candles.py): 150/215 rungs have candles in [T-3d, T]; 136 two-sided at T. Side ask median 0.95, 38% > 0.97
  (ca1_score.py). Gate "all far rungs > .97" NOT triggered (unlike mortgage), so scored.
- **CA1 dev result: 74 bets / 28 events, 8 losses, net ROI +2.05%, event-cluster bootstrap 95% CI [-10.5%, +13.2%]**
  (ca1_score.py -> ca1_dev_bets.csv). 2 losses in the first 20 (kill not hit) but CI crosses 0 -> **not distinguishable
  -> DEAD per prereg** (n >= 30 met). By kind: ADS 32 bets pnl +1.80 (3 losses), APP 39 bets -0.67 (5 losses), POS 3
  bets +0.20; CC far rungs: 67 far, only 5 two-sided, median side ask 0.98 = priced/no quote (post-hoc split, not a claim).
- edgelab: dataset kalshi_carbonarc (vault.yaml), ca1_carbonarc_persist_taker-13c39cc2 registered (family
  kalshi_data_ladders -> 66 trials, bar p < 7.58e-04) and retired; verify ok (1374 lines).

## Other families (no test possible / not E4 shape)
- Fiscal.ai quarterly KPIs (incl. KXJBLU): settle on the company's own report; markets close at that release (KXBA
  26OCTDELIV closed 10-13, KXF 26OCTUSSALES 10-02, KXHOOD 26NOV* 10-27, KXLUV 10-21; markets_fiscalai.json). Partial free
  nowcasts exist for a few (Robinhood monthly operating data, Boeing monthly orders&deliveries page) but each is ~1 event
  per quarter -> a handful of events in the next 3 months; at min_n 30 a forward test needs years. JetBlue: no free
  timely source for load factor/fare (BTS T-100 lags ~3 months), 4 events, 0 settled. => watch only.
- TikTok Shop (RHODE/RARE/REM/HAUS/FENTY TTS + CELEBTTSARRAY): Carbon Arc, 0 settled, first events close 12-05/06,
  lifetime vol 200-9,183. Same family as CA1 (persistence dead); no free signal. Retail FT/CC named targets: 1 open
  event each, 0 settled, vols 5-14,829.
- KXSPRLVLDATE: 1 event / 3 markets, refill-policy forecast; KXSPRLVL weekly already dead (SP1, phys_ladders.md).

## Untested leads (each needs its own prereg; not run)
- APP downloads vs Google Trends YoY (free, has history): APP is where persistence misses (launch-driven jumps);
  41 settled APP events but only 2 months per series -> one or two cross-sections, weak power.
- Carbon Arc free access: if a free Carbon Arc account shows month-to-date prisms, that is the real E4 source
  (subscribers see the panel daily). Unverified: the prisms page is JS-only. Ask the user whether to try a signup.

## Learnings
### A free "last month's value" nowcast does not beat Carbon Arc KPI ladders
CA1 (prev settled expiration_value as nowcast, |z| >= 2 far rungs, taker at T = prev close + 24h): 74 bets ROI +2.05%
[-10.5%, +13.2%] (results/altdata_kpi/ca1_dev_bets.csv). Dead; the YoY indices jump (ln sd 0.19 vs MAD 0.076).
### Carbon Arc KPI markets settle on a paywalled panel that subscribers see daily
168 Kalshi series (CC, FT, APP, POS, ADS, TTS, arrays) settle on Carbon Arc prisms; rules say "the panel updates daily".
The informed side is the subscriber, not us. Check for free month-to-date access before any further work.
### Fiscal.ai company-KPI markets close at the company's own release
KXBA, KXF, KXHOOD, KXLUV quarterly KPI markets closed on the release date; at ~1 event per quarter per series no rule
reaches min_n 30 in a useful time, even where monthly company data gives a partial nowcast.

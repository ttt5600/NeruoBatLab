# pm_full_census -- every active Polymarket market (global + Polymarket US), tested vs untested (2026-11-18/19)

Agent: pm_full_census (user: "look at all polymarkets"). Read-only public GETs; no orders, no git, no sign-ups.
Scripts: scripts/pm_full_census_pull.py (Gamma keyset events + PM US gateway markets), scripts/pm_full_census_bbo.py
(PM US per-market /bbo: sharesTraded, openInterest, bid/ask; all non-sports + 1/8 random sports, weight 8).
Scratch: results/pm_full_census/.

## Question
1. What is listed and active on Polymarket (global, gamma-api) and Polymarket US (gateway.polymarket.us), by category,
   with volume and depth? Which global events are on PM US?
2. Which categories with volume have NO test in FUND_BOOK 3c/3d / LEARNINGS?
3. For the best 2-3 untested: is there an external reference + history? Pre-registered dev test.

## Do-not-redo read first
FUND_BOOK 3c/3d PM rows: H1/H2 longshot-NO maker (mention/culture), OVR overreaction fade, PM-T favourite timing,
XV-PM in-game vs Kalshi + pmus_lag, negRisk overround (display artefact), M1/M2 Kalshi-vs-PM elections & sports
(0.5-1c), Kalshi-vs-PM weather 0/48, PM US vs Pinnacle 0/116, E1p/E2p/E3p earnings, F4 PM US weather, pm_wallet_signal,
LX1 logical arb (1/5), LX3 PM US liquidity rewards (parked), smallmarket_calibration (FLB ~= cost venue-wide).
Kalshi-side tests that cover the same underlying: L22 politics favourites (dead), Q012 crypto hourly (dead),
options_implied_tails / crypto_barrier (in flight), econ reject / N2 CPI (dead), esports ESP1/cs2_model (in flight).

## Log
- 2026-11-19 ~05:45Z: Gamma /events offset caps at 2,500 ("use /events/keyset") -> switched to keyset cursor.
  PM US /v1/markets?closed=false returned 74,548 markets (22 already RESOLVED status). No volume in the listing;
  volume only via per-market /bbo (sharesTraded, openInterest) -> sampled pull.
- Gamma keyset pull: 18,949 active events / 328,856 market rows (incl. 135,798 closed legs inside open events and
  ~27k inactive); open+active+accepting = 166,110 markets (results/pm_full_census/gamma_markets.parquet groupby).
  Sum of event volume1wk over active events = $202.0M, volume24hr $57.5M, liquidity $592M (gamma_events.parquet).
  CAVEAT: this is volume of events still open; short-lived recurring markets (games, crypto 5/15-min up/down, daily
  temps) that already closed are not in it -> separate pull of events closed in the last 7 days (gamma_closed7d.parquet).
- Categorisation: scripts/pm_full_census_cat.py (priority rules on tag slugs + title regex). First pass mis-sorted
  2028 nominee events into "earnings" via a rewards tag `earn-4` and "Largest IPO" into weather via a generic tag ->
  fixed with title-first rules (nominee/election, commodities) and dropped those tags.
- **PM US fee changed**: every open PM US market now has feeCoefficient 0.095 (pmus_markets.parquet, all 10 categories,
  min=median=max). notes/venues.md / contract_census 10-29 recorded 0.0695. Taker fee at p=.5 is now 2.4c not 1.7c.
  Anything modelled with 0.0695 (F4, pmus_lag) understates PM US taker cost by ~37%.
- PM US open markets by category (count): sports 65,061 (FOOTBALL 38,164, SOCCER 11,122, BASKETBALL 6,647, HOCKEY
  3,427, TABLE TENNIS 2,003, ...), politics 4,545 (USCONGRESS 3,494), culture 2,323 (AWD 1,843), crypto 1,405 (hourly
  BTC/ETH/SOL/XRP price + range ladders), macro 501 (gas/diesel weekly+monthly, commodities, inflation), technology 226
  (AI model rankings), geopolitics 200 (ship traffic Hormuz/Suez/Malacca, UN SG), climate 146 (US city highs/lows),
  finance 131, science 10.

## Candidate selection (written before any return/outcome data is pulled)
PM US is the only venue the user can trade, and its price-history gives book-derived longPrice = YES ask and
shortPrice = NO ask (scripts/polymarket_us.py docstring) -> executable historical prices exist ONLY on PM US. Global PM
history (CLOB prices-history) is a mid/last display price -> fine as a REFERENCE, not as an execution price.
Untested angle with an obvious external reference: **the global Polymarket book itself**. PM US is a separate, thinner
order book listing many of the same questions; the global book is far deeper (event volume1wk sum $202M). If PM US
lags the global book on non-game markets, buying PM US toward the global price is the XV-PM idea on a NEW market set
(XV-PM tested only in-game sports vs Kalshi, where Kalshi led by 10-20 s and PM US real ask killed it).

## PRE-REGISTRATION T1 "XV-US": PM US vs global Polymarket, non-in-game markets (written before outcomes)
- Universe: PM US markets that RESOLVED, category != in-game sports (exclude moneyline/spread/total/props of a single
  game; keep futures, politics, culture, crypto non-hourly, macro, tech, geopolitics, finance, climate), matched 1:1 to
  a global Gamma market (outcome name normalised-exact + event title token similarity; 30 random pairs eyeballed
  before any outcome is read; ambiguous matches dropped).
- Prices: PM US /v1 price-history longPrice (YES ask) and shortPrice (NO ask), hourly. Global reference = CLOB
  prices-history (fidelity 60) last point at or BEFORE t. PM US execution = first PM US point at or AFTER t, within 60 min.
- Decision grid: hourly t, only while t <= resolution/endDate - 24 h (avoid resolution races). First signal per market
  only (one bet per market).
- Rule: buy YES if g - (yes_ask + fee) >= 0.03; buy NO if (1-g) - (no_ask + fee) >= 0.03; ask in [0.05, 0.95];
  fee = 0.095 * p(1-p) (current coefficient; also report at 0.0695). 1 contract, hold to resolution.
- Split: resolved markets sorted by resolution time; earliest 60% of EVENTS = DEV, latest 40% = SEALED (not opened).
- Stats: ROI per $ cost; event-clustered bootstrap 2,000 draws, 95% CI.
- SUCCESS (dev): CI lower > 0, n >= 100 bets over >= 30 events, no single event > 25% of P&L.
- KILL: CI crosses 0 -> "not distinguishable" -> dead; n < 100 -> "no capacity".
- Placebos: (P1) same rule with the global price lagged 7 days (stale reference) -- should not win;
  (P2) reverse sign (buy the side PM US prices >= 3c ABOVE global) -- should lose. A P1 win = FLB/selection artefact.
- Too-good guard: any dev ROI > +10% -> inspect 20 top rows (timestamps in s vs ms, after-resolution points) first.
- AMENDMENT A1 (before any outcome/return is read; data-shape only): PM US price-history INTERVAL_ALL returns DAILY
  points at 00:00 UTC (int seconds; tec-mls-winner-...-vwh: 245 points, median step 86,400 s). So the decision grid is
  daily 00:00 UTC, not hourly. Some points are crossed (first MLS point longPrice .245 / shortPrice .5 -> implied YES bid
  .5 > ask .245 = an empty side shown as a default) -> validity filter: keep a point only if
  yes_ask + no_ask >= 0.999 and yes_ask + no_ask - 1 <= 0.10. Global reference = CLOB prices-history last point <= t.
  Matching: require event-title Jaccard >= 0.6 (the 30-pair eyeball of open matches found 1/30 wrong contract,
  "League Phase Top Finisher" -> "finish top 8", at Jaccard 0.50; results/pm_full_census/match_open.parquet).
- AMENDMENT A2 (before outcomes): global matches for RESOLVED PM US markets come from gamma /public-search
  (events_status=closed) on each PM US event question, then the same matcher (Jaccard >= 0.6). Resolution-race guard:
  PM US endDate is the scheduled end, and "by date" markets can resolve early, so a stale PM US display quote after the
  real outcome is known would look like a huge lag. Extra guards: drop t at or after the first day the global price is
  >= 0.97 or <= 0.03 ("decided"), and t <= endDate - 24 h. Sensitivity (reported, not the decision): also drop the 7
  days before "decided".
- PM US volume census (results/pm_full_census/pmus_bbo.parquet; /bbo sharesTraded = LIFETIME shares, notional =
  shares x last price; sports game markets sampled 1/8 and weighted x8). Lifetime notional of currently open markets:
  sports futures ~$349M (11,432 mkts), politics ~$47.6M, finance ~$22.4M, culture ~$15.6M, crypto ~$10.1M, macro
  ~$7.9M, technology ~$6.4M, open sports game markets ~$4.7M (53,296 mkts, only 12% ever traded, median spread 14c),
  geopolitics ~$1.0M, science ~$0.7M, climate ~$0.5M. Two-sided book share: politics 17.5%, technology 22.6%, culture 49.7%.
- Global flow (events CLOSED in the last 7 days, lifetime volume; results/pm_full_census/closed7d_by_cat.csv): 42,283
  events, $3.58B; fed_macro $1.04B (35 events), geopolitics $894M, sports games $555M (14,732 events), sports futures
  $384M, esports $197M (644), crypto price $173M, crypto up/down $120M (23,910 events), weather $12.3M (596 events).
- LIVE cross-venue snapshot (results/pm_full_census/xvus_live_gaps.parquet; 956 open matched markets with two-sided
  books on both venues, Jaccard >= 0.6): median |PM US mid - global mid| 2.2c, p90 21c; PM US median spread 1.0c vs
  global 2.4c (on matched markets). 70 markets show a PM US ask+fee >= 3c through the global bid/ask, BUT the largest
  (>20c) are contract mismatches (weekly ship-traffic windows, "impeached and convicted" vs "impeached", "closer race").
  Plausible genuine ones are thin PM US books: e.g. Bundesliga top scorer Schick US ask .02 vs global bid .145; Oscar
  VFX "Avengers: Doomsday" US bid .21 vs global ask .005. Snapshot only: says nothing about returns.

## Census table (global = gamma active events, vol1wk = sum of event volume1wk; PM US = open markets, lifetime notional)
| category | global events / vol1wk | PM US open mkts / notional | tested? (cite) |
|---|---|---|---|
| sports games | 9,310 / $27.3M (+14,732 closed in 7d, $555M lifetime) | ~53.3k / ~$4.7M | TESTED dead: M2 (FUND_BOOK 3d l.346), PM US vs Pinnacle 0/116 (l.348), XV-PM in-game (l.247), PM-T (l.243), pm_wallet_signal (LEARNINGS l.182) |
| sports futures | 1,377 / $8.3M | ~11.4k / ~$349M | UNTESTED as a strategy (smallmarket: sports 7d calibration excess +0.064 "mostly futures legs", exploratory) -> T1 |
| elections | 1,080 / $35.2M | politics 4,545 / ~$47.6M | TESTED: M1 (l.345), L22 politics favourites dead (l.204), L18 |
| politics "by date" / cabinet | 236 / $2.6M | (in politics) | partly: smallmarket politics 7d excess ~0; LX2 tail-NO Kalshi (l.186) -> T1 |
| geopolitics | 468 / $23.0M | 200 / ~$1.0M | UNTESTED (PW ship traffic dead at data gate, l.245) -> T1 |
| fed / macro | 259 / $19.4M | macro 501 / ~$7.9M | TESTED: fed_vs_futures dead, econ reject (l.337), N2 CPI (l.334), N6 gas (l.335); PM US gas weekly too new (2 closed events, far rungs at 1c) |
| crypto price ladders (hit/above) | 451 / $22.3M | crypto 1,405 / ~$10.1M (hourly price ladders) | Kalshi analogues: Q012 hourly dead, crypto_barrier / options_implied_tails in flight; PM itself UNTESTED |
| crypto up/down 5-15 min | 3,543 / ~$0 open (23,910 closed in 7d, $120M) | none | UNTESTED; global only (not CA-legal); manipulation pitfall (pm_literature_scout) |
| equities / commodities / IPO | 630 / $18.2M | finance 131 / ~$22.4M | UNTESTED on PM (options_implied_tails = Kalshi) -> T1 for PM US finance |
| tech / AI | 332 / $17.3M | 226 / ~$6.4M | UNTESTED (LMArena source 403 in info_gap_census) -> T1 |
| culture / awards | 338 / $24.4M ($19.3M = one Stranger Things event) | 2,323 / ~$15.6M | TESTED: H1/H2 culture NO maker dead (l.241), MEDIA (l.244) -> also in T1 |
| weather | 290 / $1.3M (+596 closed in 7d) | climate 146 / ~$0.5M | TESTED US cities: F4 (l.169), Kalshi-vs-PM 0/48 (l.347); international cities UNTESTED (global only) |
| tweets / post counts | 21 / $1.7M | none | Kalshi Q014 Truth Social forward (in flight); PM Elon counts UNTESTED, global only |
| esports | 551 / $0.9M (+644 closed, $197M) | sports ESPORTS 244 | Kalshi ESP1 / cs2_model in flight; PM UNTESTED |
| mentions | 9 / $0.1M | none | TESTED: H1 (l.207/241), Kalshi M1/M2 mention NO (l.330) |
| earnings | 7 / ~$0 | none | TESTED: E1p/E2p/E3p (l.361-362) |
- AMENDMENT A3 (before outcomes): 30-pair eyeball of RESOLVED matches (Jaccard >= 0.6) found ~4/30 wrong contracts
  (RBC Canadian Open R3 leader -> U.S. Open R3 leader; "European Masters" -> the 2025 edition, 371 d apart; "CeeDee
  Lamb" -> "traded?"; Women's Open -> CPKC Women's Open, 22 d apart). New filter (scripts/pm_full_census_a3.py):
  |PM US endDate - global event endDate| <= 14 d AND every non-generic PM US question token appears in the global
  event title or market question. 8,244 -> 4,992 markets / 293 events (sports futures 4,232 [golf, NASCAR, drafts],
  culture 487, politics 202, technology 56, macro 15). Re-eyeball of 30 new random pairs: 30/30 same contract.
  The universe is therefore mostly golf/NASCAR outrights + awards; the T1 verdict speaks for that mix.
- AMENDMENT A4 (data-shape; found at the timestamp sanity check, before any bet/outcome was tabulated): CLOB
  prices-history `interval=max, fidelity=60` returns NOTHING for markets resolved before ~2026-10-19 (global g_t
  range in the first panel = 2026-10-19..11-18), i.e. only the last ~30 days have hourly history. `interval=all,
  fidelity=720` returns the full life (12-hour points; e.g. SD Gov GOP primary 408 points 2025-12-11..2026-07-28).
  Global reference switched to 12-hour points (reference age still capped at 3 days). Panel rebuilt for the A3 set only.

## T1 XV-US result (DEV only; sealed = events ending after 2026-09-15 02:37Z, NOT opened)
Command: `../.venv_quant/bin/python scripts/pm_full_census_xvus_score.py` -> results/pm_full_census/xvus_dev_report.txt;
bets in results/pm_full_census/xvus_dev_bets.parquet. Panel: 62,878 PM US daily points over the A3 set; 10,592 valid
after the pre-registered filters (dev 5,042 / sealed 5,550 rows). Dev markets: sports 860, culture 96, politics 65,
technology 7, macro 3.
| rule | n bets | events | ROI | 95% CI (event bootstrap) |
|---|---|---|---|---|
| T1 main, fee .095 | 470 | 96 | +1.73% | [-12.59, +15.93] |
| T1, fee .0695 | 476 | 98 | +2.25% | [-11.30, +15.80] |
| sensitivity: drop 7 d before "decided" | 302 | 80 | +8.88% | [-7.65, +25.27] |
| P1 placebo: global lagged 7 d | 180 | 73 | +11.12% | [-3.64, +24.79] |
| P2 placebo: reverse sign | 399 | 113 | -11.25% | [-19.81, -4.09] |
| POST-HOC (after outcomes): drop global prices in [.49,.51] | 409 | 95 | +6.93% | [-7.97, +20.71] |
**Verdict: DEAD on dev -- not distinguishable** (CI crosses 0; pre-registered kill). The stale-reference placebo P1
earns as much as the live rule, so whatever positive drift there is would not be a lag effect anyway. Sealed unused.
Row inspection (too-good guard): the biggest winners were YES longshots (NASCAR/golf/R3 leader) bought at 5-9c where
the global "reference" was ~0.495 -- the CLOB history of an untraded global market sits at its 0.5 starting price.
11.4% of dev rows have a global price in [.49,.51]. Mean claimed entry edge YES 0.308 / NO 0.183 = mostly that
placeholder, not a real gap. Dropping those rows (post hoc, labelled as such) still crosses 0.
What P2 does show: when PM US is priced >= 3c ABOVE global, buying it loses -11% [-19.8, -4.1]. The global book is the
better price where they disagree, but PM US's cheap side is not cheap enough to pay spread + 0.095 fee.
Not registered in edgelab (needs a dataset adapter); pre-registration is in this file above the result, with A1-A4.

## Tests not run (and why)
- PM US gas/diesel weekly (AAA): only 2 closed events per series on PM US (pmus_closed_ns.parquet); far rungs YES
  bid .01 / ask .02 (pmus_bbo) -> nothing to buy, nothing to backtest. Same shape as state_gas.
- International daily highs (HK, Tokyo, Guangzhou... $35-70k/city-day), PM crypto hit ladders, Elon tweet counts,
  esports, crypto up/down: global-only (not CA-tradeable), and global history is mid/last, not ask. Leads only.
- AI-model ranking markets (PM US technology 226 mkts, ~$6.4M): resolve off LMArena, a public leaderboard visible
  before close -> information-lag lead; LMArena returned 403 in info_gap_census. Untested.

## Learnings
### Polymarket US now charges 0.095 p(1-p) taker on every market
All 74,548 open PM US markets carry feeCoefficient 0.095 (results/pm_full_census/pmus_markets.parquet), up from 0.0695
recorded 10-28 (notes/venues.md). Any PM US model costed at 0.0695 understates the taker fee by ~37%.
### PM US does not lag global Polymarket on outrights/awards/politics enough to pay
T1 XV-US dev: +1.73% [-12.6, +15.9] n470/96 events; stale-reference placebo +11.1% [-3.6, +24.8]. Dead.
(notes/pm_full_census.md, results/pm_full_census/xvus_dev_report.txt)
### Untraded Polymarket markets report a 0.5 price in CLOB history
A global market with no trades shows ~0.495-0.5 in prices-history; used as a "fair" reference it fakes 30c+ edges on
longshots. Filter references by traded volume / quote presence, never trust a 0.5 print.
### CLOB prices-history: fidelity=60 with interval=max returns nothing for markets resolved > ~30 days ago
Use interval=all with fidelity=720 (12 h) for full history of resolved markets.
### Gamma /events offset caps at 2,500; use /events/keyset with next_cursor
Also the PM US gateway pages past 2,500 fine, but stop on an EMPTY page, not a short one (one page returned 499).
### Name-based cross-venue matching is ~10-13% wrong without an end-date check
30-pair eyeball at Jaccard >= 0.6 had ~4 wrong contracts (other tournament, other year); end-date gap <= 14 d plus
all-distinctive-tokens rule fixed 30/30.

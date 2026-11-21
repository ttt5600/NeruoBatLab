# count_nowcast -- Kalshi series that settle on a free, public, intra-period-observable count or chart (agent, 2026-11-20)

Question: which live Kalshi series settle on a FREE public count/chart that can be observed (partially) before settlement,
are not in FUND_BOOK 3d / LEARNINGS dead list, and score well on STRATEGY_v2's rubric? For the top 2: census, nowcast
accuracy from free point-in-time data, bracket staleness vs nowcast, pre-registered rule, edgelab registration, dev or
forward-only spec. No git, no orders, no launchd.

Scratch: results/count_nowcast/ (series_sweep.py/.txt, series_live.json (16,768 series), open_events.json (22,920 open
events), census.py + census_*.csv, mk/<SERIES>.json, live_book.py/.json).

## 1. Sweep (results/count_nowcast/series_sweep.txt, open_events.json)
Keyword sweep of all 16,768 series (settlement source name/url + title) for Spotify/kworb/Netflix/box office/App Store/
YouTube/Wikipedia/Trends/FlightAware/FAA/ERCOT/CAISO/PJM/MISO/EIA/Steam/OpenRouter/TSA/count words -> 1,494 matches; then
cross-checked against the 22,920 events open on 2026-11-21 to keep only LIVE daily/weekly/monthly series.

**Most of the brief's examples are dormant on Kalshi** (census.py output, `last` = latest close_time of any market):
| series | settled ev | last | lifetime contracts | why not testable now |
|---|---|---|---|---|
| KXSPOTIFYD (US daily #1) | 592 | 2026-07-02 | 21.8M | no open events since July; whole Spotify family (W, 2D, GLOBALD, ARTISTW/D, ALBUMW, cities, SPOTSTREAMSUSA) stopped 2026-07-02/03 |
| KXSPOTIFYW (US weekly #1) | 72 | 2026-07-03 | 4.9M | same |
| KXFLIGHTJFK/ORD/LAX (FlightAware delays+cancels) | 53/53/52 | 2025-11-14 | 0.10-0.18M each | dormant; FlightAware behind Cloudflare (robots.txt returned a 308 challenge) |
| KXAPPRANKFREE | 302 | 2025-11-14 | 3.9M | dormant (and no free rank history) |
| KXTOPSELLERS (Steam) | 8 | 2025-01-19 | 2.9k | dormant |
| KXNETFLIXMOVIERANKD / SHOWRANKD (FlixPatrol daily) | 1 / 1 | 2025-05-17 | <1k | dormant; FlixPatrol = Cloudflare JS challenge (not bypassed) |
| KXYTVIEWSD | 119 | 2026-09-29 | 0.15M | dormant; YouTube views = MEDIA dead row |
| Box office | -- | -- | -- | still not listed (only KXSLATECOUNT = count of wide releases, 1 event, 2027) |
| Wikipedia pageviews / Google Trends | -- | -- | -- | no live periodic series (only annual KXRANKLISTGOOGLESEARCH*) |
kworb.net robots.txt = `Allow: /` (usable if Spotify series are relisted); charts.spotify.com needs login for downloads.

Already covered / dead (dropped): ERCOT load KXTXERCOTPEAKD (Q017), ERCOT/PJM/MISO hub weekly/monthly price (PW1 + LRN
l.256), TSA daily/weekly (ufc_tsa / B4), AAA state gas + diesel (N6/state_gas), OpenRouter share/tokens (agent ai_nowcast
started 2026-11-21T04:26Z -- not duplicated), Truth Social post counts KXTRUTHSOCIAL(D) (Q014 dead), YouTube weekly/monthly
(MEDIA), Billboard/album units/pure albums (billboard_top, album_units), Carbon Arc arrays (altdata_kpi), PortWatch
Hormuz/Suez/Panama (PW), SpaceX count (SX1), RT (Q013), econ prints (econ_reject/N2).

## 2. Rubric scores of what is left (live, not dead). Census: census.py; book: live_book.py (snapshot 2026-11-21 ~05Z)
R1 free+exact settlement source · R2 sharper input · R3 far-rung neglect · R4 settled ev/month · R5 spread+fee · R6 who sees first
| series | settles on | ev 90d | lifetime ct | med spread (two-sided rungs) | far rungs no 24h vol | R1-R6 = score |
|---|---|---|---|---|---|---|
| KXTRUMPNOMNUM (weekly nominations sent to Senate) | whitehouse.gov Presidential Actions dates | 12 | 0.70M | .03 (4) | 6/10 | 1,1,2,1,1,1 = 7 |
| KXKEVCOUNT (CISA KEV additions, monthly) | CISA KEV catalog dateAdded (free JSON) | 1 | 0.15M | .02 (6) | 2/4 | 1,1,2,0,2,1 = 7 (R4=0: 1 event ever) |
| KXSOFRD (daily SOFR) | NY Fed | 62 | 0.48M | .64 (1) | 6/6 | 1,1,2,2,0,0 = 6 (not a count; dealers see repo first) |
| KXEOWEEK (weekly executive orders signed) | Federal Register signing_date (free API) | 12 | 14.6M | .01 (3) | 0/2 | 1,1,0,1,2,1 = 6 (largest volume of the set) |
| KXTGAW (TGA balance on a date) | Daily Treasury Statement (free API) | 6 | 0.11M | .295 (2) | 2/2 | 1,1,2,1,0,1 = 6 (closes before DTS publishes: forecast, not nowcast) |
| KXTRUMPNUMSTATES (monthly) | official schedules / press | 4 | 0.92M | .02 | 2/3 | 0,1,1,0,2,1 = 5 |
| KXBILLSSIGNED (monthly) | congress.gov / WH | 2 | 0.04M | .07 | 4/6 | 1,1,2,0,0,1 = 5 |
| KXPARDONSTRUMP (monthly) | WH / DOJ | 4 | 0.33M | .09 | 4/5 | 1,1,2,0,0,1 = 5 |
| KXJUDGECOUNT (monthly) | congress.gov | 4 | 0.16M | .08 | 1/2 | 1,1,1,0,0,1 = 4 |
| KXTRUMPENDORSEMENTS (weekly) | Truth Social, judgment call | 12 | 1.18M | none two-sided | 0/1 | 0,1,0,1,0,1 = 3 (Q014 sibling) |
| KXNETFLIXRANKSHOW/MOVIE (weekly) | Tudum weekly | 13/13 | 18.1M/12.3M | --/.02 | 0 | 1,0,0,1,1-2,0 = 3-4 (daily signal = FlixPatrol, blocked) |
Gate: none scores >= 8. Top 2 chosen for testing: **KXEOWEEK** (exact free PIT settlement source, the only one with real
volume; R3 = 0 is its weak item) and **KXTRUMPNOMNUM** (best score among weeklies). KXKEVCOUNT has 1 settled event:
forward-log only.

## 3. KXEOWEEK -- contract (read from mk/KXEOWEEK.json, 145 markets, 59 events, 2025-09..2026-12)
"If the President signs above k executive orders during <Sun> to <Sat>" (strike_type greater, k = 0..2 until 26OCT03,
0..4 since 26OCT10). Settlement: signing date as recorded in the Federal Register; markets close early once determined
(close_time - week end: median 6 d, max 15 d). fee quadratic, multiplier 1.

## PRE-REGISTRATION EO1 (written 2026-11-21 BEFORE any KXEOWEEK price/candle was fetched)
Seen before writing: event list with settled results (expiration_value) and close times (needed to see the structure);
FR API sample of 3 EOs (signing->publication lag 3-6 days). No price looked at.
Mechanism: the Federal Register IS the settlement source; once FR has published n EOs signed inside the week, every rung
k < n is determined YES. Question: does the YES ask on determined rungs stay below 1 - fee while the rung is still open?
(E4 pattern = settlement source observable before the book reprices.) Also descriptive: the week-end+1d price vs the
signing-date count (staleness), flagged as liberal (assumes WH discloses on the signing date).
- PIT nowcast L(T) = # EOs with signing_date in [week start, week end] and FR publication_date such that
  publication_date 12:00Z <= T (FR posts ~06:00 ET; 12:00Z is >= 1 h after in both EST and EDT).
- Trial EO1-Y (taker): for each rung k, T_k = first publication 12:00Z at which L(T) > k. If the rung's close_time > T_k,
  buy 1 YES at the yes ask of the last hourly candle ending in (T_k - 6 h, T_k + 1 h]; require 0.50 <= ask <= 0.98 (ask
  > .98 counts as "no trade, priced"). Cost = ask + ceil_cent(0.07 * ask * (1 - ask)). Payout 1 if result yes.
- Placebo (base-rate shape): same T_k, YES on every OTHER open rung of the same series (any week) with ask in the same
  0.50-0.98 band that is NOT determined (L(T) <= k). Rule ROI must exceed placebo ROI.
- Split by week end: dev <= 2026-08-31; holdout 2026-09-01..2026-11-14 (not opened without the lead's yes); forward
  2026-11-15+.
- Success (dev): n >= 30 bets, ROI > 0, week-clustered bootstrap 95% LCB > 0, market-null p (prod of ask over bets, the
  probability all win if prices were fair) < edgelab family bar, beats placebo. Kill: n < 30, or ROI <= 0, or any loss
  whose FR record contradicts L (= settlement mismatch -> stop).
- Data gate (before prices): FR signing-date weekly counts must reproduce Kalshi results on >= 95% of settled rungs.
- Power note: zero-loss rule -> evidence is run length; at ask .96 need ~75 straight wins for p < .05/1 (L39 lesson).

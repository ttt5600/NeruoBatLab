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

## RESULT EO1 (dev scored once, 2026-11-21; holdout candles never fetched)
results/count_nowcast/eo1.py -> eo1_frame.parquet; edgelab trial **eo1_locked_yes-6f1de23e** (family kalshi_data_ladders,
70 trials, bar 7.14e-4) evaluate dev -> **dead, "no bets"**.
- Data gate PASS: FR signing-date counts reproduce Kalshi on 121/123 settled rungs (eo_fr.py; NaN-strike single markets =
  "above 0"). The 2 misses (26FEB28-0, 26MAR07-1) settled YES with FR count 0 / 1; the market had YES bid .96 / .99 on
  the Sunday after -- Kalshi and traders counted from whitehouse.gov, not FR. FR republishes corrections (14388 twice):
  dedupe by EO number.
- Dev 88 rungs / 44 weeks (2025-10-19..2026-08-29): 45 never determined, **28 YES ask > .98 (all 31 open rungs had ask
  1.00 at the last quote)**, 12 closed before FR publication, 3 no candle in the window -> 0 bets (< min 30). Placebo moot.
- Descriptive (eo_desc.py): first YES bid >= .90 came a median 13 h after 00:00 ET of the signing day; 42/43 = 97.7%
  within 24 h (28% earlier still = expectation or lower rung). FR publication lag median 5 d (max 48). So the window
  FR-lag never exists: the book prices off the White House posting.
- NO side after the week (liberal signing-date count, not a trial): YES bid at Sun 17Z median .01; 12 rungs with YES
  bid .03-.50, all NO -- 12 cases in 10 months, unpowered; the 2 FR-vs-Kalshi misses would have been NO-side losses had a
  rule used FR. Verdict EO1: **dead (priced)**; no forward logger proposed.

## KXTRUMPNOMNUM -- contract and gate (mk/KXTRUMPNOMNUM.json, 77 markets, 18 events 2026-07-19..11-28)
"Trump has sent at least k nominations to the Senate from <Sun> through <Sat>" per whitehouse.gov Presidential Actions
publication dates (checked 10:00 ET the Sunday after). greater_or_equal; ladders {2,3,5,9/10} -> {1,3,5,10} -> {1,10,20,
30/40} -> {6,10,15,20,30,40}. Free PIT proxy: congress.gov API v3 /nomination/119 (DEMO_KEY; 2,250 records pulled before
the rate limit, complete for receivedDate >= 2025-06-30 because the list is sorted by updateDate desc), civilian only,
by receivedDate. whitehouse.gov wp-json returns 403 (not bypassed); robots.txt allows the HTML listing.
Gate (no prices): civilian weekly count consistent with all 16 settled events (zero weeks <-> "no" on rung 1/2;
Jul 19 count 8 in (5, 9); Aug 2 21, Sep 13 41, Sep 27 17, Nov 8 45 all >= top YES rung).
Shape: bimodal -- 0 when the Senate is out, >= 8 otherwise. Fit window 2025-07-06..2026-06-21 (before Kalshi listed):
22 weeks with 0 civilian noms Sun-Wed, 1 of them got noms Thu-Sat (2026-01-25 week, 16) -> P = 1/22, Jeffreys 0.065
(results/count_nowcast/nom_week_fit.csv).

## PRE-REGISTRATION NM1 (written 2026-11-21 BEFORE any KXTRUMPNOMNUM candle/price was fetched)
Seen before: settled results per event (for the gate) and the live book snapshot (spread .03, 6/10 far rungs no 24h vol).
- T = Thursday 15:00Z of the week. C = civilian nominations with receivedDate Sun..Wed (congress.gov updates by morning).
- NM1-N: if C == 0, for each rung k >= 1 still open at T: p_yes = 0.065; NO ask = 1 - yes_bid (last hourly candle in
  (T-6h, T]); buy NO if (1 - 0.065) - NOask - ceil_cent(0.07 a (1-a)) >= 0.02 and NOask >= 0.50.
- NM1-Y: if C >= k: buy YES at yes_ask if 0.50 <= ask <= 0.98.
- Placebo (base-rate shape): NO on the lowest open rung at T in EVERY week whose NO ask is in the same band, ignoring C.
- Split: all settled weeks 2026-07-19..2026-11-14 are dev (16 events; no holdout possible); forward 2026-11-15+.
- Success: n >= 30, ROI > 0, week-clustered 95% LCB > 0, beats placebo, p < family bar. Kill: n < 30 or ROI <= 0.
- Expected: n < 30 (16 weeks) -> kill on power regardless; the descriptive numbers decide whether a forward logger is worth it.

## RESULT NM1 (dev scored once, 2026-11-21)
results/count_nowcast/nm1.py -> nm1_frame.parquet; edgelab **nm1_nowcast-1fa68f4f** (family kalshi_data_ladders, 72 trials,
bar 6.94e-4) evaluate dev -> **dead**: n 10 bets (< min 30), net ROI -7.04% [-74.0, +31.0], p .616, ROI ex top week -32.7%.
- Nowcast accuracy (Sun..Wed civilian count C vs full-week count): 14/16 weeks C already equalled the week total; the 2
  misses were 26AUG02 (C 0 -> 21) and 26NOV08 (41 -> 45).
- All 10 bets were the N arm (C == 0 -> NO): 7 wins, 3 losses, and the 3 losses are the one week (26AUG02) where noms came
  Thu-Sat. The market had YES bid .20-.46 on exactly that week (NO ask .54-.80) -- it knew something the base rate did
  not (rule PA again: where model and market disagree most, the market is right).
- Y arm (C >= k): YES ask 1.00 on the median rung at T (11 rungs; 0 in the .50-.98 band) -> priced.
- Placebo (lowest rung NO in every week, same band): n 4, 3 wins, ROI +5.6% -- the rule did not beat the placebo.
- C == 0 rungs: YES bid median .02 (45 rungs), 4 resolved YES.
Verdict NM1: **dead (priced + too few events: 16 weeks, ~4/month)**; no forward logger.

## Proposals (not done; no launchd)
- Spotify relisting watch: the Spotify family (KXSPOTIFYD 592 settled, 21.8M contracts; KXSPOTIFYW 72; GLOBALD 510)
  stopped listing 2026-07-02. kworb.net allows all crawlers and keeps per-track daily stream history, so a weekly-#1
  nowcast (Fri..Tue of the Fri..Thu week observed by Wednesday) is free and point-in-time. If series_watch sees a new
  KXSPOTIFY* event, launch a dev test on the archived 72 weekly + 592 daily events first (candles are on /historical).
- KXKEVCOUNT (CISA KEV monthly, 1 settled event): forward-log only if it reaches >= 6 events; KEV JSON has dateAdded.

## Learnings
### Kalshi count markets price off the fastest official page, not the named settlement source
KXEOWEEK settles on Federal Register signing dates, but YES bid reached .90 a median 13 h after the signing day began,
42/43 within 24 h, while FR publication lagged 5 days (median). At FR publication the YES ask was 1.00 on all 31 open
determined rungs -> EO1 0 bets. Source: results/count_nowcast/eo1.py, eo_desc.py. A slow official source is never the
stale one if a faster official page (whitehouse.gov) shows the same count.
### The Spotify / flights / app-rank / Steam / Netflix-daily count series are dormant on Kalshi
Spotify family last close 2026-07-02/03 (KXSPOTIFYD 592 settled, 21.8M contracts); flights 2025-11-14; app rank
2025-11-14; Steam 2025-01; Netflix daily 1 event. Box office, Wikipedia, Google Trends have no periodic series. Source:
results/count_nowcast/census_*.csv, open_events.json (22,920 open events 2026-11-21). Check open events before planning
a test on any "chart" series.
### A base-rate NO rule on a bimodal weekly count loses on the one week the market flags
KXTRUMPNOMNUM: 0 civilian nominations Sun-Wed was followed by more later in 1/22 fit weeks, yet the NO arm lost 3 of 10
bets, all in the single week where the market already priced YES at .20-.46. ROI -7.0% [-74, +31] n10. Source:
results/count_nowcast/nm1.py, edgelab nm1_nowcast-1fa68f4f.
### Kalshi settlement sometimes disagrees with the named source; the market follows what Kalshi will count
2/123 KXEOWEEK rungs settled YES although the Federal Register lists 0 / 1 EOs for the week; the book had them at .96-.99.
A rule keyed to the named source would have lost on exactly those. Source: results/count_nowcast/eo_fr.py.

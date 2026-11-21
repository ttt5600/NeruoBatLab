# cbank_thin -- Kalshi thin central-bank decision markets vs free public consensus

Agent: cbank_thin (research subagent, 2026-11-20). Read-only; no orders, no git, no launchd.
Prior work read: STRATEGY_v2 #6 (26 KXCBD* series, ~16 settled/month aggregate), fed_vs_futures.md (Kalshi Fed tails
NOT rich vs ZQ futures; 7 bets in 22 meetings -> no capacity), FUND_BOOK 3d (econ consensus-reject: quoted tails fairly
priced; N2/N6: ladders only quote rungs inside ~3 sd).

## Question
Are Kalshi's thin non-Fed central-bank decision markets mispriced versus free public consensus (economist polls as
reported in free news / Trading Economics consensus / bank guidance)? Specifically: when public consensus is near-
unanimous, does Kalshi still price the non-consensus outcomes rich enough to sell after fees?

## Census plan
Series found by /series keyword scan (results/cbank_thin/series_hits.json -> cb_series.json): 29 decision series with
events (KXCBD* + KXCBDECISION*), plus old KXBOE (2023-24). Events per series: results/cbank_thin/events.json.
Note: Kalshi history for these series starts 2025-11/12 (most series 2026-07+ only) -- see census numbers below.

## Pre-registration (written 2026-11-20 BEFORE any Kalshi price, outcome or consensus value was looked at)
Disclosure: general knowledge of some 2026 decisions is possible but nothing was looked up for this table.

Decision time T in {D-7, D-1}: D = market close_time (announcement); T = close_time - 7 d and - 1 d.
Kalshi price at T: last hourly candle ending <= T, within the 24 h before T (else no quote -> skipped); yes_bid close
and yes_ask close.

Consensus C (point in time): the latest free-news report of an economist poll (Reuters / Bloomberg / local-press poll,
or Trading Economics consensus field when timestamped) published BEFORE T. Recorded: URL, publication date, poll
median bucket, strength s = share of polled economists in the median bucket ("all"/"unanimous" -> 1.0; "26 of 28" -> .93;
point forecast only with no dispersion -> s unknown). If no poll report before T -> excluded (logged). Bank forward
guidance is recorded as a note, not used as C. Consensus is collected BLIND to outcomes (outcomes kept in a separate
file not printed until consensus file is frozen with its sha256 noted here).

Rule CB1 (primary, taker): for every event with s >= 0.90 at T, for every market whose bucket != C bucket and whose
yes_bid >= 0.04: buy NO at no_ask = 1 - yes_bid. Fee per Kalshi formula on a C = 10 contract order:
ceil_to_cent(0.07 * 10 * p * (1-p)) / 10 per contract, p = NO price. Stake = price + fee; payout 1 if NO.
One bet per (market, T); D-7 and D-1 pooled (same meeting clustered).
Placebo P0 (no external info): same rule with C replaced by Kalshi's favorite at T (highest yes mid) and s ignored.
CB1 must beat P0 to credit consensus; if both pass it is a longshot-NO effect, not a consensus effect.

Success (dev): n_bets >= 20 AND net ROI > 0 with 95% CI lower > 0 from a MEETING-cluster bootstrap (10k) AND
CB1 ROI - P0 ROI on matched meetings CI > 0. Kill: n < 20 -> "no capacity" (dead); CI crosses 0 -> "not distinguishable".
Time split: dev = decisions <= 2026-08-31; sealed = 2026-09-01 .. 2026-11-20 (untouched unless dev passes + lead yes).
Calibration (descriptive, all quoted markets): Kalshi mid at D-1 vs outcome by bin; consensus surprise rate vs outcome.
Expected failure mode stated up front: small n (Kalshi history ~1 year, ~29 series, minority buckets quoted < 4c).

## Log
- 2026-11-20: series + events census (above). Prereg written before prices/outcomes.
- Amendments (all before any outcome was read; outcomes sit unprinted in results/cbank_thin/outcomes_HIDDEN.json):
  (a) KXBOE (2023-24, cumulative "Above X%" ladder, 24h volume 0) excluded: "minority bucket" undefined on a cumulative ladder.
  (b) T anchor: close_time is sometimes AFTER the announcement (KXCBDPOLAND-26SEP02 closes 09-09, 7 days late;
      AUSTRALIA-26FEB03 closes 02-04). Anchor A = min(close, ticker date 00:00 UTC) (Canada tickers have no day -> close);
      T = A - 7 d / A - 1 d. Without this, D-1 on POLAND-26SEP02 would have been after the decision (lookahead).
  (c) Price units: /historical candles carry "close" in DOLLARS (MEXICO-26FEB05-HOLD bid .97/ask .98), live candles
      "close_dollars". First frame divided by 100 -> favourite mids 0.009; fixed in results/cbank_thin/frame.py.
  (d) Consensus collection scope: every dev event where any non-favourite bucket has yes_bid >= .04 at D-7 or D-1
      (27 events; these are the only events where CB1 can bet unless consensus disagrees with Kalshi's favourite) plus a
      seeded random sample of 7 other dev events (seed 20261120) to measure consensus-vs-favourite agreement.
      Targets: results/cbank_thin/consensus_targets.json (34 events).
- Census (results/cbank_thin/frame.py output): 613 closed markets, 115 events, 27 series; Kalshi history starts 2025-10
  (Canada) / 2025-11..12 (Asia/EU/LatAm majors); most small KXCBD* banks only from 2026-07..09. Events by month:
  2026-09 20, 2026-10 16, 2026-11 11 (to 11-20); earlier months 3-9. Dev (anchor <= 2026-08-31, ex-BOE) 62 events;
  sealed 47. Spread (ask-bid) at T: median .02-.03, 75th pct .04-.05 (quotes_q.csv).
  Minority (non-favourite) buckets with yes_bid >= .04: dev D1 28 cells/23 events, D7 19/16; sealed D1 22/20, D7 20/17.
  Median favourite mid at D-1: dev .965, sealed .8275.
- Consensus frozen 2026-11-20 BEFORE outcomes: results/cbank_thin/consensus.json sha256 a2326609fda4af63da81edce2224be5ec6057218c0e5baeeeb86cd8cd8b1d79a
  (merged from consensus_part{1,2,3}.json, 3 helper agents, web search; 68 (event,T) rows, 49 with a pre-T poll, 18 with
  s >= .90). Only lead edit: AUSTRALIA-26MAR16 D1 -> Finder survey (02:07Z, hold 23/37) replaces Reuters (00:38Z, hike
  23/30) under the prereg "latest pre-T poll" rule (both s < .90; CB1 unaffected). Known weaknesses: Israel + SARB no
  pre-T poll found (403s / only post-decision citations); Korea Reuters polls land ~00:40 UTC on T_D1's date, i.e. just
  after T, so local Korean surveys used; several "latest" articles carry only "majority" wording (s null -> excluded
  even when an earlier pre-T article had counts, e.g. RUSSIA-26MAR20 D1 TASS 20/20 vs Interfax "overwhelming majority").

## Result (dev, results/cbank_thin/eval.py -> eval_dev.log, bets_CB1_dev.csv, bets_P0_dev.csv, calibration_D1_dev.csv,
## consensus_vs_outcome_dev.csv)
- CB1 (prereg): n 6 bets / 6 meetings, 0 losses, mean NO price .932, ROI +6.78% [+4.68, +9.53] meeting bootstrap (degenerate
  with 0 losses); Clopper-Pearson loss-rate upper .459 vs break-even .064. **n < 20 -> KILL: no capacity (dead on dev).**
  D1 5 bets, D7 1. The 6: CHINA-25DEC22 cut 9c (poll 25/25 hold), MEXICO-26MAY07 hold 5c (32/35 cut), MEXICO-25DEC18 hold 5c
  (29/29 cut), JAPAN-25DEC18 hold 4c (50/50 hike), KOREA-26JUL15 hold 6c (10/10 hike), INDIA-26AUG05 D7 hike 12c (68/72 hold).
  24 h volume 0 .. 38,751 contracts (median of the 6 ~ a few hundred).
- Placebo P0 (fade every Kalshi non-favourite with bid >= .04, no outside info): n 47 / 27 meetings, 10 losses, ROI -8.02%
  [-22.87, +4.62] -> not distinguishable, negative point. Losses = contested meetings (Russia, Brazil, Mexico-26MAR26, SARB).
  CB1 - P0 on the 6 matched meetings: [-0.8, +1.5] pp -> not distinguishable (on those meetings P0 bet the same markets).
- Consensus vs outcome (pre-T polls only): s >= .90 hit 18/18. All-strength D1 hit .833 (n 30), D7 .706 (n 17).
  Kalshi favourite hit D1 .877 (n 65), D7 .818 (n 44) (different, larger set). Consensus == Kalshi favourite in .851 of 47
  cells; on the 7 disagreements Kalshi was right 4 (4 meetings), consensus 3 (2 meetings) -> no evidence polls beat Kalshi.
- Kalshi D1 calibration (dev, 309 market-cells): mid <= .02 realised 0/171; .02-.05 mid .032 realised .029 (n 35);
  .05-.10 mid .075 realised .062 (n 16); .4-.6 mid .487 realised .278 (n 18, 7 meetings); upper bins .9-1.0 all hit (n 47).
  Tails roughly calibrated; middle bins too few meetings to read.
- Data caveats: 2 (event,T) cells had the winning bucket unquoted at T (INDIA-26JUN05 D7, ISRAEL-26JUL06 D7), so "favourite"
  there was chosen among quoted markets only. Consensus for Israel/SARB unobtainable pre-T (paywall/403).
- edgelab: cb1_cbank_consensus_fade-c7f44bb9 registered before prices/outcomes (family kalshi_data_ladders, bar p < 7.46e-4),
  retired dead. Sealed 2026-09-01..11-20 (47 events) untouched.
- Not run (forking path, not prereg): using an earlier count-bearing pre-T poll when the latest article only says "majority"
  (e.g. RUSSIA-26MAR20 D1 TASS 20/20; would add at most a handful of bets, still n < 20).

## Literature: how often central banks surprise consensus
- Bloomberg Economics (2018, "Emerging Central Banks Surprise Most on Policy, Fed the Least",
  https://www.bloombergquint.com/global-economics/emerging-central-banks-surprise-most-on-policy-fed-the-least): G20 since
  2010; BRICS banks averaged 1.7 surprises/year (2.5 in policy-change years); Fed least.
- IMF WP/24/224 (high-frequency MP shock dataset, 29 countries): surprises centre on zero; EMs show more dispersion and
  fatter tails. Checo, Grigoli & Sandri (IMF 2024 / BIS WP 1170): EM forecast-error shocks from Consensus Economics.
- Here: unanimous-ish polls (s >= .90) never missed in 18 dev cells; split polls missed often (D1 overall .833).

## Proposal (not done; needs lead/user)
- If anyone wants this revisited: a forward paper logger that snapshots KXCBD* quotes at D-1 and records a poll per meeting.
  Expected rate: dev produced 6 CB1 bets in ~9 months on 62 events; even at the sealed-period pace (~17 events/month) 20 bets
  takes months, at a few hundred contracts of 24 h volume and ~+4-6c/contract. Low priority: consensus collection is manual.

## Learnings
### Kalshi thin central-bank markets: consensus-fade has no capacity
CB1 (fade non-consensus buckets when a pre-T poll is >= 90% one way, bid >= 4c) found 6 bets in 62 dev meetings (2025-10..
2026-08), 0 losses, ROI +6.8% -- dead by prereg (n < 20). Source: results/cbank_thin/eval_dev.log. Why: unanimous polls are
usually matched by a 95c+ Kalshi favourite; minority buckets >= 4c mostly occur on genuinely split meetings.
### Fading Kalshi's non-favourite CB buckets blindly loses
P0 n 47, ROI -8.0% [-22.9, +4.6]; the 4c+ minority prices sit on contested meetings (Russia, Brazil, Mexico) and hit 10/47.
Source: results/cbank_thin/bets_P0_dev.csv. Kalshi CB tails look roughly calibrated (mid .032 vs realised .029, n 35).
### Kalshi close_time on CB decision markets can be days after the decision
KXCBDPOLAND-26SEP02 closes 09-09; AUSTRALIA-26FEB03 closes 02-04. Anchor decision-time snapshots on the meeting date
(ticker date 00:00 UTC), never close_time minus k days. Source: results/cbank_thin/event_dates.csv.
### Kalshi /historical candles use "close" in dollars, live ones "close_dollars"
Dividing "close" by 100 gave favourite mids of 0.009. Check units on both endpoints. Source: results/cbank_thin/frame.py.
### Free pre-decision economist polls exist for most banks but not all
49/68 (event,T) cells found a pre-T poll via free news (Reuters reprints on fxstreet/radio sites, Citi Mexico survey, Focus,
Vedomosti/Izvestia, Korean press). Israel and SARB failed (403/paywall); Reuters BOK polls land ~00:40 UTC decision-eve.
Trading Economics guest API is discontinued. Source: results/cbank_thin/consensus.json.

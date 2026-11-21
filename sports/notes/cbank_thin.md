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

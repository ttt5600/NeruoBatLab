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

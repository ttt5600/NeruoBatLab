# Polymarket EV by time-to-resolution (user idea 2026-11-06)

Question: does the return of buying at price p depend on the time left to the SCHEDULED resolution?
Code: `scripts/pm_timing.py` (enum / prices / panel / describe / frame build). Rule: `edgelab/rules/pm_timing.py`.
Outputs (gitignored): `data/pm_timing/panel.parquet`, `tables.json`, `frame.parquet`, `trades.parquet`.

## Panel (DEV only, 2025-05-01..2026-04-30, the 92 smallmarket sampled days)
19,955 resolved binary markets / 4,375 events; 62,350 market-horizon rows after dropping settled prints (<0.5c) and
untraded default 0.50 displays. Rows per horizon: 30d 2,219 / 7d 5,957 / 3d 10,002 / 1d 13,793 / 6h 15,183 / 1h 15,196.
S = min(endDate, gameStartTime) as stored by gamma; t = S - h; row only if open at t and a display price <= t within
6 h (asserted). Price = CLOB display (mid, or last trade if spread > 10c). 95% CIs = event-cluster bootstrap.

Wrong turn worth keeping: the first pass showed a big 35-50c "longshot" gap. It came from p = 0.50 exactly (the
default display of an untraded book): folding put token0 ("Yes") on the longshot side, and Yes loses ~56%. Dropped.

## Result: favourite return at the MID (no costs), all categories
| h | fav return [CI] | longshot gap (won - price) [CI] | mean lockup |
|---|---|---|---|
| 30d | +5.0% [+2.6, +7.3] | -3.4 pp [-4.9, -1.7] | 31.7 d |
| 7d | +4.9% [+3.1, +6.5] | -3.3 pp [-4.3, -2.2] | 9.0 d |
| 3d | +2.7% [+1.4, +4.1] | -2.0 pp | 4.7 d |
| 1d | +3.0% [+2.0, +4.1] | -2.1 pp | 2.5 d |
| 6h | +1.8% [+0.7, +2.9] | -1.3 pp | 1.6 d |
| 1h | +1.7% [+0.6, +2.8] | -1.2 pp [-1.9, -0.5] | 1.4 d |
Per trade, the favourite-longshot gap roughly triples from 1h to 7d (CIs at 1h and 7d do not overlap). Per DAY of
capital it is the reverse (shorter is far better), so "lockup discount" is consistent but not the whole story.
By category the long-horizon gap is mostly culture_tech (-7 pp at every horizon, owned by the H2a test) and sports
futures at 30d (-11 pp, 76 events). Politics is flat ~-2 to -3 pp at every horizon. Token0 (Yes / first-listed side)
priced 35-65c: -8.7 pp at 7d vs -3.1 pp at 1h.
Kalshi (smallmarket k_priced2, mids, 7d/1d/1h): favourite +1-6% at mid in every horizon, no monotone time pattern;
at the ask + fee every cell is <= ~0 (e.g. 7d 10-20c -3.3% [-7.1, +0.2]).

## Trials (family polymarket_calibration, 8 trials -> bar p < 0.00625), registered before any tape was pulled
Maker bid on the favourite (display 0.65-0.95) at the display rounded down, fill only on a print THROUGH the bid,
categories except mentions/culture_tech. edgelab dev verdicts:
- pm_t1_fav7d_maker-7db3611f (S-7d): FAIL, n 1357, net ROI -2.5% [-4.8, -0.3], p .98, halves -3.4/-1.6%, CLV -0.0003 -> dead
- pm_t2_fav30d_maker-b127fc20 (S-30d): FAIL, n 671, net ROI +0.1% [-4.1, +4.9], p .47, halves +0.5/-0.4% -> dead
Mechanism (adverse selection): 7d, 1,766 bids posted, 77% filled. Unfilled favourites won 100% (price ran away), filled
ones won 78.6% at a mean price of 0.806. The whole mid-price edge (+3.3% on all posted) sits in orders that never fill.
Sealed (2026-05..10, 46 sampled days, mentions/culture excluded) was never unsealed.

## Open
Taker at the real ask cannot be backtested (no historical Polymarket asks); a forward book logger at S-7d would be the
only honest test. Polymarket US (regulated, US users) lists mostly sports; the long-dated politics/culture markets
behind the gap are on the international venue, which US users cannot use.

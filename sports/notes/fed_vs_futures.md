# fed_vs_futures -- Kalshi FOMC decision tails vs fed-funds-futures implied probabilities

Agent: fed_vs_futures (research subagent, 2026-11-18). Read-only; no orders, no git.
Prior work checked: FUND_BOOK 3d, LEARNINGS_2026-11 (LX1 = KXFED vs KXFEDDECISION internal consistency only, 0 hits;
econ-event ladders dead: seeded by makers; OT1 extension: Kalshi crypto tails only 1-2c rich, eaten by spread + fee).

## Question
E4 pattern on Fed markets: are Kalshi FOMC-decision TAIL outcomes (hike, 50bp+ cut, or whichever outcome futures put
near zero) priced richer than futures-implied after fees, and is the rich price stale (little trading)?

## Data (availability checked before any outcome/price was computed)
- Kalshi: FEDDECISION-23MAY .. FEDDECISION-24NOV (12 meetings) and KXFEDDECISION-24DEC .. -25DEC (9) = 21 dev meetings,
  5 markets each: Hike >25, Hike 25, Hold (H0), Cut 25, Cut >25. Prices via /historical/markets/{tk}/candlesticks
  (fallback /series/.../candlesticks), hourly, through scripts/kx_throttle.py. 2026 meetings (26JAN..26OCT) SEALED.
- Futures: Yahoo `ZQ=F` (30-day fed funds, CBOT, continuous front month) daily close, 2022-01..; expired single
  contracts (ZQZ24.CBT etc.) are NOT on Yahoo (probe returned 0 rows), so only the CURRENT-month contract is usable.
- EFFR: FRED `EFFR` daily (raw/effr.csv).
- Atlanta Fed MPT (raw/mpt_histdata.xlsx, SOFR-options, 3-month IMM windows) downloaded; NOT used for the primary rule
  (windows are 3-month averages, not per-meeting). Robots.txt allows; possible later cross-check only.
- CME settlement pages not used (scraping ToS); FedWatch historical not free.

## Pre-registration (written before any Kalshi price or futures-implied probability was computed)
Disclosure: I know from general knowledge some realised decisions (e.g. 2024-09 was a 50bp cut). The rule below is fixed
from the brief, not tuned.

Futures-implied P (method F1, "same-month ZQ arithmetic", FedWatch-style):
- Announcement day A (Kalshi close_time date, ET). New rate effective A+1. Meeting month M = month of A.
- Decision day D = A - k calendar days, k in {14, 7, 1}; decision instant D 15:00 America/New_York.
- Inputs known at decision: ZQ=F close of the last trading day < D (no same-day close -> no lookahead), EFFR values dated
  <= D-2 (FRED publishes day t at ~09:00 ET t+1).
- Valid only if that ZQ close date is inside month M (then ZQ=F is the month-M contract). Otherwise the (meeting, k) cell is
  skipped (expected: early-month meetings at k=14/7). Roll check: verify ZQ=F on the first trading days of a month ~
  100 - EFFR level (contract switched), else mark invalid.
- Calendar EFFR path for month M: observed EFFR (ffill to calendar days) for days <= D-2; days after that through A at the
  last observed EFFR r0; days A+1..end at r_post. implied avg = 100 - ZQ. Solve r_post; delta = r_post - r0 (bp).
- Map delta to P: delta/25 = k0 + f (k0 = floor); P(move 25*k0) = 1-f, P(move 25*(k0+1)) = f, all other outcomes 0.
  Kalshi buckets: Hike>25 = moves >= +50, Hike25 = +25, Hold = 0, Cut25 = -25, Cut>25 = <= -50.

Kalshi price at decision: the last hourly candle ending <= decision instant; yes_bid close, yes_ask close.
Skip if no candle in the 24 h before the decision instant.

Rule FT (taker, primary): for each of the 5 outcomes with P_f <= 0.05 and Kalshi yes_bid >= P_f + 0.03 and yes_bid >= 0.03:
buy NO at 1 - yes_bid, fee 0.07*P*(1-P) per contract (exact), stake = price + fee, payout 1 if outcome NO.
One bet per (market, k). Pooled over k = 14, 7, 1.

Rule FM (maker, secondary): offer YES at max(yes_bid + 0.01, yes_ask - 0.01) when that >= P_f + 0.03 and P_f <= 0.05;
fee 0; filled iff any later hourly candle before the next decision time / close has a trade price high >= our price.
Report filled vs unfilled loss rates (adverse-selection check, as E1/QM2).

Success (dev): n_bets >= 20 AND net ROI > 0 with 95% CI lower > 0 from a MEETING-cluster bootstrap (10k), AND exact
Clopper-Pearson 95% upper bound of loss rate < break-even loss rate (mean stake). Report the exact binomial CI on losses.
Kill: n_bets < 20 -> "no capacity" (dead); ROI CI crosses 0 -> "not distinguishable" (dead on dev).
Staleness report (descriptive): volume in the 24 h before decision for flagged markets; how long the bid persisted.
Capacity: yes_bid resting size is not in candles; report volume-based upper bound.
Sealed: 2026 meetings untouched; would be scored once only with the lead's/user's yes.

## Log
- 2026-11-18: data availability probed (above). Prereg written. Next: build frame (results/fed_vs_futures/).
- 2026-11-18 data: scripts/fed_vs_futures_fetch.py -> results/fed_vs_futures/{markets.json (110 mkts, 22 dev events incl
  23MAY's 6-bucket layout), candles.json, zq_front.csv (755 rows)}. Pre-2025-05 candles exist only for hours with trades
  (tails often 0-20 candles in 20 d); 2025-05+ near-hourly. Prereg's "candle within 24 h" filter applies.
- Amendment (math necessity, before results printed): d2 = 0 (meeting on the month's last day, e.g. 24JAN31) -> invalid
  (contract carries no post-meeting days). First run crashed on it.
- ZQ=F roll check (non-meeting months, n 293 days): |ZQ - (100 - month-avg EFFR)| median 0.13 bp, max 5.3 bp (Mar 2023
  SVB week) -> ZQ=F is the current-month contract; method valid.

## Result (scripts/fed_vs_futures_eval.py -> results/fed_vs_futures/{frame.csv, bets_FT.csv, bets_FM.csv})
- Coverage: 44/66 (meeting, k) cells valid (22 invalid: early-month meetings at k=14/7, month-end meetings); 219 market-
  cells, 167 with a quote.
- Calibration (quoted cells): futures P_f = 0 (n 89): Kalshi mid .010, bid .005, realised 0. P_f in (0,.05] (n 15): mid .023
  vs P_f .024, realised 0. P_f (.05,.2] (n 14): Kalshi mid .050 vs P_f .085, realised 0. Upper bands track each other
  within 2-8c. **Kalshi tails are NOT priced richer than futures; if anything Kalshi is more confident (cheaper) on
  futures' 5-20% outcomes.**
- FT (prereg taker): n = 7 bets / 5 meetings, 0 losses, ROI +3.7% (meeting bootstrap degenerate with 0 losses;
  Clopper-Pearson loss rate [0, .41] vs break-even .036). n < 20 -> **KILL: no capacity (dead on dev).** All 7 are
  1-pt-wide quotes at 3-6c with 24 h volume 33k-1.17M contracts -> nothing stale; this is a liquid, tight book.
- FM (maker): 12 candidates, all 12 "filled" by later trade prints, 0 losses; ~+4.1c/contract on 12 -- same tiny n.
- Sanity rows: 24SEP T-1 futures 57.5% for 50bp vs Kalshi 56-61c (realised 50bp); 24DEC T-14 futures 71.5% cut vs
  Kalshi 71-72c. The two venues agree closely near the meeting.
- Descriptive only (post hoc, not a rule): 24SEP T-7 futures 30% for 50bp vs Kalshi bid 5c -> 50bp happened. Kalshi
  under-priced a futures-supported tail once; the (.05,.2] band went the other way 0/14. Too few meetings to test.
- edgelab: fed_ft_tail_no-3d150d6a registered before outcomes (family kalshi_data_ladders, bar p < 7.94e-4), retired dead.
- Sealed 2026 meetings untouched.

## Learnings
### Kalshi FOMC-decision tails are not rich versus fed-funds futures
ZQ-implied P = 0 outcomes: Kalshi mid 1.0c (n 89 quoted market-cells 2023-05..2025-12); P_f 5-20%: Kalshi mid 5.0c vs
P_f 8.5%. Only 7 taker NO bets passed P_f <= .05 & bid >= P_f + 3c in 22 meetings. Source: results/fed_vs_futures/frame.csv.
Why: the E4 stale-tail pattern needs a neglected book; Fed books trade 30k-1M+ contracts/day at 1c spreads.
### Free per-meeting fed-funds futures history = Yahoo ZQ=F front month only
Yahoo drops expired ZQ contracts (ZQZ24.CBT etc. return 0 rows); ZQ=F is the current-month contract (roll check median
0.13 bp error). So futures-implied P only exists for decision days inside the meeting month; 22/66 cells lost. Atlanta
Fed MPT (mpt_histdata.xlsx, SOFR options) is free but on 3-month IMM windows, not per meeting.
### Kalshi pre-2025 candles only exist for hours with trades
FEDDECISION tails have 0-20 hourly candles over 20 days before 2025-05; from 2025-05 they are near-hourly. Any quote-based
backtest on older Kalshi must require a recent candle and will lose most tail rows.

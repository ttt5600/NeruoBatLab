# ForecastEx daily weather beyond E5 highs: lows, international highs, legacy highs (DEV, 2026-10-28)

## Verdict
**All 3 families FAIL. The real finding is that they cannot be tested from the daily files: these markets trade on the event day itself, not the day before.**
- **Precipitation is not testable at all.** ForecastEx has no daily rain contract; rain is monthly only (TR*).
- **F1, US daily lows (UL), 24 airports:**
  - 6,132 rule candidates. Only 77 of them traded on D-1.
  - T1 (last trade): +2.67u, ROI +3.5%, p .35.
  - T2 (day-high trade): -4.63u, ROI -6.0%.
  - FAIL.
- **F2, international daily highs (SH), 8 cities:**
  - 6,056 candidates. 7 bets had a D-1/D-2 trade.
  - Fails the loss check (Wilson upper bound 35% against a 10% break-even).
  - FAIL, with nothing to read from it.
- **F3, legacy US daily highs (DH), Nov 2025 to Feb 2026:**
  - 1,938 candidates. 0 traded on D-1, so there are 0 bets.
  - FAIL, untestable.
- **Is this the dead Kalshi mechanism? Partly yes, and the data can't say more.**
  - The consensus forecast itself is accurate on lows. Across all candidates, regardless of whether they traded, the bought side lost:
    - UL: 1.9% [1.6, 2.3]
    - SH: 0.6%
    - DH: 6.8%
  - So the model side works here, as it did on Kalshi.
  - The problem is price. The few UL picks that did trade the day before lost 18% of the time (14/77), against 1.9% for all candidates. That is adverse selection: a lows contract that trades below about 0.80 the day before is a contested one. It is the same pattern that killed Kalshi X2 (lows): the cheap prints are the informed ones.
  - The price that matters, roughly 0.88-0.90 on an *untraded* candidate, does not exist in daily files.
- **No sealed ask.** The only thing that could resolve this is the step E5 already needs: a forward paper logger on live IBKR bid/ask. Adding UL and SH to that logger costs nothing.

## Pre-registration
Full text is the docstring of `scripts/forecastex_wx2.py`. It was written and sha-hashed (`48951024…`, 02:56 UTC) before any price was joined to a forecast or an outcome.

**Families:**
- **F1, UL lows:** 24 K-airports. Model value = min over local hours 0-23, chosen a priori because the contract is the calendar-day minimum. Kalshi X2 had used hours ≤ 18.
- **F2, SH highs:** the 8 catalogued cities with identified stations: EDDF, LFPO, RJTT, WSSS, YSSY, CYHC, CYUL, CYYZ. Strikes are in Celsius, margin 5/3 C. Five delisted SH codes have no identifiable station and were excluded a priori.
- **F3, DH legacy highs:** 22 airports. NYC, TOR, YUL and YVR were excluded because their stations are ambiguous.
- **DL legacy lows:** under 10 event days, so dropped and not counted as a trial.

**Rule:** E1/E5 frozen. Five Open-Meteo lead-1 models (4 outside CONUS), at least 3 required, margin 3F.

**Contract direction:** taken from settlement ladders only.
- UH-style "exceed" for SH and DH: 100% of informative ladders were non-increasing.
- **UL is "below X"**: YES iff low < X, with 2,886/2,886 ladders non-decreasing. The rule was mirrored for UL as pre-registered.

**Pricing:**
- T1: the last trade in the D-1 file (D-2 for Europe, Asia and Sydney, to avoid lookahead), plus $0.01.
- T2: that day's high trade, plus $0.01.

**Bar and verdict:**
- 6 trials, so p < .0083.
- A pass also needs the Wilson loss upper bound to be below break-even.
- Fewer than 500 bets means PASS-UNDERPOWERED at best.

**Ledger:** 6 rows, dataset `forecastex_wx2`, all marked fail.

**Code fixes after hashing (no rule change):** `df.sem` resolved to pandas' `DataFrame.sem` method instead of the column. The first run therefore marked every station-day non-monotone and scored 0 bets. I fixed it to `df["sem"]`; the final sha is `07e53ca1…`.

## Results (DEV, event dates < 2026-08-15; input = Q007 scratch pull of contracts expiring < 08-15)

| family | station-days | candidates | traded on price date | T1 bets / losses | T1 units, ROI, p | T2 units, ROI | verdict |
|---|---|---|---|---|---|---|---|
| F1 UL lows | 2,983 (Feb 13 to Aug 13) | 6,132 | 77 | 77 / 14 (18.2%, b/e 20.6%) | +2.67, +3.5%, .35 | -4.63, -6.0% (b/e 14.1%) | FAIL |
| F2 SH intl highs | 1,390 (Feb 12 to Aug 14) | 6,056 | 7 | 7 / 0 | +0.84, loss check fails | +0.84 | FAIL |
| F3 DH legacy highs | 1,468 (Nov 6 to Feb 10) | 1,938 | 0 | 0 | none | none | FAIL |

**F1 detail:**
- July: +6.95/34. August: -4.28/43.
- 19 of the 77 bets were at LAS (-1.2u) and 18 at OKC (+1.1u).
- Mean T1 cost was 0.79, so the trades that existed were cheap, contested prints.

**Where the trading actually happens** (contract rows with pair_quantity > 0, lag = event date minus trade date):

| family | lag +1 (D-1) | lag 0 (event day) | lag -1 (file dated D+1) |
|---|---|---|---|
| UL | 3,024 rows / 72k pairs | 29,998 / 3.65M | 8,504 / 1.38M |
| DH | 0 | 13,250 / 224M | 9,328 / 46M |
| SH | 804 / 47k | 2,166 / 289k | 182 / 9k |

- D-1 trading on UL starts only in July 2026. The same thing happened on UH, where 128 of E5's 159 DEV bets were in Jul to Aug.
- Event-day prices cannot be used. The daily file has no timestamps, so a day-D last trade can come after the minimum or maximum has been observed.

## Side observation (UNVERIFIED, matters for E5 timing)
- **The evidence:** UL and DH contracts expire at about 00:00-03:00 Central on D+1, yet carry a lot of trading in files dated D+1 (lag -1).
- **What it suggests:** the daily-file trade date may be **UTC**, not Central as assumed in E5. If so, for US stations the D-1 file ends at 19:00-20:00 ET / 16:00-17:00 PT on D-1. That means E5's prices are staler than assumed, with **no** East-coast lookahead. That is conservative for E5.
- **Not settled:** SH's 182 lag -1 rows don't fit cleanly with an 18:00 CT expiry. Treat this as a hypothesis to check against live IBKR timestamps.

## Next
- No sealed test is asked for: there is nothing to seal.
- When the E5 forward paper logger on live IBKR bid/ask exists, log UL and SH consensus candidates in it too, at the same 22:00 local decision time.
  - Pre-register them as separate arms: UL "below" semantics, SH Celsius margin 5/3.
  - The all-candidate accuracy (UL 1.9% and SH 0.6% losses) is the reason to bother. Whether they are buyable at 0.88-0.90 is the open question.

Files:
- `scripts/forecastex_wx2.py`
- `data/forecastex_wx2.parquet` (gitignored data dir)
- `lab/ledger.jsonl` (6 rows)
- Nothing committed, no orders.

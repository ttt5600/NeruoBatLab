# More volume for E1/E2: new Kalshi daily weather series (2026-10-28, DEV-only, no orders)

## Verdict
**Nothing can be added to the live bot. Added nightly volume: $0.** As of 2026-10-28, Kalshi lists no daily-high city
we don't already trade and no other forecastable daily weather series. No trials were scored (0 qualifying series), so
the taker and maker backtests never ran: there was no history to run them on.

## What was checked (scripts/kalshi_wx_new.py `census`; pre-registration in its docstring, written before any data)
- 458 "Climate and Weather" series on the public API, plus a ticker scan of every other category (only legacy
  `HIGHNY0`/`KXHIGHNY0` and old city rain series turned up, all inactive).
- **Our 24 KXHIGH* series:** all open, 12 markets each (2 event-days x 6 brackets), ~1.18M contracts of 24h volume combined.
- **Other daily-high tickers:** none tradeable.
  - Never listed a single event: 21 international `KXHIGHT*` (London EGLL, Paris LFPG, Toronto CYYZ, Tokyo RJTT, Seoul,
    HK, Singapore, Sydney, Beijing, Shanghai, Mumbai, Dubai, Mexico City, Sao Paulo, Istanbul, Geneva, Berlin,
    Frankfurt, Brussels, Amsterdam), plus `KXHIGHTKSAN`, `KXDVHIGH` (Death Valley), `KXDENHIGH`, `KXHOUHIGH`, `KXHIGHOU`.
  - Retired: `KXHIGHHOU` (70 days, 2024-12..2025-01; Houston now trades as KXHIGHTHOU, which we already have),
    `KXCITIESWEATHER` (2025-02..03), `KXHIGHUS` (2023).
- **Other daily weather:** there is no daily snowfall, max wind or gust series. Snow is monthly or seasonal only, and
  the L39 monthly logger already covers that. Daily rain (KXRAIN etc.) and lows were skipped as already dead.
  `KXBIGGESTQUAKE` is daily but not weather-forecastable.
- **Near-miss, not daily:** weekly average temperature `KXAVGTK*` (11 cities; OKC and AUS each had about 270-300k
  contracts). It ran only 2026-08-24..09-27 (8 events per city) and nothing is open now. Too short to test, and nothing to trade.

## Where volume could actually come from (observation, not a trial)
The live paper log (`~/.claude/quant-kalshi/data/kalshi_paper.jsonl`, 148 picks, 12 nights 10-16..10-28) shows a
median of 14 picks per night and a median of **~175 contracts resting within 1c** per pick (25th percentile 42). For the
current picks, the binding limit is our own caps ($15-30/day, 1-3 contracts/order), not market depth or the number of
series. The median pick's NO price is 0.99, though, so most of that depth earns well under 1c per contract. Scaling up
means raising caps and switching to maker (E4, already a sealed pass), not adding cities.

## Re-check trigger
If Kalshi starts listing events in the international `KXHIGHT*` series, run `census` again. Any series with >= 86
settled days (60 DEV days at 70%) is then scored under the frozen pre-registration above. International stations
would also need a check of their settlement source and units (C vs F) before the 3F margin applies.

Ledger: one `kalshi_wx_new` row, status `no_trial`, 0 trials (lab/ledger.jsonl).

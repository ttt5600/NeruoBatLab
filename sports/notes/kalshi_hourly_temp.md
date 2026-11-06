# Kalshi weather sibling census + HT1 hourly temperature maker (2026-11-06)

**Verdict (edgelab, verbatim): `ht1_hourly_maker-9c9e238c` dev FAIL -> dead.** n 342, net ROI +1.16% [-1.08, +2.96],
p 0.130 vs bar 1.11e-3 (45 trials in kalshi_weather_temp), halves +0.16% / +2.16%, top week 2026-W37 = 59.8% of profit,
CLV +1.6c. No sealed run.

## Census (scripts/wx_census_all.py -> data/wx_census/{series,markets}.json; counts and volume only)
501 Climate-and-Weather series (daily 144, custom 141, annual 75, monthly 72, one_off 36, weekly 19, hourly 14).
Only one untested family has n in the thousands: **hourly temperature** (KXTEMP*H to 09-10, then *HS; MIAH runs both).
| series | freq | settled events | lifetime contracts | status |
|---|---|---|---|---|
| KXRAIN (multi-city daily rain) | daily | 108 | 79.9M | R1 sealed FAIL (already dead) |
| KXTEMPMIAH | hourly | 1,965 | 26.0M | live; HT1 |
| KXTEMPLAXH / NYCH / CHIH / AUSH / DCH | hourly | ~1,500-3,800 each | 3.7-10.1M | ended 09-10; HT1 dev |
| KXTEMPNYCHS / CHIHS / LAXHS | hourly | ~1,320-1,340 | 5.0-8.0M | live since 09-10; HT1 |
| KXSNOWSTORM | one-off | 6 | 10.6M | too few events |
| KXNYCSNOWM + other *SNOWM / KXRAIN*M | monthly | 4-33 | 0.4-6.6M | = L39 (forward lead) |
| KXHMONTH / KXHMONTHRANGE (global temp) | monthly | 32 / 28 | 4.2M / 1.2M | too few for a 1e-3 bar |
| KXTORNADO | monthly | 40 | 4.1M | too few |
| KXHURCAT | custom | 95 | 3.6M | hurricane category; too few |
| KXRAIND* (per-city daily rain) | daily | 0-3 | ~0 | new, no history |

## HT1 rule (edgelab/rules/kalshi_wx_hourly.py, frozen before any price/outcome pull)
Decide at T = market open + 15 min (H - 45 min). Input = the series' own published value for H-2 (only when its
settlement_ts <= T, which held for 45% of events; median settle lag 81 min) + each Open-Meteo model's lead-1 change
fc(H)-fc(H-2). All >= 3 models miss K by > 3 F -> maker bid on that side at min(floor mid, ask-1c), q in [.50,.99],
filled iff >= 10 contracts print at <= q in (T, H]. Split (vault kalshi_temp_hourly): dev 04-16..09-30,
sealed 10-01..11-05, forward 11-07+.

## Why it fails / capacity
3,800 candidate events -> 8,996 selected tickers (all splits) -> only 342 dev fills in 71 days (~5 fills/day at 10 contracts):
far strikes sit at 0/1c with no prints. 8 losses (2.3%) at mean price 0.965, so the edge is ~1c on ~50 contracts/day (~$0.50/day expected profit at best).
MIAH = 172 of the 342 fills. All picks were NO (YES-side never cleared the quote filter or never filled).
Data: data/kalshi_hourly_temp/ (markets, forecasts, quotes, tape, frame). Builder: scripts/kalshi_hourly_temp.py.

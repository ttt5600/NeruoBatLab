# Physical/mechanical Kalshi ladders beyond weather -- census + SX1 SpaceX + SP1 SPR (2026-11-06)

**Verdicts (edgelab, verbatim status rows):**
- `sx1_spacex_maker  kalshi_data_ladders (34, 1.5e-03) dead  dev  11 +1.58% [+1.11%,+2.27%]  2.5e-05 +0.0155  dead (dev fail): 5c conc>30%,5d n<min`
- `sp1_spr_maker     kalshi_data_ladders (34, 1.5e-03) dead  dev   3 +3.45% [+3.45%,+3.45%]  2.5e-05 +0.0333  dead (dev fail): 5c conc>30%,5d n<min`
Sealed never opened for either. Prereg commit 72369aa (rules + specs registered before any history/quote/tape/outcome pull).

## Census (scripts/phys_census.py -> data/phys_census/markets.json; weather side = data/wx_census, scripts/wx_census_all.py)
15,900 Kalshi series scanned by keyword. Settled events / lifetime contracts (all settled markets):

| series | what | settled ev | contracts | free PIT source / partial observability | note |
|---|---|---|---|---|---|
| KXSPACEXCOUNT (monthly) | SpaceX launches in month | 14 monthly (+3 annual) | 22.9M | Launch Library 2 liftoff times; count observed live | **SX1**; biggest capacity |
| KXNATGASD/W/MON | Henry Hub price | 123/32/7 | 9.8M/2.0M/0.2M | price series | financial, not physical (commodities = covered) |
| KXTORNADO | SPC prelim tornado count / month | 40 | 4.1M | SPC daily storm-report CSVs (observed daily) | best untested lead; per-rung books thin now (100s of contracts), outbreak tails fat |
| KXHURCAT | storm max category | 95 | 3.6M | NHC advisories | ~5-60k/storm; the counterparty reads the same NHC forecast |
| KXMEASLES | annual US cases | 4 | 2.7M | CDC weekly | too few events |
| KXSPRLVL | weekly SPR stock (EIA WPSR) | 20 | 0.9M | EIA WCSSTUS1 (= settlement on all 15 printed values) | **SP1**; 2026 drain dmax 9.9 Mbbl makes rungs reachable |
| KXLAUNCHES / KXLAUNCHCOUNTM | FAA-licensed launches / month | 4 / 1 | 0.7M / 0.6M | LL2 | started 2026-07; forward only |
| KXTXERCOTPEAKD | ERCOT daily peak load | (Q017) | -- | ERCOT forecast | already Q017 |
| KXPJM*/KXMISOIND/KXERCOT* hubs W/M | ISO power price | 2 each (new 10-23) | 10-22k | ISO day-ahead LMPs (observed daily in-week) | too new; forward lead |
| KXGSTORM, KXSEAICE, KXCO2LEVEL, KXEVMKT | geomagnetic / sea ice / CO2 / EV share | 9 / 2 / 0 / 4 | <0.1M | NOAA SWPC / NSIDC / NOAA GML | 2021-22 only or never listed |

Absent on Kalshi: Kp/sunspot dailies, river gauges, wildfire acres, AQI, pollen. Monthly global temp (KXHMONTHRANGE,
28 events) is in the weather census.

## SX1 (edgelab/rules/spacex_count.py)
Daily 16:00Z: c = LL2 SpaceX liftoffs this ET month, L = days left, cap = fastest L-day burst in trailing 365 d;
NO maker on rung K when K+1-c >= cap+2 (first such T, spread <= 10c, q = min(floor NO mid, NO ask-1c) in [.50,.99]),
filled iff >= 10 contracts print at <= q before month end. 27 tickers selected (all splits), 21 filled; dev 11 fills,
0 losses, price .96-.99. Settlement audit (descriptive): LL2 month count = Kalshi expiration_value on 9/10 2026 months;
2025 months LL2 +1..+2 (overcount -> rule fires LATER, conservative). Capacity: 1k-52k contracts printed at <= q per
selected rung -> ~2 rungs/month x ~1c = a few $/month. Real but tiny: the market already prices the cap at ~1c.

## SP1 (edgelab/rules/spr_level.py)
2026 SPR is in an emergency drain (455 -> 282 Mbbl); trailing-104-wk max |weekly change| = 9.92 from 2026-07 on, so
only APR08 rungs (dmax 1.76) were out of reach: 3 fills, one event. The rule cannot fire while the drain dominates dmax.

## TC1 KXTORNADO + PW1 weekly ISO power (2026-11-09)
**TC1 `tc1_tornado_maker-6dcaab12`: DEAD at the pre-registered gate, never scored.** SPC month total as it stands today vs
Kalshi expiration_value, dev months, |diff| <= max(3, 3%): 17/26 = 0.654 < 0.90 (results/tornado_gate.json). Every miss is
SPC above Kalshi (+4..+55; 24DEC 145 vs 90, 25MAR 300 vs 268, 24MAY 571 vs 542): the archive keeps growing after the
10:00 ET first-of-month expiration read, so c from today's archive is not the count anyone could see. Same trap as PortWatch.
Any retry needs a forward first-print logger of NAT_<MON>.json. All rungs are 'greater', so a "passed" rung has no NO leg.
SPC facts: summary/<y>/ruf/NAT/NAT_<MON>.json has daily counts back to 2004 = the filtered daily CSV (raw before filtering existed).
**PW1 `pw1_power_week_maker-25925930`: forward paper, registered, logger NOT installed.** Settlement = all-hours average
of REAL-TIME prices Fri 5 PM ET -> Fri 5 PM ET; day-ahead is not in the settlement, but realised intervals pin part of the week and DA forecasts the rest.
MISO + ERCOT keyless; PJM settlement feed returns 401 without a key, excluded. scripts/power_week.py tick|settle; install
command in scripts/com.jonathanwang.quant-power-week.plist.template. Loader prefers ~/.claude/quant-sports/data/live.

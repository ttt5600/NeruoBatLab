# climate_nowcast -- Kalshi climate-index series vs free nowcasts (agent climate_nowcast, started 2026-11-20)

Question: Kalshi series that settle on a published physical index (NOAA monthly global temp, NSIDC sea ice, NOAA GML CO2)
-- after the measured period ends and a free source (Copernicus ERA5 Climate Pulse, ~2-day lag) already pins the value,
are the far brackets still priced as if uncertain? Mechanism class = NOWCAST (STRATEGY_v2 s.1, the only one that replicated).
Do-not-redo checked: FC1/FC2 ForecastEx GTM/MACD (fx_climate.md, dead n2/n0 on trade-only files), LX2c climate tail
(season-settled), KXSEAICE too few events (phys_ladders.md), tropical/NHC, snow makers. This is the Kalshi version of FC1
with real hourly bid/ask candles (the thing FC1 lacked).

## Census (scripts/climate_nowcast.py census -> results/climate_nowcast/census.json)
| series | what | settled events (+active) | lifetime volume | note |
|---|---|---|---|---|
| KXHMONTH (+HMONTH) | "is month M the hottest M ever" (NOAA cag) | 33 (+1) | 4.51M | binary, 1 mkt/event |
| KXHMONTHRANGE (+HMONTHRANGE) | NOAA monthly land+ocean anomaly, 6 brackets (~0.06-0.07 C wide) | 29 (+1) | 1.27M | ladder |
| KXGTEMP | hottest year ever (GISS) | 4 (+2) | 1.34M | annual |
| KXHOTYEAR / KXWARMING | annual GISS / Berkeley | 1 / 1 | 0.13M / 0.02M | one-offs |
| KXARCTICICEMAX / KXARCTICICEMIN | NSIDC annual max / min extent | 3 (+1) / 4 | 0.31M / 0.27M | annual, too few |
| KXSEAICE, KXARTICICE, KXCO2LEVEL, KXLANINA, KXELNINO | sea ice / CO2 / ENSO | 2,1,1,1,1 | < 0.03M each | too few |
Only the two monthly NOAA series have > 4 events. Sea ice / CO2 / annual = forward-log at most (R4 power 0).
Combined monthly NOAA = ~29-30 month clusters over 2024-05..2026-10 (~1 cluster/month). That is the expected failure mode.

## Pre-registration CN1 (frozen 2026-11-20, BEFORE any Kalshi price or ERA5->Kalshi-settlement comparison was looked at;
## seen so far: ticker/bracket metadata, close times, and -- in the census print -- the expiration_value of 26SEP/26OCT
## (1.33 / 1.30001) and HMONTH 26MAY..26OCT results. No candles pulled.)
Rule `edgelab/rules/climate_nowcast.py:cn1_taker`, builder `scripts/climate_nowcast.py build`, dataset `kalshi_climate_nowcast`.
- pred(M) = mean over all days of M of Climate Pulse ERA5 daily global 2 m anomaly (1991-2020) + c_cal(M), c_cal = mean
  (NOAA cag land_ocean - ERA5 monthly) per calendar month over 2001-01..2023-12 (ends before the first Kalshi event);
  sd = pooled residual sd of that fit x 1.25 (tail inflation: model tails were what killed past rules, STRATEGY_v2 s.2).
- RANGE bracket (lo, hi]: p_yes = Phi((hi-pred)/sd) - Phi((lo-pred)/sd) (open ends -> +-inf; Kalshi bounds as listed).
- HMONTH: p_yes = 1 - Phi((rec - pred)/sd), rec = max NOAA archive value for that calendar month over years < year(M).
- Decision T = 16:00 UTC on day 4 of M+1 (ERA5 last day of M public ~day 2). inputs_as_of = day 3 00:00 UTC.
  No bet if market close_time <= T or month has < all days of ERA5.
- Quote = last hourly candle ending in (T-6h, T]; yes_ask / yes_bid close; null -> no bet. YES ask = yes_ask, NO ask = 1-yes_bid.
- Buy side s if p_model(s) - ask(s) - fee(ask) >= 0.05 and 0.02 <= ask <= 0.97. Fee = Kalshi taker 0.07*C*p*(1-p)
  rounded up to the cent on a C = 10 contract order. Several brackets per month allowed; cluster = month M.
- CLV close = our side's ask on the last candle ending <= min(close_time, day 8 of M+1 00:00Z).
- Split by date = first day of M+1: dev 2024-06-01..2025-09-30 (months 2024-05..2025-08), sealed 2025-10-01..2026-11-15
  (months 2025-09..2026-10, not opened without user yes), forward 2026-11-16+.
- SUCCESS (dev): n >= 30 bets AND >= 10 month clusters AND net ROI cluster-bootstrap 95% CI lower > 0 AND rule ROI > both
  placebos. KILL: CI upper < 0, or n < 30 -> dead as a dev trial, forward-only logger spec (no launchd).
- Placebos (descriptive, same rule, not registered): P1 base-rate shape = pred replaced by the centre of the listed ladder
  (Kalshi's own prior at listing); P2 persistence = ERA5 anomaly of M-1 + c_cal(M) (info available at listing).
- Settlement audit (descriptive, after the above): Kalshi expiration_value vs NOAA cag current archive vs pred.

## Log
- 2026-11-20 fit (scripts/climate_nowcast.py fit -> data/climate_nowcast/inputs.json): c_cal 0.557..0.657 C by calendar
  month, pooled residual sd 0.0394 C (n 276 months 2001-2023); sd used 0.049. Bracket width is 0.06-0.07 C.
- Build: 214 contracts (180 RANGE + 34 HMONTH), 187 with candles (data/climate_nowcast/frame.parquet, candles.json).
  Early HMONTH events (2024) closed at month end, before T -> no bet (rule).
- Registered `cn1_gtemp_nowcast_taker-ae732f36` (family kalshi_data_ladders, 69 trials, bar p < 7.25e-4) before any
  candle was examined. Vault split `kalshi_climate_nowcast` added to edgelab/vault.yaml; loader in edgelab/loaders.py.

## Result CN1 dev (edgelab evaluate --split dev): DEAD
`cn1_gtemp_nowcast_taker dead dev 23 +17.47% [-60.01%,+105.95%] p 3.1e-01 CLV +0.0804 -- 5a p>=bar, 5b half<=0, 5c conc>30%, 5d n<min`.
n 23 bets / 12 month clusters (min_n 30); half1 -14.0%, half2 +85.2%; top cluster 2025-06 = 134% of profit, ROI ex-top
-6.4%. Placebos (descriptive, my scorer at C=10 fee): rule +19.2% n23; P1 ladder centre -15.9% n30; P2 persistence -4.7% n29.
Rule beats both placebos but its CI spans -60..+106: not distinguishable from zero. Sealed not opened (needs dev pass).

### Why: two model errors, both found only after the dev look (so no rule change is allowed to claim dev)
1. **NOAA revises after first print.** Today's NOAA archive puts the value outside the Kalshi-settled bracket in 8 of 16
   dev months (e.g. 24MAY settled 1.17-1.23, archive now 1.15; 25FEB settled 1.24-1.30, archive 1.22). So any fit on
   the current archive is fit on a different series than the one Kalshi settles on. First prints have to be logged
   (or scraped from NCEI monthly-report pages) -- the PortWatch / tornado trap again.
2. **The ERA5->NOAA offset drifted.** Archive minus pred over dev months: mean -0.060 C, sd 0.030 (n 16) -- a bias of
   1.2 x the sd we used. Post-hoc trailing-12-month offset (months M-13..M-2): mean -0.004, sd 0.051 (n 16).
The market was partly right where we were wrong (e.g. 24DEC T1.30 YES at .33 with p .77 lost; 25JUL B1.065 at .11 lost).

### Is the book stale? Partly
CLV mean +0.080 (n 23, coverage 1.0): between day 4 and day 8 prices moved toward the ERA5 side, i.e. the book reprices
around the Copernicus monthly bulletin (~day 6-8), not on Climate Pulse dailies. That is the only positive sign, and it
does not survive our own model bias into ROI.

## Power verdict (the expected failure)
~1 independent month cluster per month across both series (they share one outcome). 16 dev months gave 23 bets in 12
clusters; edgelab min_n 30 and a family bar of 7.25e-4 cannot be reached from this series in under ~2-3 years. Sea ice
(KXARCTICICEMAX/MIN: 3-4 settled events each, annual), CO2 (1), annual GISS (4) are worse. R4 power = 0 per the rubric.

## Proposal (lead / user decides; nothing installed)
CN2 forward-only paper logger: `scripts/climate_nowcast.py log` (written, smoke-run 2026-11-20 = no-op on a non-decision
day; decision path untested until 2026-12-04). Runs daily ~16:05Z, acts on day 4 (decision) and day 8 (CLV close) of
M+1; saves each NOAA CSV vintage (so first prints accumulate); pred = ERA5 month mean + trailing offset M-13..M-2,
sd 0.064 (post-hoc dev trailing sd 0.051 x 1.25 -- chosen after the dev look, so forward-only); same edge rule.
Writes data/climate_nowcast/cn2_paper.jsonl. Not registered in edgelab (needs a forward loader; one family-bar count) --
register only if the user wants the logger installed. Expected capacity tiny: ~1-2 bets/month at 6-30c.
KXARCTICICEMAX-27APR01 (active, 11 rungs, "any NSIDC day Dec 18-Apr 1 above X") is a reach-style nowcast like SX1;
one event/year -> fold into the same logger only if CN2 is installed.

## Learnings
### NOAA monthly global temperature is revised after first print; Kalshi settles on the first print
Today's NOAA cag archive lies outside the Kalshi-settled bracket in 8 of 16 months 2024-05..2025-08 (mostly lower by
0.02-0.08 C). Fit nowcast mappings on logged first-print vintages, not the current archive. Source: notes/climate_nowcast.md,
data/climate_nowcast/noaa_land_ocean.csv vs frame.parquet results.
### ERA5 ran ~0.06 C warm relative to NOAA in 2024-25 versus the 2001-2023 offset
Archive minus (ERA5 + 2001-2023 calendar-month offset) = -0.060 mean, sd 0.030 over 16 dev months; a trailing 12-month
offset removes the bias (mean -0.004, sd 0.051). A fixed long-window offset between two climate datasets is a model-tail
error waiting to happen. Source: same.
### Kalshi global-temp brackets reprice around the Copernicus monthly bulletin, not Climate Pulse dailies
CN1 bets at day 4 had CLV +0.080 (n 23) to day 8, yet ROI +17.5% [-60, +106] n.d. Source: edgelab cn1_gtemp_nowcast_taker-ae732f36.
### Monthly climate series give ~1 independent cluster per month: dev trials cannot pass the family bar
KXHMONTH + KXHMONTHRANGE share one outcome per month; 16 dev months -> 23 bets / 12 clusters (min_n 30). Sea ice / CO2 /
annual GISS have <= 4 settled events. Forward-log only. Source: results/climate_nowcast/census.json.

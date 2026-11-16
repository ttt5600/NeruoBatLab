# snow_markets -- KXSNOWTHANKSGIVING + KXSNOWDAYS (agent `snow_markets`, started 2026-11-16)

Leads from RESEARCH_QUEUE 'From series watch' 2026-11-15. No orders, no git, no launchd. Scratch: `sports/results/snow_markets/`.
Census: `results/snow_markets/census.py` -> `census.json` (read-only Kalshi GETs via kx_throttle, 2026-11-16).

## (1) Exact rules (from market JSON rules_primary/secondary, census.json)

### KXSNOWTHANKSGIVING-26NOV26 (contract terms SNOWOVERTIME.pdf), 21 binary markets, NOT mutually exclusive
- YES iff "total new snowfall at <station> on November 26, 2026 is strictly greater than 0 inches, **including a reported
  trace**". Station's local reporting day (NWS climate day = local standard time). Snow on ground does not count.
- Stations (custom_strike.station): KBOS KPWM KPVD KNYC KEWR KTTN KPHL KPIT KBWI KDCA KCLE KCMH KDTW KIND KORD KMKE KMSN
  KMSP KDSM KDEN KSEA.
- close 2026-11-27 04:59Z (ET cities) .. 07:59Z (SEA); expected expiration 11-27 15:00Z; **closes early if it snows** (YES).
  Sources: The Weather Company (weather.com/kalshi) + NWS. fee_type quadratic (maker 0, taker 0.07 p(1-p)).
- Depth 2026-11-16: thin. Spreads 3-20c, typical yb .01-.27 / ya .15-.40 with 50-lot quotes; lifetime volume 0-209
  contracts per city (MSP 209, PIT 113, DTW 107; 9 cities 0). This is a ~$100s market.
- KEY RULE FACT: **trace counts YES**. A model "P(measurable snow)" (>= 0.1 in / >0.01 in liquid) UNDERSTATES this:
  flurries reported as T at a staffed/ASOS site are far more frequent than measurable snow.

### KXSNOWDAYS-01DEC26-01APR27 (contract terms SNOWDAYSSCALAR.pdf), 22 SCALAR markets (one per city)
- YES pays linearly: $0 at `scalar_floor` qualifying days -> $1 at `scalar_cap`, capped [0,1], **rounded down to $0.01**;
  NO pays 1 - YES. Qualifying day = official daily snowfall >= threshold at the airport/CLI station.
  **Threshold differs by city: 1.0 in for CHI(KORD), DTW, CLE, DEN, MSP, PIT; 0.1 in for the other 16.** Trace = 0.
- Window Dec 1 2026 .. Apr 1 2027 inclusive (122 days); days midnight-midnight EST fixed (= NWS LST climate day ->
  ACIS daily snow matches). Close 2027-04-02 04:59Z+; "expiration may move earlier when the payout is definitively
  determined" (e.g. count >= cap). price step 0.001 (deci_cent). fee_type quadratic.
- Floors/caps (city floor-cap): ATL 0-4, AUS 0-2, BAL(KBWI) 1-18, BOS 11-31, CLT 0-4, CHI 6-16, CVG 9-31, CLE 6-25,
  DFW 0-6, DEN 6-17, DTW 7-19, HOU(KHOU Hobby) 0-2, IND 10-33, KC(KMCI) 5-18, MSP 5-19, NASH(KBNA) 1-9, NYC(KNYC) 2-19,
  PHL 2-20, PIT 5-21, PDX 0-7, SEA 0-14, DC(KDCA) 1-16.
- Depth 2026-11-16: **deep and tight -- 1c (often 0.1-1c) spreads with 1.4k-7k contracts per side on every city**
  (a market maker quoting all 22). Lifetime vol mostly 0-200; ATL 3.2k, AUS 0.8k. Mid prices .18 (HOU) - .52 (CLT).
  A tight two-sided MM quote = the mid is someone's model; edge only if our climatology/conditional model beats theirs.
- Mid-season this becomes L39-like: observed count so far is pinned; only the remaining days are uncertain.

## (2) PRE-REGISTRATION -- written 2026-11-16 BEFORE any ACIS climatology, ensemble value or outcome was computed
(I have seen only the Kalshi quotes above.)

### SD1 -- KXSNOWDAYS climatological fair value, pre-season (forward paper, one season)
- Data: ACIS StnData daily `snow` at each settlement station (sid = 3-letter id; KNYC -> NYC Central Park), seasons
  1996-97 .. 2025-26 (30 most recent complete seasons). Season window Dec 1 .. Apr 1 inclusive. 'T' = 0; 'M' = missing;
  a season with > 5 missing days in the window is dropped; need >= 25 seasons else city skipped.
- N_s = # days with snow >= city threshold. payout_s = floor_cent(clip((N_s - floor)/(cap - floor), 0, 1)).
  FV = mean_s payout_s. (Sensitivity, NON-decisional: 1991-2026 all seasons; last 15 seasons.)
- Decision snapshot = first run of the logger on 2026-11-16 (and re-logged daily; only the FIRST fire per city-side counts).
  Taker: buy YES if FV - (yes_ask + fee(yes_ask)) >= 0.10; buy NO if (1 - FV) - (no_ask + fee(no_ask)) >= 0.10,
  no_ask = 1 - yes_bid. fee = 0.07 p(1-p) per contract (exact). 1 unit cost per bet.
- Score at settlement (April 2027): units = sum(payout - cost)/sum(cost). n <= 22 and the cities share one winter
  (ENSO / NAO move all eastern cities together) -> effectively ONE draw. **Cannot pass any bar; forward observation only.**
  Kill: units <= 0 at settlement -> SD1 dead. "Pass" only means: keep the logger for the 2027-28 season.
- Prior: an MM quoting all 22 at 1c is probably climatology-aware; expect few/no fires. The 30-season mean ignores
  ENSO and the warming trend (fewer snow days), which the MM may include.

### SD2 -- KXSNOWDAYS conditional (L39-like) fair value in-season (forward paper)
- At each daily logger run on date d (Dec 2 .. Mar 31): N_obs = qualifying days Dec 1..d-1 from ACIS (current season;
  any missing day -> skip city that day). Remaining distribution = same-30-season counts over d..Apr 1.
  FV_d = mean_s payout(N_obs + R_s). Same taker rule and 0.10 margin as SD1; first fire per city-side after Dec 1.
- Plus a 'pinned' sub-rule SD2p: if N_obs >= cap (YES certain) or (cap reached impossible?) -- only the mathematically
  decided case counts: N_obs >= cap -> payout = 1 for sure; buy YES if yes_ask + fee <= 0.98. Log only.
- Score as SD1 at settlement; forward observation only.

### TG1 -- KXSNOWTHANKSGIVING forecast+climatology probability (forward paper, ONE event)
- Climatology C = fraction of years 1991..2025 with snow >= T (trace or more, ACIS 'T' or > 0) on days Nov 19..Dec 3
  (15-day window around Thanksgiving) at the station, over non-missing days.
- Ensemble E = fraction of members with Nov 26 local-day (LST) snowfall > 0 at the station lat/lon, pooled over
  Open-Meteo ensemble API models `gfs025` (GEFS 31) and `ecmwf_ifs025` (ECMWF-ENS 51), latest run at logger time.
  (Any modelled snowfall > 0.0 cm counts, mirroring trace = YES.)
- P = w E + (1 - w) C with w by lead L = days from logger date to Nov 26: L >= 7 -> 0.5; 3 <= L <= 6 -> 0.75; L <= 2 -> 0.9.
- Taker rule: buy YES if P - (yes_ask + fee) >= 0.10; buy NO if (1 - P) - (no_ask + fee) >= 0.10. Logger runs daily
  ~15Z 11-16..11-26 (read-only); first fire per city-side counts; all snapshots logged.
- Score after 11-27 from Kalshi `result`. 21 correlated cities, one day -> **not a test**; report units and Brier of P vs
  market mid vs C only as an observation. Kill: no rule follow-up unless repeated over several events (Christmas?
  if Kalshi lists one).

## Progress log
- 2026-11-16: census + rules (above). Pre-registration written before any climatology/ensemble pull.
- 2026-11-16 11:07Z: ACIS cache `results/snow_markets/acis_<STN>.json` (30 stations, 1991-01-01..2026-11-15; names match
  the rules' stations, KNYC = NY CITY CENTRAL PARK). First snapshot `scripts/snow_markets.py log` ->
  `results/snow_markets/snow_log.jsonl` (43 rows: 21 TG1 + 22 SD1).
- Registered in edgelab (dataset `kalshi_snow_markets` added to edgelab/vault.yaml, forward only, rule forward:unwired):
  `tg1_snow_thanksgiving_taker-9bc2a57c`, `sd1_snowdays_climo_taker-9131f5c2`, `sd2_snowdays_conditional_taker-15607f25`
  -> family kalshi_weather_precip 14 trials, bar p < 3.57e-03. DISCLOSED: SD1 was registered just AFTER its first
  snapshot fired (no outcome exists until Apr 2027, so no lookahead).
- Implementation fix after the first snapshot (before any outcome; that snapshot had 0 TG1 fires): the ensemble filter
  counted only the 50 ECMWF perturbed members (ECMWF control key ends `_ecmwf_ifs025_ensemble` and was missed; GEFS 0.25
  keys are all null beyond its 240 h horizon). Now ECMWF 51 + GEFS 31 once Nov 26 is inside 240 h; a member needs all
  24 LST hours. Checked: MSP -> (0.039, 51 members).

## Results (2026-11-16 snapshot)

### TG1 Thanksgiving -- 0 fires
Every city's P sits inside or near [yb, ya] (snow_log.jsonl first 21 rows). E (ECMWF only, lead 10 d) 0-0.20.
C (trace-or-more, Nov19-Dec3 1991-2025, ~525 station-days each): MSP .38, PIT .33, MSN .33, CLE .31, DTW .31, MKE .30,
DSM .25, DEN .25, ORD .23, CMH .23, IND .20, PWM .18, BOS .11, PVD .11, EWR .08, TTN .07, NYC .06, PHL .06, SEA .06,
BWI .05, DCA .03. Spreads 3-20c on 50-lot quotes; lifetime volume ~650 contracts total. Fires need a sharp forecast move
in the last 2-5 days. Capacity: tens of dollars per city.

### SD1 Snow days -- 5 YES fires on the first snapshot, all the same direction
DTW cost .297 vs FV .425; NASH .345/.447; MSP .345/.499; BOS .265/.412; IND .236/.408 (cost = ya + taker fee).
HOU, DEN, SEA skipped (< 25 complete seasons: ACIS snow missing for whole early seasons).
**Why the gap (descriptive; `results/snow_markets/sens.py` -> `sens.txt`, 19 cities with FV):**
mean(FV - Kalshi mid) / MAE: FV30 (1996-2026) +.085 / .089; FV15 +.048 / .069; **FV10 (2016-17..2025-26) +.006 / .051**;
FV all 1991-2026 +.093 / .105. The market maker (1.4-7k contracts per side, 1c spreads, all 22 cities) prices roughly the
LAST-10-WINTER climatology. Recent winters were low-snow (IND 0.1-in days, last 10: 10,21,16,13,21,10,10,10,16,21 vs
30-yr mean 19.6). So SD1's fires are ONE bet that this winter reverts toward the 30-yr mean instead of the recent decade
(trend / ENSO), not a pricing error. Not a "too good" bug: the same FV code reproduces the mids when fed the last decade.
The 5 fires share one Great-Lakes/Northeast winter -> one draw. Observation only.

### SD2 -- starts Dec 2 (needs the daily logger)
In-season is where structure may exist: once N_obs is pinned, the remaining-days distribution narrows, and an MM that
reprices slowly (cf. L39) could lag after a big storm. Untested.

## Proposal for the lead (launchd change -> lead/user OK needed)
Daily read-only logger ~15:30Z through 2027-04-01: `cd sports && ../.venv_quant/bin/python scripts/snow_markets.py log`
(~45 Kalshi GETs/day via kx_throttle; 22 ACIS calls/day from Dec 2; 21 Open-Meteo calls/day until 11-26). Pattern:
scripts/install_kalshi_wx_monthly_agent.sh. Scoring: after 11-27 (TG1) and after 2027-04-02 (SD1/SD2) from Kalshi
result / settlement value. An adapter (edgelab/adapters/snow_fwd.py) is still to be written; until then the trials sit on
forward:unwired (n = 0, fail-closed).

## Learnings

### Kalshi KXSNOWTHANKSGIVING counts a trace as YES
Rules: "strictly greater than 0 inches, including a reported trace" (results/snow_markets/census.json). A model
P(measurable snow) understates it; ACIS 1991-2025 trace-or-more frequency around Thanksgiving is .03 (DCA) to .38 (MSP).

### KXSNOWDAYS is a scalar market with city-specific thresholds and a maker on all 22 cities
Payout linear floor->cap, rounded down to the cent; threshold 1.0 in for ORD/DTW/CLE/DEN/MSP/PIT, 0.1 in elsewhere; EST
climate day Dec 1..Apr 1 inclusive; 1-7k contracts per side at 1c spreads (census.json, 2026-11-16).

### The KXSNOWDAYS maker prices last-10-winter climatology, not the 30-year normal
mean(FV - mid) = +.006 using 2016-26 seasons vs +.085 using 1996-2026 (19 cities, results/snow_markets/sens.txt). A gap
from 30-yr normals is a single bet on a snowier-than-recent winter. Check the averaging window before calling a
climatology gap an edge.

### Open-Meteo ensemble: ECMWF control key ends `_ecmwf_ifs025_ensemble`; GEFS 0.25 is null beyond 240 h
Filtering on "member" drops the control; GEFS keys exist but are all null past 10 days (results/snow_markets/ens_probe.json).
Log the member count per model in every row.

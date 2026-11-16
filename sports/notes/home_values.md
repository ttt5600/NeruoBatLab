# home_values -- Kalshi per-city Zillow ZHVI ladders (*HOME series)  (agent home_values, started 2026-11-16)

## Question
Kalshi listed 18 per-city home-value series 2026-11-09..11 (KXAUSHOME KXCHIHOME KXDALHOME KXHOBHOME KXHOUHOME KXJCHOME
KXLAHOME KXMEMHOME KXMIAHOME KXNYCHOME KXPALISADESHOME KXPALOHOME KXPHLHOME KXPHXHOME KXSANTAMONICAHOME KXSDHOME
KXSEAHOME KXVENICEHOME; list: results/home_values/series.json, markets: results/home_values/markets.json, fetched
2026-11-16 by results/home_values/fetch_markets.py). Is the next ZHVI print predictable enough from public data that
far-from-nowcast rungs are mispriced (E4 shape: stale quotes on the near-certain side)?

## Facts read before any modelling (2026-11-16)
- Rules (assets.kalshi.com/contract_terms/HOMEVALUE.pdf, saved results/home_values/raw/HOMEVALUE_terms.pdf):
  underlying = MIDDLE-TIER, SMOOTHED, SEASONALLY ADJUSTED ZHVI for the home type (16 series SFR, KXHOUHOME condos/co-ops)
  for the region ID; value for month M = the one shown "Data through <end of M>"; whole dollars, Research CSV rounded
  half-up; the Zillow Home Values page with that home type governs, Research CSV is the fallback; ONLY the first
  non-preliminary value counts (later restatements ignored); "above" = strictly greater.
- Matching Zillow file: files.zillowstatic.com/research/public_csvs/zhvi/City_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month.csv
  (Venice 21056 and Pacific Palisades 19810 are in the Neighborhood_ file; Houston condo in City_..._condo_...).
  Downloaded 2026-11-16: last column 2026-09-30 => October 2026 not yet published on 11-16.
- Events: 16 x "<city>-30NOV26" (Nov-2026 value; close 2026-12-17 15:00Z, expected expiration 12-19), KXMEMHOME-31DEC26
  (Dec 2026), KXHOUHOME-28FEB27 and -31DEC27 (condo, LT strikes, all quoted 1c/99c placeholders, OI 0).
  98 markets, 4-7 strikes per event, strikes spaced ~0.1-1% of value. No settled market exists in any series.
- Rules text says last trading = end of the period, but API close_time = 2026-12-17 15:00Z (around the expected
  release). The close time is what the API enforces; treat a mid-December release-day as the true resolution moment.
- Subtitles carry YoY vs the Nov-2025 value (e.g. Austin GT530000 "-1.1% YoY" <-> CSV Nov-2025 535,711).

## Pre-registration (written 2026-11-16 BEFORE any nowcast-error or price-vs-model computation)
Nowcast N0 (frozen): x = log ZHVI (latest vintage at decision time, first-published latest month L).
  Target month T = L + h. Forecast mu_h = x_L + sum_{k=1..h} d_k with d = damped momentum: d_k = phi^k * d_L, where
  d_L = x_L - x_{L-1}, phi = 0.85 (set a priori: smoothed ZHVI monthly changes are ~0.9 autocorrelated; not tuned).
  Error scale: sigma_h = c_h * s_city, s_city = std of second differences (d_t - d_{t-1}) over the last 36 months;
  c_h estimated ONCE on the full city panel (all Zillow cities, revised data, 2015-2025) as the 0.95-quantile of
  |error|/s_city at horizon h divided by 1.96 (=> sigma for a normal-equivalent tail), then x1.5 safety for revisions
  of the target month (first-published vs revised) unless the Wayback vintage check (below) shows a larger factor.
  P(YES) for "above K" = 1 - Phi((log K - mu_h)/sigma_h).
Rule HV1 taker (E4-style far rung): at decision time (each logger tick; the key tick = first tick after a Zillow
  release), for each open rung with h <= 2: side = YES if p >= 0.95, NO if p <= 0.05; buy the side at its ask iff
  side_prob - ask - taker_fee >= 0.05 and ask in [0.50, 0.97], spread <= 15c. One bet per market (first firing tick),
  1 contract nominal.
Rule HV2 maker: same side selection; post at q = min(floor(side mid), side ask - 1c) in [0.50, 0.97]; filled iff >= 10
  contracts print at side price <= q before close; unfilled = no bet; maker fee 0.
Cluster = release month (all cities move with one macro/mortgage-rate shock and share Zillow's method revisions).
Success (forward, both rules separately): n >= 40 bets over >= 4 release months AND net ROI 90% cluster-bootstrap CI > 0
  AND far-rung hit rate >= 93%.
Kill: HV1/HV2 losses_in_first n=20 max_losses=2 (a p>=.95 rule losing 3 of 20 is ~1.6% likely if calibrated), or
  ROI CI upper < 0 after 40.
No settled history exists => FORWARD ONLY. Dev work allowed: calibration of N0 on Zillow history (no Kalshi prices).

## Plan
1. Panel calibration of c_h on the current (revised) city file; revision check with Wayback vintages
   (web.archive.org has City_..._sm_sa_month.csv snapshots 2022-10-27, 2023-04-12, 2023-05-15).
2. Snapshot today's quotes vs N0 (descriptive only, NOT scored; scored rows come from the logger).
3. Who is quoting: orderbook depth + trades.
4. Register HV1/HV2 in edgelab (forward, rule forward:unwired until a logger + adapter exist); logger proposal.

## Amendment A1 (2026-11-16, after the Zillow-only calibration, before any Kalshi outcome exists; none exists)
The prereg allowed replacing the x1.5 revision safety if the vintage check showed a larger factor. It does (below), so:
sigma_h = sqrt((c_h * s_city)^2 + r^2), with r = 1.47% = q95 |one-release level shift| 2.884% / 1.96 (Wayback City
vintages 2023-04-12 -> 2023-05-15, SizeRank <= 3000, n 2,881). If the Metro vintage panel (wb_fetch.py) gives a larger
r, the larger one is used; never a smaller one. Rules HV1/HV2 otherwise unchanged.

## Log
### 2026-11-16 calibration (results/home_values/nowcast_calib.py -> calib.json)
- Revised panel, 2,900 cities, origins 2015-01..: h=1 |error| median 0.147%, q95 0.590%; c_h 1.215 / 3.220 / 5.786
  (h=1/2/3). Tail width varies by era: q95|z| h=1 1.95-2.15 in 2015-19, 2.8-3.5 in 2020-22, 1.32-1.55 in 2023-26.
- REAL-TIME check (forecast from the 2023-04-12 vintage, target = first-published Apr-2023 in the 2023-05-15 vintage):
  |z| median 3.14, q95 7.79 vs 2.07 for the same cities/origin on revised data; mean z +3.06 (one-sided bias).
- Cause: every release re-levels the WHOLE history of each city by a near-constant %: the 2023-05 vintage shifted
  every month Oct-2022..Mar-2023 by the same amount per city (all cities: median +1.32%, 5-95% [-1.06%, +5.44%];
  SizeRank <= 3000: median +0.875%, q95 |shift| 2.884%; contract cities: NYC -3.52, Hoboken -2.20, Seattle -1.39,
  SD +0.40, LA +0.50, Memphis +0.43, Phoenix +0.72, JC +0.89, Dallas +1.16, Palo Alto +1.20, Santa Monica +1.57,
  Austin +1.69, Miami +1.71, Philly +2.99, Chicago +3.70). Month-on-month moves are ~0.15%; the restatement is ~6x
  bigger, so the next print is NOT predictable to the strike spacing (0.1-1%) from the current vintage.
- Long-run restatement (first-published vs today's file): Sep-2022 median |diff| 3.08% (q95 13.3%), Apr-2023 4.99%.
### 2026-11-16 governing page vs Research CSV
- zillow.com/home-values/10221/austin-tx/ (server-rendered JSON, all-homes) on 11-16 already showed data through
  2026-10-31 (Oct 488,880) while the Research CSVs still ended at Sep. The page's Sep value 487,890 vs CSV
  City sfrcondo Sep 491,960 = -0.83%: the page is a NEWER vintage, re-levelled. => The Research CSV is a stale input;
  anyone with a browser sees the new month (and the SFR toggle) days before the CSV.
- After one GraphQL POST (the page's own zhviRangeDataQuery, metadataKey 'sfr') Zillow's PerimeterX blocked this IP:
  GraphQL -> captcha, then every page GET -> 403 (fetch_pages.py, results/home_values/pages/20261116T111830Z.json).
  Not bypassed. A scripted logger cannot reliably read the governing source.
### 2026-11-16 market state (books.json, trades.json, snapshot_rev0.0.csv; descriptive only, NOT scored)
- 264 trades / ~51k contracts since 11-10, 42.7k of them on 11-16; many sizes (200, 100, 600, 456, 1...). Spreads
  10-30c, depth 100s-1000s per level. This is an actively traded new market, not a stale maker-seeded one.
- Strikes sit within ~+-1 sigma of N0 (CSV-based, h=2); ZERO markets meet HV1 (p >= .95 / <= .05 with >= 5c edge).
  Apparent big 'edges' (Austin/Dallas/Philly YES) vanish once the page's re-levelling is applied: Austin SFR Sep CSV
  521,944 x 0.9917 = ~517.6k, +0.2% Oct => ~518.7k, so GT521000 at ~.28 is sensible. The market is ahead of the CSV.
  (Snapshot used the pre-A1 sigma, i.e. NARROWER than A1; with the A1 restatement term even fewer rungs qualify.)
### 2026-11-16 registration
- edgelab: hv1_home_far_taker-eb20dcd4, hv2_home_far_maker-9796f4ae (family kalshi_data_ladders, 60 trials -> bar
  p < 8.33e-04), dataset kalshi_home_values (vault.yaml, forward only from 2026-11-17), rule forward:unwired (fail
  closed: n = 0 until a logger + adapter exist, which will be a NEW trial hash by design).
- Metro vintage panel for A1: results/home_values/wb_fetch.py (57 Wayback Metro ZHVI snapshots 2021-09..2026-11) was
  rate-limited by the Internet Archive (429; 1/57 downloaded at hand-off). Re-run `../.venv_quant/bin/python -I
  results/home_values/wb_fetch.py` then `results/home_values/metro_vintages.py` -> metro_vintages.json (r_pct);
  A1 says r = max(1.47, r_pct). Tells us whether the 2023-05 re-level was a one-off (methodology) or every month.

- UPDATE: fetch finished with 27/57 snapshots; metro_vintages.py -> metro_vintages.json: 8 consecutive-release pairs
  (2022-01..2026-04), metros SizeRank <= 300. EVERY release restates the prior month: per-release q95 |shift| 0.64,
  0.96, 0.81, 0.71, 3.63, 1.29, 1.46, 2.40%; medians -0.4..+2.0% (common, US-level moves -0.51..+1.69%). Pooled q95
  2.39% -> r_pct 1.22% < 1.47%, so A1 keeps r = 1.47% (city-level restatements are larger than metro, as expected).
  => The re-levelling is routine, not a one-off; two of eight releases moved the level ~2% in one go.

## Verdict (2026-11-16): LEAD, weak. The E4 shape does not apply as written.
1. No stale certain side: strikes are listed within ~+-1 sigma of the current level and the book is actively traded
   (42.7k contracts on 11-16). HV1/HV2 would fire on 0 of 98 markets today.
2. The binding uncertainty is Zillow's whole-history re-levelling at each release (~1% typical, q95 ~2.9%; evidence =
   ONE city vintage pair + the Austin page, so 'every release' is not yet shown), several times the ~0.15% monthly
   move and unknowable before the release. Momentum nowcasts look great on revised data (h=1 median error 0.15%)
   and fail real-time (|z| q95 7.8 vs 2.1) -- the classic vintage trap.
3. Information edge runs AGAINST a scripted trader: the governing page shows the new month (and the SFR toggle)
   before the Research CSV, and blocks bots (PerimeterX 403 after one GraphQL call).

## Lead worth one cheap forward look: release-day lock-in (NOT registered; needs the timing fact first)
Kalshi close_time for the Nov events = 2026-12-17 15:00Z (can_close_early True). If the page shows the Nov value
before Kalshi halts trading, rungs whose strike is > ~0.3% from the published value are decided while still quoted.
Read-only proposal (lead decides; nothing deployed):
- logger: hourly 12-08..12-19 UTC, Kalshi orderbook + trades for all *HOME markets via kx_throttle (~100 GETs/h),
  plus once a day the Research CSV last column to timestamp the CSV release (fetch_books.py / fetch_markets.py reuse).
- The page itself must be read by a human or a real browser (Claude in Chrome) every ~2 h on 12-10..12-17 to
  timestamp the page flip -- scripted GETs are blocked; do not try to bypass PerimeterX.
- Question: is there any window with the page value published AND the market open AND a decided rung quoted <= 97c?
  If yes on >= 3 cities in December, pre-register a lock-in taker (family lock_in) for January.

## Learnings
### Zillow ZHVI re-levels the whole history at a release, so momentum nowcasts fail in real time
In the 2023-04 -> 2023-05 vintage pair each city's entire history shifted by one constant % (median +0.88%, q95 |shift|
2.88% for the 3,000 largest cities) while month-on-month moves are ~0.15%. Real-time h=1 |z| q95 7.8 vs 2.1 on revised
data (results/home_values/calib.json). A Kalshi ZHVI ladder with strikes 0.1-1% apart is mostly a bet on the restatement.
### Kalshi HOMEVALUE settles on the Zillow page, which leads the Research CSV and blocks bots
On 2026-11-16 the Austin page showed October (Sep re-levelled -0.83% vs the CSV) while the CSV ended at September; one
GraphQL call got the IP 403'd on every page. Treat the CSV as a stale proxy; read the page with a real browser.
### New Kalshi data ladders are not always maker-seeded on the certain side
The 18 *HOME series (listed 11-09..11) traded ~51k contracts in the first week with strikes inside +-1 sigma; the E4
stale-tail check came back negative here. Check the book and the strike placement before assuming a seeded tail.

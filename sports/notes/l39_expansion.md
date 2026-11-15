# L39 expansion (agent `l39_expansion`, started 2026-11-15)

Lead L39 = monthly rain/snow brackets that are (forecast-)unreachable in the last 7 days, bought as NO maker.
Questions from the lead: (1) capacity: other Kalshi monthly/seasonal weather series where "decided before close" logic
applies; (2) an earlier-in-month probabilistic version from climatology; (3) settlement-risk audit of every L39
loss / near-loss. No orders, no git, L39 sealed window (rain Jul 2026-Jan 2027, snow Mar 2026 + Dec26-Feb27) untouched.

Scratch: `sports/results/l39_expansion/`.

## Correction to the brief (matters for the risk audit)
L39 is NOT "mathematically unreachable". Rain/snow over the remaining days is unbounded. The frozen rule
(`scripts/kalshi_wx_winter.py` docstring) is `X - OBS_d > 2*FMAX + 0.5 in` (rain; snow +1.0 in), with FMAX = wettest
of 5 Open-Meteo models over the rest of the month. Its protection is a forecast margin. That is why one loss exists.
Truly mathematical decisions only exist for monotone running statistics with a physical bound (running min/max temp,
cumulative HDD with a daily cap, "already reached" YES).

## (3) Settlement-risk audit -- dev picks (data/kalshi_wx_winter_dev_picks.parquet) + forward (22 picks)
Command: inline script, output `results/l39_expansion/dev_audit.parquet` (203 unique NO tickers: 167 rain, 36 snow).
- **The one loss: KXRAINSFOM-25DEC-4** (d 12-27, 4 days left). OBS 3.45 in, FMAX 0.016 in, ACIS final 4.39 in -> 0.94 in
  fell after the decision (59x FMAX). Market had YES 9-16c. **Not a settlement quirk**: ACIS agrees with the Kalshi
  result (the earlier audit matched 1,259/1,259 settled markets). It is a forecast miss (late-Dec atmospheric river);
  backtest forecasts are lead j+1 previous-runs, older than the live 18Z run.
- **Near-losses (final slack < 0.5 in):**
  KXRAINMIAM-25NOV-2 slack 0.24 in (rest 1.59 in vs FMAX 0.50, 3.2x; YES bid 15c);
  KXRAINMIAM-26MAY-7 slack 0.38 (rest 1.69 vs FMAX 0.53);
  KXRAINHOUM-26APR-6 slack 0.46 (rest 0.56 vs 0.17);
  KXRAINLAXM-26APR-1 slack 0.49 (rest 0.30 vs 0.11).
  Snow: KXCHISNOWM-26JAN-12 slack 0.70 (rest 2.6 in vs FMAX 0.44 -> 6x; lake effect).
- **Forecast-miss frequency:** remaining precipitation exceeded FMAX on 45/203 picks (22%) and 2*FMAX on 25/203 (12%).
  The additive margin (0.5 in rain / 1.0 in snow), not the 2x multiplier, is what protects the book. Losers and
  near-losers cluster in convective/subtropical cities (MIA, HOU, SFO atmospheric rivers) and lake-effect snow.
- **Settlement-source quirks found:** (a) KXRAINWPBM rules cite "CLIDJT" (no such NWS CLI; WPB is PBI) ->
  `rain_station` returns DJT; ACIS has no DJT, so the logger skips it (safe by accident; do not trade until Kalshi
  clarifies). (b) KXRAINNAPAM is a Nov-Mar SEASONAL ladder at KAPC (ASOS, not a CLI station) -> no ACIS id, skipped.
  (c) Resort snow series (Mammoth, Park City, Stowe, Beaver Creek, Keystone, Northstar, Alta, Big Sky via OnTheSnow)
  settle on resort self-reports: no independent archive to audit, no backtest. (d) Precipitation climate day is local
  STANDARD time (known caveat). (e) Kalshi settles LOCKED ladders early (close_time moves); W1 guards on T < close.
- **Forward 22 picks (Oct 2026)**: 22-0, min slack not computed per pick here (all `result` no). Note: in the deployed
  picks file `pnl_maker` is populated even when `maker_filled` is False (rows 4,5,8,14,18,20,21) -- check that the
  `report` maker P&L filters on fill (not verified here).

## (1) Capacity census (read-only, 2026-11-15)
Files: `results/l39_expansion/census_A.json`, `hist_A.json` (scripts census.py / hist.py), `open_2026-11-15.json`.
- **Truly-decidable series exist but have no history.** Monthly/period min temp (KXLOWTPERIOD 5 events all open
  Nov 12-15, vol 48k; KXMSNLOWT 1, KXSJCLOWT 1 ~7.5k each; KXMINNYC 5 events 2021-22), cumulative HDD (KXHDDCVG Oct-Dec,
  1 event, 24k), KXHOBBYTEMP (1), KXAVGTEMP (US monthly avg: 2 events 2022-23, 3.6k vol, dead), KXYEARLOW* (10 cities,
  0 markets), weekly avg-temp KXAVGTK* (8 events each Aug-Oct 2026, 12-61k), seasonal resort snow (1 open event each,
  0.4-18k; Big Sky monthly 146k). None has enough settled events for a dev test; all would be forward-paper only.
- **The real capacity lever is the rain/snow ladders L39 already trades:** Nov-2026 has **37 monthly rain series
  open** (vs 15 with October ladders, which is all the October forward run saw). `rain_station` resolves 36/37
  (WPB = DJT quirk; Napa seasonal excluded). Dec-2026 monthly snow: 15 series open; `W.SNOW` lacks **MKE, MSP, PIT,
  TTN** (ACIS ids MKE, MSP, PIT, TTN) -> logger will silently skip them in Dec.
- Volumes on Nov rain ladders (lifetime, all brackets): LAX 40.8k, PHX 51.1k, LAS 26.6k, ABQ 21.4k, SAT 13.9k, SEA
  10.0k; most others 0.4-9k. Snow Dec: PHIL 79k, SEA 28k, DET 13.7k, MKE 12.6k, CHI 11.3k.

## (2) PRE-REGISTRATION P1m -- written 2026-11-15 BEFORE any early-window quote, climatology or outcome was pulled
Question: earlier in the month (7-20 days left), does a CLIMATOLOGY bound make some brackets safe enough to buy NO as
maker at a profit?
- Universe: settled ladders in `data/kalshi_wx_winter.parquet` (same events/strikes as W1), DEV ONLY: rain months
  < 2026-07, snow months < 2026-03. Sealed rows are never fetched.
- Decision: T_d = 22:00 station-local, days d with 7 <= days_left <= 20 (disjoint from the L39 window 0-6).
- State: OBS_d = ACIS total over days 1..d-1 (pcpn for rain, snow for snow); any missing ACIS day in the month ->
  city-month dropped.
- Climatology: ACIS daily at the same station, years 1991..(event year - 1) (no lookahead). For window length
  L = end - d + 1, take every L-day window starting at the same calendar date shifted by k in -10..+10 days in each
  past year; windows with a missing day are dropped. Need >= 200 windows. Q = 99.5th percentile (numpy
  method="higher").
- Fire: X - OBS_d > Q; market open at T_d; last hourly candle <= T_d tradeable (yes_bid > 0, yes_ask < 1, spread <=
  10c, yes_ask > 0.01). First qualifying d per ticker. Rain and snow pooled = ONE trial.
- Execution: maker M3 exactly as W1m (q = min(floor_cent(NO mid), NO ask - 1c), filled iff >= 1 contract printed at
  no_price <= q after T_d and before close; fee 0). Stake 1 unit cost.
- Pass (all): filled n >= 60; cluster-bootstrap p (over city-months) < bar AND market-null p < bar, bar = 0.05 / 11
  = 0.00455 (family kalshi_weather_precip has 10 trials; this is the 11th); ROI > 0 in both chronological halves.
- Kill: ROI <= 0, or any failure of the pass rule -> dead (no re-tuning of the quantile, window or margin).
- Diagnostics declared NON-decisional (printed, not trials): taker version; Q at 99% / 99.9%; overlap with W1 tickers.
- Expected: market NO asks ~.97-.98 -> break-even loss rate ~2-3%; a nominal 0.5% climatological tail must survive
  fat tails (the SFO loss was a 1-in-many event). Prior: probably fails on n or market-null p.

## Progress log
- 2026-11-15: registered `l39_p1m_climo_maker-4b39ec50` in edgelab (family kalshi_weather_precip -> 11 trials, bar
  4.55e-03) BEFORE the dev score. New: scripts/l39_p1m.py, edgelab/rules/l39_climo.py, loader kalshi_wx_monthly_early,
  vault entry kalshi_wx_monthly_early (dev 2021-12..2026-06; builder writes DEV rows only), spec
  edgelab/specs/l39_p1m_climo_maker.yaml.
- Build: results/l39_expansion/p1m_frame.parquet, 13,384 rows, 100 rain + 57 snow dev city-months; quoted 92%.

## (2) RESULT P1m -- FAIL (dead), direction positive (results/l39_expansion/p1m_dev.json, p1m_picks_q995.parquet)
| variant | picks | fills | losses | ROI | bootstrap p | market-null p (1/city-month) | halves |
|---|---|---|---|---|---|---|---|
| **q995 maker (DECISION)** | 105 | 94 (90%) | 0 | **+4.28%**, units +4.02 [+3.13,+5.01] | <1e-4 | **.021** (.117) | +5.2% / +3.6% |
| q995 taker (diag) | 105 | - | 0 | +2.08% | <1e-4 | .119 | +2.6% / +1.7% |
| q990 maker (diag) | 128 | 113 | 0 | +4.42% | <1e-4 | .0073 | +5.1% / +3.9% |
| q999 maker (diag) | 88 | 77 | 0 | +4.60% | <1e-4 | .032 | +5.6% / +3.2% |
- Fails the prereg on market-null p (.021 > .00455). edgelab independently reproduces n 94, ROI +4.28% [+3.61,+5.00]
  and marks it dead: concentration > 30% (KXRAINDENM 22/94 fills), CLV null (no close price in frame), too-good guard.
- **Why it cannot pass: arithmetic, not luck.** With zero losses the market-null p is just prod(cost) = .96^94 =
  .0216. Clearing .00455 needs >= 132 loss-free fills at a .96 cost (213 at W1m's .975). Every "certain side at
  96-98c" strategy on this ladder is power-bound; dev history cannot reach the bar.
- Row inspection: 46/105 picks fire at 20 days left (first eligible day). Degenerate climatology: MIA/AUS/HOU snow
  where q995 = 0, so any bracket > 0 fires. **Near-loss: KXAUSSNOWM-26JAN-1.0** (q995 0.60, final 0.80 in, slack
  0.2; YES 14-21c). Others are wide: median final slack 3.93 in. 37/105 tickers are also W1 picks later in the
  month, so P1m adds about 68 new tickers over 157 city-months (~0.43 per city-month).
- Verdict: dead as a stand-alone trial (pre-registered kill). Do not re-tune. It could join L39's forward paper log as
  an extra column at no cost, but it cannot be promoted without a new pre-registered trial.

## Expected $/month at realistic depth (L39 W1m, with the expansion fixes)
Inputs: forward Oct 2026 = 22 picks from 15 series, fill 15/22, +3.8c per filled contract at mean q .966 (deployed
picks file); dev W1m +3.6%; depth at decision: median yes_bid_size 50 contracts, median 24h volume 12 contracts
(deployed picks file). Nov 2026 has 36 tradeable rain series (vs 15 in Oct) -> about 45-55 picks, about 30-45 fills.
Dec-Mar adds about 15 snow series -> about 15-25 fills.
- At $3/bracket: about +$4-6/month rain plus about +$2-3/month snow (winter) = **about $6-9/month expected**.
  One loss costs about $3 (about 25 wins). $5/bracket is the most a 12-contract/day tape plausibly fills
  (unverified; no historical depth), so at most about $10-15/month.
- Under the current cap (<= $15 capital per month): about $15 x 3.9% = **about $0.6/month**. Lifting the cap to
  about $150/month (about 45 brackets x $3) is the only material lever. That is a sizing decision for the user, and
  the 1-in-150 loss rate (CI up to 3.8%) argues against it before the sealed shot.

## Proposals for the lead (no deploys done)
1. Add MKE, MSP, PIT, TTN to `kalshi_wx_winter.SNOW` (ACIS ids MKE, MSP, PIT, TTN) BEFORE Dec 25. Otherwise the
   Dec snow window silently skips 4 of 15 series. This is a logger change, so it is the lead's deploy.
2. Hard-exclude KXRAINWPBM (CLIDJT) and any series whose station is not an NWS CLI site (Napa KAPC, resort snow).
3. Check that `report` maker P&L filters on `maker_filled` (deployed picks carry pnl_maker on unfilled rows).
4. Forward-log the P1m signal as a column (no money). Mathematically decided series (KXLOWTPERIOD, KXHDDCVG,
   KXSJCLOWT) have no history: forward-paper only, and each event is $7-48k lifetime.

## Learnings
### L39 is a forecast-margin rule, not a "mathematically unreachable" rule
Rain/snow over the remaining days is unbounded. The protection is X - OBS > 2*FMAX + 0.5 in. Remaining precip
exceeded FMAX on 45/203 dev picks (22%) and 2*FMAX on 25/203 (12%) (results/l39_expansion/dev_audit.parquet). The
additive margin, not the multiplier, is what saves the book.
### The only L39 loss was weather, not a settlement quirk
KXRAINSFOM-25DEC-4: 0.94 in fell after the decision against FMAX 0.016 (59x). ACIS matched Kalshi on 1,259/1,259
settled markets. Near-losses cluster in MIA/HOU (convective), SFO (atmospheric rivers) and CHI lake-effect snow.
### Zero-loss 96c strategies are power-bound: n >= ln(bar)/ln(cost)
With no losses, market-null p = prod(cost). At cost .96 and bar .00455 you need 132 loss-free fills (213 at .975).
P1m's 94 loss-free fills give p .0216 (notes/l39_expansion.md). Compute this before building a test on near-certain
contracts. It tells you whether the history can ever pass.
### Kalshi weather capacity is in rain/snow breadth, not new series types
Nov 2026: 37 monthly rain series open, against 15 with October ladders. The truly decidable types (period min temp,
HDD, annual low) have 0-5 settled events each (results/l39_expansion/hist_A.json). The logger misses 4 Dec snow
series (MKE/MSP/PIT/TTN) because the station map is hard-coded.
### Settlement-source traps in Kalshi weather rules
KXRAINWPBM cites "CLIDJT" (not a real CLI). Napa settles at KAPC ASOS over Nov-Mar. Resort snow settles on resort
self-reports (OnTheSnow). Parse the station from rules_primary and refuse anything that is not an NWS CLI / ACIS
station.

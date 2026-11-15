# consensus_reject_census -- Kalshi ladders with public, timestamped forecasts (E4 analogue) (agent, 2026-11-15)

Question: besides weather (E4), which Kalshi ladders have >= ~40 resolved events AND a public forecast that is
timestamped before resolution and retrievable as-of decision time, so that an E4-style "NO on rungs every forecast
rules out" rule can be backtested? Then pre-register and dev-test the best 1-3.

Do-not-redo honoured (FUND_BOOK 3d): CPI (N2), econ claims/U3/payrolls/GDP/PCE (econ_reject), gas, crypto hourly,
ERCOT (Q017), PortWatch/EIA, MEDIA weekly far-bracket (YouTube/Netflix; charts DROPPED there at the data gate, never
tested), tornado, SpaceX/SPR, hourly temp, daily rain/lows, TSA, RT/Metacritic (Q013 forward logger).

## Census (scripts: results/consensus_reject_census/census.py, hist_census.py; output hist_KXALBUMEQUIV.json)
Series search over data/phys_census/series_all.json (15,900 series, 11-06) by keyword; settled counts from
/historical/markets + /markets?status=settled (throttled via scripts/kx_throttle.py).

| candidate | Kalshi series | settled ev | lifetime contracts | public forecast / PIT vintages | verdict |
|---|---|---|---|---|---|
| box-office opening weekend | none (only KXGGBOFILM award) | 0 | -- | -- | absent on Kalshi |
| CPI / payrolls / claims / GDP | KXCPI etc. | many | -- | Cleveland Fed, consensus | DEAD already (N2, econ_reject) |
| RT / Metacritic | per-title KXRT*/MC* | per-title 1 | -- | RT live score | covered by Q013 logger |
| air quality | KXPMLA | 12 (Sep 2021) | 30k | AirNow fcst | too few, 2021 only |
| Lake Mead / river | MEAD, CORIVER | 0 | -- | Reclamation 24-mo study | never listed |
| seasonal / monthly snow | KXMAMMOTHSNOW, KXALTASNOW, KXASPSNOWM | 0-1 each | 13k | NWS/OpenSnow | too few (city monthly snow = L39) |
| app-store rank | KXAPPRANKFREE | 302 (to 2025-11) | 3.9M | none public | no forecast; series ended |
| Spotify daily top-song streams | KXSPOTSTREAMSUSA | 59 (to 2026-04) | 0.25M | none (kworb no vintages) | no forecast; ended |
| artist weekly streams | KXARTISTSTREAMS | 429 (to 2026-07) | 0.34M (median 105/ev) | none | thin; ended |
| Spotify weekly #1 | KXSPOTIFYW | 72 | 4.9M | none | no forecast |
| **album first-week units** | **KXALBUMEQUIV** | **95 (2026-05-03..11-08)** | **7.7M (median 28.6k/ev, 10 rungs, all 'greater')** | **HitsDailyDouble chart-preview / chart-update / chart-story articles with `publishedAt` + `_updatedAt` (Sanity CMS)** | **BEST -> test** |
| Billboard 200 #1 / Hot 100 #1 | KXTOPALBUM / KXTOPSONG | 103 / 178 (since 2023) | 3.0M / 5.0M | HDD projects the #1 | categorical; second |
| Hot 100 top-10 entrants | KXRANKLISTSONGTOP10 | 50 | 2.7M | weak (X accounts) | no PIT source |

Note: only ONE public forecaster (HDD) is reachable for album units; X chart accounts are not fetchable. So "consensus"
here = 1 source. Disclosed before the rule: I saw one settlement (Nickelback EVE26NOV05 = 19,130 units) while reading rules.

## PRE-REGISTRATION (written 2026-11-15 BEFORE any HDD number was joined to a Kalshi price or settlement)
Contract: KXALBUMEQUIV-<ALB><YYMONDD>-<K>K: YES iff Luminate album-equivalent units for the Fri..Thu tracking week
ending <YYMONDD> > K; settles on Luminate API at 10:00 ET the Sunday after; Kalshi close = that Sunday 14:00-15:00Z.
Forecast F = the latest HDD number for that album in any HDD /news/charts/ article with publishedAt AND _updatedAt
< T (articles edited after T are used only via a Wayback capture taken < T, else dropped). Numbers are parsed from
the sentence that names the album ("NNk", "NN,NNN", ranges -> midpoint, "NNk+" -> NN).
Decision time T = Saturday 12:00Z before the Sunday close (after HDD's Friday chart story).
Rejected rungs: NO if K >= 1.30 F; YES if K <= F / 1.30. Both sides one contract per rung, first quote at T.
Trials (k = 2, family kalshi_data_ladders unless the lead says media):
- AU-T1 taker: buy the rejected side at the ask from the last hourly candle ending <= T, + Kalshi taker fee
  (quadratic, 0.07 p (1-p) rounded up to the cent); require ask <= 0.98.
- AU-M1 maker: post q = floor_cent(min(mid, ask - 1c)) on the rejected side at T, filled iff a later tape trade
  prints THROUGH q before close; maker fee 0. Also report filled-vs-unfilled loss rates (adverse selection check, E5/RD2).
Split by tracking-week end: dev 2026-05-01..2026-08-31; sealed 2026-09-01..2026-11-08 (NOT opened); forward 11-09+.
Success (dev): ROI > 0 with week-clustered bootstrap 95% LCB > 0 and p < family bar, n >= 30 bets.
Kill: dev ROI <= 0, or n < 30, or the data gate fails.
Data gate (before any price): for dev weeks, |F - settled units| / units <= 25% on >= 80% of albums with an HDD number
(settlement read only for this gate). Fail -> stop.

## AMENDMENT 1 (2026-11-15, BEFORE any HDD number was joined to a Kalshi price or settlement)
Found structured HDD vintages: HDD's public Sanity CMS (project 8aky18h3, read-only GROQ via apicdn; script
results/consensus_reject_census/hdd_q.py; raw data/consensus_reject_census/hdd_charts_all.json) has one `chart` doc per
week per type with `_createdAt`/`_updatedAt`: **midweek-20** (37 docs, created Tuesdays ~19-21Z = HDD's projection for
the CURRENT Fri..Thu week, top 20 with units) and **hits-top-50** (37 docs, created Fridays ~20Z = near-final units for
the week that ended the day before, i.e. ~37 h before Kalshi's Sunday close). These replace prose parsing.
Mapping: midweek doc -> week ending the next Thursday on/after created date; top-50 doc -> the Thursday before created.
A doc is usable at T only if `_updatedAt` < T (latest edit before T => current content = content at T).
Match: Kalshi album title + artist parsed from rules_primary ("If <TITLE> by <ARTIST> has above") vs chart row
"ARTIST | TITLE", normalized (upper, strip punctuation/"(DELUXE..)"); title match required.
Absent album -> upper bound U = smallest units on that doc (only if the doc has its full 20 / 50 rows), but ONLY if
neither the normalized title nor the artist string appears anywhere on the doc (guard against name mismatches).
Trials (k = 3; replaces AU-T1/AU-M1 above):
- AU-L-T taker, T = Saturday 12:00Z before the close, F = hits-top-50: NO if K >= 1.15 F (present) or K >= 1.15 U50
  (absent); YES if K <= F / 1.15. Buy at ask (last hourly candle ending <= T) + taker fee, ask <= 0.98.
- AU-L-M maker, same selection, q = floor_cent(min(mid, ask - 1c)) on the chosen side, fill iff a later trade prints
  through q before close; fee 0. Report filled-vs-unfilled loss rates.
- AU-E-T taker, T = Wednesday 12:00Z of the tracking week, F = midweek-20: NO if K >= 1.30 F or K >= 1.30 U20;
  YES if K <= F / 1.30; ask <= 0.98.
Data gate (settlement read only here, dev weeks only): top-50 F within 10% of settlement on >= 90% of present albums;
midweek F within 30% on >= 80%; absent-bound violated (settled > U) on <= 5% of absent albums. Fail -> stop that trial.
Split unchanged (dev weeks ending 2026-05-01..08-31). Clustering unit = tracking week.

## Log
- 11-15: census done (table above). HDD homepage fetch 200; story JSON carries publishedAt/_updatedAt; sitemap 404;
  Wayback CDX 502 on first try. Enumerating HDD chart articles next.
- 11-15: `scripts/album_units.py build` -> data/consensus_reject_census/au_frame.parquet (1,168 markets, 95 events;
  per stage: late present 587 / absent 260 / edited_after_T 255 / ambiguous 49 / nodoc 17; early present 528 /
  absent 318 / edited 239 / ambiguous 66 / nodoc 17). Two regex bugs fixed before any outcome (rules text has
  irregular spaces "Aug 20, 2026 , tracking", "-July 16").
- 11-15 DATA GATE (`album_units.py gate`, dev weeks only, one row per event):
  * LATE (hits-top-50): 22 present, within 10% on 0.864 (need 0.90) -> **FAIL; AU-L-T and AU-L-M stop, never scored.**
    Median |err| 4.7%, and every large miss is HDD BELOW Luminate (ICEMAN 460k vs 495k; Lost Weekend 113k vs 126k;
    DIRTY BLONDE 16.9k vs 19.3k): HDD's Friday Top 50 is NOT the settlement number (different universe/methodology),
    so the "near-final leak" reading is wrong. Absent bound 0/8 violated.
  * EARLY (midweek-20): 27 present, within 30% on 0.963 (need 0.80) -> PASS; median |err| 6.4%; absent bound 0/12
    violated (largest absent album 19.3k vs U20 25.7k). AU-E-T proceeds to prices.
- 11-15 edgelab: registered au_e_taker-9305b25d, au_l_taker-157b13e4, au_l_maker-ce4dc9f8 (family kalshi_data_ladders
  -> 41 trials, bar p < 1.22e-03); retired both late trials at the data gate (never scored). New code:
  edgelab/rules/album_units.py, loader kalshi_album_units (edgelab/loaders.py), vault.yaml dataset kalshi_album_units,
  specs edgelab/specs/au_*.yaml. Candles: data/consensus_reject_census/au_candles.json (333 keys, 323 with candles).
- 11-15 **RESULT AU-E-T dev: DEAD (n < min).** `python -m edgelab evaluate au_e_taker --split dev`: n 7, ROI +2.88%
  [+2.06,+3.71], p 2.5e-05 < bar, 0 losses, CLV +1.7c -- but n 7 << min_n 30 -> dead (d_min_n).
  Why so few: the forecast is RIGHT and the market already knows it. 333 selected dev rungs (38 events, 14 weeks),
  **0 losses** (`album_units.py score dev`); but at Wednesday 12:00Z our side's ask is > 0.99 on 300/319 quoted rungs,
  0.98-0.99 on 12, <= 0.98 on only 7 (results/consensus_reject_census/au_scored_dev.parquet). The album-units book
  prices HDD's midweek projection fully by Wednesday; the 1c left is fee + tick. Unlike weather (E4), the far rungs
  here are not stale lottery tickets.
  Not run (not pre-registered, would be new trials): maker at 0.99 on the 300 rungs (max +1c/contract, capacity tiny);
  KXTOPALBUM/KXTOPSONG #1 vs midweek-20 (same source, same book; expect the same).
- Sealed (weeks 2026-09-01..11-08) NOT opened.

## Learnings
### HitsDailyDouble chart vintages are queryable with timestamps
HDD's public Sanity CMS (project 8aky18h3, apicdn GROQ, read-only) holds one `chart` doc per week for midweek-20
(Tue projection) and hits-top-50 (Fri), each with _createdAt/_updatedAt, back to ~2026-03 (37 each). Point-in-time
condition: use a doc only if _updatedAt < T (several docs are edited 1-7 days later). Script results/consensus_reject_census/hdd_q.py.
### HDD's Friday Top 50 is not the Luminate settlement number
On 22 dev albums HDD top-50 units were within 10% of Kalshi's Luminate settlement on 0.864, and every large miss was
HDD BELOW (ICEMAN 460k vs 495k). Do not treat it as a "leak" of the Sunday number.
### HDD midweek projection is accurate, and the Kalshi album-units book already prices it
Midweek-20 within 30% of settlement on 0.963 of 27 dev albums (median error 6.4%); a 1.30x reject margin had 0
losses on 333 rungs, but 300/319 of those rungs were already > 0.99 at Wednesday 12:00Z. A good public forecast is
not an edge when the book is made by people reading the same forecast (contrast E4 weather).
### No other Kalshi ladder has a public timestamped forecast with >= 40 events (2026-11-15 census)
Box office: no Kalshi series. Air quality KXPMLA 12 events (2021). Lake Mead/river: never listed. Snowfall
seasonal: 1 event each. App rank / Spotify streams: no public forecast. CPI/econ already dead. Census table above.

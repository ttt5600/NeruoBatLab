# source_shift -- systematic search for P-1 (settlement-definition shift) opportunities (agent source_shift, 2026-11-21)

Brief (lead): find more P-1 cases (IDEA_PATTERNS.md P-1: OpenRouter default chart switched to BYOK-excluded mid-week; bots
repriced in minutes). (1) Kalshi history: mid-event far strike listings, rules-text changes, settlement holds / >= 85c
favourite flips -> what happened at the source, did the book / settlement follow the OLD or NEW definition -> base rate.
(2) Build read-only scripts/source_shift_watch.py (multi-definition snapshots + Kalshi strikes/rules, ALERT lines). (3) Report
base rate + open P-1 candidates; append patterns to IDEA_PATTERNS.md. NO git, NO orders, NO launchd.
Scratch: results/source_shift/. Watch output: ~/.claude/quant-sports/data/source_shift/snapshots.jsonl.

## Pre-registration (written 2026-11-21 ~06:05Z, BEFORE any settled outcome / flip / hold was computed)
Seen before writing: API field list (one KXTOKENUSE market), the live KXTOKENUSE-26NOV23 ladder (strikes T137..T200 created
11-21T04:10Z, T310..T335 created 11-18T22:13Z), one-day market counts by series (no outcomes).

**Question.** When a Kalshi contract's settlement source changes how it measures (methodology / default view / vintage /
correction / outage / definition dispute) while the contract is open, does Kalshi settle on the OLD definition (the one used by
the series' earlier settlements or in force at listing) or on the NEW one (what the source shows at the observation time)? And
did the final pre-close book price the old or the new?

**Universe.** Settled markets from /historical/markets + /markets?status=settled for 11,080 series (cached series list minus
Sports, Crypto and hourly/daily/one-off Financials/Commodities price ladders); script results/source_shift/census.py.

**Detectors (frozen):**
- (a) LATE FAR STRIKE: in an event with a numeric ladder (floor/cap strikes), a market created >= 6 h after the event's first
  market whose strike lies outside the initial ladder [min, max] by more than 2 x the initial median strike step. Score =
  distance outside / initial ladder span. Report the count per series and the top cases by score in non-weather series.
- (b) RULES CHANGE: within an event, two creation batches (created_time differs by >= 6 h) whose normalised rules templates
  (numbers, months, sub-titles masked) differ = mid-event rules change. Across events of a series: list template changes of
  rules_secondary (descriptive, the date Kalshi changed the contract).
- (c) HOLD / FLIP / ODD SETTLEMENT: hold = event settlement_ts - close_time > max(48 h, series median + 24 h) where the series
  median is over its own events; flip = event favourite (highest last_price, >= 0.85, market volume >= 100 contracts) settled
  NO, or a market with last_price <= 0.15 and volume >= 100 settled YES in a mutually-exclusive ladder; odd = a market whose
  settlement_value is neither 0 nor 1 or result not in {yes, no}.
**Case classification (manual, for every flagged non-weather candidate inspected, top-ranked first until time runs out):**
cause in {source methodology/default change, source correction/revision, source outage/missing print, definition dispute /
interpretation, Kalshi error, ordinary news (no source change)}. For source-change causes record OLD vs NEW for (i) settlement,
(ii) final pre-close price. Weather holds are taken from winners_transfer P4 (delays.parquet), not re-pulled.
**Base-rate statistic.** k/n = settlements that followed the NEW definition among source-change cases; Jeffreys 95% interval.
Pre-set wording: n < 10 -> "anecdotal, not a base rate"; interval crossing 0.5 -> "not distinguishable from a coin flip".
No trade rule is tested here (descriptive); any trade idea that comes out goes to a separate prereg.

## Log
### 2026-11-21 05:58-06:20Z -- census, watcher, first cases
- Census: results/source_shift/census.py (per-series /historical/markets + /markets?status=settled; historical cutoff
  2026-09-21 per GET /historical/cutoff). First pass at 4 threads did ~1 series/s; restarted 06:20Z at 8 threads with
  Elections dropped (resumable). Legacy non-KX series (2021-23) are included where the API still serves them.
- Detector (c) caveat found on the first look (before any case was classified): in forecast-type series (CPI, state gas,
  claims...) the market closes BEFORE the data exist, so a >= 85c favourite losing is ordinary uncertainty, not a settlement
  anomaly (Economics: 107 of 4,868 events with fav >= .85 lost; results/source_shift/events.parquet). Flips are informative
  only with a hold, a source event, or in close-after-observation series. Inspection therefore uses holds + odd settlements +
  flips-with-holds.
- **Shutdown 2025 = the biggest source-outage cluster** (Economics holds by close month: 2025-10 15, 2025-11 12, 2025-12 8;
  analyze.py). Two different exchange behaviours, read from the API (/historical/markets/<ticker>):
  * Source published LATE -> Kalshi waited and settled on the actual print even where the book's favourite lost:
    KXU3-25SEP 4.4 / KXPAYROLLS-25SEP 119,000 settled 2025-11-20 (1,154 h hold; BLS Sep report 11-20, 4.4% / +119k per
    bls.gov empsit archive); KXPAYROLLS-25OCT settled 2025-12-16, every strike NO incl. T-100000 at .87 (BLS Oct payrolls
    -105k, JEC compilation of the 12-16 release) [critic: blank ev and -105k below the lowest rung -> indistinguishable from an
    all-No default; use KXPAYROLLS-25SEP T100000 .10 / T75000 .20 -> YES instead]; KXTSAW-25OCT12/19/26 settled 10-23..11-06 on TSA numbers (rules said
    "one week after" at most); KXPCECORE-25SEP/OCT settled 2026-01-03 / 01-23.
  * Source NEVER published the defined number -> settlement matched the book's last price on every TRADED strike (critic:
    KXU3-25OCT T3.8-T4.0 had no trades; core y/y T3.2's last trade was 8 h before close after trading .49-.75):
    KXCPI/KXCPIYOY/KXCPICORE/KXCPICOREYOY-25OCT settled 2025-11-22 02:04-03:00Z (BLS cancelled Oct CPI 2025-11-21, Bloomberg
    / Investing.com) at 0.2 / 3.3 / 0.2 / 3.3 -- NOT the last BLS print (Sep: +0.3% m/m, 3.0% y/y, core +0.2%, 3.0%; bls.gov
    cpi_10242025) -- and each strike resolved to its last-price side (e.g. CPI T0.1 .92 YES, T0.2 .25 NO; core YoY T3.2 .63
    YES, T3.3 .02 NO). KXU3-25OCT settled 12-16 at 4.4 (no Oct 2025 U3 exists: household survey not collected, BLS 12-16
    release) -- T4.3 .63 YES, T4.4 .25 NO. KXGDP-25OCT30 (named underlying "Advance Estimate", which BEA cancelled and
    replaced by an "initial estimate" released 2025-12-23 at 4.3%) settled 2025-11-24 at 3.8 (= Q2 third estimate) -- T3.5
    .64 YES, T4.0 .19 NO. Under the NEW definition (BEA's renamed initial estimate, 4.3%) T4.0 would have paid YES.
  Rules text then: CPI/CPIYOY secondary had a shutdown clause ("latest Expiration Date will be extended to the sooner of the
  release of the Underlying or six months after the end of the government shutdown"); U3 had no secondary text; GDP said
  "expire at the first 10:00 AM following the release ... or 3 months following". Kalshi settled the never-published ones
  well inside those windows.
- Watcher built + tested (see section below).

### 2026-11-21 ~06:20-06:28Z -- more cases (LMArena, clarifications, weather switch)
- **NWS -> Weather Company switch (Kalshi's own source change).** KXHIGHDEN-26AUG13 (listed 08-12 09:30Z, NWS text) settled on its
  NWS text; the first TWC event KXHIGHDEN-26AUG14 was listed late (08-13 17:10Z vs the usual 09:30Z) with new text
  (results/source_shift/rules_text.jsonl, census_markets.jsonl). => open contracts kept the OLD source; only new events moved.
- **LMArena default switched to style control on 2025-05-16** (HF dataset card lmarena-ai/leaderboard-dataset). Kalshi's
  KXLLM1/KXTOPMODEL text gained "check the 'Remove Style Control' toggle" only for events created from 2025-12-31 (7 months
  later), i.e. it pinned the PRE-switch view. In 2026, 21 KXTOPMODEL events had different #1 models in the two views; the 19
  with a YES market all settled on the no-style-control model, 0 on the style-control default (results/source_shift/lmarena/
  lma_check.py -> lma_vs_kalshi.csv). Critic: these fall in 4 contiguous stretches (effective n ~4), 5 rows use a leaderboard
  dated the settlement day itself, and the contract TEXT told traders which toggle -- so this shows Kalshi following its own
  text, not a mid-contract source change. The HF history matches neither view for all 10 YES events Dec-2024..May-2025:
  unusable before June 2025. The events open across the 2025-05-16 switch (KXLLM1-25MAY31, KXTOPMODEL-25MAY) cannot
  be judged: the HF history lacks the Gemini-2.5 rows Kalshi settled on (dataset top = o3 on 2025-05-22 in both views) --
  the free history is itself a revised vintage. Also KXLLM1-25DEC31: "This event is not affected by the amendment that is
  effective in April 2025" (tie-break rule change grandfathered for an open event).
- **Mid-contract clarifications found in rules text** (grep of rules_text.jsonl for Clarification/amendment; 31 distinct):
  * GTEMP-23: "NASA updated Land-Ocean Temperature Index values for previous years ... revising down 2020 from 1.02 to 1.01 ...
    The 2023 ... market will resolve based on whether or not 2023's value is greater than 1.02, as its rules say" -> OLD
    (literal number kept). Outcome 1.17 YES (did not matter).
  * TROPSTORM-23DEC01 (7/7/23): NHC's post-analysis added an unnamed January storm -> "encompassed by the Payout Criterion"
    -> NEW (revised count). Settled expiration_value 20 = 19 named storms Arlene..Tammy + the unnamed one (my count).
  * KXARCTICICEMIN-25OCT01 (7/1/25): DoD stopped the data NSIDC uses; "If the NSDIC replaces this dataset with something else
    by the Expiration Date, that data will be used ... [else] Rule 7.2 ... JAXA" -> NEW (replacement dataset). Settled 10-02.
  * GDP-23OCT26 (9/21/23) quotes Rulebook Rule 7.2: "if a Contract's Source Agency is unavailable or undergoes significant
    modifications, Kalshi retains the authority to designate a new Source Agency and Underlying for that Contract and to
    change any associated contract specifications after the first day of trading." Kalshi CAN switch definitions mid-contract.
  * Others are interpretation clarifications (FOMC cancelled -> all No; SEP not published -> all No; PJM alert counting;
    FEMA county definition; NWS alert link) -- definition disputes resolved by the exchange's reading, not a source change.

## Source-shift case table -- REVISED after critic review (2026-11-21 ~06:50Z; first-draft rows are described in the log above)
A fresh critic subagent re-read the files and ~19 Kalshi records; it broke the first base-rate count (3 of 5). Changes: GDP moved
to "never published" (the Advance Estimate never existed; BEA's 4.3 arrived inside the expiry window, latest_expiration_time
2026-01-29, and Kalshi used neither); GTEMP dropped (1.02 is a constant in the contract; NASA's revision touched 2020, not the
2023 value measured; no payout effect); TROPSTORM dropped (rungs step by 3, so the added storm changed no payout, and the ev
path T15 @ ev 18 on 10-06 vs T18 @ ev 19 on 10-24 is internally inconsistent -- unresolved); sea ice -> undetermined (no file
shows a replacement or JAXA used; strikes kept settling on the usual series); added KXWAYMOCITIES and KXALBUMSALES (both NEW).
| case | date | what changed at the source | Kalshi settled on | payout effect | class |
|---|---|---|---|---|---|
| KXALBUMSALES (ROM, SON) | 2026-01-26 | Hits Daily Double moved its chart date Fri -> Thu | NEW, applied to OPEN events listed 01-08 / 01-23 ("Kalshi will resolve future markets using the chart dated Thursday ... pursuant to Rule 7.2(a)") | yes (which week's chart) | methodology/schedule |
| KXWAYMOCITIES-25 | 2025-12-08 | source app renamed 'Waymo One' -> 'Waymo' | NEW ("clear continuity ... continue to track ... through name changes"); settled 3 | possible | rename |
| KXAAAGASD | 2026-08-05 | AAA display 3 -> 4 decimals | NEW: KXAAAGASD-26AUG05 (listed 08-04) settled 4.0801; 08-04 was 4.089 | none (only ties; none recorded) | display |
| Weather material errors (winners_transfer P4) | 2026-08..11 | TWC display error later corrected | corrected value; 3 of 15 holds flipped the >= 85c favourite (NY 11-13 $410k) | yes | correction |
| KXGDP-25OCT30 | 2025-11-24 | "Advance Estimate" cancelled; BEA "initial estimate" 4.3% on 12-23, inside the window | neither: 3.8 (= Q2 third estimate = every traded strike's last-price side) | yes (T4.0 .19 NO; 4.3 would be YES) | never published / renamed |
| LMArena KXLLM1/KXTOPMODEL | 2025-05-16 | default view -> style control | open events undetermined (HF history unusable before 2025-06); Kalshi later wrote the OLD view into NEW contracts (from 2025-12-31) | -- | default view |
| Kalshi's own NWS -> TWC | 2026-08-14 | exchange changed source | OLD for open events (all 39 KXHIGH/KXLOWT: 26AUG13 listed 08-12 09:30Z settled at the old ~5.2 h delay; 26AUG14 listed 08-13 17:10-17:12Z) | n/a | exchange switch |
| Shutdown, late prints | 2025-10..12 | source delayed | the real print, against the book: KXPAYROLLS-25SEP (ev 119,000) T100000 .10 -> YES, T75000 .20 -> YES; KXU3-25SEP T4.3 .31 -> YES | yes | outage, published late |
| Shutdown, never printed (CPI Oct x4, U3 Oct, Eggs Oct) | 2025-11/12 | no number, ever | every TRADED strike on its last-price side; 3 of 6 also equal the last published print (U3 4.4, core m/m 0.2, GDP 3.8); only the CPI headline m/m, headline y/y, core y/y rule that out -- and those 3 settled the same night (one decision); ev 3.3 looks like a label (y/y ladder tops at T3.1) | -- | outage, never published |
| KXOPENROUTER KXTOKENUSE-26NOV23 / KXTOPUSAGEAI-26NOV16 | 2026-11-20/21 | default chart -> BYOK-excluded | PENDING (11-23) | yes | default view |
KXPAYROLLS-25OCT is NOT evidence for "waited for the real print": its expiration_value is blank and -105k is below the lowest
rung, so it is indistinguishable from an all-No default (critic). KXEOWEEK's FR-vs-WH rungs are a named-vs-faster-page dispute,
not a change over time; excluded.
**Base rate as pre-registered** (settlements that followed the NEW definition among source-change cases with a determinable
OLD/NEW on an open contract; corrections included, outages and Kalshi's own switch reported separately): NEW 4 of 5 decisions
(ALBUMSALES, Waymo, AAA, weather corrections; GDP not NEW), Jeffreys 95% [0.37, 0.98]; 6 of 7 if the three corrected weather
events count separately, [0.50, 0.98]. n < 10 -> **anecdotal, not a base rate**; the decision-level interval crosses 0.5.
Deviation note: the first draft counted revisions but left out corrections; this version follows the pre-registration.
**What the rows say (pattern, not a rate):** on OPEN contracts Kalshi has followed the source's current presentation every time
it was observable and payout-relevant (album chart date via Rule 7.2(a), renamed app, corrected weather values). OLD appears only
when Kalshi itself changes the source (grandfathering) or when it writes a definition into NEW contracts (LMArena toggle).
When no defined number ever appears, settlement matched the book's last price (and in half the cases the last print too).
For the live P-1 bet this leans NEW (= BYOK-excluded) -> the T238 YES is more likely a loser than P-1's 10-20% suggested.

### Coverage and stopping point (census stopped ~06:52Z; resumable: `cd results/source_shift && ../../../.venv_quant/bin/python census.py`)
2,515 of 8,342 universe series pulled: Economics 924 (537 with markets), Climate 584 (278), S&T 397 (263), Entertainment 610
(389, partial). NOT covered: rest of Entertainment, Companies, Health, Transportation, Social, World, Commodities, Financials
(weekly+), Mentions, Politics; Elections and the big daily weather ladders were excluded for speed (weather holds come from
winners_transfer). So the case table is a lower bound; the clarification grep should be rerun when the census completes.
KXDEBTSHRINK-28NOV11 (clarification 4/22/25 pins the Q4-2024 debt baseline at $36,218,605M) is not classified.

## Watcher: scripts/source_shift_watch.py (read-only; NOT installed -- lead decides; suggested cadence hourly)
Output: ~/.claude/quant-sports/data/source_shift/snapshots.jsonl (one line per run) + state.json (baselines, rolling
histories, per-period values, per-event strikes/rules hashes). Prints `ALERT <kind> <source> <detail>`.
| source | definitions logged each run | alerts |
|---|---|---|
| OpenRouter token chart | default, byok=organic, byok=all weekly totals (last 3 weeks), #1 model per view | default view switches (default==organic/all/neither), organic/all ratio leaves its band, completed week revised, #1 model differs between views |
| OpenRouter shares | author-market-share and open-closed (default/organic/all); TEXT REQUEST share (modality-chart?routeSegment=text marketShareData, the Kalshi share-series source) | a BYOK filter appears on a share chart; completed-week share revised > 0.05 pp |
| Vercel AI Gateway | labs export (CC-BY) and the page's own groupBy=chef endpoint; weightClass (open weights) | export vs page gap > 0.1 pp; a day >= 2 days old revised > 0.05 pp; schema change |
| AAA | Current/Yesterday/Week/Month/Year rows, all grades, decimals | precision/grade/row change; yesterday's print differs from what we logged as Current |
| TSA | tsa.gov is NOT fetched (403 to scripts; not worked around); Kalshi KXTSAD first prints | a first print changes |
| Weather | Weather Company display (weather.com/kalshi/api/climate/primary, d-2..d) vs NWS CLI (api.weather.gov) for the 7 E4 stations | TWC vs final CLI max differ >= 1F (NY 11-13 signature); revised reports; official value changes; unknown status |
| NOAA | NCEI land+ocean CSV sha + last 24 months | new month (first print = Kalshi vintage); any of last 24 months revised |
| Kalshi (39 series) | open events: strikes, rules template hashes, bids/asks; series settlement_sources / contract URLs | new strikes (FAR if > 2 steps outside the old ladder), rules template change mid-event or between events, source/URL change, new OpenRouter-series event |
Test (2026-11-21 06:08-06:28Z): 4 real runs, 0 alerts after baselines. A synthetic-state run (state edited to remove
KXTOKENUSE T137..T200, fake a KXTSAW template, fake default=all, fake AAA 11-19 = 3.7000, ratio history at 1.0, fake KXHIGHNY
template) fired all 6 alert kinds correctly (KALSHI_NEW_STRIKES with FAR [137..193], KALSHI_RULES_CHANGED x2, FORMAT,
REVISION, DIVERGENCE); the real state was restored afterwards and the synthetic line removed from snapshots.jsonl.
Wrong turn: first version used a browser User-Agent; AAA and tsa.gov answer differently by UA. Switched to an honest UA
(every source except tsa.gov answers it) and dropped tsa.gov rather than work around its block.

### 2026-11-21 ~06:30Z -- detector pass 2 (census 1,817 of 8,342 series: Economics, Climate, S&T done; rest partial)
- results/source_shift/analyze.py on 224,355 markets / 32,366 events: (a) late far strikes 480 events / 124 series (Economics
  424, Climate 35, S&T 21); (b) 76 events whose later-listed markets carry a new template (18 with new secondary text), 654
  series-level template changes in 173 series; (c) holds 1,696 (1,331 Climate -- mostly long-lived legacy events), odd
  (scalar) settlements 22.
- (a) in S&T is the P-1 signature only for the live case; settled examples are ordinary moves (KXTOKENUSE-26OCT26 +5 strikes
  above, settled 188T inside the new region; KXANTHVSPEND-10NOV26 strikes to 65.8 when spend share spiked, settled 45.9).
  Late far strikes are mostly the exchange chasing a moving number, so the detector needs a source-divergence check to be useful
  -- that is what the watcher adds.
- (b) mid-event template changes in S&T are almost all KXLLM1 (tie-break amendment 2025-04, style-control text 2025-12-31,
  model-name edits) plus KXARCTICICEMIN-25OCT01 (the DoD-data clarification). No other mid-event source switch found.
- **More "no data -> book" settlements (P-5):** scalar (fair-price) settlements after multi-day holds when the display had no
  valid value: KXHIGHTDC-26FEB02/05/24, KXLOWTATL-26APR14, KXLOWTMIA-26APR11/JUL07, KXHIGHMIA-26APR11 (e.g. KXLOWTMIA-26JUL07
  B78.5 settled 0.72 with last .75, B74.5 0.24 with last .24), KXRAIN-26NOV03-PHX 0.01 after 7 days while the other 32 cities
  settled normally. Weather rules say exactly this ("all markets will resolve to the last fair price determined by Kalshi").
  KXCPICORE-25DEC settled at scalar 0.75/0.13/0.08/0.04 on 2025-12-29, two weeks BEFORE the Dec CPI print (KXCPI-25DEC settled
  0.3% on 2026-01-13) -- looks like an exchange-side early termination at fair price; cause not found (flagged, not classified).

## Open P-1 candidates (2026-11-21 ~06:33Z)
1. **KXTOKENUSE-26NOV23** (our $14.08 T238 YES; settles 11-23 10:00 ET). Organic (default) week-so-far 156.0T vs byok=all
   257.7T (watcher 06:28Z). For BYOK-inclusive: every past settlement used it; LMArena precedent (Kalshi pinned the pre-switch
   view); contract names no view. Against: "as observed" page reading (AAA display precedent), Kalshi listed T137..T200 at
   04:10Z, sibling KXTOPUSAGEAI says "as displayed", book T238 .01/.05, and every observable payout-relevant mid-contract
   change on an open contract went NEW (album chart date via Rule 7.2(a), Waymo rename, weather corrections). Estimate: below
   P-1's 10-20% (no number defensible at n ~5); hold the $14.08, do not add.
2. **KXTOPUSAGEAI-26NOV16** (closes 11-23 04:59Z): #1 model organic = deepseek-v4.1-flash 49.2T -> DEEP; byok=all =
   tencent/hy4-flash 96.8T -> TENC. Book DEEP .88/.99, TENC 0/.14. Same question as 1 at a worse price -> no trade.
3. **KXTOKENUSEM-26NOV30** (monthly Nov tokens, settles ~12-01): same BYOK question on a month; book illiquid (bids 0-.20,
   asks .70-1.00 on most rungs, snapshot) -> nothing tradeable; record which view settles 26NOV23 first.
4. **OpenRouter share series** (KXANTHSHARE/GOOG/DEEP/OPENSOURCESHAREOR, week of 11-16): source = text REQUEST share
   (BYOK-inclusive, no toggle). Book matches it (anthropic 5.71 vs T5.6 .33/.37; deepseek 25.2 vs all rungs >= 28.5 at
   0/.01). Only a NEW organic toggle on that chart before Mon 10:00 ET would move them (+~15% relative if tencent's 13.2% is
   all BYOK). No evidence it is coming -> watch (watcher FORMAT alert on byok_filterable).
5. **Weather TWC display vs NWS CLI**: forward watch only (I1 revived via the free display JSON); 0 disagreements in 4 runs.
None of these is a trade recommendation; the base rate cannot carry a bet (P-4).

## Learnings
### On open contracts Kalshi has followed the source's current definition whenever it mattered
Pre-registered count after critic review: NEW 4 of 5 decisions (album chart date moved Fri -> Thu, applied to open events via
Rule 7.2(a); Waymo app rename; AAA 3 -> 4 decimals; weather corrections) vs GDP (settled on neither), Jeffreys [0.37, 0.98], n < 10
= anecdotal. OLD appeared only when Kalshi changed the source itself or wrote a view into new contracts (LMArena). Source:
notes/source_shift.md revised case table. The first draft (3 of 5 with GTEMP/TROPSTORM/sea ice) was wrong -- a critic broke it.
### A cancelled data release settles near the book (and often the last print); a delayed one settles to the real print
2025 shutdown, never published: every TRADED strike settled on its last-price side (CPI 0.2/3.3; U3 4.4; GDP 3.8), but U3, core
m/m and GDP also equal the last published print; only one same-night CPI decision rules that out. Published late: Kalshi waited
and settled on the real print against the book (KXPAYROLLS-25SEP T100000 .10 and T75000 .20 -> YES; KXU3-25SEP T4.3 .31 -> YES).
Source: results/source_shift/census_markets.jsonl, /historical/markets rules text.
### Kalshi grandfathers open contracts when it changes the source itself
NWS -> Weather Company (2026-08-14): KXHIGHDEN-26AUG13 kept NWS text; the first TWC event was listed late (08-13 17:10Z).
KXLLM1-25DEC31: "not affected by the amendment that is effective in April 2025". Source: rules_text.jsonl.
### The Weather Company display Kalshi settles weather on has a free date-addressable JSON
weather.com/kalshi/api/climate/primary?date=YYYY-MM-DD returns 66 stations with max/min and official/revised/preliminary status
for any date (2026-11-13 NYC official 47 = the settled value after the 155 h hold). Found by grepping the page's Next.js chunk.
Enables a forward settlement-error watcher (preliminary display vs NWS CLI final). Source: scripts/source_shift_watch.py.
### OpenRouter share series settle on text REQUEST share, not token share
modality-chart?routeSegment=text -> marketShareData; week 11-09 anthropic 6.27 -> KXANTHSHARE-26NOV16 settled 6.3. The byok
param is ignored there and on author-market-share/open-closed (BYOK-inclusive only, 2026-11-21). Source: watcher snapshots.
### With the BYOK split, "top model by usage" has two different winners
Week of 11-16: #1 model = tencent/hy4-flash 96.8T with BYOK vs deepseek-v4.1-flash 49.2T without; KXTOPUSAGEAI-26NOV16 book
DEEP .88/.99 (prices the organic default, "as displayed"). Source: watcher snapshot 2026-11-21T06:28Z.
### Mid-contract "Clarification" text is Kalshi's archive of past disputes
31 distinct clarifications/amendments in stored rules text (NASA revision, NHC added storm, DoD sea-ice data stop, FOMC
cancelled, SEP not published, Rule 7.2 source substitution). Grep rules text before assuming how an ambiguity resolves.
Source: results/source_shift/rules_text.jsonl.
### Free leaderboard histories can be revised vintages
The LMArena HF dataset has no Gemini-2.5 rows in May 2025 although Kalshi settled KXTOPMODEL-25MAY on Gemini-2.5-Pro; it cannot
judge events open across the 2025-05-16 style-control switch. Same trap as NOAA first prints. Source: lmarena/lma_check.py.

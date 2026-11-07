# Information-gap census (2026-11-07) -- where a free public feed publishes the resolution quantity before the market reprices

READ-ONLY research. No orders, no code or registry edits, nothing committed. Every number below was read from an API or a
file during this session (snapshot window 2026-11-07 ~20:20-20:50 UTC). Scratch scripts and raw pulls live in
/tmp/infogap/ (not in the repo). **One snapshot is an anecdote, not evidence**: every gap below is a hypothesis for a
forward logger, never a result.

## Method
1. Universe: Kalshi `/events?status=open&with_nested_markets=true`, paginated, giving 21,911 open events
   (/tmp/infogap/open_events.json), aggregated by series on `volume_24h_fp`, `open_interest_fp` and the median YES spread of
   two-sided markets (agg.json). Polymarket gamma `/events?active=true&closed=false&order=volume24hr` returned the top 2,000
   events (pm_events.json). Polymarket US gateway `/v1/events` returned 3,000 active events (pmus_events.json). Polymarket
   **international is not US-tradable** (FUND_BOOK 2.1, notes/pmus_lag.md), so a family that lives only there counts as reference only.
2. Filter: non-sports families whose resolution is a published count, tally or leaderboard. I dropped everything in FUND_BOOK
   3d unless I had a new angle. About 25 families were scanned (table). Already covered elsewhere: Q013 RT, Q014 Truth Social,
   Q017 ERCOT, E1X-T TSA (loggers), L26 intraday station max, N6/Q016 gas, R1 rain, Q012 crypto hourly, PW PortWatch,
   SX1 SpaceX, MEDIA YouTube/Netflix and M3 mentions (dead).
3. For each family: `rules_primary`/`rules_secondary` plus `settlement_sources` from the event payload; a curl probe of the
   leading feed (HTTP status recorded; 403 and bot walls were skipped, never bypassed); one snapshot of source vs mid.
4. Ranking: likely gap × capacity × how cheaply it can be logged forward, with a freshness-risk judgement taken from the
   election-needle lesson (notes/election_needle_2026.md).

## Headline
- The only family with real size right now is the **post-election canvass**: margin brackets, uncalled winners, seat
  counts, popular vote and turnout. Its feeds are free and fetchable. Every contested market sits AWAY from the raw count, but
  **Kalshi and Polymarket agree with each other** in every cross-checked case. By lesson 4 of the needle note, that means the
  markets are pricing the composition of the outstanding ballots, not lagging. The free gap against the naive count is not an
  edge. An edge would need a better outstanding-ballot model or a faster county-level drop than the state or AP aggregators.
- Lesson 11's stale-feed trap is live and measurable. The NYT model block (`nyt_model_estimates.nyt_updated_at`) is frozen at
  **2026-11-04T08:08Z** for every Senate race, while the AP counts in the same file kept updating through
  2026-11-07T20:22Z. The Maine counted margin (6.297) is already above that frozen model's top quantile (6.26). The CA SoS
  `api.sos.ca.gov/returns/status` endpoint serves **2024-09-16** timestamps (a stale county-status feed). Use neither.
- The "running count" families on Kalshi (presidential actions, EOs, Trump golf) are watched. On the snapshot the market
  matched the resolution page exactly, so the only gap would be a seconds-to-minutes latency race.

## Ranked table

| # | family | venue / series | resolution source (rules) | leading feed + cadence | fetch OK? | snapshot gap (one look) | volume / spread | freshness risk | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Canvass margin brackets** | Kalshi KXMIDTERMMOV (48 events with a quoted 3-97c market), KXAKMOV, KXVOTEGENERAL, KXCAWEALTHTAXMOVD; PM US `ushrmov-ca-*` (open) | state SoS / certifying authority, e.g. "Mississippi Secretary of State", "California Secretary of State", "State of Alaska (final RCV round)" | AP counts in NYT JSON `static01.nyt.com/elections-assets/pages/data/2026-11-03/results-{senate,governor,house}.json` (per-race `updated_at`, minutes to days); CA SoS `api.sos.ca.gov/returns/*` (daily, `ReportingTime` 2026-11-06 23:00) | NYT 200 (1.40 MB senate, 1.59 MB governor, 10.2 MB house); CA api 200; CA `electionresults.sos.ca.gov` Incapsula (skipped); AK elections.alaska.gov timeout; MS no feed found | ME Sen: counted Jackson +6.297 at eevp 93.58 (707,000 of 755,524); Kalshi "6+" 0.04/0.18 (v24 3,123), "5.5+" 0.86/0.94. MS Sen: counted Colom +0.270 at eevp 96.92 (953,314 of 983,599); Kalshi "0.6+" 0.80/0.92, "1+" 0.13/0.26; PM "Colom 0.6-0.9%" 0.551/0.709. FL Gov: counted Donalds +1.296 at eevp 99, 0 of 23,561 expected provisionals counted; Kalshi "1+" 0.63/0.66 (v24 220, OI 79,561). CA Prop 40: counted No +7.470 (Yes 4,541,175 / No 5,274,431); Kalshi "No ≥7%" 0.19/0.24 (v24 9,922). CA Gov: counted Hilton 34.7%; Kalshi "≥36%" 0.25/0.27 | KXMIDTERMMOV 24h 361,170 contracts across 518 events (5,172 mkts); per bracket 0-8k/24h; median quoted spread 0.54 on the whole series (wide tails), 1-17c near the money | HIGH: AP total_expected_vote is an estimate; counties drop in batches; the NYT model is frozen (above); provisionals are uncounted until canvass | **TOP, time-limited.** Forward logger now through certification; judgment trades only under the needle rules |
| 2 | **Uncalled winners** (provisional / cure outstanding) | Kalshi KXHOUSERACE (NC-03, NC-05), KXCAELECTION (CA-04, CA-07), KXAPCALLMSSEN; PM `nc-03-house-election-winner` | sworn-in member (Library of Congress); AP call for KXAPCALLMSSEN | NC SBE `er.ncsbe.gov/enr/20261103/data/results_0.txt` (Last-Modified Fri 06 Nov 20:10:38 GMT); AP vote-type split in NYT JSON | NC SBE 200 (191,621 B); NYT 200 | NC-03: SBE and AP show Murphy (R) 117,583 vs Smith (D) 117,350 (R +233); AP expected 240,925 vs counted 237,904; absentee-mail split Smith 2,804 vs Murphy 1,160; NYT race `updated_at` 2026-11-05T17:30Z. Kalshi R 0.17/0.19 (v24 1,250,124), D 0.82/0.83 (v24 455,728); PM Smith 0.816/0.817. NC-05: Hubbard (D) +99 (139,034 vs 138,935); Kalshi D 0.93/0.95, PM D 0.903/0.942 | NC-03 is the single most-traded non-sports market today | HIGH: both venues price the trailing candidate as favourite. The feed is fresh but incomplete (provisional = 0 counted), so the composition of what is outstanding is the information | **Part of #1.** Lesson 4 in action: the market is not lagging the count |
| 3 | National aggregates | Kalshi KXHOUSEPOPVOTEMARGIND, KXHOUSETURNOUT, KXDSENATESEATSH, KXCAHOUSEDEM; PM `2026-midterms-house-popular-vote-margin-of-victory-224` | Clerk of the House (certified; resolves 2027-02) | NYT house JSON national row `US-G-H-2026-11-03` (`updated_at` 2026-11-07T20:24Z) | 200 | Counted D-R +12.951 at eevp 94.11 (114,332,706 of 121,487,271 expected); Kalshi "≥14%" 0.88/0.96, "≥14.2" 0.58/0.68, "≥14.5" 0.16/0.19; PM "14-16%" 0.876/0.886. Turnout: expected 121.49M vs Kalshi "120-124.99M" 0.92/0.94 | ~2-5k contracts/24h per bracket | MED: long horizon; CA is the swing; the Clerk count includes unopposed races that AP reports as 0 (e.g. FL-10 total_votes 0) | Fold into #1 logger (cheap: one row) |
| 4 | Post counts via xtracker | PM intl `elon-musk-of-tweets-*`, `donald-trump-of-truth-social-posts-*`, Zelenskyy etc.; Kalshi KXTRUTHSOCIAL(D) (covered by Q014) | PM: xtracker "Post Counter"; Kalshi: Roll Call Factbase | `xtracker.polymarket.com/api/users` (`lastSync` 20:28-20:33Z, i.e. minutes) | 200 | Elon Nov 3-10: count 184 at 99.7 of 168 h, naive pace 310; PM modal brackets 280-299 0.22/0.23, 300-319 0.227/0.234 (consistent). Trump Nov 3-10: count 52, naive pace 88; PM 100-119 0.212/0.231, 80-99 0.145/0.157 (market above naive pace, consistent with bursty posting) | PM Elon weekly event 24h $295,095; Kalshi KXTRUTHSOCIALD 24h 3,842 contracts | LOW: the feed IS the PM resolution source and syncs in minutes | Not US-tradable. Use as a **2nd-counter freshness check for Q014** (amendment proposal below) |
| 5 | OpenRouter weekly tokens | Kalshi KXTOKENUSE (1 event, 13 strikes) | OpenRouter rankings value for 7 UTC days, read 10:00 ET next day | `openrouter.ai/rankings` HTML embeds per-model weekly totals ("Usage data through Nov 6") | 200 (1.54 MB) | Embedded top-20 models (view=week, dated 2026-11-06) sum to 161.34T; the market strikes total all models (210T 0.69/0.90, 214T 0.21/0.22, 223T 0.03/0.05). The full total is not in the HTML (only via an undocumented frontend call, not pursued) | 24h 19,845 contracts; 1-4c spreads | MED: 1-day reporting lag; revision behaviour unknown | LEAD, data gate first (stub C) |
| 6 | Album first-week units | Kalshi KXALBUMEQUIV (14 open events, 140 markets) | Luminate API value Sunday 10:00 ET | free precursor: industry mid-week projections (HitsDailyDouble); not fetched | not probed | none taken | 24h 23,954 contracts; median spread 0.04 | MED: projections get revised; Luminate is paid | LEAD-low (stub E, data gate first) |
| 7 | White House presidential actions | Kalshi KXTRUMPACT | whitehouse.gov "Presidential Actions" dates at 10:00 ET Sunday | the same page | 200 | Page shows 3 items dated Nov 1-7 (Nov 3 x2, Nov 5); Kalshi "≥3" 0.99/1.00, "≥4" 0.01/0.02 | 24h 28,054 | LOW | Watched. Latency race only. No logger |
| 8 | Executive orders / week | Kalshi KXEOWEEK | Federal Register signing date | FR API `/api/v1/documents.json?conditions[presidential_document_type]=executive_order` (count 1 signed ≥ 10-20: EO 14438 signed 10-22, published 10-28); WH EO page newest 2026-10-22 | 200 / 200 | Week 11/1-11/7: no EO on either feed; Kalshi "above 0" 0.02/0.03 | 24h 21,756 | MED: FR lags ~6 days and a backdated EO can appear | Watched; the FR lag is known to traders. No logger |
| 9 | Trump golf days | Kalshi KXTRUMPGOLFD | Factbase calendar JSON `golf` flag or a pool photo | `media-cdn.factba.se/rss/json/trump/calendar-full.json` (named in rules) | not probed | Kalshi SAT 0.96/1.00, SUN 0.69/0.79 | 24h 5,133 | LOW | Too thin |
| 10 | Vercel open-weights share (daily) | Kalshi KXOPENSOURCESHARE (5 events) | Vercel AI Gateway chart value for day D, read D+1 10:00 ET | `vercel.com/ai-gateway/leaderboards/...` (HTML embeds per-model daily shares) | 200 (1.68 MB) | Not computed (open-vs-closed series not located in the HTML within the time cap) | 24h 22,879 across 5 events; median spread 0.17 | MED | Low: a single-day number, not observed before the day ends |
| 11 | LMArena top model | Kalshi KXLLM1 (OI 9,204,870) | LMArena leaderboard, "remove style control" | arena.ai / lmarena.ai | **403** (skipped) | -- | 24h 131,698 | -- | Skip (blocked; event-driven, not a count) |
| 12 | App downloads (Oct/Nov) | Kalshi KXGPTAPP, KXCLAUDEAPP, ~15 others | Carbon Arc YoY index (paywalled "prisms") | none free | -- | -- | 24h 117,813 (GPT); spreads 0.05-0.2 | -- | Skip (no free resolution path) |
| 13 | Polymarket box office | PM `*-opening-weekend-box-office` (e.g. Ramayana 24h $36,647) | Box Office Mojo | BOM `/weekend/` and The Numbers daily chart | 200 / 200 | -- | -- | -- | Not US-tradable; no open Kalshi box-office series found |
| 14 | Polymarket intl daily highs | PM `highest-temperature-in-<city>-*` (dozens; 24h $10-42k each) | Wunderground station | `aviationweather.gov/api/data/metar` | 200 | -- | -- | -- | Not US-tradable; L26 (Kalshi) already showed traders see obs in 1-5 min. No new angle |
| 15-25 | Dead or covered, re-checked for liquidity only | KXAAAGASW/D (24h 423,715/218,623), KXRAIN (309,343), KXBTCD (1,480,787) and KXBTC15M (530,902), KXHORMUZ* (75,313), KXSPACEXCOUNT, KXYTVIEWSW, KXNETFLIXRANK*, KXTSAW/KXTSAD (1,650/375), KXRT (629,435; Q013), KXTRUMPSAY (mentions) | -- | -- | -- | -- | -- | -- | Dead per FUND_BOOK 3d, or already logged. No new angle found |

### What the canvass snapshot does and does not show
- In every contested market, prices sit where a forecast of the OUTSTANDING ballots would put them, not at the raw count. The
  direction is plausible from the AP vote-type split. In NC-03 the absentee-mail ballots run Smith 2,804 vs Murphy 1,160, and
  0 provisionals of ~3,021 outstanding votes are counted. At that mail-ballot share, Smith would net several hundred to ~1,200
  votes against a 233-vote deficit. That is my arithmetic on AP's estimates, not a model.
- Polymarket agrees with Kalshi to within about 1c on NC-03, MS and the House popular vote. Per the needle lesson this is the
  "both venues disagree with the feed, so the feed is incomplete" case, not an edge.
- The candidate genuine disagreements are FL Gov "1+" (0.63/0.66 while counted +1.296 at eevp 99) and ME Sen "6+"
  (0.04/0.18 while counted +6.297 at eevp 93.58). In FL, Jolly would need roughly 78% of the two-party share of the ~44k
  outstanding votes for the margin to fall below 1.0. That is my arithmetic from AP's `total_expected_vote` (8,311,226) and
  counts. It could be wrong if the FL Division of Elections count (the resolution source) differs from AP's. I could not read
  that count: `results.elections.myflorida.com` returned 200 on the landing frame and 500 on my guessed download URL.
  The ME bracket is thinly traded, and its quantity depends on which towns are outstanding. Both are anecdotes.

## Pre-registration STUBS -- PROPOSALS ONLY, nothing has run, not added to QUESTIONS.md

Power reference (from notes/contract_census_2026-10-29.md, `scripts/edge_math.py power`): +2% at 93c needs ~871 bets; +3% at
90c needs ~574. Brackets within a race are strongly correlated, so **races, not brackets, are the unit**.

### Stub A [proposal] -- Canvass margin brackets vs a count-plus-outstanding projection (G-CANVASS)
question:  From now until certification, at fixed 6-hourly snapshots, does a projection (AP current count + AP outstanding
           expected vote, allocated by vote-type shares already counted in the same race and, where available, by county/town)
           price the margin brackets better than the Kalshi mid?
mechanism: The settlement value is a deterministic sum of ballots, part of which is already public. The only uncertainty is the
           size and composition of the outstanding ballots. The other side: professional election traders with county-level
           desks (DDHQ-style), partisans, and late-arriving retail. Prior: the market already beats the naive count
           (this snapshot), so an edge needs better composition modelling or faster county drops.
baseline:  (i) Kalshi mid at the snapshot; (ii) naive "final = current margin". Show (ii) loses before claiming anything.
change:    the single projection above, frozen before the first logged snapshot. No NYT model block (frozen 11-04 08:08Z).
freshness gate: use a row only if the race's AP `updated_at` is < 6 h old AND the market's last trade is < 6 h old; log, don't
           drop, the rest.
metric:    Brier per bracket vs the certified value (state SoS certification; AP final as an audit), aggregated per race;
           race-clustered bootstrap. Secondary: paper ROI at the real ask + fee for |model - ask| >= 8c, one entry per race.
success:   proposal: projection Brier < mid Brier with the race-clustered 95% CI excluding 0 over >= 30 races, AND paper ROI LCB > 0.
kill:      proposal: projection Brier >= mid Brier; OR (ii) naive beats (i) market (would mean the logger or the join is broken);
           OR fewer than 30 races pass the freshness gate.
power:     about 40-50 contested statewide or district canvass markets this cycle (48 KXMIDTERMMOV events quoted 3-97c today, plus
           winners). That is enough for a coarse Brier comparison only; ROI is **underpowered for 2-3% edges within one cycle**.
           Run descriptive now; pool with future primaries, specials and international canvasses for any sealed claim.
cost:      free. One fetch of three NYT JSONs, CA SoS and NC SBE every 30-60 min (well under any rate limit). Kalshi
           `/events?series_ticker=` polled for held or contested races only (lesson 14: 85 series @ 20 s caused 429s).

### Stub B [proposal] -- Uncalled-winner markets with outstanding provisional/cure ballots (G-UNCALLED)
question:  For races with |margin| < 0.5 pts and provisionals uncounted, does a vote-type composition model (provisional and
           late-absentee share by party, taken from the same race's counted absentee split) give a better win probability than the
           mid, and does it move before Kalshi does when the state file updates (NC SBE `results_0.txt` Last-Modified)?
mechanism: The counting authority publishes the file, and the market reprices on the AP/NYT relay. A lag window may exist
           between state-file publication and repricing, as in AZ-08 on 11-03.
baseline:  mid at the same timestamp; "current leader wins".
metric:    (a) latency: minutes from state-file Last-Modified change to the first Kalshi mid move >= 5c; (b) Brier vs result.
success:   proposal: median latency >= 10 min on >= 10 file updates AND model Brier <= mid. Below that, nothing to trade.
kill:      proposal: median latency < 5 min, or model Brier > mid.
power:     a handful of races (NC-03, NC-05, CA-04, CA-07, AK) gives tens of file updates. **Descriptive only.** It decides
           whether a needle-style judgment bot is worth arming at the next canvass.

### Stub C [proposal] -- KXTOKENUSE weekly OpenRouter tokens vs the partially observed week (G-TOKENS)
data gate (FIRST, no prices): reconstruct the all-model 7-day total from the public rankings page (or a documented API) for each
           settled KXTOKENUSE week, and compare with the settled `expiration_value`. Pass if it matches within 1% on >= 90% of
           weeks. Fail means dead at the data gate (same as PW).
question:  On day 5-6 of the week, with 4-5 days observed, are strikes more than 3 sd from the projection mispriced?
baseline:  mid; naive projection = observed days × 7/n.
metric:    ROI at the real ask + fee, clustered by week; Brier vs mid.
success / kill: proposal: DEV ROI LCB > 0 at p < .05/#trials / DEV ROI <= 0 or < 20 settled weeks.
power:     one event per week. Settled history length unknown (check `/markets?series_ticker=KXTOKENUSE&status=settled`).
           Likely **underpowered** (cf. MEDIA ytv_m1_maker n 18), so a forward logger at most.

### Stub D [proposal, amendment to Q014 -- PM to decide] -- xtracker as a second counter for Truth Social
question:  Does the xtracker running count (Polymarket's resolution source, syncing every few minutes) agree with the Factbase
           count Q014 uses (Kalshi's source)? Are Kalshi KXTRUTHSOCIALD brackets slower than the Polymarket brackets on the same
           window after a posting burst?
use:       a freshness gate (lesson 11) plus a cross-venue check (lesson 4). Not a stand-alone strategy: PM intl is not US-tradable.
metric:    count agreement rate per day; lag (minutes) of the Kalshi mid vs the PM mid after a burst of >= 5 posts in 30 min.
kill:      proposal: counts disagree on > 10% of days (the second counter is useless) or lag < 2 min.

### Stub E [proposal] -- KXALBUMEQUIV vs free mid-week projections (G-ALBUM)
data gate (FIRST): is there a free, timestamped, ToS-compatible mid-week projection source? Do archived projections match the
           settled Luminate value within ~15% on >= 80% of settled events? Also re-run the census's "far rungs one-sided"
           check at the decision time (it killed N6 and E1X-T taker).
question / metric / kill: as Stub C, clustered by album; ROI at the real ask; kill if the data gate fails.
power:     ~14 events open now (weekly cadence of releases), so the event count may reach 100+ over months. Unknown until the
           settled list is pulled.

## Surprises / things to remember
- NYT JSON mixes a **fresh AP count** with a **frozen model** in the same object. A logger that reads `nyt_model_estimates` would
  repeat the stale-needle loss. Key on `top_reporting_unit.provider_updated_at`, not on the model block.
- The NYT house file contains a national aggregate race (`US-G-H-2026-11-03`). Summing the districts naively doubles the House vote
  (my first sum gave 228.7M; the national row says 114.3M). Leakage/bug list candidate (FUND_BOOK s6 item 6).
- CA SoS `api.sos.ca.gov/returns/status` returns 2024-09-16 county timestamps, while `/returns/ballot-measures` and
  `/returns/governor` carry 2026-11-06 23:00. These are endpoint-specific stale feeds on one host.
- Kalshi has no open box-office series and no Elon post-count series. The biggest count markets (Elon tweets) live on PM intl only.

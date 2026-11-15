# Real-time sharp odds feeds for a CA retail user (sharp_feed_research, 2026-11-15)

Question (from lead): which real-time (seconds, not ~15-min cached) sharp-book odds feeds for NFL/NBA/NHL/soccer,
pre-game AND in-game, can one California retail user legally access, at what cost; cheapest way to test
"Kalshi lags a real-time sharp price"; cost of a 2-week test. Desk research only: no accounts, no spend, no trials started.

Why: free Pinnacle guest odds are Cloudflare-cached (max-age ~905 s, median 440 s old at read) -> xv_live_lag_t03
untestable (EDGES_AND_AGENTS.md line ~350; KS12 in LEARNINGS_2026-11.md). PM US REST is cached max-age=30 (pmus_lag.md).

Pre-registered decision rule (written before searching): recommend the cheapest source that (a) carries Pinnacle or an
exchange-grade sharp price in-play for >=2 of NFL/NBA/NHL/soccer, (b) push or >=1 req/s polling, (c) no US-barred
account needed by us, (d) published price. Kill: if none meets (a)-(d) under $500/mo, report "no affordable feed".

## Candidates (all latency figures are VENDOR CLAIMS; none independently measured)
| Source | Access from CA | In-play | Latency | Cost | Notes / source |
|---|---|---|---|---|---|
| Pinnacle official API | NO: public API closed 2025-07-23; bespoke access only (api@pinnacle.com); Pinnacle bars US customers since 2007 (UIGEA) | yes | push | n/a | github.com/pinnacleapi/pinnacleapi-documentation ; sportsapis.dev/pinnacle-api ; en.wikipedia.com/wiki/Pinnacle_Sports |
| Betfair Exchange API | NO: US on restricted list, API key needs funded verified account | yes | stream | n/a | britishracecourses.org guide quoting Betfair T&C; betfairsquare.pages.dev (secondary, unofficial) |
| pinnapi (Pinnacle reseller) | data-only, no betting account | yes, separate live/prematch streams | claims 15-40 ms to SSE; SSE = drop ALERTS (5% default threshold), full frames need WS add-on | free 100 REST/day; REST $99/mo 10 req/s; Edge $149; Scale $229; raw WS +$99 | sportsapis.dev/apis/pinnapi (review 2026-11-02); pypi.org/project/pinnapi ; sourcing method undisclosed |
| SharpAPI | data-only | paid plans "real-time and live/in-play" | not published | Pinnacle needs Sharp $399/mo; streaming +$149; 3-day free trial (card) | sharpapi.io/pricing ; sharpapi.io/sportsbooks/pinnacle-odds-api |
| The Odds API | data-only | yes | REST only; competitor table says 30-60 s | free 500 credits; ~$30-249/mo | the-odds-api.com bookmaker page: Pinnacle "Odds are from public website which may incur a delay" -> likely same staleness as our guest feed |
| OddsPapi | data-only | yes (WS) | targets ~1 s for Pinnacle live, no measured value published | free tier; paid not stated | oddspapi.io/en/coverage/latency |
| OpticOdds / OddsJam API | B2B sales | yes | Optic claims <800-900 ms | not published; competitor/community reports ~$5k/mo (Optic), $999-5k+/mo (OddsJam) | opticodds.com/pricing ; sportsgameodds.com comparisons (competitor, unverified) |
| SportsDataIO / Sportradar | enterprise | yes | ? | not found | [unsourced] |
| Circa | no official API (aggregators resell) | Circa in-play limited | ? | via SharpAPI/OddsPapi/SGO | sportsapis.dev/circa-api ; oddspapi.io/blog/?p=3108 |
| Novig (CFTC exchange) | yes, all states except NV/AZ/MI (per review) | yes | own WS; free keys for account holders; data may not be redistributed | $0 data; taker fee 0.06*P*(1-P) | sportsapis.dev/novig-api (2026-10-28); docs.novig.com |
| ProphetX | CA reportedly available | yes | Trading API keys for verified accounts since 2026-10-07 (per review) | $0 data | sportsapis.dev/prophetx-api ; actionnetwork.com review |
| Polymarket US | yes | yes | WS needs PM US API key; REST cached 30 s | $0 | notes/pmus_lag.md (already tested: dead) |
| BetOnline / Bookmaker.eu | offshore, unlicensed in CA; no public API | - | - | - | not recommended [unsourced on legality detail] |

## Recommendation
pinnapi is the only source meeting (a)-(d) at <$200/mo. Novig/ProphetX are free and legal but are US exchanges that
plausibly FOLLOW Kalshi (same lesson as PM US), so they are not an independent sharp reference.
Plan: (1) pinnapi free tier (100 REST/day) during one live NBA game: confirm in-play prices change on a seconds scale and
compare the reported timestamps to wall clock; (2) optionally SharpAPI 3-day Sharp trial ($0 if cancelled) as a second,
independent Pinnacle pipeline -- if both agree tick-for-tick, freshness is real; (3) 2-week test on pinnapi REST
($99, 10 req/s ~ 1 poll/s for ~8-10 concurrent games) or REST+WS ($198) for push frames. Expected 2-week cost $99-198
(one monthly bill; no proration assumed [unsourced]). Needs user approval (spend).

## Learnings
### Pinnacle's public API is closed and Pinnacle bars US customers, so a CA user can only get Pinnacle via resellers
Closed 2025-07-23 (github.com/pinnacleapi/pinnacleapi-documentation); US excluded since 2007 UIGEA (Wikipedia). Any
"real-time Pinnacle" for us is a third-party pipeline with undisclosed sourcing; verify freshness before trusting it.
### The Odds API's Pinnacle comes from the public website and is REST-only, so it likely repeats our stale-guest-feed failure
the-odds-api.com bookmaker list note: "Odds are from public website which may incur a delay". Do not buy it for lag tests.
### Cheapest real-time Pinnacle candidate is pinnapi at $99/mo REST (+$99 WS), with a free 100 req/day tier to validate first
sportsapis.dev/apis/pinnapi (2026-11-02). Latency 15-40 ms is vendor-claimed only. SSE streams are drop ALERTS, not full books.
### SharpAPI needs the $399 Sharp plan for Pinnacle but has a 3-day free trial, usable as a second pipeline cross-check
sharpapi.io/pricing; sharpapi.io/sportsbooks/pinnacle-odds-api. Streaming is +$149/mo extra.
### Betfair Exchange and Pinnacle accounts are not legally available to US residents; Novig/ProphetX/PM US are, but are followers not sharp references
Betfair T&C (via britishracecourses.org); Novig keys free for account holders (sportsapis.dev/novig-api). PM US lag test already dead (notes/pmus_lag.md).

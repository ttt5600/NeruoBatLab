# infolayer -- information-arbitrage connection layer (design, 2026-11-07)

Goal (user 11-07): a general layer that connects public information sources (official tallies, running counts,
social/online feeds, published statistics) to Kalshi/Polymarket markets and flags gaps fast. Generalises the
election NYT-needle bot (notes/election_needle_2026.md) so each new source/market is a small plug-in.

## Pipeline
    SOURCE adapters  ->  Observation(quantity, value, observed_at, source_updated_at, fresh?)
    FAMILY plug-in   ->  maps markets <-> observations, fair probability per market (from the resolution rule)
    GAP check        ->  fair vs executable price after Kalshi fee; FRESHNESS gate; 2nd-venue cross-check
    PAPER ledger     ->  every flag to ~/.claude/quant-kalshi/data/infolayer/<family>.jsonl (timestamps, book, fair)
    GRADER           ->  settle flags vs outcomes; price drift after flag (CLV); feeds edgelab evidence
    EXECUTOR         ->  NOT BUILT. Only for families that pass edgelab (GOAL.md rule); caps from the user.

## Rules baked in (from the election lessons)
1. A flag needs a FRESH source (family defines freshness; stale -> no flag, logged as stale).
2. The source must be the resolution path or a near-deterministic precursor -- not a model opinion.
3. If a 2nd venue exists and also disagrees with the source, mark `venue_agrees_with_market` (the source is the
   likely stale side) -- recorded, used by the grader to test the rule.
4. EV after the exact Kalshi taker fee (0.07*p*(1-p) per contract) and at the executable side of the book.
5. No orders anywhere in this package. Paper first; money only via edgelab pass + user caps.
6. Deploy from outside ~/Desktop (TCC) and poll only the series a family needs (429s).

## Where the edge can and cannot be (honest expectations)
- Speed races on big, watched releases (CPI, jobs, scores in big leagues) are lost to HFT/market makers: skip.
- Edges live in THIN markets keyed to slow-moving or awkward-to-read public data: canvass counts, running post/mention
  counts, weekly stats pages, niche schedules. Capacity is small per market; breadth (many families) is the lever.
- Social sources (X/Instagram): only via official APIs / public pages without bypassing logins or bot walls. X API is
  paid; Polymarket's xtracker already counts posts for its own markets (free JSON).

## Families (status)
| family | markets | source | status |
|---|---|---|---|
| canvass_margin | Kalshi KXMIDTERMMOV (race margin >= K pts) | AP counts in NYT results JSON (counts only; model block frozen 11-04) | paper logger built 11-07 |
| next (census stubs) | uncalled winners, House popvote/turnout, xtracker post counts (Q014 2nd counter), KXTOKENUSE, KXALBUMEQUIV | notes/info_gap_census_2026-11-07.md | proposed |

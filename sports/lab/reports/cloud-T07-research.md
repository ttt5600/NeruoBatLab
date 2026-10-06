# T07 — Closing line value, open-vs-close efficiency, how/when lines move, steam and reverse line movement

## What I did

Searched the topic in four pieces — (1) CLV as a measure of skill, (2) opening vs
closing line efficiency / the mechanics of how and when lines move, (3) steam
moves, (4) reverse line movement — and for each, deliberately searched for the
refutation or decay side as well as the original claim, per standing orders.
Checked `lab/registry.py list` first (36 existing entries, S001-S036 in the
main registry) and confirmed none of them cover CLV, line-movement mechanics,
steam, or RLM directly, so nothing here duplicates prior work. Pulled
`lab/run.py schema nfl_spread`, `nfl_total`, `nfl_moneyline`, and `mlb_k_props`
before writing `testable_with` for any entry. Added 9 entries (S001-S009 in
`lab/registry/incoming/T07.jsonl`).

Several numbers surfaced by the search tool's own summaries did not survive a
direct fetch of the underlying page and were dropped rather than used: an
"Oswald (2022)" NFL-market-efficiency paper with a specific 64.7% underdog
stat turned out, once I fetched the actual thesis page, to be authored by
Elijah Costa (2025), not "Oswald" at all, and the 64.7% figure never appeared
on the page I could actually read — I only kept what Costa's own abstract page
stated (sharp-action-driven predictability of in-week odds movement, and a
separate home-field-advantage strategy I did not register because it is not a
line-movement claim). Similarly, a vague "CLV trackers get 2-3x the ROI of
win-rate trackers" line from one search summary had no traceable source and
was discarded. Both are flagged here, not silently fixed, per the no-fabrication
rule.

## The central, structural finding

Every entry below ended up with `testable_with: []`. That is not an accident
of laziness — it is the single most important finding of this topic for this
project. `nfl_spread`, `nfl_total`, and `nfl_moneyline` each store exactly one
closing-ish price per game (confirmed via `lab/run.py schema`); there is no
opening-line column, no intra-week odds history, and no public-bet-percentage
column anywhere. `mlb_k_props` similarly stores one strikeout-prop line taken
~4h before first pitch. CLV, open-vs-close efficiency, steam, and RLM are all
*defined* as a comparison between at least two time-stamped prices (or a price
and a public-money split) for the same game. With single-snapshot data, none
of them can be computed, let alone tested, no matter how good the underlying
research is. This is a data-infrastructure gap, not a research dead end.

## What I found, grouped by sub-topic

**CLV as skill.** The practitioner consensus (Pinnacle's own educational
content, well-known tipster Joseph Buchdahl) is that beating the closing line
by X% should produce roughly X% ROI over turnover long-run, and Buchdahl's own
~20,000-bet soccer system came in close (3.4% actual vs. 4.0% expected,
S001). But this is contradicted in degree, not direction, by two more
rigorous, independent checks I found: Karl Whelan's decile analysis of 3,670
NBA moneyline games found 3 of 5 *positive*-CLV deciles were unprofitable on
average and only the top decile (10%) cleared real money (S002) — CLV measures
relative timing, not whether you beat the actual vig-inclusive margin. And Jay
Pinho's multi-season Pinnacle backtest found the relationship itself has
decayed recently in soccer (4.3%→3.5% actual-vs-expected 2019/20+, falling to
4.2%→1.9% since 2023/24, S003) — a decay story structurally identical to the
HuBERT-plateau pattern this lab has seen before: a clean theoretical story
(closing line = best estimate) that only partially survives contact with
recent, real data.

**How and when lines move.** Two peer-reviewed results point the same
direction: closing lines are *more* informative than opening lines on
average (Miller & Rapach 2013, 1972 NFL bookmaking data, S008) but line
movement itself is not a clean monotonic improvement — Jay Simon's 2024
Management Science paper on 3,681 MLB games found real-time line sequences
are significantly negatively autocorrelated (overreaction then partial
reversal) and get *worse*, not better, in the 90 minutes before weekend day
games despite more information being available (S007). A 2025 Claremont
senior thesis (Costa, high-frequency NFL moneyline data 2020-2024) adds a
mechanism: in-week movement direction is predictable and driven by sharp
action, not injury news (S004). An older arXiv working paper (Szalkowski &
Nelson 2012) found the raw open-to-close difference alone retrodicts
divisional winners at ≥75% and flags a profitable home-underdog rule (S009) —
flagged in its own skeptic_note as likely overlapping the already-registered
Borghesi late-season home-underdog effect (S003 in the *main* registry).

**Steam and reverse line movement.** RLM is the one sub-topic with a direct
practitioner-vs-academic conflict in the *same* kind of claim: a Sports
Insights/BetLabs NFL backtest since 2003 shows a modestly profitable RLM rule
(290-249 ATS, 4.8% ROI, S005), while Francisco & Moore (2019, Journal of
Economics and Finance) tested RLM-based strategies on NCAAF totals
(2005-2016) and found them *not* profitable. I registered this as one entry
with both results in `claimed_edge`, since it is the same strategy type
tested on different sport/markets with opposite conclusions — genuinely
contested, not refuted. Steam-chasing (S006) has no empirical study behind it
at all that I could find, only a consistent practitioner argument that by the
time a steam move is visible on a tracking service, the value that caused it
already belongs to whoever moved first — graded `claim`, the weakest entry
here, explicitly because nobody has measured it.

## The 2-3 most promising entries, and why

None are testable *now* — that is the honest headline for this topic — but if
forced to rank which would be cheapest to unlock:

1. **S004 (NFL moneyline, sharp-action-driven weekly movement)** — only needs
   a daily (not even hourly) NFL moneyline odds-history feed, which several
   free/cheap odds-API services provide going forward; of everything here it
   has the clearest mechanism and the most recent, most relevant sample
   (2020-2024).
2. **S005 (reverse line movement)** — needs public bet-percentage data, which
   several vendors (Sports Insights/BetLabs, Covers.com) already publish
   going forward; this is the only entry where a *cheap, specific* new data
   source (not a full historical rebuild) would make it testable, and it is
   also the entry with the sharpest head-to-head conflict in the literature
   worth resolving independently.
3. **S007 (MLB autocorrelation/overreaction)** — the dataset gap is the
   biggest (needs a full intra-week odds-history sequence per game across
   multiple books, not just the single ~4h-out snapshot `mlb_k_props` has),
   but it is the most rigorous, most recent (2024, Management Science) peer-
   reviewed result in the whole set, and the autocorrelation structure it
   describes is a concrete, falsifiable statistical signature, not just a
   win-rate claim.

## What data would unlock this topic

One thing, essentially: a timestamped odds-history feed (open through close,
ideally hourly or better, across 2+ books) for whichever of NFL
spread/total/moneyline or MLB moneyline this project adds next, plus — for
RLM specifically — a parallel public bet-percentage/handle feed for the same
games. Every entry in this topic reduces to "compare price A to price B for
the same game" or "compare price to the public-money split"; a single
closing snapshot, no matter how accurate, cannot answer any of it.

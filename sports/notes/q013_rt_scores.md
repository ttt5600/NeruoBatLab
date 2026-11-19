# Q013 Rotten Tomatoes / Metacritic score brackets vs the live score: FORWARD paper logger (pre-registration)

Filed 2026-10-29 (PT) before the first logger tick and before any price was joined to any score. Paper only, no orders.
Code: `scripts/q013_logger.py` (tick / report / map). Tests: `tests/test_q013_q014_loggers.py`.
Agent: launchd `com.jonathanwang.quant-q013`, every 900 s, code copy in `~/.claude/quant-kalshi`
(launchd cannot read ~/Desktop). Data: `~/.claude/quant-kalshi/data/q013/` (gitignored).

## 1. Why forward-only
Kalshi has 85 settled KXRT and 19 settled KXMC events (census 2026-10-29), but there is no free point-in-time history of
the review count and score. RT and Metacritic show only the current state. So the reference can only be timestamped
by our own clock, going forward.

## 2. Question
At a fixed time before close, given the live score and the review count so far, does a beta-binomial (RT) or
normal-mean (MC) forecast of the final score price the brackets better than the Kalshi book, by enough to clear a
5c edge after the taker fee?

Mechanism: the settlement variable is mostly observed. Each new review moves the score by about 1/N, and retail may
price the headline number rather than that arithmetic. The other side includes known scrapers and LIP market makers,
so the prior is that there is no edge.

## 3. Data sources (checked 2026-10-29, all public, polite)
- **Kalshi**: public `/markets?series_ticker=KXRT|KXMC&status=open`, with backoff on 429. The shared IP does get 429s.
- **RT**: the film page `rottentomatoes.com/m/<slug>` embeds `<script id="media-scorecard-json">`, whose
  `criticsScore` holds {score, reviewCount, likedCount, notLikedCount}. Slug resolution uses RT's own `/search?search=<title>`
  (a movie row whose normalised title equals the Kalshi title and whose release-year is the close year or the year before).
  `data/q013/overrides.json` can hand-pin a slug. All 18 open KXRT events resolved to 2026 pages on 2026-10-29.
  The settlement value is the Tomatometer, i.e. the all-critics score; Top Critics are not used.
- **Metacritic**: the game page's ld+json `aggregateRating` {ratingValue, reviewCount} is the default-platform Metascore
  that the rules name. The per-review scores come from `backend.metacritic.com/reviews/metacritic/critic/games/<slug>/web`,
  the endpoint the site itself calls. It returns about 10 items per page whatever `limit` says, so the code pages by the
  returned count.
- Politeness: a score page is refetched every 15 min only when close is <= 8 days away, otherwise hourly. It is always
  refetched inside a decision window. There is a 1 s sleep between pages.

## 4. Frozen rule
- Decision times: tau = close_time - 24 h (`T24`) and close_time - 6 h (`T6`). The pick is made at the first tick in
  [tau, tau + 60 min). If the agent misses that window, there is no pick. There is one row per (decision, market), logged
  even when no bet is made (with the reason).
- Eligibility: review count n >= 10, and the market's strike is parseable. Rules: "Above X" is strict, and "between" is
  inclusive. The quote must be two-sided, with 0 < yes_bid < yes_ask < 1.
- Remaining reviews: m = max(2, ceil(f x n)), with f = 0.08 at T24 and 0.04 at T6. **This is a guess, frozen as is.** The
  logger records the review counts, so whether m is realistic can be checked descriptively afterwards. Any refit is a NEW trial.
- RT model: K fresh among the m remaining reviews ~ BetaBinomial(m, liked + 1, n - liked + 1). Final score =
  round-half-up(100 (liked + K) / (n + m)). Rounding was checked on 5 live films on 2026-10-29 and matched round-half-up
  every time (44/52 shows 85, where floor would give 84). The logger re-checks this on every fetch.
- MC model: final = current Metascore + delta, with delta ~ Normal(0, sd^2) and
  sd^2 = (m/(n+m))^2 s^2 (1/m + 1/n) + 1/12. Here s is the sd of the per-review scores from the review API, falling back
  to 10 if the API fails, and the 1/12 term is the displayed-integer rounding. The pmf is over integer scores 0-100.
- Bet: buy YES at the ask if p - ask - 0.07 ask(1-ask) >= 0.05. Otherwise buy NO at (1 - bid) if
  (1-p) - (1-bid) - 0.07 (1-bid) bid >= 0.05. One contract per (decision, market), with cost = price + fee.
- Settlement: the Kalshi `result` of each market.

## 5. Trials, statistics, bar
- Two trials, T24 and T6, Bonferroni one-sided p < 0.025 each. The ALL pool is descriptive.
- ROI = sum P&L / sum cost. The CI and the one-sided p come from an EVENT-clustered bootstrap (10,000 resamples,
  seed 1013). The observed ROI is quoted, not the bootstrap mean.
- Calibration: model p_yes vs outcome in 6 bins. Brier of the model vs Brier of the mid, on the same decision rows.
- CLV: entry price vs the last logged pre-close mid of the same market.
- PASS (per trial): >= 30 events with settled bets, ROI > 0, p < 0.025, and model Brier < mid Brier.
- Too-good guard: ROI > +30% or a win rate far above the model -> audit the rows first (score timestamp vs tau, quote time).

## 6. Kill
- After 30 events with settled bets on a trial: ROI <= 0, or model Brier >= mid Brier -> that trial is DEAD.
- If the RT rounding mismatch is > 2% of fetches, or the MC default-platform score is not the one Kalshi settles on,
  fix the semantics first and count no trial until then.
- If either source blocks us (401/403), the logger stops fetching it and says so. Blocks are never bypassed.

## 7. Power (stated before any data)
There are about 18 KXRT + 3 KXMC events open now. Close dates run to 2026-12-28, roughly 1-3 films a week. 30 events with
bets therefore needs months, and only if the rule fires at all. Strikes within an event are correlated, so the event
count is what binds. This logger can detect a LARGE mispricing only. A 2-3% edge is undetectable at this n.

## Ops 2026-11-18 (agent `ops_q013_fix`; user-approved paper-logger fix; no orders, no git)
**Bug.** RT renamed `criticsScore.score` -> `criticsScore.scorePercent` ("88%") at ~2026-11-03 22:41Z (last legacy row
ts 1793735314 = 11-03 19:48Z, IPL 87). `rt_score` stored score=None from then on, and the pick gate (`score is None` ->
skip) logged 170 KXRT decision rows with no model evaluation: CAT, EBE, GOD, HOWT, WILD x {T24, T6}, 15-19 strikes each
(deployed `data/q013/picks.jsonl`, rows with ts >= 1793745660, all reason "no score / n < N_MIN / unknown strike").
MC (KXMC) was unaffected.

**Fix (`scripts/q013_logger.py`, deployed to `~/.claude/quant-kalshi/scripts/`, backup `q013_logger.py.bak-2026-11-18-prescorepct`).**
- `parse_rt_page`: score = legacy `score` -> `scorePercent` (strip %) -> round-half-up(100 liked/(liked+not_liked)).
  New field `score_src` says which. The report's rounding check now skips `score_src == "calc"` rows (trivially equal).
  Semantics unchanged: the frozen rule always meant the displayed Tomatometer; the calc fallback matched it 1,377/1,377.
- Resolver: RT `/search` removed (robots.txt disallows it for `User-agent: *`, re-fetched 11-18). New `rt_resolve`
  tries slug candidates `/m/<title_words>[_<year>|_<year-1>]` (hyphen kept / dropped) and accepts a page only if its
  ld+json Movie name (minus a trailing "(YYYY)") equals the Kalshi title and its dateCreated year is close-year or
  year-1. 2 s between RT requests (score fetches too). `overrides.json` still wins. Live check 11-18: HUN, IPL, FOC,
  FUR, HEX (-> /m/hexed_2026 after stripping "(2026)"), ZER, VIO, DUNE all resolved to the same paths the old search
  found (state.json). No login, no /napi, 401/403 still never retried.
- Tests: `tests/test_q013_q014_loggers.py` +6 (15 pass): captured page fixture
  `tests/fixtures/rt_page_scorepercent_2026-11-18.html` (live /m/i_play_rocky, trimmed), legacy-shape fixture
  `rt_page_score_legacy.html` (same page with criticsScore rewritten to the 11-03 logged values; NOT a raw capture --
  no legacy page was saved), calc fallback, resolver never GETs /search, slug candidates, health streak.
  Fixture builder: `results/ops_q013_fix/make_fixtures.py`.

**Health check (`scripts/picks_health.py`, deployed; wired into `fund_brief.py` section 6, daily 07:30, backup
`fund_brief.py.bak-2026-11-18-prehealth`).** Eligible day = UTC day with >= 1 decision row; ALERT if the 3 most recent
eligible days all have 0 rows with p_yes (q013, q014). Plus a q013 field-null check: RT fetches in the last 24 h with
n >= 10 and score None. Alerts go to the brief, `~/.claude/quant-kalshi/ALERTS_picks_health.txt` and a macOS
notification. Run on deployed data before the fix tick: q013 ALERT (11-09 0/50, 11-15 0/35, 11-16 0/35) and
184/184 null-score fetches; q014 ok (26/26 each day). The q013 streak alert stays on until the next eligible day
(HUN/IPL T24 = 2026-11-22 15:00Z) produces evaluated rows -- expected, not a new failure.

**Proposal for lead (not done).** The 170 skipped rows can be reconstructed point-in-time: at each decision tick
scores.jsonl holds liked/not_liked/n (score_calc) from the same ts and the pick row holds the bid/ask. A backfill
(`decision` rows tagged `backfill: true`, frozen model unchanged) would recover 5 events for the T24/T6 trials. It
touches the trial's data, so it is the lead's / user's call. `~/.claude/quant-sports/scripts/q013_logger.py` is a
stale identical copy of the old code (no launchd job uses it); left untouched.

**Verified (deployed, launchd tick ts 1795059524 = 2026-11-19 03:38Z).** `data/q013/scores.jsonl`: KXRT-HUN score 74
(score_src scorePercent, n 101, score_calc 74), KXRT-IPL 88 (n 157, calc 88). HOWT/WILD are closed (no open markets),
so HUN/IPL are the current films. No errors in q013_logger.log/.err. Dry run of the frozen model on those rows
(not written): P(>70) at T24 = 0.997 (HUN), 1.000 (IPL), i.e. the gate now evaluates. First real picks: 2026-11-22
15:00Z (T24 for HUN/IPL). Field-null check changed to judge each film's LATEST fetch so it clears on the first good
tick: now 0/2 films null (ok); the zero-pick streak alert stays until 11-22 as expected.

## Learnings
### RT renamed criticsScore.score to scorePercent on 2026-11-03 and Q013 skipped every RT pick for 15 days
170 KXRT decision rows on 5 films (CAT, EBE, GOD, HOWT, WILD) logged "no score" (deployed data/q013/picks.jsonl).
A logger that logs None instead of failing loudly hides a dead trial; parse with a fallback chain and record the source.
### A picks-health alarm catches silent skips within 3 eligible days
scripts/picks_health.py: 0 model-evaluated rows on 3 consecutive eligible days, or a film's latest fetch losing its
score field. On the pre-fix data it fired on q013 (11-09 0/50, 11-15 0/35, 11-16 0/35) and stayed quiet on q014.
### RT film slugs can be resolved without /search, which robots.txt forbids
Guess /m/<words>[_<year>] and verify the page's ld+json Movie name + dateCreated year; RT appends "(YYYY)" to the name
when it disambiguates (Hexed (2026)). Matched the old search mapping on 8/8 open films (2026-11-18).

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

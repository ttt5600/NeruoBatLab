# rt_review_lag -- Rotten Tomatoes pre-aggregation review lead (forum scout idea #4, Q013 extension)

Agent topic `rt_review_lag`, started 2026-11-19 ~02:00Z. Read-only. No orders, no git. Scratch: `sports/results/rt_review_lag/`.

## 0. Prior work (why Q013 is where it is)
Q013 (`notes/q013_rt_scores.md`) is a forward paper logger (launchd `com.jonathanwang.quant-q013`, 15 min, live since
2026-10-30) that models the residual drift of the LIVE Tomatometer at T-24h / T-6h. It went forward-only because there is
no free point-in-time history of RT review counts. Its trial needs 30 settled-bet events (months). It does NOT use any
information that RT has not yet ingested. This agent's idea is that extra information: reviews already published at the
outlets but not yet counted by RT.

As a by-product Q013 has logged RT {score, n, liked} and the full KXRT book every 15 min since 10-30
(`~/.claude/quant-kalshi/data/q013/{scores,changes,snapshots}.jsonl`). That is a point-in-time joint history that did not
exist when Q013 was designed, and it is what this agent tests on.

## 1. Question
Does information published before RT ingests it (outlet reviews; embargo lift) predict the KXRT settlement better than
the Kalshi book, inside the window before close, by enough to clear ask + fee?

Decomposed into two necessary conditions. Either failing kills the lead:
- **C1 ingestion lag**: RT counts a review a meaningful time after the outlet publishes it.
- **C2 market does not already know**: Kalshi prices do not move ahead of RT's counted score (if they do, someone is
  already trading the outlet reviews / embargo wave and the pre-aggregation information is priced).

## 2. Pre-registration (written before any outcome below was computed)
Dev window = Q013 logs from 2026-10-30 to 2026-11-11 23:59Z. **Sealed** = Q013 logs from 2026-11-12 onward; not opened
by this agent (it is also the Q013 trial's own forward data).

**M1 Kalshi-vs-RT lead-lag (tests C2).** Event = an RT tick change for a KXRT film where |delta score| >= 2 points or
delta n >= 5 between consecutive Q013 fetches, with the change first seen at tick t and previous fetch at t_prev
(gap <= 20 min; otherwise the event is dropped). For each event take the KXRT strike whose floor is nearest the new
score; Kalshi mid m(.) from snapshots (nearest snapshot at or before the time). Direction s = sign of the RT score change
(score up -> YES on "above" strikes up). Signed moves:
  pre  = s * (m(t_prev) - m(t_prev - 60 min))     # before RT showed the change
  post = s * (m(t + 60 min) - m(t))               # after we saw it
  tot  = s * (m(t + 60 min) - m(t_prev - 60 min))
Statistic: film-level pre-share = sum(pre)/sum(tot) over that film's events with tot > 0; reported as the median
over films and a film-clustered bootstrap CI of the pooled ratio sum(pre)/sum(tot) (10,000 resamples, seed 4).
- C2 FAIL (market already knows) if pooled pre-share >= 0.5 with CI lower bound > 0.25.
- C2 OPEN (market follows RT) if pooled pre-share < 0.25 AND post-share sum(post)/sum(tot) CI excludes 0.
- otherwise "not distinguishable".
Too-good guard: if post-share > 0.5 (Kalshi lags RT by > 15 min), look at the rows (snapshot time vs score fetch time)
before believing it.

**M2 ingestion lag (tests C1).** Per review: outlet publication time (the article's own `article:published_time` /
ld+json `datePublished`) vs the time RT shows it (RT review `creationDate`, if it carries time-of-day; else Q013's tick
on which n incremented). Success = median lag >= 30 min on >= 40 reviews from >= 5 films; kill = median lag < 10 min
(scout #4 thresholds). If neither the RT feed nor outlet pages give time-of-day, M2 is untestable on history -> forward.

**M3 (only if C1 and C2 both pass)** = the scout #4 trial: model with pre-ingestion reviews vs Kalshi mid, Brier on >= 40
films, then ROI at ask + fee, film-clustered. Not run unless M1/M2 pass.

## 3. Politeness / ToS
RT robots.txt (fetched 2026-11-19): `User-agent: *` disallows `/search`, `/m/*/pictures`, `/critics/self-submission/`,
`/user/account/*`; `/napi/*` is disallowed only for Google-Extended. Polite: >= 2 s between RT requests, no logins.

## 4. Log
### 2026-11-19 step 1 -- KXRT rules census (`scripts/rt_review_lag.py census` -> results/rt_review_lag/kxrt_events.json)
- Series KXRT, source Rotten Tomatoes, fee quadratic x1. Rule text (KXRT-HOWT): "If <film> has a Tomatometer score of
  above X on <Mon> at 10:00 AM ET" -- strictly above; measured **the Monday after wide release, 10:00 AM ET**; close =
  that time; settlement timer ~1 h. No review-count minimum in the text. Prohibitions include "Rotten Tomatoes reviewers".
- 105 events returned (settled+open), 39 with nested markets (27 finalized 2026-09-14..11-16, 12 active to 12-28);
  66 older events come back without markets. Markets open weeks before release (e.g. KXRT-CAT open 10-08, close 11-09),
  so **the review embargo lifts while the market is trading.** 10-23 strikes per film (5-pt grid, some 2-3-pt fills).
- Volume (volume_fp, contracts) per film: median 825,866, total 38.4M over the 39 films (range 56k ZER .. 3.1M DIG).

### step 2 -- M1 lead-lag on Q013 logs (dev <= 11-11)  (`rt_review_lag.py leadlag` -> leadlag_pre.json)
- **Q013 DATA BUG FOUND**: from 2026-11-03 22:41Z RT's `media-scorecard-json.criticsScore` no longer has `score`; it
  now has `scorePercent: "88%"` (checked live on /m/i_play_rocky 11-19). Q013 stores score=None for every film since,
  and its pick gate (`sc.get("score") is None` -> skip) has logged **zero eligible RT picks since 11-03** (CAT, GOD,
  WILD, EBE, HOWT all "no score / n < N_MIN"). liked/notLiked are still there: score_calc matched score on 1,377/1,377
  earlier rows, so the fix is `score = scorePercent` or score_calc. Not fixed by me (deployed bot copy; lead/user).
- Also: q013_logger resolves slugs via RT `/search`, which RT robots.txt disallows for `User-agent: *`. Use overrides /
  the sitemap instead.
- M1 therefore uses score_calc where score is None (forced by the bug, not by outcomes).
- Result: 22 RT-change events on 4 films (8 dropped for fetch gap > 20 min, 4 no mid). Pooled signed Kalshi move
  pre / concurrent / post / total = 0.235 / 1.31 / 0.44 / 1.985 -> **pre-share 0.12, film-bootstrap CI [-0.24, 0.42];
  post-share 0.22 CI [-0.19, 0.41]; concurrent share 0.66.** Pre-reg verdict: **not distinguishable** (pre < 0.25
  but post CI crosses 0). At 15-min resolution two-thirds of the Kalshi move lands inside the same tick as the RT
  change, so the tick cannot tell who leads. 4 films is far too few anyway.
- Minute-level tape at the two embargoes in dev (`rt_review_lag.py trades`, `scripts/rt_tape.py`):
  - HOWT, embargo 14:00Z 11-11: RT n = 0 at the 13:58 fetch, 6 at 14:15, 14 at 14:47, 22 at 16:08, 33 at 22:01.
    Kalshi: 10 contracts 13:45-13:59, then 1,574 at 14:01, 10,107 at 14:04, 48,264 at 14:14. >65 fell 0.14 -> 0.07
    by 14:11. **The market reprices within 1-4 minutes of the embargo, before our RT fetch showed any review.**
  - CAT, embargo 17:00Z 11-03 (RT fetch gap 10:02 -> 17:25, so RT timing unknown): >65 0.69 at 16:57 -> 0.19 at 17:08;
    23,386 contracts 17:00-17:09.
  - 10 of the 16 MC-listed CAT reviews carry outlet timestamps 16:55-17:01Z; RT had 11 reviews at 17:25, 22 next
    morning, 57 by 11-09. HOWT: 9 of 30 MC-listed reviews published 13:55-14:26Z. So **C1 holds at count level**:
    a big stock of reviews is public within a minute of the embargo while RT counts them over hours to days.
  - But C2: someone trades the embargo minute (the venue's own blog tells traders to read outlet reviews, scout C12).
    Speed is taken. What could be left is classification quality in the hours after the embargo.
- M2 per-review ingestion lag is **untestable on history**: RT review lists are client-rendered from
  `/napi/rtcf/v1/movies/<emsId>/reviews`, which returns **401** to an anonymous GET (recorded, not worked around), and
  RT shows dates only. Metacritic's critic feed (robots allows; date only) + each outlet page's own
  `article:published_time` give outlet times (`scripts/rt_outlet_times.py`; NYT 403, The Times/WSJ robots-disallow,
  skipped).

### M4 pre-registration (written 2026-11-19 before computing any nowcast or Brier; only HOWT/CAT tapes seen so far)
Question: 30 minutes after the review embargo, does a nowcast built ONLY from outlet reviews already published
(Metacritic-listed reviews with an outlet timestamp <= t*) price the KXRT strikes better than the Kalshi tape?
This is the cheap historical version of scout #4 (RT's own fresh/rotten labels per review are not reachable).
- Films: dev = finalized KXRT events with markets, close <= 2026-11-16, EXCEPT KXRT-EBE (kept unseen); older events
  only if `/markets?event_ticker=` returns their markets. Forward films = later trial.
- MC slug = slugify(title) or slugify(title)+"-2026"; accepted only if the feed's reviewed title matches and review
  dates lie in [close - 120 d, close]. Otherwise the film is dropped (logged).
- Embargo E = start of the first 10-min window containing >= 3 outlet timestamps (time-of-day present). No such
  window -> drop. t* = E + 30 min.
- Nowcast: k = MC reviews with outlet time <= t*, mean MC score s. Film-level map fitted leave-one-film-out on the dev
  films: logit(RT_final/100) = a + b (Metascore_final - 50)/10 (RT_final = midpoint of the settled Kalshi bracket;
  Metascore_final = feed criticScoreSummary.score). Predictive logit ~ N(a + b (s-50)/10, sigma_fit^2 +
  b^2 var_s/(100 k)); P(RT > strike) by 20,000 draws, displayed score = round(100 * inv_logit).
- Kalshi price per strike = VWAP of that strike's trades in [t* - 10, t* + 10] min (no trade -> strike dropped).
  Outcome = market `result`. Strikes with VWAP outside [0.03, 0.97] dropped (both priced it).
- Statistic: per film, mean over strikes of (nowcast - outcome)^2 - (vwap - outcome)^2; mean over films, film
  bootstrap 95% CI (10,000, seed 4).
- PASS (lead survives on history): >= 20 films and CI upper < 0. KILL: point estimate >= 0 (the outlet-score nowcast
  does not beat the tape 30 min after the embargo). Otherwise not distinguishable.
- Too-good guard: Brier gain > 0.05 per strike -> inspect rows (outlet timestamps vs E; VWAP window; bracket).
- Descriptive only, not a test: same with t* = E + 3 h.

### step 3 -- M4 result (`scripts/rt_m4.py collect|score` -> results/rt_review_lag/m4_films.json, m4_rows_t30.json)
- 92 KXRT events collected (live + `/historical/markets`; close 2026-01-26..11-16; EBE excluded). MC slug unresolved
  on 39; 45 without an embargo cluster in all -> **47 films with E, 46 scored, 232 strikes** at t* = E + 30 min.
- **Pre-registered verdict: KILL.** Film-mean Brier diff (nowcast - tape) **+0.1008, 95% CI [+0.0610, +0.1415]**; the
  nowcast was better on 11/46 films. Pooled Brier nowcast 0.2362 vs tape 0.1362. E + 3 h (descriptive): +0.1055
  [+0.0519, +0.1614], 12/45.
- Why (rows inspected): the early (embargo-wave) reviews average **+5.5 MC points above the final Metascore**, and the
  pre-registered map was fitted on the final Metascore, so the nowcast overshoots (its 0.7-0.9 bin hit 48%; the tape's
  0.7-0.9 bin hit 77%). corr(early mean, final Metascore) 0.971; corr(final Metascore, RT final) 0.912; median 17
  outlet reviews by t*.
- POST-HOC (descriptive, not a trial): map fitted directly on the early mean (LOFO) -> +0.0261 [-0.0019, +0.0556],
  19/46 better, pooled 0.159 vs 0.136. Even repaired, the outlet-score nowcast does not beat the tape.
- Registered in edgelab: `rt1_outlet_nowcast_embargo` (family kalshi_media, 1 trial, bar p < .05), retired dead.
  `edgelab status` notes the dataset has no vault entry (scored outside edgelab; a Brier gate, not bets).

## 5. Verdict and forward logger decision
- C1 (a count-level stock of published-but-uncounted reviews right after the embargo): holds.
- C2 fails in practice: the tape reprices within 1-4 minutes of the embargo hour, and 30 min later it beats an
  outlet-review nowcast by 0.10 Brier on 46 films. Traders already read the reviews (the venue's own RT kit says to).
- **Lead: DEAD on history for the score-based version.** The one untested variant is an own text classifier that
  predicts RT's fresh/rotten per review; given the tape's 0.136 Brier at E+30m and the minute-level reaction, prior is low.
- Forward logger `scripts/rt_lag_logger.py` (read-only: RT page + MC movie feed + outlet published_time + Kalshi book,
  5-min ticks while hot; never calls RT /search; pauses a site 24 h on 401/403) is written, and one sandbox tick ran
  clean (12 events, 0 errors; data in results/rt_review_lag/fwd_test/). **Not installed under launchd**: the idea was
  testable on history and failed its pre-registered gate, so a forward run would only serve the low-prior classifier
  variant. Install only if the lead wants that variant (needs an install script + qs_job entry).
- **Q013 FIX NEEDED (lead/user)**: q013_logger rt_score must read `criticsScore.scorePercent` (or use score_calc);
  since 11-03 22:41Z every RT pick is skipped. Also stop calling RT /search (robots-disallowed for all agents).
- Not yet critic-reviewed (kill verdict; lead may send it to `critic`).

## Learnings
### KXRT settles on the Tomatometer the Monday after wide release at 10:00 ET, and the embargo lifts while it trades
Rule text from /series/KXRT + markets (results/rt_review_lag/kxrt_events.json); median film volume 826k contracts.
Embargo clusters start at :55 because a few outlets post about 5 minutes early.
### The Kalshi RT book reprices within 1-4 minutes of a review embargo
HOWT 11-11: 10 contracts 13:45-13:59Z, then 1,574 at 14:01 and 10,107 at 14:04, before our 13:58 RT fetch showed any
review (scripts/rt_tape.py). Speed on embargo reviews is already taken.
### An outlet-review nowcast 30 min after the embargo loses to the KXRT tape
46 films: Brier diff +0.1008 [+0.061, +0.142] (pre-reg kill); repaired post-hoc map +0.026 [-0.002, +0.056]
(scripts/rt_m4.py). Embargo-wave reviews run +5.5 Metacritic points above the final Metascore, so a map fitted on final
scores overshoots: calibrate a nowcast on the same-time sample it will see live.
### RT changed its scorecard JSON on 2026-11-03: criticsScore.score became scorePercent
This silently zeroed Q013's RT picks for 2+ weeks (picks.jsonl reasons). Loggers should alarm when a field goes null
on rows where it used to be present, not just log None.
### RT review lists are 401 to anonymous clients; Metacritic's feed + outlet published_time is the polite path
`/napi/rtcf/v1/movies/<emsId>/reviews` -> 401 (not bypassed). The MC feed is date-only but carries outlet URLs; the
outlet pages' article:published_time gave minute timestamps for most outlets (NYT/AP/RogerEbert 403; The Times/WSJ
robots-disallow, skipped).

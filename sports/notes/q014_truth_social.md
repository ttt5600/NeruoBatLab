# Q014 Truth Social post-count brackets vs the running count: FORWARD paper logger (pre-registration)

Filed 2026-10-29 (PT) before the first logger tick. Paper only, no orders.
Code: `scripts/q014_logger.py` (tick / report / model). It imports helpers from `scripts/q013_logger.py`.
Tests: `tests/test_q013_q014_loggers.py`. Agent: launchd `com.jonathanwang.quant-q014`, every 900 s, code copy in
`~/.claude/quant-kalshi`. Data: `~/.claude/quant-kalshi/data/q014/` (gitignored).

## 1. Question
With part of the window elapsed and the running count known, are KXTRUTHSOCIAL (weekly) and KXTRUTHSOCIALD (daily)
brackets priced off a sensible forecast of the remaining posts? The forecast is a frozen per-hour rate with
negative-binomial overdispersion. Mechanism: a count process that is partly observed before close; retail may anchor
on round numbers. The prior is no edge.

## 2. Contract semantics (rules_secondary, read 2026-10-29)
- Windows run 12:00:00 AM ET to 11:59:59 PM ET: one ET day (daily), or Sunday through Saturday (weekly, e.g. Oct 25-31).
  Weekly markets close the next morning (13:59Z). The code parses the window from the rules text, not from the ticker.
- Truths, ReTruths and Quote Truths count. Replies count "if captured by Roll Call". "Between" is inclusive. Posts deleted
  "as of 12:00 AM ET" do not count.
- Settlement source: Roll Call Factba.se.

## 3. Data sources (checked 2026-10-29)
- **Count = the settlement source's own JSON**: `rollcall.com/wp-json/factbase/v1/twitter?platform=truth social&sort=date
  &sort_order=desc&page=N&format=json`. This is the endpoint the Factba.se archive page itself calls. It returns 50 posts
  per page, each with a post timestamp (ET offset), an id (= Truth Social status id) and `deleted_flag`. The full archive goes
  back to 2022 (`meta.total_hits` 38,743 on 2026-10-29).
- Cross-check: the trumpstruth.org RSS (`/feed`, the latest 100 items, each carrying `truth:originalId` = the same id).
- truthsocial.com's own API returned a Cloudflare 403. It is not used, and it is not bypassed.
- Politeness: a full sweep of the needed range (back to the earliest open window start) runs hourly. Otherwise the sweep
  is incremental and stops at the first page of already-known posts. There is a 1 s sleep between pages, capped at 20 pages.
- Count rule used by the model: RC posts dated in [ws, now] with deleted_flag false that have not vanished from the last
  full sweep. The count including deleted posts is logged too.

## 4. Frozen rule
- **Calibration, stated plainly**: the rate model is fitted ONCE, on the posts dated in the logger's own first 7 days
  [first_tick, first_tick + 7 d). It is then frozen in `data/q014/model.json` and never refitted. No picks are made
  during calibration. No Kalshi price enters the fit. Note that the calibration week spans the 2026-11-01 DST change,
  which the hour-of-day counts handle.
- Model: r_h = the mean posts per hour for each ET hour-of-day h over the calibration week. Day-level overdispersion
  comes from the 7 consecutive 24 h block totals: k = mean^2 / (var - mean) if var > mean, else Poisson. The remaining
  count in [now, window end) is the sum over ET days of NB(mu_day, k), where mu_day = sum of r_h over the remaining
  hours of that day. These are moment-matched to one NB with mean M and var V = sum(mu + mu^2/k).
  P(YES) = P(count_so_far + R satisfies the bracket).
- Decision slots: weekly markets at 12:00 ET on each of the 7 days of the window; daily markets at 12:00 ET and
  18:00 ET. The pick is made at the first tick in [slot, slot + 60 min). There is one row per (slot, market), logged
  without a bet when there is no edge or no quote.
- Bet: the same as Q013. Buy YES at the ask if p - ask - 0.07 ask(1-ask) >= 0.05. Otherwise buy NO at (1-bid) if
  (1-p) - (1-bid) - fee >= 0.05. The quote must satisfy 0 < bid < ask < 1. One contract per row, settled on Kalshi `result`.
- Shadow (NOT the test): the same rule with the Poisson probability, logged as `p_yes_poisson`. It shows whether
  overdispersion matters.

## 5. Trials, statistics, bar
- Two trials, W (weekly) and D (daily), Bonferroni one-sided p < 0.025 each.
- ROI = sum P&L / sum cost. The CI and p come from an EVENT-clustered bootstrap (10,000 resamples, seed 1013). The
  observed ROI is quoted. Calibration bins plus Brier model vs mid. CLV vs the last pre-close mid.
- PASS (per trial): ROI > 0, p < 0.025, model Brier < mid Brier, with >= 30 daily events (D) or >= 12 weekly events (W)
  with settled bets.
- Too-good guard: ROI > +30% -> audit the rows (count timestamp vs slot, RC lag, deleted handling) first.

## 6. Kill
- D: after 40 settled daily events with bets, ROI <= 0 or Brier >= mid -> DEAD. W: the same after 12 weekly events.
- If the RC count at window end disagrees with Kalshi's `expiration_value` on > 10% of settled events, fix the count
  semantics (replies, deletions) before counting either trial.

## 7. Power
There are about 7 daily events a week and 1 weekly event a week, each with several brackets. That is ~30 daily events in
about 5 weeks after calibration, and 12 weekly events in about 3 months. Only a large mispricing is detectable.
The weekly trial is the underpowered one.

## 8. Note for PM
The settlement source's JSON has the full post history with timestamps, so the COUNT side of this study IS backtestable
(37 weekly + 28 daily settled Kalshi events, census 2026-10-29). The binding gap is historical Kalshi quotes (candles).
Per the brief this logger calibrates on its own first week. A historical-rate variant would be a separate pre-registered trial.

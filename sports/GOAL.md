# GOAL -- find edges that are real when evaluated, not edges that look real (user, 2026-11-02)

Mandate: double a $1-2k unlevered bankroll in 12 months, proven live (notes/FUND_ARCHITECTURE.md). An "edge" exists
ONLY when the harness below says so. Agents (including the lead) may not declare edges, change bars, or promote
strategies by hand; they run `edgelab` and report what it says. This file is the contract; `sports/edgelab/` enforces it.

## The one metric (same for every strategy, every venue)
**Net ROI per $ risked on bets that were decided BEFORE their outcome existed, priced at a price we could actually get,
after all costs** -- reported with a cluster bootstrap interval (cluster = event-day), plus **CLV** (closing price minus
our price, same side) as the leading indicator. Wins/losses, hit rate, Brier, "model beats the close" are diagnostics,
never the verdict.
- Price we could get: taker = ask + fee; maker = fill modeled from the public trade tape at our posted price (no
  fills that never printed); sportsbook = the quoted price, never "-110 assumed" for a verdict.
- Costs: venue fee schedule in code (Kalshi 0.07*p*(1-p) taker), spread, and no free cancels.

## Rules enforced in code (edgelab)
1. **Register before you look.** A trial = a spec (hypothesis, dataset, rule module + its git hash, decision time,
   price model, split, bar). `edgelab register` stores the spec hash. Any later change to the rule, filters, threshold
   or split = a NEW trial. Amendments are allowed only before the first data access (access is logged).
2. **Every trial counts.** The bar for a family = alpha / (number of trials EVER registered in that family, dead ones
   included). Nothing can be un-registered. Re-running a dead idea with a tweak raises the bar for everything in it.
   (Since 2026-11-18 this is the DEV bar; the sealed bar is rule 5(a) and forward promotion is rule 6.)
3. **Vault.** Data splits are defined once per dataset in `edgelab/vault.yaml` by DATE. Dev loads freely; sealed data
   loads only through `edgelab`, only for a trial whose dev already passed, and only ONCE (second unseal refused).
4. **Point-in-time.** Every bet row carries `decided_at` and the inputs' `as_of`; the evaluator rejects any row whose
   inputs are timestamped after `decided_at`, or whose price is not from <= decided_at (or the modeled fill time).
5. **Pass = all of:** (a) one-sided p < family bar on net ROI; (b) observed ROI > 0 in both time halves; (c) no single
   event-day/week contributes > 30% of total profit (concentration); (d) n >= pre-set minimum; (e) mean CLV >= 0.
   Too-good guard: ROI > +10% on n > 200, or Sharpe-equivalent implausible -> status `needs_audit`, not pass.
   **Amendment (user, 2026-11-18):** a market that settles at value v pays v per YES / 1 - v per NO contract (Kalshi
   `scalar` settlements: politics metadata, cricket no-result, ties/draws at $0.50); it is scored at that value, not as
   a loss or a push. edgelab SCORING_VERSION 2 enforces it (notes/ops_edgelab_scalar.md); evaluations before 2026-11-18
   are v1 and no recorded verdict changes under v2.
   **Amendment (user, 2026-11-18, "approval on everything"; bar_calibration_v2 + critic edits a-g; edgelab BAR_VERSION
   2, notes/ops_goal_amendment.md):** (a) stage bars: dev = the rule-2 family bar; the j-th sealed shot in a family is
   judged at .05 / 2^j, j = 1 + the family's sealed shots already taken (pre-edgelab spent unseals included; weather's
   next shot = .05/32), so a family's sealed false passes sum to <= .05. (c) is replaced by: observed net ROI > 0 after
   dropping the top-profit week AND > 0 after dropping the top-profit cluster (each separately); the profit share is
   reported only. Too-good guard: ROI > +10% on n > 200 -> needs_audit stays for every trial; the Sharpe trigger no
   longer applies when the mean all-in cost is >= .90. (b), (d), (e) unchanged.
6. **Promotion ladder (state machine):** registered -> dev_pass -> sealed_pass -> forward_paper -> live_small ->
   live. `forward_paper` -> `live_small` needs the trial's forward sequential test to accept H1 on forward data alone
   (amendment 2026-11-18 below; it replaced "100 forward bets passing rule 5"), then a user audit (acceptance sets
   needs_audit). Only the user moves anything to `live_small`/`live` or changes size. The daily ticket lists only
   strategies at >= forward_paper (labelled PAPER) and says so.
   **Amendment (user, 2026-11-05):** before forward eligibility, the user may approve ONE size step of at most 2x for a
   strategy at forward_paper (so it passed sealed) that already trades live and has a pre-set kill (rule 7). Any
   further size step needs the forward test to accept H1 (2026-11-18 wording). edgelab enforces this (`approve --size`).
   **Amendment (user, 2026-11-18; edgelab BAR_VERSION 2, edgelab/forward_test.py):** only a trial that reached
   sealed_pass can be promoted by forward data; a trial that failed dev or sealed (e.g. X1, L32) may log forward paper
   but its forward data cannot promote it (a named user exception would be charged to that family's forward budget of
   .05 per calendar year; not implemented in edgelab). Forward test parameters are fixed at registration (spec
   `forward_test`); trials already scoring forward bets on 2026-11-18 use the defaults below, fixed at adoption.
   - Near-certain (mean all-in cost >= .90): Wald SPRT on filled losses; H0 loss prob = 1 - c_i (c = fill price +
     fee), H1 = 0.70 x (1 - c_i); alpha .005, beta .10; looks after each event-day cluster; LLR divided by
     1 + f x .031, f = fills per night fixed at registration (default 55/30). No crossing within 36 months -> stays at
     forward_paper and the user decides.
   - Sports ROI (every bet priced 40-60c; calibrated at 49-53.5c): Wald SPRT on win/loss vs break-even at cost;
     H1 = +3% ROI (+5% if registered); alpha .05, beta .20; looks per date; LLR divided by 1 + (b - 1) x .05,
     b = bets per date fixed at registration (default 7). No crossing within 24 in-season months -> the user decides.
     Any other price mix has no forward test (cannot be promoted by forward data) until its own placebo calibration
     is registered.
   - Net CLV is a diagnostic for the forward test, never part of its pass or kill (pre-set kills that already use
     CLV stay in force, rule 7).
   - Disclosed residual risks: the SPRT calibration already looked at E4/X1 forward data (the live orig7 fill mix), so
     those two trials' forward tests are not fully pre-registered; month-level loss dispersion comes from 16 months
     of one pick stream (re-estimate after 12 forward months); expected time to accept an E1-sized edge is ~2 years
     at 55 fills/month and a +2% sports edge cannot be promoted within the 12-month mandate.
7. **Kill rules are code too:** each forward strategy has a pre-set kill (e.g. CLV < 0 with p < .05 after 100 bets).
   Since 2026-11-18 (user) the forward test's lower-boundary (H0) crossing is a kill IN ADDITION to the pre-set kill;
   any one triggering kills the strategy.
8. **The harness tests itself.** Placebo strategies (random sides, shuffled signals) are run through the same pipeline;
   their false-pass rate must be <= alpha. Golden cases must reproduce: E1 weather consensus-reject = sealed pass;
   L2 NBA team-rating sides = fail. If either breaks, the harness is broken, not the market.

## What the lead does every cycle
Ask `edgelab status` -> score anything finished -> register the next idea from QUESTIONS.md -> run -> report the
harness verdict verbatim. Summaries to the user quote edgelab, not impressions.

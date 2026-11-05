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
3. **Vault.** Data splits are defined once per dataset in `edgelab/vault.yaml` by DATE. Dev loads freely; sealed data
   loads only through `edgelab`, only for a trial whose dev already passed, and only ONCE (second unseal refused).
4. **Point-in-time.** Every bet row carries `decided_at` and the inputs' `as_of`; the evaluator rejects any row whose
   inputs are timestamped after `decided_at`, or whose price is not from <= decided_at (or the modeled fill time).
5. **Pass = all of:** (a) one-sided p < family bar on net ROI; (b) observed ROI > 0 in both time halves; (c) no single
   event-day/week contributes > 30% of total profit (concentration); (d) n >= pre-set minimum; (e) mean CLV >= 0.
   Too-good guard: ROI > +10% on n > 200, or Sharpe-equivalent implausible -> status `needs_audit`, not pass.
6. **Promotion ladder (state machine):** registered -> dev_pass -> sealed_pass -> forward_paper -> live_small ->
   live. `forward_paper` -> `live_small` needs >= 100 forward bets passing rule 5 on forward data alone. Only the user
   moves anything to `live_small`/`live` or changes size. The daily ticket lists only strategies at >= forward_paper
   (labelled PAPER) and says so.
   **Amendment (user, 2026-11-05):** before 100 forward bets, the user may approve ONE size step of at most 2x for a
   strategy at forward_paper (so it passed sealed) that already trades live and has a pre-set kill (rule 7). Any
   further size step still needs the 100 forward bets. edgelab enforces this (`approve --size`).
7. **Kill rules are code too:** each forward strategy has a pre-set kill (e.g. CLV < 0 with p < .05 after 100 bets).
8. **The harness tests itself.** Placebo strategies (random sides, shuffled signals) are run through the same pipeline;
   their false-pass rate must be <= alpha. Golden cases must reproduce: E1 weather consensus-reject = sealed pass;
   L2 NBA team-rating sides = fail. If either breaks, the harness is broken, not the market.

## What the lead does every cycle
Ask `edgelab status` -> score anything finished -> register the next idea from QUESTIONS.md -> run -> report the
harness verdict verbatim. Summaries to the user quote edgelab, not impressions.

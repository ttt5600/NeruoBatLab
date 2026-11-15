# ot1b_monotone -- OT1b: OT1 with a monotone, arbitrage-free implied distribution (FORWARD only)

Agent: ot1b_monotone (2026-11-15). Read-only; no orders, no git, no .env / kill changes. Adds a SECOND rule column to the
existing OT1 forward logger (scripts/ot1_forward.py, launchd quant-sports-ot1; user approved read-only paper loggers).

## Why
The frozen OT1 implied P(S > K) = central difference -dC/dK (h = 0.001 K) on a piecewise-linear-in-moneyness IV smile
is not monotone in K: 432 of 1,020 DEV crypto events have >= 1 reversal; 427 of 13,615 rungs clipped to exactly 0/1
(notes/ops_ot1_forward.md). A forward pick can land on a clipped-to-0 rung (live 11-15: T67249.99 = 0.0 while
T67749.99 = .0042). OT1 (trial ot1_crypto_taker-18d38e5c) stays frozen; OT1b is a NEW trial.

## Pre-registration (written 2026-11-15 BEFORE any OT1b number was computed; window from 2026-11-16)
- Question: does the OT1 taker rule make money forward when P(YES) comes from a monotone, arbitrage-free distribution?
- Construction (frozen in edgelab/rules/opt_tails_mono.py, function `p_above_grid` / `p_above_mono`):
  same smile tuple (F_T, tau_D, xs, sig) from the frozen scripts/options_implied_tails.py:smile, same sqrt-time map.
  1. Grid: 401 strikes uniform in log(K/F_T) over exactly the smile's covered range mapped to the Kalshi horizon,
     log(K/F_T) in [xs[0] * sqrt(tauK/tauD), xs[-1] * sqrt(tauK/tauD)] (no extrapolation, as OT1).
  2. Call prices C_i = Black-76(F_T, K_i, sigma(x_i), tauK) with sigma linearly interpolated in x (as OT1).
  3. Call-spread probabilities q_j = -(C_{j+1} - C_j)/(K_{j+1} - K_j) at grid midpoints.
  4. Isotonic (PAVA) non-increasing regression of q_j in K, weights K_{j+1} - K_j (= the derivative of the greatest
     convex minorant of C, i.e. the closest butterfly-arbitrage-free call curve); then clip to [0, 1].
  5. P(S > K) = linear interpolation of that step-monotone curve at K over the midpoints; K outside -> NaN.
  P(YES) from P(S>K) exactly as OT1 (oit.p_yes: greater -> P(>floor), less -> 1 - P(>cap), between -> difference).
- Frame / rule / timing / quote / settle / fee / depth: IDENTICAL to OT1 (notes/ops_ot1_forward.md pre-registration):
  same Deribit pull and same one Kalshi GET per event (same snapshot), amendment-2 frame recomputed with the OT1b
  p_yes (oit.tail_candidates), rule edgelab/rules/opt_tails.py:taker unchanged (edge >= .03 after exact fee, ask in
  [.50, .99], spread <= 5c), 1 contract / $1 of cost, cluster crypto|date, bet only if snapshot in [11:00, 11:20) ET.
- Logged as `kind: pick_b` rows in the same ot1.jsonl (status bet | outside_frame | off_time); day row gets
  `bets_b`, `best_of_day_b`; quote rows get `p_yes_b`. OT1 rows/fields are unchanged.
- Judge: KILL if the ROI CI (ask + exact fee) is entirely below 0 after 150 bets (min_n 150); edgelab trial
  ot1b_crypto_taker_mono-7d75fa46 (dataset kalshi_opt_tails_mono: dev null, sealed null, forward
  [2026-11-16, null]; adapter edgelab/adapters/ot1b_fwd.py). NO dev scoring (the DEV window was already used by OT1; scoring OT1b on it would be
  tuning the smoother on seen outcomes). Only descriptive DEV diagnostics below (counts of changed picks; no P&L,
  outcomes never loaded).
- Overlap: OT1 and OT1b share most picks; they are NOT independent evidence. Report both; the pair counts as two
  trials in family kalshi_data_ladders.

## Log
- 2026-11-15 started; inbox `started` row.
- Frozen module edgelab/rules/opt_tails_mono.py written; trial registered `ot1b_crypto_taker_mono-7d75fa46`
  (family kalshi_data_ladders now 49 trials, bar p < 1.02e-3). Any edit of opt_tails_mono.py = a new trial.
- vault.yaml: new dataset `kalshi_opt_tails_mono` (dev null, sealed null, forward [2026-11-16, null], forward adapter
  edgelab/adapters/ot1b_fwd.py = ot1_fwd.py reading `pick_b` rows). OT1's dataset / adapter / rule untouched.
  `edgelab status` next-gate text for it reads "no dev/sealed window: user must pre-register a sealed forward window"
  (generic text for forward-only datasets, same as nhl_om_a_kft); `evaluate --split forward` works regardless.
- Logger scripts/ot1_forward.py: additive `ot1b()` after OT1's picks per event, on a COPY of the same snapshot frame;
  import + call wrapped so a failure writes `ot1b_error` in the snapshot and OT1 still logs. Diff vs the pre-change
  copy results/ot1b_monotone/ot1_forward.pre_ot1b.py: only added lines (+ day row gains *_b fields, quote rows p_yes_b).
  Installer scripts/install_ot1_agent.sh now also copies opt_tails_mono.py; re-run -> launchd reloaded, heartbeat
  08:06:15 PST rc=0.
- Scorecard `card_ot1b` in scripts/scorecards.py (too-early, 0 of 150).

## DEV diagnostics (descriptive only; outcomes dropped on read, no P&L) -- results/ot1b_monotone/dev_diag.py
- 1,020 crypto events with a smile; events with >= 1 reversal of P(>K) on `greater` rungs: OT1 432, OT1b 0
  (per-event counts results/ot1b_monotone/dev_reversals.csv).
- Rungs with finite p: OT1 13,615, OT1b 13,868 (grid midpoints reach slightly further than K +- 0.1%).
- p exactly 0 or 1: OT1 306, OT1b 162 (OT1b zeros are monotone flat tails from PAVA pooling + clip, not reversals).
- |p_OT1b - p_OT1| on rungs with both: median .00013, p90 .0052, p99 .030, max .198.
- Amendment-2 candidates: OT1 5,066, OT1b 5,153, common 4,865; 288 OT1b candidates have no DEV quote (never fetched).
- Rule picks on quoted candidates: OT1 387 (matches DEV n), OT1b 380, common 360, dropped 27, added 20 (+ unknown
  picks among the 288 unquoted). ~93% of OT1's DEV picks are also OT1b picks: the forward trials overlap heavily.

## Verification (2026-11-15)
- Worktree forced decide (SUFFIX _ot1btest, never scored/settled): both rule columns written; 120/120 quote rows carry
  p_yes_b; KXBTCD 31 rungs: OT1b monotone True (OT1 also monotone on this snapshot); 0 picks both (no live edge).
- Deployed copy (~/.claude/quant-sports venv) forced decide (SUFFIX _ot1btest_deployed): OT1b import OK, same result,
  120 p_yes_b rows. Test files ot1_ot1btest*.jsonl / ot1_ot1btest_deployed*.jsonl are not read by settle/adapters.
- Pick path + adapter sandbox (results/ot1b_monotone/sandbox_test.py; synthetic smile, one quote edited, book GET
  stubbed): 1 pick_b row (NO .94, edge .041) -> adapter ignores a `pick` (OT1) row in the same file -> resolve_outcomes
  + schema.enforce pass (inputs 16:05:36 < quote 16:05:41 <= decided 16:05:42 < event_start 21:59:59).
- First real OT1b decision: 2026-11-16 11:00 ET, same launchd run as OT1.
- Note: edgelab status shows OT1 (ot1_crypto_taker) as state `dead` (dev p .0058 >= bar), consistent with LEAD /
  forward-paper-only; OT1b is `registered`.

## Learnings
### PAVA on call-spread probabilities removes every OT1 reversal at a small cost in p
Weighted isotonic regression of -dC/dK over a 401-point log-strike grid gives 0 reversals vs 432 of 1,020 DEV events
for the frozen central-difference construction; median |dp| .00013, p99 .030 (results/ot1b_monotone/dev_diag.py).

### A smoother fix changes few picks: OT1b keeps 360 of OT1's 387 DEV picks
27 dropped, 20 added on quoted candidates (+ up to 288 unquoted new candidates). OT1 and OT1b forward results will be
highly correlated; do not count them as two confirmations (dev_diag.py output).

### Register a fix to a frozen trial as a forward-only dataset so the vault blocks dev scoring
A vault.yaml dataset with dev/sealed null (kalshi_opt_tails_mono) makes `evaluate --split dev` impossible rather than
merely discouraged; the tuning-on-seen-data risk is enforced in code, not memory.

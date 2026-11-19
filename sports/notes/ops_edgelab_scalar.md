# ops_edgelab_scalar -- edgelab scoring of scalar / $0.50 Kalshi settlements (agent started 2026-11-15)

## Question
edgelab maps a market result to an outcome in {0, 0.5, 1}; 0.5 = push (stake back). Kalshi markets can settle at a
SCALAR value v in (0,1) (politics metadata markets 0.25/0.50; cricket NR / NFL tie / UFC draw $0.50). Such a contract
really pays v (YES) or 1-v (NO). (1) Where does edgelab map results to P&L, and which registered trials / datasets
contain non-{0,1} settlements? (2) How many past verdicts would change if scored at the true settlement value?
(3) A minimal, versioned, OFF-by-default fix + tests; apply steps for the user.

## Pre-registered rules (written before any recompute)
- "Verdict changes" = the sandbox re-score under the fix gives a different `verdict` (pass/fail/needs_audit) than the
  registry's recorded evaluation for the same (trial, split). Also reported: changes in failed-check set and net ROI.
- Scope: only (trial, split) pairs that already have an `evaluate` event in lab/edgelab_registry.jsonl (no new data
  window is opened). Recompute in a temp-dir sandbox registry; the real registry is never written.
- The fix must leave every binary (0/1/void) row's P&L bit-identical (test), and must be off unless enabled.

## Plan
1. Read evaluator / vault.resolve_outcomes / loaders / adapters for every result -> outcome -> pnl path.
2. Scan each evaluated dataset's raw rows for result values outside {yes,no,<side>} and settlement values not in {0,1}.
3. Sandbox recompute of affected (trial, split) pairs, old vs new scoring.
4. Fix behind a flag + tests; run the suite.

## Log
### 1. Where a result becomes P&L (read 2026-11-15)
- `edgelab/vault.py resolve_outcomes` maps hidden results -> outcome in {0, 0.5, 1}:
  `result_equals_side` (24 datasets): 1 if result == side else 0 -> a Kalshi `result == "scalar"` is a LOSS for BOTH
  sides, even when the value is 0.00 (NO actually won in full) or 0.50. `result_side` (forward adapters): 'void' -> 0.5.
  `won_column` / `won_push_columns` / `win_push_columns` / `l2_side`: binary + push only.
- `edgelab/schema.py normalize` rejects any outcome outside {0, 0.5, 1}.
- `edgelab/evaluator.py cost_and_pnl`: outcome 1 -> n(1-cost), 0 -> -n*cost, anything else -> 0 (push).
- Upstream (outside edgelab) many loaders/builders DROP non-yes/no markets: loaders kalshi_tsaw, portwatch, ytviews,
  album_units, top_album, opt_tails, wx_monthly_early, golf_topn; builders lx2_tail_no_carry, phys_ladders,
  kalshi_hourly_temp, cbb_ingame, kalshi_ingame_lowscore; Polymarket builders drop 50/50 resolutions. Dropping by
  outcome is a (small) selection on the outcome; lx2 dev: 26 scalar of 19,216 binary markets settled < 2025-10-01
  (data/lx2_tail_no_carry/markets.parquet) -- not traced to bets.
- kalshi_cricket_nr: vault outcome kind `none` (script authoritative); rq_cricket_c1/c2 are registered, never
  edgelab-evaluated (registry has only their `register` events).

### 2. Which evaluated data contain non-binary settlements (results/ops_edgelab_scalar/scan.py -> scan.json)
Every (dataset, split) with an evaluate event in lab/edgelab_registry.jsonl (85 events), raw pre-filter rows:
ONLY kalshi_politics_fav dev has them: 121 `scalar` rows of 17,940. All other datasets: binary results (+ book
pushes, which are real voids). Bets hit: pol_t7 7 of 337, pol_t3 10 of 506, pol_t14 0 of 273 (all KXHORMUZMAX,
values 0 / .25 / .50).

### 3. Sandbox re-score (results/ops_edgelab_scalar/sandbox_rescore.py -> sandbox_rescore.json)
Temp copy of the registry, save_rows=False, real registry sha256 asserted unchanged (9fe24bff6432). 15 dev trials
on 7 Kalshi datasets re-run under v1 and v2; v1 reproduces every recorded verdict / n / ROI exactly.
| trial | registry = v1 ROI [CI] | v2 ROI [CI] | p v1 -> v2 | verdict |
|---|---|---|---|---|
| pol_t7_fav_taker | -3.00% [-9.99, +3.36] | -0.56% [-6.50, +4.99] | .807 -> .563 | fail -> fail |
| pol_t3_fav_taker | -3.74% [-8.99, +0.96] | -1.93% [-6.91, +2.44] | .937 -> .795 | fail -> fail |
| pol_t14_fav_taker | -9.26% | -9.26% (no scalar bets) | same | fail -> fail |
| 12 others (tsa t1/t2, ytv, spacex, rain rd1/rd2, lx2 a/b, ot1-4) | bit-identical | | | unchanged |
**Verdicts that change: 0.** Failed-check sets identical (a,b,c,e still fail for T7/T3). v2 T7 -0.56% matches the
pol_favourites script's scalar-aware number (-0.56% [-6.28, +5.07], notes/pol_favourites.md) -> cross-check OK.
Not re-run (no scalar rows per scan, and v2 is bit-identical on binary rows by test): the other 70 evaluate events.

### 4. Fix (written, OFF): SCORING_VERSION in edgelab/__init__.py (= 1)
- vault.resolve_outcomes(..., scoring_version): v2 runs `_settle_values` for result_equals_side / result_side: a bet
  whose hidden result is 'scalar' gets payout = v (YES) or 1 - v (NO) from the outcome's `value_col`; outcome = payout.
  Missing value / non yes-no side -> VaultError (never silent). Rules may not set `payout`.
- schema: optional `payout`; a fractional outcome is legal only when it equals payout.
- evaluator.cost_and_pnl: payout rows -> contracts x (payout - cost) (dollar staking: stake x (payout/cost - 1)).
- run.evaluate / score_only take scoring_version (None = package flag); v2 events carry `scoring_version: 2`; v1
  events unchanged.
- tests/test_edgelab_scalar.py (12 pass): flag default 1; v1 legacy; politics .25 both sides; cricket .50 not a push;
  binary bit-identical v1 vs v2 (contract + dollar staking, full score); void push unchanged; result_side scalar;
  loud failures. Full suite tests/: 528 pass, 2 fail in tests/test_bet.py (bet.py order-body asserts; does not import
  edgelab; not touched here -- pre-existing).

### 5. APPLY STEPS (for the USER; a scoring change is GOAL-level)
1. Approve amending GOAL.md rule 5: "a market that settles at value v pays v per YES / 1 - v per NO contract".
2. Politics: loader `kalshi_politics_fav` adds `scalar_value = ticker.map(data/pol_favourites/scalar_values.json)`
   (+ that file in sources); vault.yaml kalshi_politics_fav: `hidden: [result, scalar_value]`,
   `outcome: {kind: result_equals_side, key: ticker, value_col: scalar_value}` (hidden, so rules never see it).
3. Cricket: vault.yaml kalshi_cricket_nr `outcome: {kind: result_equals_side, key: ticker, value_col: sv}` (hidden is
   already [result, sv]). Do NOT edit rules/cricket_nr.py (a rule edit = a new trial hash).
4. Set `SCORING_VERSION = 2` in edgelab/__init__.py; run `pytest tests/test_edgelab*.py` and `python -m edgelab selftest`.
5. Optional record: `python -m edgelab evaluate <pol_t7 id> --split dev` (and pol_t3) -> appends v2 evaluate events;
   state stays dead, verdict stays fail. Cricket c1/c2 can then get their first edgelab dev evaluation.
6. Not covered: kx_fwd emits 'void' for NFL ML ties (Kalshi pays $0.50; v2 needs result 'scalar' + value). Changing
   the adapter after its first scored forward evaluation is refused by the adapter lock -> new trial or user decision.
   No forward voids exist yet (grep "void" ~/.claude/quant-sports/data/kx_forward/*.jsonl: none).

## Learnings
### edgelab v1 scored every Kalshi `scalar` settlement as a loss for BOTH sides
Even v = 0.00 (NO won in full) counted as a NO loss. Only kalshi_politics_fav dev was hit (121 / 17,940 rows; 17 bets).
Source: results/ops_edgelab_scalar/scan.json. Matters for any new dataset with disputed or partial settlements.
### Fixing the scalar bug changes no past edgelab verdict
Sandbox re-score: pol_t7 -3.00% -> -0.56% [-6.50, +4.99], pol_t3 -3.74% -> -1.93% [-6.91, +2.44], still fail; 12
other trials bit-identical (results/ops_edgelab_scalar/sandbox_rescore.json).
### Many Kalshi builders drop non-yes/no markets (outcome-conditioned exclusion)
lx2 dev: 26 scalar of 19,216 markets dropped. Tiny here, but a drop by outcome is a selection bias; prefer scoring at
the settlement value (SCORING_VERSION 2 + value_col) over filtering.
### A full no-patch sandbox re-score confirms v2 changes 0 of 65 reproducible dev verdicts
Applied code, every dev-evaluated trial: 65 reproduce v1 exactly, 0 verdict changes; only pol_t7/pol_t3 ROI moves
(results/ops_edgelab_v2_apply/sandbox_rescore_applied.json). See "Applied 2026-11-18" below.
### A line-push 'void' is a sportsbook concept; Kalshi never pushes a strike market
kx_fwd now defers voids to kalshi_result (yes/no/scalar at value). Log the exchange settlement value next to the result.
### GOAL.md hash pin was already stale before the 11-18 amendment
fund.yaml pins f4a17a15f067 while GOAL.md was 8f2d97bd0945 (11-05 rule-6 amendment never re-locked); re-lock after
every GOAL amendment or `hub doctor` FAILs.

## Applied 2026-11-18 (agent ops_edgelab_v2_apply; user approval 2026-11-18: "you have my approval on everything")
- **GOAL.md rule 5** amended (dated user amendment: a market settling at v pays v / 1 - v; SCORING_VERSION 2 enforces).
  GOAL.md sha256 now `796004f8606638ed5861e67f62af20af445b0986c9dc645e9ca0c2bffa61ec3b`. fund.yaml NOT edited: it still
  pins f4a17a15f067 (already stale before today: pre-edit sha was 8f2d97bd0945, from the 11-05 rule-6 amendment).
  **USER must update the mirror sha in ~/agent-harness/goals/fund.yaml and re-lock (`sudo install.sh --lock`).**
- Politics: loader `kalshi_politics_fav` adds hidden `scalar_value` from data/pol_favourites/scalar_values.json (file in
  sources; 96/96 scalar tickers in priced.parquet covered); vault.yaml `hidden: [result, scalar_value]`,
  `value_col: scalar_value`.
- Cricket: vault.yaml kalshi_cricket_nr outcome `none` -> `{kind: result_equals_side, key: ticker, value_col: sv}`
  (cricket_frame: 194 scalar rows, sv = 0.5 on all; yes/no rows sv 1/0). rules/cricket_nr.py untouched.
- `SCORING_VERSION = 2` in edgelab/__init__.py. tests/test_edgelab_scalar.py: `test_flag_is_off_by_default` ->
  `test_flag_is_v2_after_user_approval`, + config test (value_cols present and hidden), + kx_fwd void test.
- Checks: pytest tests/ 537 pass, 2 fail = the pre-existing tests/test_bet.py order-body asserts (no edgelab import);
  `python -m edgelab selftest` ok; `python -m edgelab verify` ok (first_break null).
- **Sandbox re-score with the APPLIED code, no monkeypatches** (results/ops_edgelab_v2_apply/sandbox_rescore_applied.py
  -> sandbox_rescore_applied.json, sandbox.log): all 68 trials with a dev evaluation. 65 ran: v1 reproduces the
  registry on all 65; **0 verdicts change under v2**; only pol_t7 (-3.00% -> -0.56%) and pol_t3 (-3.74% -> -1.93%) move.
  3 cannot run under either version (lx2_a_t90, lx2_b_t180, nd1_nyc_nowcast_maker: RegistryError "rule module changed
  since registration" -- pre-existing, refused by design; none has scalar rows per scan.json). Real registry sha
  asserted unchanged; lab/ files hashed before/after: identical. **No real-registry re-evaluation done** (none required;
  step 5 optional; future evaluations record `scoring_version: 2`).
- **kx_fwd void fix** (edgelab/adapters/kx_fwd.py): scripts/kx_forward.py grades 'void' when the final lands exactly on
  the line S (a book push; it is a strike-ladder void, not an ML tie), but Kalshi settles the ticker yes/no or `scalar`.
  A void now defers to `kalshi_result` (yes/no -> that side; 'scalar' -> scored at hidden `sv` = settle row
  `kalshi_value`, missing -> VaultError); no kalshi_result -> stays 'void' push. Loader report counts
  `void_resolved_by_kalshi` / `void_kept_push`. Lock: no kx_fwd trial has any registry event (grep = 0) and 0 void settle
  rows exist in ~/.claude/quant-sports/data/kx_forward/*.jsonl -> lock not engaged, no score changes.
  PROPOSAL (not done; live launchd paper job): scripts/kx_forward.py market_result() should also log Kalshi's
  settlement value as `kalshi_value`, so a scalar void scores instead of raising.
- **Scalar-dropping builders: listed, NOT fixed** (not trivial: a loader edit changes registered trials' data; scan.json
  shows none of their evaluated windows hold scalar rows). edgelab loaders filtering result in {yes,no}: kalshi_tsaw,
  kalshi_portwatch, kalshi_ytviews, kalshi_album_units, kalshi_top_album, kalshi_opt_tails, kalshi_opt_tails_ext,
  kalshi_wx_monthly_early, kalshi_golf_topn. scripts/: cbb_ingame, dl_portwatch, e1x_media, e1x_tsa(+logger),
  e1x_quake(+maker), econ_reject, kalshi_cpi, kalshi_gas, kalshi_rain, kalshi_hourly_temp, kalshi_longshot,
  kalshi_soccer(+sharp), kalshi_fav_maker, kalshi_ingame_lowscore, kalshi_wx_winter, kalshi_wx_monthly_logger,
  kalshi_paper, lx2_tail_no_carry, crypto_barrier_forward, live_edges, fund_stats. New datasets: keep scalar rows and
  set `value_col` instead.

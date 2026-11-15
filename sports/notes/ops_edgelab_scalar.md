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

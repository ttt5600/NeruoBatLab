# ops_goal_amendment -- GOAL rules 5/6/7 amendment v2 applied to GOAL.md + edgelab (agent, 2026-11-18)

Question: implement the user-approved (2026-11-18, "approval on everything") amendment at the end of
notes/bar_calibration_v2.md plus critic edits a-g (inbox row `critic_amendment_v2`) in GOAL.md and edgelab.
Pre-set checks (written before running anything): existing 42 edgelab tests still pass; selftest ok (placebo
corrected false-pass <= .05, golden E1 pass / L2 fail, cheaters refused); registry hash chain ok; sandbox forward
re-evaluation: expect NO trial to become eligible for live_small. Kill: any golden case breaks or any trial promotes.

## What changed
- GOAL.md: rule 2 note (dev bar), rule 5 amendment (stage bars, concentration, Sharpe guard), rule 6 rewritten
  (forward test replaces 100 bets; sealed-pass gate; 11-05 size-step criterion reworded; tests + horizons; CLV
  diagnostic; residual risks incl. the SPRT panel having seen E4/X1 forward data), rule 7 (SPRT H0 kill additive).
  New sha256 f405c49ecec8c6349b08eca9e8f3ba18ed1f7dae50223840b9d1f708e3047db7 (before: 796004f8...). User must re-lock
  ~/agent-harness/goals/fund.yaml (not touched).
- edgelab/__init__.py: BAR_VERSION = 2 (v1 kept for sandboxes: run.evaluate(..., bar_version=1)).
- edgelab/forward_test.py (new): near-certain loss SPRT (rho1 .7, a .005, b .1, deff 1 + f x .031, f default 55/30,
  36 months) and sports ROI SPRT (r1 .03/.05, a .05, b .2, deff 1 + (b-1) x .05, b default 7, 24 in-season months,
  scope: every price in [.40,.60]); LLRs identical to results/bar_calibration_v2/sim2.py nc_task/sp_task. Spec key
  `forward_test` validated at registration; no key -> adoption defaults.
- registry.py: sealed_shots / sealed_bar (.05/2^j; shots = vault spent_unseals + non-replay sealed accesses).
- evaluator.py: version=2 -> (c) ROI > 0 ex top week AND ex top cluster; Sharpe trigger only if mean cost < .90.
- run.py: sealed bar, forward_test on forward evals, kill rule `forward_sprt_h0` appended to preset/spec kills,
  forward_eligible_for_user = H1 accepted AND reached sealed_pass; acceptance -> needs_audit + forward_audit_required.
- cli.py approve --to live_small: needs eligibility and `--audit-note` (writes a `user_audit` event). report.py:
  next-gate text + `_forward_test` in evidence (EVIDENCE_FIELDS unchanged for the hub verifier).
- tests/test_edgelab_amendment.py (13 tests).

## Interpretation choices (flag to lead/user)
1. Sealed bar = .05/2^j REPLACES the rule-2 family bar at the sealed stage (intent of bar_calibration v1/v2: "dev keeps
   the family bar; sealed at ..."). Weather next shot: .05/32 = .0016, looser than the old .05/73 = .00068. If the user
   wants min(family bar, .05/2^j) it is a one-line change in run.evaluate.
2. Near-certain = mean all-in cost >= .90 (same threshold for the test kind and the Sharpe exemption); sports scope =
   every bet's price in [.40,.60], else out_of_scope (no accept, no SPRT kill; existing kills still apply).
3. Looks = clusters (event-days) ordered by last decided_at; first crossing is final; push/scalar rows carry no LLR.
4. The X1/L32 user-exception path (family .05/yr budget) is not implemented (refused by construction).

## Results
- Tests: edgelab 55/55 pass (`pytest tests/test_edgelab*.py`). Full suite 550 pass, 2 fail in tests/test_bet.py
  ('0.5500' vs '0.55' price formatting; no edgelab import -- unrelated to this change).
- Sanity (seeded iid streams; tests/test_edgelab_amendment.py helpers, rng 99): near-certain placebo 1,000 streams of
  36 months: accept .002, kill .802; E1-size ratio .684 (300): accept .373, kill .040; sports placebo 24 mo x 60/mo
  (1,000): accept 0, kill .203.
- `edgelab selftest` ok: placebo corrected false-pass 0.0 (real and fair batches, 210 each; uncorrected .0048 / .052),
  golden E1 sealed pass at bar .025 (j=1), L2 fail; cheaters refused. lab/edgelab_selftest.json rewritten.
- `edgelab verify`: ok, 1359 lines (real registry not written by this agent).
- Sandbox forward re-eval (results/ops_goal_amendment/sandbox_forward.py -> sandbox_forward.json; registry copies, sha
  checked unchanged): 0 state changes v1 vs v2, 0 eligible. e4_maker_e1 forward_paper n 32: near_certain continue,
  LLR +0.24 in [-2.30, +5.19]. e4_maker_x1 (dead) LLR +0.22 continue; tsa_t2 (dead) n 4 continue; 4 trials error
  identically under v1 and v2 (nba.py rule module changed since registration x2, tsa_t1 adapter missing,
  kalshi_prop_ladders KeyError 'result') -- pre-existing.
- `edgelab status` still shows the v1 forward line for e4_maker_e1 until the lead's next real forward evaluation.

## Learnings
### An amendment's stage bars must say whether they replace or stack on the old bar
The text ".05/2^j for the j-th sealed shot" did not say if rule 2's alpha/n_trials still applies at sealed; replacing
it makes weather's next sealed bar .0016 vs .00068 before. Source GOAL.md rule 5(a), edgelab/run.py.
### Keep the old rule runnable behind a version switch when changing a verdict function
bar_version=1 in run.evaluate let the sandbox show 0 forward state changes v1 vs v2 on the same registry copy.
Source results/ops_goal_amendment/sandbox_forward.json.
### A calibrated near-certain forward SPRT on a null usually kills, rarely accepts
iid placebo over 36 months: accept .002, kill .80 -- the forward test is mostly a decay detector.
Source tests/test_edgelab_amendment.py helpers (rng 99).

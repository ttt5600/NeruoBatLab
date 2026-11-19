"""edgelab -- the enforced evaluation harness for sports/GOAL.md.

    cd sports && ../.venv_quant/bin/python -m edgelab status
    python -m edgelab register edgelab/specs/<spec>.yaml
    python -m edgelab evaluate <trial_id> --split dev|sealed|forward
    python -m edgelab selftest
    python -m edgelab evidence <strategy> --out f.json

Modules: registry (append-only, hash-chained trial log), vault (date splits + access log + unseal gate), schema
(bet rows + point-in-time checks), fees, evaluator (GOAL.md rule 5), ladder (rule 6 state machine), selftest (rule 8),
migrate (existing strategies -> registry), evidence (hub verifier JSON).
"""
ALPHA = 0.05

# Settlement scoring version (ops_edgelab_scalar 2026-11-15; notes/ops_edgelab_scalar.md). GOAL-level switch: the USER
# flips it, never an agent.
#   1 = legacy (every registry verdict through 2026-11-15): outcome in {0, 0.5, 1}; 0.5 = push (stake back); a Kalshi
#       market settled `scalar` resolves to 0 for BOTH sides under result_equals_side.
#   2 = settlement-value scoring: a bet whose hidden result is 'scalar' pays the venue's settlement value for our side
#       (YES v, NO 1 - v; v = the dataset's outcome `value_col`, a hidden column), P&L = contracts x (value - price)
#       - fee. Binary rows are bit-identical to v1. Evaluations under v2 carry `scoring_version: 2` in the registry.
SCORING_VERSION = 2   # flipped 2026-11-18 on the user's approval (GOAL.md rule 5 amendment)

# Bar version (ops_goal_amendment 2026-11-18; notes/ops_goal_amendment.md). GOAL-level switch: the USER flips it.
#   1 = rules 5-7 as written 2026-11-02/05: every stage judged at the rule-2 family bar; concentration = 30% profit-share
#       cap; Sharpe > 5 audit trigger for all trials; live_small needs >= 100 forward bets passing rule 5.
#   2 = GOAL.md rules 5/6/7 amendment (user, 2026-11-18): sealed bar .05/2^j; concentration = ROI > 0 after dropping the
#       top week and, separately, the top cluster; no Sharpe trigger at mean cost >= .90; forward promotion by a
#       pre-registered sequential test (edgelab/forward_test.py) whose H1 acceptance -> needs_audit; its H0 crossing is
#       a kill added to the pre-set kills. Evaluations under v2 carry `bar_version: 2` in the registry.
BAR_VERSION = 2   # adopted 2026-11-18 on the user's approval ("approval on everything")

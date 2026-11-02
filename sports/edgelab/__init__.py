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

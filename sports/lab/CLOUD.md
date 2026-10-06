# Running lab agents in the cloud

Cloud agents work on a fresh clone of branch `worktree-quant-trader`. They
get the DEV split only (`lab/devdata/`, committed); the lockbox never leaves
the local machine, so no cloud agent can see it even by accident.

Their results are not trusted as-is. Back on the local machine:

```
python lab/registry.py merge lab/registry/incoming/<TOPIC>.jsonl      # re-validated, deduped, re-numbered
python lab/run.py merge-ledger lab/ledgers/<AGENT>.jsonl              # their trials COUNT toward the bar
python lab/run.py dev lab/strategies/<file>.py                        # local re-score is the record
```

## Setup (first thing in every cloud agent)

```
cd sports
pip install -q numpy pandas scipy scikit-learn pyarrow requests
```

Use `python3` wherever the standing orders say `../.venv_quant/bin/python`.

## Where a cloud agent writes

| agent | writes | via |
|---|---|---|
| research, topic T | `lab/registry/incoming/T.jsonl` | `LAB_REGISTRY=lab/registry/incoming/T.jsonl python3 lab/registry.py add '<json>'` |
| strategy, name A | `lab/strategies/A_*.py`, `lab/ledgers/A.jsonl` | `LAB_LEDGER=lab/ledgers/A.jsonl python3 lab/run.py dev lab/strategies/A_x.py` |

Read the shared registry and ledger first (`python3 lab/registry.py list`,
`python3 lab/run.py ledger`) so you do not duplicate what exists.

## Final message (required)

Your final message is how results come home. Include, verbatim, the full
contents of every file you created, each in a fenced block headed by its
path. Then your report. If you also can commit and push to a new branch
named `lab-cloud/<your agent name>`, do so and say which branch.

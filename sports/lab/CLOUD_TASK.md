# Cloud task card

You were started with one line naming your ROLE and ID. Do this:

1. `git fetch origin worktree-quant-trader && git checkout worktree-quant-trader`
2. `cd sports && pip install -q numpy pandas scipy scikit-learn pyarrow requests` (use `python3`)
3. Read `lab/CLOUD.md`.

## ROLE research, ID = topic id (e.g. T10)

- Your topic text is the entry with that id in `lab/registry/topics.json`.
- Follow `lab/RESEARCH_AGENT.md` exactly.
- Run `python3 lab/registry.py list` first; do not duplicate existing entries.
- Add entries ONLY via `LAB_REGISTRY=lab/registry/incoming/<ID>.jsonl python3 lab/registry.py add '<one-line JSON>'`
- Report: `lab/reports/cloud-<ID>-research.md`. Do NOT run `registry.py topic-done`.
- Files to return: `lab/registry/incoming/<ID>.jsonl`, the report.

## ROLE strategy, ID = agent name (e.g. sides)

- Your assigned registry entries are listed after your ID in the start line.
- Follow `lab/STRATEGY_AGENT.md` exactly; write up to 4 strategies, one principled version each.
- Read `python3 lab/run.py ledger` and `python3 lab/registry.py list --testable` first.
- Name files `lab/strategies/<ID>_<short>.py`.
- Score ONLY via `LAB_LEDGER=lab/ledgers/<ID>.jsonl python3 lab/run.py dev lab/strategies/<ID>_<short>.py`
  -- every run is appended and counted; never delete ledger lines.
- Report: `lab/reports/cloud-<ID>-strategy.md`, with the harness's numbers copied verbatim.
- Files to return: every strategy file, `lab/ledgers/<ID>.jsonl`, the report.

## Finish (both roles)

Commit your files and push to a new branch `lab-cloud/<ROLE>-<ID>` (or any branch
you are permitted to push; say which). Your FINAL MESSAGE must include, verbatim
in fenced blocks headed by path, the full contents of every file listed under
"Files to return". Never invent a URL, study, author or number.

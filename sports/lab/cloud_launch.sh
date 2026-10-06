#!/bin/bash
# Launch ONE research topic as a claude.ai/code cloud session (billed to cloud
# session credits, not local usage). Prints whatever the CLI returns -- the
# session URL -- so results can be collected later from the pushed branch.
#   bash lab/cloud_launch.sh T08
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
T=${1:?topic id}
TOPIC=$(../.venv_quant/bin/python -c "import json; print(next(t['topic'] for t in json.load(open('lab/registry/topics.json')) if t['id']=='$T'))")
PROMPT="Research agent for a sports-betting lab. Work on branch worktree-quant-trader of this repo.
Read sports/lab/CLOUD.md and sports/lab/RESEARCH_AGENT.md first and follow them exactly.
Agent name: research-$T. Topic $T: $TOPIC
Write registry entries ONLY via: cd sports && LAB_REGISTRY=lab/registry/incoming/$T.jsonl python3 lab/registry.py add '<one-line JSON>'
Run python3 lab/registry.py list first to avoid duplicating the existing entries.
Report to sports/lab/reports/cloud-$T-research.md. Do NOT run registry.py topic-done.
When done, commit both files and push to a new branch lab-cloud/research-$T. Never invent a URL, study, author or number."
claude --cloud "$PROMPT" < /dev/null 2>&1 &
pid=$!
for _ in $(seq 1 45); do sleep 2; kill -0 $pid 2>/dev/null || break; done
kill $pid 2>/dev/null && echo "(cli still attached after 90s; detached it -- the cloud session keeps running)"

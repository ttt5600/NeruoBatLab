#!/bin/bash
# Run agent iterations for the betting lab.
#   bash lab/loop.sh research [N]     # N research topics (default 1)
#   bash lab/loop.sh strategy [N] [K] # N strategy iterations, K strategies each (default 1, 3)
# Env: LAB_MODEL (default sonnet), LAB_MAX_PER_DAY (default 12)
#
# Agents run headless with a TOOL ALLOWLIST, not bypassPermissions: research
# agents can search the web and write to the registry only through its CLI;
# strategy agents can write strategy files and run the dev scorer, nothing
# else. The lockbox command is not on either list.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1          # -> sports/
MODE=${1:?research|strategy}
N=${2:-1}
K=${3:-3}
MODEL=${LAB_MODEL:-sonnet}
MAX=${LAB_MAX_PER_DAY:-12}
PY="../.venv_quant/bin/python"
mkdir -p lab/logs lab/reports
LOCK=lab/.loop.lock
if [ -s "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
    echo "a lab loop is already running (pid $(cat "$LOCK"))"; exit 0
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

for i in $(seq 1 "$N"); do
    today=$(ls lab/logs 2>/dev/null | grep -c "^$(date +%Y%m%d)-") || today=0
    if [ "$today" -ge "$MAX" ]; then echo "daily cap $MAX reached"; break; fi
    STAMP=$(date +%Y%m%d-%H%M%S)
    REPORT=lab/reports/$STAMP-$MODE.md
    LOG=lab/logs/$STAMP-$MODE.log
    if [ "$MODE" = research ]; then
        TOPIC=$($PY -c "import sys; sys.path.insert(0,'lab'); import registry as r; t=r.next_topic(); print(f\"{t['id']}: {t['topic']}\" if t else '')")
        [ -z "$TOPIC" ] && { echo "no pending topics"; break; }
        PROMPT="$(cat lab/RESEARCH_AGENT.md)

## This iteration
Topic $TOPIC
Report file: $REPORT"
        TOOLS=(WebSearch WebFetch Read "Bash($PY lab/registry.py:*)" "Bash($PY lab/run.py schema:*)"
               "Edit($REPORT)")
    else
        PROMPT="$(cat lab/STRATEGY_AGENT.md)

## This iteration
Strategies to write and test: up to $K
Report file: $REPORT"
        TOOLS=(Read Glob Grep "Edit(lab/strategies/**)" "Edit($REPORT)"
               "Bash($PY lab/registry.py list:*)" "Bash($PY lab/run.py dev:*)"
               "Bash($PY lab/run.py ledger:*)" "Bash($PY lab/run.py schema:*)")
    fi
    echo "$(date '+%F %T') $MODE iteration $i/$N -> $REPORT"
    claude -p --model "$MODEL" --allowedTools "${TOOLS[@]}" -- "$PROMPT" > "$LOG" 2>&1
    echo "  exit $? ; report $( [ -s "$REPORT" ] && echo written || echo MISSING )"
done

#!/bin/bash
# Start ONE research-agent iteration: headless Claude Code with harness/AGENT.md as its orders.
# tick.py calls this when there is something to act on; you can also run it by hand:
#   bash harness/run_agent.sh "manual: submit E2"
# One agent at a time (lock file), at most MAX_RUNS_PER_DAY per calendar day (cost bound).
set -uo pipefail
H=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$H")
MAX_RUNS_PER_DAY=${HARNESS_MAX_RUNS_PER_DAY:-6}
mkdir -p "$H/logs" "$H/reports"
REASON=${1:-manual run}

LOCK=$H/.agent.lock
if [ -s "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
    echo "$(date '+%F %T') agent already running (pid $(cat "$LOCK")); not starting: $REASON"
    exit 0
fi
today=$(ls "$H/logs" 2>/dev/null | grep -c "^agent-$(date +%Y%m%d)-") || today=0
if [ "$today" -ge "$MAX_RUNS_PER_DAY" ]; then
    echo "$(date '+%F %T') daily cap reached ($today/$MAX_RUNS_PER_DAY); not starting: $REASON"
    exit 0
fi

STAMP=$(date +%Y%m%d-%H%M%S)
LOG=$H/logs/agent-$STAMP.log
REPORT=harness/reports/$STAMP.md
PROMPT="$(cat "$H/AGENT.md")

## This iteration
Started $(date '+%Y-%m-%d %H:%M %Z') because: $REASON
Write your report to $REPORT"

nohup bash -c '
    echo $$ > "$1"
    cd "$2" || exit 1
    claude -p --permission-mode bypassPermissions ${HARNESS_MODEL:+--model "$HARNESS_MODEL"} "$3"
    rc=$?
    rm -f "$1"
    msg=$(grep -m1 -v "^#" "$2/$4" 2>/dev/null | cut -c1-150 | tr "\"" " ")
    osascript -e "display notification \"${msg:-no report written (exit $rc)}\" with title \"Research agent finished\"" 2>/dev/null
' _ "$LOCK" "$ROOT" "$PROMPT" "$REPORT" > "$LOG" 2>&1 < /dev/null &
echo "$(date '+%F %T') agent started (log $LOG): $REASON"

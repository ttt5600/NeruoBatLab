#!/bin/bash
# Daily paper step, run by launchd at 14:30 local (17:30 New York) on weekdays.
# Paper only: no broker, no credentials. A non-zero exit raises a notification,
# because the failure that cost three weeks of forward evidence exited 0.
cd "$(dirname "$0")/.." || exit 1
mkdir -p results/logs
LOG=results/logs/paper_step.log
{ echo "=== $(date -u +%FT%TZ)"; ../.venv_quant/bin/python scripts/paper_run.py step; } >>"$LOG" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "exit $rc" >>"$LOG"
  osascript -e "display notification \"paper step failed (exit $rc); see quant/$LOG\" with title \"quant paper\"" 2>/dev/null
fi
exit $rc

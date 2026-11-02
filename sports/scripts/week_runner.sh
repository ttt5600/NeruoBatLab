#!/bin/bash
# Week runner (until launchd deploy): every 10 min 08:00-21:30 local -> NBA early-form tick, L32b tick, daily ticket.
# Paper only. Prevents idle sleep only during game hours. Ends 2026-11-08 23:59.
cd /Users/jonathanwang/Desktop/vocalizations_lab/.claude/worktrees/quant-trader/sports
PY=../.venv_quant/bin/python
end=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-08 23:59" +%s)
last_refresh=""
while [ "$(date +%s)" -lt "$end" ]; do
  h=$(date +%H%M); d=$(date +%Y-%m-%d)
  if [ "$h" -ge 0800 ] && [ "$h" -le 2130 ]; then
    echo "== $d $(date +%H:%M)"
    if [ "$last_refresh" != "$d" ]; then
      $PY scripts/nba_early_form_logger.py tick 2>&1 | grep -E "LOGGED|Error|Traceback" ; last_refresh=$d
    else
      $PY scripts/nba_early_form_logger.py tick --no-refresh 2>&1 | grep -E "LOGGED|Error|Traceback"
    fi
    $PY scripts/l32b_logger.py tick 2>&1 | tail -2
    $PY scripts/daily_ticket.py --date "$d" > /dev/null 2>&1
    caffeinate -i -t 600
  else
    sleep 600
  fi
done
echo "== runner done $(date)"

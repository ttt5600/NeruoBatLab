#!/bin/bash
# Kalshi prop-ladder paper logger runner (until launchd deploy): tick every 10 min 08:00-21:30, grade at the first
# tick of each day. Paper only (notes/kalshi_prop_ladders.md). Ends 2026-11-08 23:59.
cd /Users/jonathanwang/Desktop/vocalizations_lab/.claude/worktrees/quant-trader/sports
PY=../.venv_quant/bin/python
end=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-08 23:59" +%s)
graded=""
while [ "$(date +%s)" -lt "$end" ]; do
  h=$(date +%H%M); d=$(date +%Y-%m-%d)
  if [ "$h" -ge 0800 ] && [ "$h" -le 2130 ]; then
    echo "== $d $(date +%H:%M)"
    if [ "$graded" != "$d" ]; then $PY scripts/kalshi_prop_logger.py grade 2>&1 | tail -2; graded=$d; fi
    $PY scripts/kalshi_prop_logger.py tick 2>&1 | tail -3
  fi
  sleep 600
done

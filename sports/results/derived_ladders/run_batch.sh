#!/bin/bash
cd /Users/jonathanwang/Desktop/vocalizations_lab/.claude/worktrees/quant-trader/sports
for s in "$@"; do ../.venv_quant/bin/python -u results/derived_ladders/fetch_ev.py $s --max-events 800 || echo "FAIL $s"; done
echo BATCHDONE

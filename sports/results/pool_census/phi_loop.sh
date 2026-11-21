#!/bin/bash
# Poll week_plan.json; run tape_phi on series measured since the last pass (top by measured tail first). Stops when
# run_week.log shows STOP/done and every planned series has a phi row (or after MAXITER passes).
cd "$(dirname "$0")"
PY=../../../.venv_quant/bin/python
for i in $(seq 1 200); do
  SER=$($PY - <<'PYEOF'
import json, os, collections
plan = json.load(open('week_plan.json')) if os.path.exists('week_plan.json') else {}
phi = json.load(open('phi_week.json')) if os.path.exists('phi_week.json') else {}
tail = collections.Counter()
for l in open('week_markets.jsonl') if os.path.exists('week_markets.jsonl') else []:
    x = json.loads(l); tail[x['series']] += x['tail'] + x['mirror']
todo = [s for s in plan if s not in phi and tail[s] > 0]
todo.sort(key=lambda s: -tail[s])
print(','.join(todo[:5]))
PYEOF
)
  if [ -n "$SER" ]; then $PY tape_phi.py week_markets.jsonl "$SER" phi_week.json; else
    if grep -qE "STOP|done|Traceback" run_week.log; then echo "phi_loop finished"; break; fi; sleep 30; fi
done

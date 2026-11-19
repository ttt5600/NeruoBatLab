"""SANDBOX re-score: copy the real registry to a temp file, re-evaluate already-evaluated (trial, dev) pairs under
SCORING_VERSION 1 and 2. The real registry and lab/edgelab_runs are never written (save_rows=False, temp registry).
ops_edgelab_v2_apply 2026-11-18: same as results/ops_edgelab_scalar/sandbox_rescore.py but with NO monkeypatches -- uses
the APPLIED loader + vault.yaml (scalar_value / sv value_cols). Default targets = every dataset with a dev evaluation.
Run from sports/: ../.venv_quant/bin/python results/ops_edgelab_v2_apply/sandbox_rescore_applied.py"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, '.')
from edgelab import loaders, paths, run, vault  # noqa: E402
from edgelab.registry import Registry  # noqa: E402

REAL = paths.REGISTRY
real_sha = __import__('hashlib').sha256(REAL.read_bytes()).hexdigest()

ev = [json.loads(l) for l in REAL.read_text().splitlines()]
spec = {e['trial_id']: e['spec'] for e in ev if e['event'] == 'register'}
recorded = {}
for e in ev:
    if e['event'] == 'evaluate' and e['split'] == 'dev' and e.get('verdict') in ('pass', 'fail', 'needs_audit'):
        recorded[e['trial_id']] = e      # last dev evaluation per trial
targets = sys.argv[1:] or sorted({spec[t]['dataset'] for t in recorded})
out = []
with tempfile.TemporaryDirectory() as td:
    sb = Path(td) / 'reg.jsonl'
    shutil.copy(REAL, sb)
    reg = Registry(sb)
    for tid, rec in recorded.items():
        if spec[tid]['dataset'] not in targets:
            continue
        row = {'trial': spec[tid]['id'], 'dataset': spec[tid]['dataset'], 'registry': {
            k: rec.get(k) for k in ('verdict', 'n', 'net_roi', 'ci', 'p', 'bar', 'reasons')}}
        for v in (1, 2):
          try:
            r = run.evaluate(tid, 'dev', reg, save_rows=False, scoring_version=v)
          except Exception as exc:
            row[f'v{v}'] = {'error': repr(exc)[:300]}; print(spec[tid]['id'], v, 'ERROR', repr(exc)[:200], flush=True); continue
          row[f'v{v}'] = {k: r.get(k) for k in ('verdict', 'n', 'net_roi', 'ci', 'p', 'bar', 'half1', 'half2',
                                                  'losses', 'reasons', 'scoring_version')}
        if 'error' in row['v1'] or 'error' in row['v2']:
            row['v1_reproduces_registry'] = row['verdict_changes'] = None; out.append(row); continue
        row['v1_reproduces_registry'] = (row['v1']['verdict'] == rec['verdict'] and row['v1']['n'] == rec['n']
                                         and (row['v1']['net_roi'] == rec['net_roi'] or abs((row['v1']['net_roi'] or 0) - (rec['net_roi'] or 0)) < 1e-12))
        row['verdict_changes'] = row['v2']['verdict'] != rec['verdict']
        out.append(row)
        print(json.dumps({k: row[k] for k in ('trial', 'v1_reproduces_registry', 'verdict_changes')}),
              'v1', row['v1']['verdict'], row['v1']['net_roi'], row['v1']['ci'],
              'v2', row['v2']['verdict'], row['v2']['net_roi'], row['v2']['ci'],
              row['v2']['reasons'], flush=True)
assert __import__('hashlib').sha256(REAL.read_bytes()).hexdigest() == real_sha, 'REAL REGISTRY CHANGED'
Path('results/ops_edgelab_v2_apply/sandbox_rescore_applied.json').write_text(json.dumps(out, indent=1, default=str))
print('real registry untouched:', real_sha[:12])

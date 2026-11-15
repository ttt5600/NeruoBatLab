"""SANDBOX re-score: copy the real registry to a temp file, re-evaluate already-evaluated (trial, dev) pairs under
SCORING_VERSION 1 and 2. The real registry and lab/edgelab_runs are never written (save_rows=False, temp registry).
kalshi_politics_fav gets its settlement values the way the proposed apply step would add them (hidden `scalar_value`
from data/pol_favourites/scalar_values.json) -- monkeypatched here only.
Run from sports/: ../.venv_quant/bin/python results/ops_edgelab_scalar/sandbox_rescore.py"""
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

_cfg = vault.config


def cfg_patched():
    c = _cfg()
    p = c['kalshi_politics_fav']
    p['hidden'] = list(p['hidden']) + ['scalar_value']
    p['outcome'] = {**p['outcome'], 'value_col': 'scalar_value'}
    return c


_pol = loaders.LOADERS['kalshi_politics_fav']


def pol_patched():
    x, src = _pol()
    f = paths.DATA / 'pol_favourites' / 'scalar_values.json'
    sv = json.loads(f.read_text())
    x['scalar_value'] = pd.to_numeric(x['ticker'].map(sv), errors='coerce')
    return x, src + [str(f)]


vault.config = cfg_patched
loaders.LOADERS['kalshi_politics_fav'] = pol_patched

ev = [json.loads(l) for l in REAL.read_text().splitlines()]
spec = {e['trial_id']: e['spec'] for e in ev if e['event'] == 'register'}
recorded = {}
for e in ev:
    if e['event'] == 'evaluate' and e['split'] == 'dev' and e.get('verdict') in ('pass', 'fail', 'needs_audit'):
        recorded[e['trial_id']] = e      # last dev evaluation per trial
targets = sys.argv[1:] or ['kalshi_politics_fav', 'kalshi_tsaw', 'kalshi_spacex_count', 'kalshi_ytviews',
                           'kalshi_longdated_tail', 'kalshi_rain_daily_nyc', 'kalshi_opt_tails']
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
            r = run.evaluate(tid, 'dev', reg, save_rows=False, scoring_version=v)
            row[f'v{v}'] = {k: r.get(k) for k in ('verdict', 'n', 'net_roi', 'ci', 'p', 'bar', 'half1', 'half2',
                                                  'losses', 'reasons', 'scoring_version')}
        row['v1_reproduces_registry'] = (row['v1']['verdict'] == rec['verdict'] and row['v1']['n'] == rec['n']
                                         and abs(row['v1']['net_roi'] - rec['net_roi']) < 1e-12)
        row['verdict_changes'] = row['v2']['verdict'] != rec['verdict']
        out.append(row)
        print(json.dumps({k: row[k] for k in ('trial', 'v1_reproduces_registry', 'verdict_changes')}),
              'v1', row['v1']['verdict'], round(row['v1']['net_roi'], 5), [round(c, 4) for c in row['v1']['ci']],
              'v2', row['v2']['verdict'], round(row['v2']['net_roi'], 5), [round(c, 4) for c in row['v2']['ci']],
              row['v2']['reasons'], flush=True)
assert __import__('hashlib').sha256(REAL.read_bytes()).hexdigest() == real_sha, 'REAL REGISTRY CHANGED'
Path('results/ops_edgelab_scalar/sandbox_rescore.json').write_text(json.dumps(out, indent=1, default=str))
print('real registry untouched:', real_sha[:12])

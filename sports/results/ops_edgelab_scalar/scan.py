"""Scan raw dataset rows (only splits already evaluated in the real registry) for non-binary settlements.
Run from sports/: ../.venv_quant/bin/python results/ops_edgelab_scalar/scan.py"""
import collections
import json
import sys

import pandas as pd

sys.path.insert(0, '.')
from edgelab import loaders, vault  # noqa: E402

ev = [json.loads(l) for l in open('lab/edgelab_registry.jsonl')]
tr = {e['trial_id']: e['spec'] for e in ev if e['event'] == 'register'}
pairs = collections.defaultdict(set)
for e in ev:
    if e['event'] == 'evaluate' and not e.get('forward_adapter'):
        pairs[tr[e['trial_id']]['dataset']].add(e['split'])
cfg = vault.config()


def raw(ds):
    """Pre-filter frames for loaders that drop non yes/no rows."""
    loaders._imp_scripts()
    D, S = loaders.DATA, loaders.SPORTS
    if ds == 'kalshi_tsaw':
        import e1x_tsa as T
        x = T.rungs(None); x['date'] = pd.to_datetime(x['wend']); return x
    if ds == 'kalshi_album_units':
        import album_units as A
        x = A.selected(pd.read_parquet(D / 'consensus_reject_census' / 'au_frame.parquet'))
        x['date'] = pd.to_datetime(x['wend']); return x
    if ds == 'kalshi_top_album':
        import billboard_top as B
        x = B.selected(pd.read_parquet(D / 'billboard_top' / 'ta_frame.parquet'))
        x['date'] = pd.to_datetime(x['wend']); return x
    if ds == 'kalshi_opt_tails':
        x = pd.read_parquet(D / 'opt_tails' / 'frame.parquet'); x['date'] = pd.to_datetime(x['date']); return x
    if ds == 'kalshi_wx_monthly_early':
        x = pd.read_parquet(S / 'results' / 'l39_expansion' / 'p1m_frame.parquet'); x['date'] = pd.to_datetime(x['month'])
        return x
    if ds == 'kalshi_golf_topn':
        x = pd.read_parquet(S / 'results' / 'rules_quirks' / 'golf_edgelab_frame.parquet')
        x['date'] = pd.to_datetime(x['date']); return x
    return loaders.LOADERS[cfg[ds]['loader']]()[0]


out = {}
for ds, splits in sorted(pairs.items()):
    kind = cfg[ds]['outcome']['kind']
    try:
        x = raw(ds)
    except Exception as e:  # noqa: BLE001
        out[ds] = {'error': repr(e)[:200]}
        print(ds, 'ERR', repr(e)[:200])
        continue
    for sp in sorted(splits):
        w = vault.window(ds, sp)
        if w is None:
            continue
        r = vault._split_rows(x, w)
        info = {'kind': kind, 'rows': len(r)}
        for c in ('result', 'won', 'push', 'win'):
            if c in r.columns:
                vc = r[c].astype(str).value_counts()
                info[c] = {str(k): int(v) for k, v in vc.items()} if len(vc) < 15 else f'{len(vc)} distinct'
        for c in ('settlement_value_dollars', 'sv', 'settlement_value', 'settlement_price'):
            if c in r.columns:
                v = pd.to_numeric(r[c], errors='coerce')
                info[c + '_not01'] = int((v.notna() & ~v.isin([0, 1])).sum())
        out[f'{ds}:{sp}'] = info
        print(ds, sp, json.dumps(info))
json.dump(out, open('results/ops_edgelab_scalar/scan.json', 'w'), indent=1)

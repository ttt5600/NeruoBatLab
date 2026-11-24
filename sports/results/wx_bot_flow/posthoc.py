"""wx_bot_flow POST HOC diagnostics (not pre-registered; descriptive only). ../.venv_quant/bin/python results/wx_bot_flow/posthoc.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, 'scripts'); sys.path.insert(0, str(Path.home() / 'agent-harness' / 'lib'))
import wx_bot_flow_score as S, traps
R = Path('results/wx_bot_flow')
f = pd.read_parquet(R / 'frame.parquet')
base = f[(f.ask16 >= .03) & (f.ask16 <= .90) & f.p12.notna() & f.p06.notna()].copy()
out = {}
# (a) paired after-minus-before on the same brackets
pb = base[(base.vol_after > 0) & (base.vol_before > 0)].copy(); pb['dnyf'] = pb.nyf_after - pb.nyf_before
out['a_paired_after_minus_before'], _ = S.boot(pb, ['edge12', 'edge06', 'ask16'], 'dnyf'); out['a_paired_after_minus_before']['n'] = len(pb)
# (b) bot cities (suislanchez/AwpDemon list) vs AUS/PHIL
for nm, cs in (('bot5', ['NY', 'CHI', 'MIA', 'LAX', 'DEN']), ('aus_phil', ['AUS', 'PHIL'])):
    A = base[(base.vol_after > 0) & base.city.isin(cs)]
    out[f'b_{nm}'], _ = S.boot(A, ['edge12', 'edge06', 'ask16'], 'nyf_after'); out[f'b_{nm}']['n'] = len(A)
# (c) small prints only (<= 20 contracts: the $2-$100 bot sizes), after window
t = pd.read_parquet(R / 'trades.parquet'); t['ts'] = traps.epoch_seconds(t.created_time)
t = t.merge(f[['ticker', 'sigdate']], on='ticker'); t['hr'] = (t.ts - traps.epoch_seconds(t.sigdate.dt.tz_localize('UTC'))) / 3600
t['signed'] = np.where(t.taker_side == 'yes', t['count'], -t['count'])
for nm, lo, hi in (('after', 16, 19), ('before', 15, 16)):
    w = t[(t.hr >= lo) & (t.hr < hi) & (t['count'] <= 20)].groupby('ticker').agg(net=('signed', 'sum'), vol=('count', 'sum'))
    base[f'snyf_{nm}'] = base.ticker.map(w.net / w.vol)
for nm in ('after', 'before'):
    A = base[base[f'snyf_{nm}'].notna()]
    out[f'c_small_{nm}'], _ = S.boot(A, ['edge12', 'edge06', 'ask16'], f'snyf_{nm}'); out[f'c_small_{nm}']['n'] = len(A)
# (d) taker side share overall in window
out['d_yes_taker_share_contracts_15_19'] = round(float(t[(t.hr >= 15) & (t.hr < 19) & (t.taker_side == 'yes')]['count'].sum() / t[(t.hr >= 15) & (t.hr < 19)]['count'].sum()), 4)
json.dump(out, open(R / 'posthoc.json', 'w'), indent=1)
for k, v in out.items(): print(k, json.dumps(v)[:330])

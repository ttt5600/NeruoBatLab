#!/usr/bin/env python3
"""E16 contrasts for finding 074 (from the critic's check, 2026-10-15): refit the bats probes at fixed layers from the
cached beans_probe.py features, then paired clip bootstraps (n 2000, seed 0) of te4 - lpte4, lpte4 - plain, the share of
the F=4 gain kept by the low-pass control, and the same at the layer-matched L0. Writes analysis/e16_contrasts.json.
Run from ~/zf_labelset/external/beans (paths are relative to the BEANS repo)."""
import numpy as np, pandas as pd, yaml, json, time, sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
print('sklearn', sklearn.__version__)
cfg={d['name']:d for d in yaml.safe_load(open('datasets.yml'))}['bats']
labels=[str(x) for x in cfg['labels']]
Y={}
for s in ('train','valid','test'):
    df=pd.read_csv(cfg[f'{s}_data']); Y[s]=np.array([labels.index(str(v)) for v in df['label']])
J=json.load(open('/Users/jonathanwang/zf_labelset/zf_detection_dataset_v1/analysis/beans_bats.json'))['models']
jobs=[('run16_compute4x',0),('run16_compute4x@te4',0),('run16_compute4x@lpte4',3),('run16_compute4x@lpte4',0),
      ('aves-base-bio',0),('aves-base-bio@te4',0),('aves-base-bio@lpte4',0)]
C={}
for tag,l in jobs:
    t0=time.time()
    E={s:np.load(f'features/bats_{s}_{tag}.npy',mmap_mode='r')[:,l].astype(np.float32) for s in Y}
    clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=4000)); clf.fit(E['train'],Y['train'])
    p=clf.predict(E['test']); c=(p==Y['test']); C[(tag,l)]=c
    print(tag,l,'test',c.mean(),'json',J[tag]['test_by_layer'][l],'%.0fs'%(time.time()-t0),flush=True)

rng=np.random.default_rng(0); idx=rng.integers(0,2000,size=(2000,2000))
OUT={'refit_test':{f'{t}|L{l}':float(v.mean()) for (t,l),v in C.items()}}
def rep(name,v):
    obs=v.mean(); d=v[idx].mean(1); lo,hi=np.percentile(d,[2.5,97.5]); print(f'{name}: {obs:+.4f} [{lo:+.4f}, {hi:+.4f}]')
    OUT[name]=dict(observed=float(obs), ci95=[float(lo),float(hi)])
for m,lchosen in (('run16_compute4x',3),('aves-base-bio',0)):
    pl=C[(m,0)].astype(float); te=C[(m+'@te4',0)].astype(float); lp=C[(m+'@lpte4',lchosen)].astype(float)
    rep(m+' lpte4-plain (chosen)',lp-pl); rep(m+' te4-lpte4 (chosen)',te-lp)
    rep(m+' stretch share minus >8k share = 2lp-pl-te',2*lp-pl-te)
    num=lp[idx].mean(1)-pl[idx].mean(1); den=te[idx].mean(1)-pl[idx].mean(1)
    sh=(lp.mean()-pl.mean())/(te.mean()-pl.mean()); ci=np.percentile(num/den,[2.5,97.5]); print(m,'share obs %.3f'%sh,'share CI',np.round(ci,3))
    OUT[m+' share kept by lpte4 (chosen)']=dict(observed=float(sh), ci95=[float(ci[0]),float(ci[1])])
    if lchosen!=0:
        lp0=C[(m+'@lpte4',0)].astype(float)
        rep(m+' L0 matched te4-lpte4',te-lp0); rep(m+' L0 matched lpte4-plain',lp0-pl); rep(m+' L0 2lp-pl-te',2*lp0-pl-te)

import pathlib
p=pathlib.Path.home()/'zf_labelset/zf_detection_dataset_v1/analysis/e16_contrasts.json'; p.write_text(json.dumps(OUT,indent=1)); print('wrote',p)

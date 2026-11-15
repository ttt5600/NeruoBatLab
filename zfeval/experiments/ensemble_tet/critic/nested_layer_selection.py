import sys, json, time
sys.path.insert(0, "/Users/jonathanwang/Desktop/vocalizations_lab/.claude/worktrees/agent-abfbebe41020c5b4e/zfeval/experiments/ensemble_tet")
import numpy as np
from joblib import Parallel, delayed
from common import task, probe, pct, FEAT
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
y, birds, classes, BOOT = task()
WB, WS, BXL, R19, E13 = "hf-whisper-base-enc", "hf-whisper-small-enc", "birdaves-bioxn-large", "run19_seed2", "run24_bioxnl12_s2"
HUB="hf-hubert-base-ls960"
E = {t: np.load(FEAT / f"ct11_{t}_cpuemb.npy", mmap_mode="r") for t in [WB, WS, BXL, R19]}
# fold identity: folds do not depend on X
outer = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros((len(y),1)), y, birds))
o2 = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.asarray(E[WB][:,0]), y, birds))
assert all((a[1]==b[1]).all() for a,b in zip(outer,o2)); print("folds X-independent OK")
for tr,te in outer: assert not set(birds[tr]) & set(birds[te])
def fit(X, tr, te):
    est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000)).fit(X[tr], y[tr])
    out = np.zeros((len(te), len(classes))); out[:, est.classes_] = est.predict_proba(X[te]); return out
def inner_acc(t, L, k):
    tr, _ = outer[k]; X = np.ascontiguousarray(E[t][:, L]); P = np.zeros((len(tr), len(classes)))
    for itr, ite in StratifiedGroupKFold(4, shuffle=True, random_state=0).split(X[tr], y[tr], birds[tr]):
        P[ite] = fit(X, tr[itr], tr[ite])
    return float((P.argmax(1) == y[tr]).mean())
t0=time.time()
jobs = [(t, L, k) for t in E for L in range(E[t].shape[1]) for k in range(5)]
acc = dict(zip(jobs, Parallel(n_jobs=4)(delayed(inner_acc)(*j) for j in jobs)))
print("inner done", time.time()-t0)
P = {}; sel = {}
for t in E:
    P[t] = np.zeros((len(y), len(classes))); sel[t] = []
    for k, (tr, te) in enumerate(outer):
        L = max(range(E[t].shape[1]), key=lambda L: acc[(t, L, k)]); sel[t].append(L)
        P[t][te] = fit(np.ascontiguousarray(E[t][:, L]), tr, te)
hit = lambda p: (p.argmax(1) == y).astype(float)
def con(pa, pb):
    ha, hb = hit(pa), hit(pb); bs = np.array([ha[s].mean()-hb[s].mean() for s in BOOT])
    return dict(a=round(ha.mean(),4), b=round(hb.mean(),4), diff=round(ha.mean()-hb.mean(),4), ci95=[round(x,4) for x in pct(bs,.05)], bonf=[round(x,4) for x in pct(bs,.05/7)])
print("nested layers", sel, {t: round(hit(P[t]).mean(),4) for t in E})
# nested comparator: per outer fold, member with best inner acc at its chosen layer
def ens_vs(mem):
    pe = np.mean([P[t] for t in mem], 0); pb = np.zeros_like(pe)
    for k,(tr,te) in enumerate(outer):
        b = max(mem, key=lambda t: acc[(t, sel[t][k], k)]); pb[te] = P[b][te]
    return con(pe, pb), {t: con(pe, P[t])['diff'] for t in mem}
for name, mem in {"E1": (WB, WS, BXL), "E2": (WB, BXL), "E3": (WB, R19), "K5wb+ws": (WB, WS)}.items():
    print(name, ens_vs(mem))
# direct family contrast at board best layers
Pb = {t: probe(t, "best", y, birds) for t in [WB, WS, BXL, R19, HUB, "run16_seed2"]}
m = lambda *ts: np.mean([Pb[t] for t in ts], 0)
print("E2 - K5 (wb+bxl vs wb+ws)", con(m(WB,BXL), m(WB,WS)))
print("E3 - K5 (wb+r19 vs wb+ws)", con(m(WB,R19), m(WB,WS)))
print("E2 - K4 (wb+bxl vs wb+hubert)", con(m(WB,BXL), m(WB,HUB)))
print("E1 vs ws (2nd member)", con(m(WB,WS,BXL), Pb[WS]))

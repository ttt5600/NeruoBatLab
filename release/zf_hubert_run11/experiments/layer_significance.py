"""Are the per-layer differences real, or noise? Paired bootstrap on the same predictions.

Eval A is grouped by recording -> cluster bootstrap over recordings.
Eval B is one continuous recording -> moving-block bootstrap (adjacent 1 s windows are
correlated; a plain window bootstrap would understate the interval).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

d = np.load("/Users/jonathanwang/.claude/jobs/63c218d9/tmp/new_dataset_feats_38678496.npz",
            allow_pickle=True)
X, y, src, grp = d["X"], d["y"], d["src"].astype(str), d["grp"].astype(str)
A, B = src == "benchmark_neg_pool", src == "soundsep_111021"
L = X.shape[1]

# regenerate the exact same predictions the job produced
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
PA = np.stack([cross_val_predict(LogisticRegression(max_iter=2000), X[A, l], y[A],
                                 groups=grp[A], cv=cv, method="predict_proba")[:, 1]
               for l in range(L)])
PB = np.stack([LogisticRegression(max_iter=2000).fit(X[A, l], y[A])
               .predict_proba(X[B, l])[:, 1] for l in range(L)])
print("A reproduced:", [round(roc_auc_score(y[A], PA[l]), 4) for l in range(L)])
print("B reproduced:", [round(roc_auc_score(y[B], PB[l]), 4) for l in range(L)])

rng = np.random.default_rng(0)
NB = 2000


def boot_cluster(P, yv, groups, n=NB):
    recs = np.unique(groups); idx = {r: np.where(groups == r)[0] for r in recs}
    out = []
    for _ in range(n):
        pick = rng.choice(recs, len(recs), replace=True)
        sel = np.concatenate([idx[r] for r in pick])
        if len(np.unique(yv[sel])) < 2:
            continue
        out.append([roc_auc_score(yv[sel], P[l][sel]) for l in range(P.shape[0])])
    return np.array(out)


def boot_block(P, yv, block=30, n=NB):
    N = len(yv); nb = int(np.ceil(N / block)); out = []
    for _ in range(n):
        st = rng.integers(0, N - block, nb)
        sel = np.concatenate([np.arange(s, s + block) for s in st])[:N]
        if len(np.unique(yv[sel])) < 2:
            continue
        out.append([roc_auc_score(yv[sel], P[l][sel]) for l in range(P.shape[0])])
    return np.array(out)


for name, BS, yv in [("A (cluster-boot over 71 recordings)", boot_cluster(PA, y[A], grp[A]), y[A]),
                     ("B (block-boot, 30 s blocks)", boot_block(PB, y[B]), y[B])]:
    print(f"\n=== eval {name}, {len(BS)} resamples ===")
    for l in range(L):
        lo, hi = np.percentile(BS[:, l], [2.5, 97.5])
        print(f"  L{l:<2} AUC {BS[:, l].mean():.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    best = int(np.argmax(BS.mean(0)))
    print(f"  best mean layer = L{best}")
    for other in range(L):
        if other == best:
            continue
        dd = BS[:, best] - BS[:, other]
        lo, hi = np.percentile(dd, [2.5, 97.5])
        sig = "SIGNIFICANT" if lo > 0 else "not distinguishable"
        print(f"    L{best} - L{other}: {dd.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  {sig}")

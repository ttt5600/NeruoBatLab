#!/usr/bin/env python
"""Where does cluster-vote accuracy actually turn over?

The first blind-selection run gave three rules a perfect score, and all three picked k=70 --
the largest k on that grid, which was also the oracle for every encoder. A rule that always
says "the biggest number you offered me" cannot be scored on a grid that never turns over.

So push k until it does. The limit is not mysterious: as k -> n_train every cluster holds one
clip and majority-vote becomes exact 1-nearest-neighbour. That 1-NN number is computed here too,
as the ceiling the curve must approach.

Only the cheap quantities are recomputed: one ward tree per fold serves every k.
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import calltype11 as C11                                                  # noqa: E402
import calltype_blindselect as BS                                         # noqa: E402

from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[2] / "paper" / "figures" / "calltype"
KS = [4, 6, 8, 11, 13, 16, 20, 25, 30, 40, 55, 70, 100, 140, 200, 300, 450, 650, 1000, 1600]
LAYER, NFOLD, SEED, B_STAB = 3, 5, 0, 10


def cut_all(Z, n, ks):
    return {k: fcluster(Z, t=k, criterion="maxclust") - 1 for k in ks if k < n}


def vote_curve(E, y, birds):
    """Leave-birds-out cluster-vote at every k, plus the 1-NN limit."""
    cv = StratifiedGroupKFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
    correct = {k: np.zeros(len(y), bool) for k in KS}
    nn_correct = np.zeros(len(y), bool)
    nclass = len(set(y))
    ntrain = []
    for tr, te in cv.split(E, y, birds):
        sc = StandardScaler().fit(E[tr])
        Xtr, Xte = sc.transform(E[tr]), sc.transform(E[te])
        ntrain.append(len(tr))
        Z = linkage(Xtr, method="ward")
        labs = cut_all(Z, len(tr), KS)
        for k, lab in labs.items():
            C = BS.centroids(Xtr, lab, k)
            maj = np.array([np.bincount(y[tr][lab == c], minlength=nclass).argmax()
                            if (lab == c).any() else np.bincount(y[tr]).argmax()
                            for c in range(k)])
            correct[k][te] = (maj[BS.assign(Xte, C)] == y[te])
        # k -> n_train limit: every clip is its own cluster
        nn_correct[te] = (y[tr][BS.assign(Xte, Xtr)] == y[te])
    out = {k: float(correct[k].mean()) for k in KS if k < min(ntrain)}
    return out, float(nn_correct.mean()), int(min(ntrain))


def stab_bird_curve(X, birds, rng):
    accum = {k: [] for k in KS}
    units = np.unique(birds)
    for _ in range(B_STAB):
        perm = rng.permutation(len(units)); h = len(units) // 2
        ia = np.isin(birds, units[perm[:h]])
        Xa, Xb = X[ia], X[~ia]
        nmin = min(len(Xa), len(Xb))
        ks = [k for k in KS if k < nmin - 5]
        la = cut_all(linkage(Xa, method="ward"), len(Xa), ks)
        lb = cut_all(linkage(Xb, method="ward"), len(Xb), ks)
        for k in ks:
            pred = BS.assign(Xb, BS.centroids(Xa, la[k], k))
            accum[k].append(float(ARI(lb[k], pred)))
    return {k: float(np.mean(v)) for k, v in accum.items() if v}


def main():
    t0 = time.time()
    y, birds, classes = BS.cohort()
    res = {"ks": KS, "layer": LAYER, "n_clips": len(y), "b_stability": B_STAB,
           "majority": float(np.bincount(y).max() / len(y)), "encoders": {}}
    names = BS.ENCODERS + ["NULL-shuffled"]
    for nm in names:
        if nm == "NULL-shuffled":
            rng0 = np.random.default_rng(1234)
            E0 = np.load(C11.FEAT / "ct11_aves_emb.npy")[:, LAYER].astype(np.float64)
            E = np.column_stack([rng0.permutation(E0[:, j]) for j in range(E0.shape[1])])
        else:
            E = np.load(C11.FEAT / f"ct11_{nm}_emb.npy")[:, LAYER].astype(np.float64)
        t = time.time()
        acc, nn, ntr = vote_curve(E, y, birds)
        stab = stab_bird_curve(StandardScaler().fit_transform(E), birds,
                               np.random.default_rng(SEED))
        kbest = max(acc, key=acc.get)
        res["encoders"][nm] = {"vote_acc": acc, "knn1": nn, "min_train": ntr,
                               "stab_bird": stab, "best_k": kbest,
                               "best_acc": acc[kbest], "seconds": round(time.time() - t, 1)}
        print(f"[{nm:20s}] {time.time()-t:5.1f}s  peak {acc[kbest]:.4f} @k={kbest}  "
              f"1-NN {nn:.4f}  stab_bird peak k={max(stab,key=stab.get)}", flush=True)
    (OUT / "calltype_kceiling.json").write_text(json.dumps(res, indent=2))
    print(f"[done] {time.time()-t0:.0f}s -> {OUT/'calltype_kceiling.json'}")


if __name__ == "__main__":
    main()

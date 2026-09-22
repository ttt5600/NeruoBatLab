#!/usr/bin/env python
"""Second calibration: swap the target and the nuisance, and see if the rule survives.

Finding 050 showed that on ZF call type, bird-held-out stability ranks five encoders in the
same order ground truth does, while every compactness criterion ranks them backwards. Its
standing caveat is that it survived exactly ONE calibration. A second corpus would help, but
the only other labelled cohort here (chick) has two call types and almost no dynamic range.

So calibrate the MECHANISM instead, on the same clips. The claim underneath 050 is not
"bird-held-out stability is magic" -- it is:

    stability computed by holding out the NUISANCE variable predicts which encoder is
    better at the TARGET.

For call type, bird is the nuisance. Invert it: make bird IDENTITY the target (48 classes),
which makes CALL TYPE the nuisance. The claim then makes two predictions that can both be
wrong:

    1. call-type-held-out stability SHOULD now rank the encoders  (nuisance held out)
    2. bird-held-out stability SHOULD now FAIL                    (it splits on the TARGET,
                                                                   destroying what we want)

Prediction 2 is the sharp one. A criterion that keeps working after you break its logic was
never measuring what the story said it was.

Split note: the call-type work used leave-birds-out, which is impossible here -- every bird
must appear in training for identity to be predictable at all. This uses a stratified random
split over clips, which is the only option, and is stated rather than hidden.
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
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score as ARI)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[2] / "paper" / "figures" / "calltype"
KS = [8, 16, 25, 48, 70, 100, 140, 200, 300, 450]
LAYER, NFOLD, SEED, B_STAB = 3, 5, 0, 12


def cohort():
    """Same clips as the call-type work, but identity is now the label."""
    rows = [r for r in C11.collect() if r[3] in C11.KEEP11]
    birds = np.array([r[1].lower() for r in rows])
    types = np.array([r[3] for r in rows])
    ub = sorted(set(birds))
    return np.array([ub.index(b) for b in birds]), types, ub


def cut_all(Z, n, ks):
    return {k: fcluster(Z, t=k, criterion="maxclust") - 1 for k in ks if k < n}


def vote_curve(E, y):
    """Stratified random split -- every bird must be in train for identity to be learnable."""
    cv = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
    correct = {k: np.zeros(len(y), bool) for k in KS}
    nclass = len(set(y))
    for tr, te in cv.split(E, y):
        sc = StandardScaler().fit(E[tr])
        Xtr, Xte = sc.transform(E[tr]), sc.transform(E[te])
        labs = cut_all(linkage(Xtr, method="ward"), len(tr), KS)
        for k, lab in labs.items():
            C = BS.centroids(Xtr, lab, k)
            maj = np.array([np.bincount(y[tr][lab == c], minlength=nclass).argmax()
                            if (lab == c).any() else np.bincount(y[tr]).argmax()
                            for c in range(k)])
            correct[k][te] = (maj[BS.assign(Xte, C)] == y[te])
    return {k: float(correct[k].mean()) for k in KS}


def held_out_stability(X, groups, rng):
    """Cluster each half independently, carry one partition onto the other, score with ARI.

    `groups` is the variable split on. Holding out the NUISANCE is the intended use; holding
    out the TARGET is the negative control this script exists to run.
    """
    accum = {k: [] for k in KS}
    units = np.unique(groups)
    for _ in range(B_STAB):
        perm = rng.permutation(len(units)); h = len(units) // 2
        ia = np.isin(groups, units[perm[:h]])
        Xa, Xb = X[ia], X[~ia]
        nmin = min(len(Xa), len(Xb))
        ks = [k for k in KS if k < nmin - 5]
        if not ks:
            continue
        la = cut_all(linkage(Xa, method="ward"), len(Xa), ks)
        lb = cut_all(linkage(Xb, method="ward"), len(Xb), ks)
        for k in ks:
            pred = BS.assign(Xb, BS.centroids(Xa, la[k], k))
            accum[k].append(float(ARI(lb[k], pred)))
    return {k: float(np.mean(v)) for k, v in accum.items() if v}


def compactness(X, labs):
    return ({k: float(silhouette_score(X, l)) for k, l in labs.items()},
            {k: float(calinski_harabasz_score(X, l)) for k, l in labs.items()},
            {k: float(-davies_bouldin_score(X, l)) for k, l in labs.items()})


def main():
    t0 = time.time()
    y, types, ub = cohort()
    print(f"[cohort] {len(y)} clips, TARGET = {len(ub)} bird identities, "
          f"NUISANCE = {len(set(types))} call types, layer {LAYER}, ward", flush=True)
    res = {"ks": KS, "layer": LAYER, "n_clips": len(y), "n_classes": len(ub),
           "target": "bird identity", "nuisance": "call type",
           "split": f"stratified random KFold({NFOLD}) -- leave-birds-out is impossible here",
           "majority": float(np.bincount(y).max() / len(y)),
           "b_stability": B_STAB, "encoders": {}}
    names = BS.ENCODERS + ["NULL-shuffled"]
    for nm in names:
        if nm == "NULL-shuffled":
            r0 = np.random.default_rng(1234)
            E0 = np.load(C11.FEAT / "ct11_aves_emb.npy")[:, LAYER].astype(np.float64)
            E = np.column_stack([r0.permutation(E0[:, j]) for j in range(E0.shape[1])])
        else:
            E = np.load(C11.FEAT / f"ct11_{nm}_emb.npy")[:, LAYER].astype(np.float64)
        t = time.time()
        X = StandardScaler().fit_transform(E)
        labs = cut_all(linkage(X, method="ward"), len(X), KS)
        sil, ch, db = compactness(X, labs)
        rng = np.random.default_rng(SEED)
        st_type = held_out_stability(X, types, rng)          # nuisance held out -- should work
        st_bird = held_out_stability(X, y, np.random.default_rng(SEED))  # TARGET held out -- control
        acc = vote_curve(E, y)
        res["encoders"][nm] = dict(
            criteria=dict(silhouette=sil, calinski_harabasz=ch, neg_davies_bouldin=db,
                          stab_calltype_heldout=st_type, stab_bird_heldout=st_bird),
            truth=dict(vote_acc=acc), seconds=round(time.time() - t, 1))
        kb = max(acc, key=acc.get)
        print(f"[{nm:20s}] {time.time()-t:5.1f}s  identity acc peak {acc[kb]:.4f} @k={kb}  "
              f"stab_type peak k={max(st_type, key=st_type.get)}  "
              f"stab_bird peak k={max(st_bird, key=st_bird.get)}", flush=True)
    (OUT / "identity_blindselect.json").write_text(json.dumps(res, indent=2))
    print(f"[done] {time.time()-t0:.0f}s -> {OUT/'identity_blindselect.json'}")


if __name__ == "__main__":
    main()

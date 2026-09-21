#!/usr/bin/env python
"""Unsupervised classification: cluster without labels, then let each cluster vote.

AMI says the discovered groups line up with call type, but it is not an accuracy and cannot be
compared with the 0.8118 / 0.8453 the supervised probe reaches. This closes that gap.

The protocol, which is label-free everywhere it matters:

  1. split by BIRD (the same leave-birds-out folds the probe uses, so the numbers are comparable)
  2. cluster the TRAIN embeddings only -- no labels touch the clustering
  3. choose k by SILHOUETTE on the train clusters -- still no labels
  4. give each cluster the majority TRAIN label (this is the only place labels enter, and it is
     one integer per cluster rather than 768 fitted weights per class)
  5. assign each TEST clip to its nearest train centroid and predict that cluster's label
  6. score accuracy on TEST

Step 4 is why this is honest but not free: a cluster-majority vote is still supervision, just an
extremely cheap one. The right reading is "how much of the probe's accuracy is already sitting in
the geometry, recoverable with one label per cluster rather than a trained decoder".

Reported against three references: the supervised probe, the majority-class floor, and the
oracle-k version (k chosen on test accuracy) which bounds how much the silhouette choice costs.
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import calltype11 as C11                                                  # noqa: E402

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score as AMI
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[2] / "paper" / "figures" / "calltype"
NFOLD, SEED = 5, 0
KS = [4, 6, 8, 11, 13, 16, 20, 25, 30, 40, 55, 70]


def cohort():
    rows = [r for r in C11.collect() if r[3] in C11.KEEP11]
    birds = np.array([r[1].lower() for r in rows])
    tt = np.array([r[3] for r in rows])
    classes = sorted(set(tt))
    return np.array([classes.index(t) for t in tt]), birds, classes


def centroids(X, lab, k):
    return np.vstack([X[lab == c].mean(0) if (lab == c).any() else np.zeros(X.shape[1])
                      for c in range(k)])


def assign(X, C):
    # nearest centroid, squared euclidean
    d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1) if X.shape[0] * C.shape[0] < 4_000_000 \
        else np.stack([((X - c) ** 2).sum(1) for c in C], 1)
    return d.argmin(1)


def run(E, y, birds, algo="ward"):
    """Per fold: cluster train, pick k by silhouette, majority-vote, score test."""
    cv = StratifiedGroupKFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
    pred_sil = np.full(len(y), -1)          # prediction under silhouette-chosen k
    per_k_correct = {k: np.zeros(len(y), bool) for k in KS}
    chosen = []
    for tr, te in cv.split(E, y, birds):
        sc = StandardScaler().fit(E[tr])
        Xtr, Xte = sc.transform(E[tr]), sc.transform(E[te])
        best_k, best_s, cache = None, -np.inf, {}
        for k in KS:
            if algo == "ward":
                lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Xtr)
            else:
                lab = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(Xtr)
            C = centroids(Xtr, lab, k)
            # majority TRAIN label per cluster; empty clusters fall back to the global majority
            maj = np.array([np.bincount(y[tr][lab == c], minlength=len(set(y))).argmax()
                            if (lab == c).any() else np.bincount(y[tr]).argmax()
                            for c in range(k)])
            p = maj[assign(Xte, C)]
            cache[k] = p
            per_k_correct[k][te] = (p == y[te])
            s = silhouette_score(Xtr, lab)          # label-free model selection
            if s > best_s:
                best_s, best_k = s, k
        chosen.append(best_k)
        pred_sil[te] = cache[best_k]
    acc_k = {k: float(per_k_correct[k].mean()) for k in KS}
    return dict(acc_silhouette=float((pred_sil == y).mean()),
                k_chosen=chosen,
                acc_by_k=acc_k,
                best_k_oracle=int(max(acc_k, key=acc_k.get)),
                acc_oracle=float(max(acc_k.values())))


def main():
    t0 = time.time()
    y, birds, classes = cohort()
    n = len(y)
    maj = float(np.bincount(y).max() / n)
    print(f"[cohort] {n} clips, {len(classes)} classes, {len(set(birds))} birds, majority {maj:.4f}")
    probe = {"run11": 0.8118405627198124, "aves": 0.8452520515826495}   # calltype11.json, L3 both
    res = {"n_clips": n, "classes": classes, "majority": maj, "ks": KS,
           "probe_acc_reference": probe, "split": f"leave-birds-out StratifiedGroupKFold({NFOLD})",
           "note": "clustering and k-selection use NO labels; each cluster gets one majority "
                   "train label, then test clips are assigned by nearest train centroid"}
    for nm in ("run11", "aves"):
        E_all = np.load(C11.FEAT / f"ct11_{nm}_emb.npy")
        assert E_all.shape[0] == n
        res[nm] = {}
        for algo in ("ward", "kmeans"):
            # layer 3 = the probe's layer, so the comparison is like-for-like
            r = run(E_all[:, 3], y, birds, algo)
            res[nm][algo] = r
            print(f"  {nm:6s} {algo:7s} silhouette-k {r['acc_silhouette']:.4f} "
                  f"(k chosen {r['k_chosen']})  oracle-k {r['acc_oracle']:.4f} "
                  f"@k={r['best_k_oracle']}  probe {probe[nm]:.4f}", flush=True)
    (OUT / "calltype_clusterlabel.json").write_text(json.dumps(res, indent=2))
    print(f"[done] {time.time()-t0:.0f}s -> {OUT/'calltype_clusterlabel.json'}")


if __name__ == "__main__":
    main()

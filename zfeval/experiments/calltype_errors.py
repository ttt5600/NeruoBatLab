#!/usr/bin/env python
"""Per-clip anatomy of the 11-class call-type probe: what each clip was called, how confidently, and
where it sits relative to every call type's cluster.

For each model it re-runs the board's probe (cv_acc: leave-birds-out StratifiedGroupKFold(5), seed 0,
StandardScaler + LogisticRegression) at the model's own reported best layer and asserts that the
overall accuracy reproduces <tag>_calltype.json exactly, so every per-clip record below belongs to
the same predictions the board scores. Nothing is re-encoded; it reads features/ct11_<tag>_emb.npy.

Per clip, all out of fold (a clip's fold never contributed to its scaler, probe or centroids):
  P[i, c]   probe probability of call type c
  D[i, c]   Euclidean distance, in the training fold's standardised space, from the clip to the
            centroid of call type c computed on the training fold
  R[i, c]   D[i, c] / radius_c, where radius_c is the median distance of call type c's own training
            members to their centroid. R = 1 means "as far from c's centre as a typical c".
  D_lda, R_lda  the same, in a 10-d shrinkage-LDA space fitted on the training fold: the geometry
            in which call types are pulled apart, rather than the raw embedding's
  fold[i]   which of the 5 folds the clip was tested in

    python zfeval/experiments/calltype_errors.py [--tags a,b,c]  -> analysis/calltype_errors.npz
"""
from __future__ import annotations
import argparse, json, sys, time, warnings
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
from calltype11 import collect, KEEP11                                     # noqa: E402
from aves_calltype import cv_acc, NFOLD, SEED                              # noqa: E402

FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
OUT = ANA / "calltype_errors.npz"
CACHE = {"aves-base-bio": "ct11_aves_emb.npy", "run11": "ct11_run11_emb.npy"}
DEFAULT = "aves-base-bio,daptv4two_2e4s2_step15000,run16_compute4x"


def reported(tag):
    av = json.loads((ANA / "aves_variants_calltype.json").read_text())["models"]
    if tag in av:
        return av[tag]["best_11"]["layer"], av[tag]["best_11"]["acc"]
    d = json.loads((ANA / ("run15_calltype.json" if tag == "run15_combined" else f"{tag}_calltype.json")).read_text())
    return d["best_layer11"], d["acc11"]


def _dist(Xtr, ytr, Xte, K):
    cen = np.stack([Xtr[ytr == c].mean(0) for c in range(K)])
    rad = np.array([np.median(np.linalg.norm(Xtr[ytr == c] - cen[c], axis=1)) for c in range(K)])
    d = np.linalg.norm(Xte[:, None, :] - cen[None], axis=2)
    return d, d / rad[None]


def centroid_distances(X, y, birds, K):
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    D, R, Dl, Rl = (np.zeros((len(y), K)) for _ in range(4)); fold = np.zeros(len(y), int)
    cv = StratifiedGroupKFold(n_splits=NFOLD, shuffle=True, random_state=SEED)   # cv_acc's folds
    for f, (tr, te) in enumerate(cv.split(X, y, birds)):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        D[te], R[te] = _dist(Xtr, y[tr], Xte, K)
        lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(Xtr, y[tr])
        Dl[te], Rl[te] = _dist(lda.transform(Xtr), y[tr], lda.transform(Xte), K)
        fold[te] = f
    return D, R, Dl, Rl, fold


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", default=DEFAULT)
    tags = [t for t in ap.parse_args().tags.split(",") if t]
    rows = [r for r in collect() if r[3] in KEEP11]
    birds = np.array([r[1].lower() for r in rows])
    tt = np.array([r[3] for r in rows]); classes = sorted(set(tt))
    y = np.array([classes.index(t) for t in tt]); K = len(classes)
    assert len(y) == 3412 and K == 11 and len(set(birds)) == 48
    out = dict(classes=np.array(classes), y=y, birds=birds,
               paths=np.array([str(r[0]) for r in rows]), source=np.array([r[4] for r in rows]),
               tags=np.array(tags))
    for t in tags:
        t0 = time.time()
        layer, acc_rep = reported(t)
        X = np.ascontiguousarray(np.load(FEAT / CACHE.get(t, f"ct11_{t}_emb.npy"), mmap_mode="r")[:, layer])
        acc, _, P = cv_acc(X, y, birds, return_proba=True)
        assert abs(acc - acc_rep) < 1e-9, f"{t}: reproduced {acc} != reported {acc_rep}"
        D, R, Dl, Rl, fold = centroid_distances(X, y, birds, K)
        for k, v in (("P", P), ("D", D), ("R", R), ("D_lda", Dl), ("R_lda", Rl)):
            out[f"{k}__{t}"] = v.astype(np.float32)
        out[f"layer__{t}"] = np.array(layer); out["fold"] = fold
        nc, ncl = float((R.argmin(1) == y).mean()), float((Rl.argmin(1) == y).mean())
        print(f"[{t}] L{layer} probe acc {acc:.4f} (reproduces reported)  nearest-centroid acc: "
              f"raw {nc:.4f}, LDA {ncl:.4f}  "
              f"{time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(OUT, **out)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

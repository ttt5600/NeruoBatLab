"""Shared board-probe plumbing for ensemble_tet (same protocol as seed_pairs_cpu.py / seed_mean_pairs_cpu.py).

Clip list from calltype11.collect() (the worktree has no datasets/ dir, so DIRS is pointed at the main checkout's
datasets/11905533; read-only). Probe = aves_calltype.cv_acc (StandardScaler + LogisticRegression, leave-birds-out
StratifiedGroupKFold(5) seed 0). probe(..., split=k) re-implements cv_acc with random_state=k for the split-sensitivity
readout; split 0 is asserted to equal cv_acc exactly.
"""
from __future__ import annotations
import json, os, sys, warnings
from pathlib import Path
import numpy as np

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore")
import calltype11                                                          # noqa: E402
from calltype11 import KEEP11                                              # noqa: E402
from aves_calltype import cv_acc                                           # noqa: E402

MAIN_DATA = Path("/Users/jonathanwang/Desktop/vocalizations_lab/datasets/11905533")
if not calltype11.DATA.exists():
    calltype11.DIRS = [MAIN_DATA / "AdultVocalizations", MAIN_DATA / "ChickVocalizations"]

FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
CPU = json.loads((ANA / "calltype_cpu_rescore.json").read_text())["models"]
N_JOBS = 4


def task():
    rows = [r for r in calltype11.collect() if r[3] in KEEP11]
    birds = np.array([r[1].lower() for r in rows])
    tt = np.array([r[3] for r in rows]); classes = sorted(set(tt))
    y = np.array([classes.index(t) for t in tt])
    assert len(y) == 3412 and len(np.unique(birds)) == 48, (len(y), len(np.unique(birds)))
    rng = np.random.default_rng(0); ub = np.unique(birds); bidx = {b: np.where(birds == b)[0] for b in ub}
    boot = [np.concatenate([bidx[b] for b in rng.choice(ub, len(ub))]) for _ in range(2000)]
    return y, birds, classes, boot


def layer_of(tag, layer):
    return int(CPU[tag]["best_layer_cpu"]) if layer == "best" else int(layer)


def feats(tag, layer):
    E = np.load(FEAT / f"ct11_{tag}_cpuemb.npy", mmap_mode="r")
    return np.ascontiguousarray(E[:, layer_of(tag, layer)])


def _cv(X, y, birds, split):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedGroupKFold
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split)
    P = np.zeros((len(y), len(np.unique(y))))
    for tr, te in cv.split(X, y, birds):
        est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000)).fit(X[tr], y[tr])
        P[te] = est.predict_proba(X[te])
    return P


def probe(tags, layer, y, birds, split=0):
    """Out-of-fold probabilities. tags: one tag, or a tuple of tags whose features are concatenated."""
    tags = (tags,) if isinstance(tags, str) else tuple(tags)
    X = np.concatenate([feats(t, layer) for t in tags], 1)
    if split == 0:
        a, _, P = cv_acc(X, y, birds, return_proba=True)
    else:
        P = _cv(X, y, birds, split); a = float((P.argmax(1) == y).mean())
    if split == 0 and len(tags) == 1:
        want = CPU[tags[0]]["acc11_cpu"] if layer == "best" else CPU[tags[0]]["per_layer_acc11_cpu"][int(layer)]
        assert abs(a - want) < 1e-9, (tags, layer, a, want)
    return P


def pct(bs, alpha):
    lo, hi = np.percentile(bs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)

#!/usr/bin/env python
"""Cache out-of-fold probe probabilities for the power / label-ceiling analysis (CPU only, no training).

Board probe exactly as aves_calltype.cv_acc (StandardScaler + LogisticRegression(max_iter=4000), leave-birds-out
StratifiedGroupKFold(5, shuffle=True)), but with the fold random_state as a parameter so the split term can be measured.
Each model is read at its CPU best layer from analysis/calltype_cpu_rescore.json (the board's readout); split 0 is
asserted to reproduce acc11_cpu exactly. Writes zfeval/experiments/power/cache/<tag>_s<split>.npy (3412 x 11 float32).

    python cache_oof.py --split-models   # seed-group models + vanilla, splits 0-9
    python cache_oof.py --all            # every model in calltype_cpu_rescore.json, split 0
    python cache_oof.py --leaky          # 3 strong models, random (not leave-birds-out) StratifiedKFold(5), for the label audit
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1"); os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import json, sys, warnings
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore")
import calltype11                                                         # noqa: E402
from calltype11 import collect, KEEP11                                     # noqa: E402
# the worktree has no datasets/ dir; read the clips from the main checkout (same files the board uses)
_DATA = Path.home() / "Desktop/vocalizations_lab/datasets/11905533"
calltype11.DIRS = [_DATA / "AdultVocalizations", _DATA / "ChickVocalizations"]

FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
CPU = json.loads((ANA / "calltype_cpu_rescore.json").read_text())["models"]
CACHE = HERE / "cache"; CACHE.mkdir(exist_ok=True)

GROUPS = {   # recipe -> seeds (confirmed seed reruns in harness/experiments.yaml)
    "run16": ["run16_compute4x", "run16_seed2", "run16_seed3"],
    "run19": ["run19_avesteacher", "run19_seed2"],
    "e13": ["run24_bioxnl12", "run24_bioxnl12_s2"],
    "run20": ["run20_fsdonly", "run20_seed2"],
    "run29": ["run29_avesrecipe", "run29_avesrecipe_s2"],
    "run30": ["run30_logmel", "run30_logmel_s2"],
    "e19": ["daptv4two_2e4_step15000", "daptv4two_2e4s2_step15000"],
    "bxl": ["daptv4bxl_2e4_step15000", "daptv4bxl_2e4s2_step15000"],
    "aveso": ["daptv4aveso_2e4_step15000", "daptv4aveso_2e4s2_step15000"],
    "v3aves": ["daptv3aves_2e4_step15000", "daptv3aves_2e4s2_step15000"],
    "v3m8": ["daptv3aves_2e4m8_step15000", "daptv3aves_2e4s2m8_step15000"],
    "vanilla": ["run31_vanilla_lr1e4", "run31_vanilla_lr1e4_s2"],
}
SINGLES = ["hf-whisper-base-enc", "birdaves-bioxn-large", "aves-base-bio"]
LEAKY = ["hf-whisper-base-enc", "birdaves-bioxn-large", "run19_seed2"]


def data():
    rows = [r for r in collect() if r[3] in KEEP11]
    birds = np.array([r[1].lower() for r in rows])
    tt = np.array([r[3] for r in rows]); classes = sorted(set(tt))
    y = np.array([classes.index(t) for t in tt])
    assert len(y) == 3412 and len(set(birds)) == 48
    return rows, y, birds, classes


def oof(tag, split, y, birds, leaky=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    out = CACHE / (f"{tag}_leaky.npy" if leaky else f"{tag}_s{split}.npy")
    if out.exists():
        return tag, split, float((np.load(out).argmax(1) == y).mean())
    L = int(CPU[tag]["best_layer_cpu"])
    X = np.ascontiguousarray(np.load(FEAT / f"ct11_{tag}_cpuemb.npy", mmap_mode="r")[:, L])
    if leaky:
        it = StratifiedKFold(5, shuffle=True, random_state=0).split(X, y)
    else:
        it = StratifiedGroupKFold(5, shuffle=True, random_state=split).split(X, y, birds)
    P = np.zeros((len(y), 11), dtype=np.float32)
    for tr, te in it:
        est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000)).fit(X[tr], y[tr])
        P[te] = est.predict_proba(X[te])
    a = float((P.argmax(1) == y).mean())
    if split == 0 and not leaky:
        assert abs(a - CPU[tag]["acc11_cpu"]) < 1e-9, (tag, a, CPU[tag]["acc11_cpu"])
    np.save(out, P)
    return tag, split, a


if __name__ == "__main__":
    _, y, birds, _ = data()
    jobs = []
    if "--split-models" in sys.argv:
        for t in [t for g in GROUPS.values() for t in g] + SINGLES:
            jobs += [(t, s, False) for s in range(10)]
    if "--all" in sys.argv:
        jobs += [(t, 0, False) for t in CPU]
    if "--leaky" in sys.argv:
        jobs += [(t, 0, True) for t in LEAKY]
    for tag, s, a in Parallel(n_jobs=4, verbose=0)(delayed(oof)(t, s, y, birds, lk) for t, s, lk in jobs):
        print(f"{tag} s{s} {a:.4f}", flush=True)

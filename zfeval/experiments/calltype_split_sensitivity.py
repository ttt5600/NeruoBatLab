#!/usr/bin/env python
"""How much does each model's 11-class call-type score move when only the fold split changes?

Same cached embeddings, same layer (each model's reported one), same probe; only
StratifiedGroupKFold's random_state changes (0..9). Seed 0 is the board's split. Reports each
model's spread and the spread of its difference from run16 (paired: both on the same split).

The reported layer was chosen on seed 0, so seed 0 is mildly favourable to each model; expect the
other seeds to sit a little lower on average.

  python calltype_split_sensitivity.py   -> analysis/calltype_split_sensitivity.json  (MPS caches: wrong, finding 076)
  python calltype_split_sensitivity.py --cpu [--only t1,t2]
        -> analysis/calltype_split_sensitivity_cpu.json: the same on the CPU caches (ct11_<tag>_cpuemb.npy), each
           model at its CPU best layer from calltype_cpu_rescore.json (seed 0 must reproduce acc11_cpu), keyed by
           tag, reference run16_compute4x. Default models: the model board's list (harness/build_board.py MODELS).
  python calltype_split_sensitivity.py --cpu --layer 3 --only t1,t2
        -> analysis/calltype_split_sensitivity_cpu_L3.json: the same at a FIXED layer for every model (no layer
           selection on split 0); seed 0 must reproduce per_layer_acc11_cpu[layer].
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from calltype11 import collect, KEEP11                                     # noqa: E402
from calltype_per_class import MODELS, FEAT, ANA                           # noqa: E402

SEEDS = range(10)


def acc(X, y, g, seed):
    P = np.zeros(len(y), int)
    for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y, g):
        P[te] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000)).fit(X[tr], y[tr]).predict(X[te])
    return float((P == y).mean())


def cpu_models(only, layer=None):
    cpu = json.loads((ANA / "calltype_cpu_rescore.json").read_text())["models"]
    if not only:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "harness"))
        from build_board import MODELS as BOARD
        only = [m[0] for m in BOARD]
    if "run16_compute4x" not in only:
        only = ["run16_compute4x"] + list(only)
    if layer is not None:
        return [(t, None, f"ct11_{t}_cpuemb.npy", (layer, cpu[t]["per_layer_acc11_cpu"][layer])) for t in only]
    return [(t, None, f"ct11_{t}_cpuemb.npy", (cpu[t]["best_layer_cpu"], cpu[t]["acc11_cpu"])) for t in only]


def main():
    cpu = "--cpu" in sys.argv
    only = next((a.split("=", 1)[1] if "=" in a else sys.argv[sys.argv.index(a) + 1]
                 for a in sys.argv if a.startswith("--only")), "")
    fixed = int(sys.argv[sys.argv.index("--layer") + 1]) if "--layer" in sys.argv else None
    models = cpu_models([t for t in only.split(",") if t], fixed) if cpu else MODELS
    refk = "run16_compute4x" if cpu else "run16"
    rows = [r for r in collect() if r[3] in KEEP11]
    g = np.array([r[1].lower() for r in rows]); tt = np.array([r[3] for r in rows])
    cl = sorted(set(tt)); y = np.array([cl.index(t) for t in tt])
    A = {}
    for lab, _, f, (layer, rep) in models:
        X = np.ascontiguousarray(np.load(FEAT / f, mmap_mode="r")[:, layer])
        A[lab] = [acc(X, y, g, s) for s in SEEDS]
        assert abs(A[lab][0] - rep) < 1e-9, f"{lab}: seed 0 {A[lab][0]} != reported {rep}"
        print(f"{lab:<22} " + " ".join(f"{a:.4f}" for a in A[lab]), flush=True)
    ref = np.array(A[refk])
    out = dict(sklearn=sklearn.__version__, seeds=list(SEEDS), reference=refk,
               embeddings="CPU (ct11_<tag>_cpuemb.npy)" if cpu else "MPS (ct11_*_emb.npy) -- wrong for 2-8 frame clips, 076",
               layer_rule=(f"fixed layer {fixed} for every model" if fixed is not None else
                           "each model's CPU best layer (calltype_cpu_rescore.json)") if cpu else "each model's reported layer",
               models={lab: dict(acc=a, mean=float(np.mean(a)), sd=float(np.std(a, ddof=1)),
                                 min=float(min(a)), max=float(max(a)),
                                 diff_vs_run16=[float(x) for x in np.array(a) - ref],
                                 diff_mean=float(np.mean(np.array(a) - ref)),
                                 diff_sd=float(np.std(np.array(a) - ref, ddof=1)))
                       for lab, a in A.items()})
    name = (f"calltype_split_sensitivity_cpu_L{fixed}.json" if fixed is not None else "calltype_split_sensitivity_cpu.json") \
        if cpu else "calltype_split_sensitivity.json"
    (ANA / name).write_text(
        json.dumps(out, indent=2))
    print(f"\n{'model':<22}{'seed0':>8}{'mean':>8}{'sd':>8}{'min':>8}{'max':>8}{'d-mean':>9}{'d-sd':>8}")
    for lab, m in out["models"].items():
        print(f"{lab:<22}{m['acc'][0]:>8.4f}{m['mean']:>8.4f}{m['sd']:>8.4f}{m['min']:>8.4f}{m['max']:>8.4f}"
              f"{m['diff_mean']:>+9.4f}{m['diff_sd']:>8.4f}")


if __name__ == "__main__":
    main()

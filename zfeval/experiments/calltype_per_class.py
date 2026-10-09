#!/usr/bin/env python
"""Per-call-type accuracy for every encoder on the board, from the cached clip embeddings.

The board reports one number per model (11-class leave-birds-out accuracy). This breaks it down by
call type: per-class recall, the confusion matrix, and the per-class difference from run16 with a
95% bird-bootstrap interval. No re-encoding: it reads features/ct11_<tag>_emb.npy, probes the layer
each model's own JSON reported, and asserts the overall accuracy reproduces that JSON exactly, so the
breakdown is guaranteed to be of the same predictions the board scores.

Differences are OBSERVED differences (not the resample mean), with percentile intervals.
Classes are small (Di has 18 clips from few birds), so read the intervals before the bars.

  python calltype_per_class.py   -> analysis/calltype_per_class.json
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
from calltype11 import collect, KEEP11                                     # noqa: E402
from aves_calltype import cv_acc                                           # noqa: E402

FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
OUT = ANA / "calltype_per_class.json"
REF = "run16"
N_BOOT, SEED = 2000, 0

AV = json.load(open(ANA / "aves_variants_calltype.json"))["models"]


def own(tag):
    d = json.load(open(ANA / ("run15_calltype.json" if tag == "run15_combined" else f"{tag}_calltype.json")))
    return d["best_layer11"], d["acc11"]


# label, group, embedding cache, (layer, reported acc11)
MODELS = [
    ("run11", "ours", "ct11_run11_emb.npy", (AV["run11"]["best_11"]["layer"], AV["run11"]["best_11"]["acc"])),
    ("run15", "ours", "ct11_run15_combined_emb.npy", own("run15_combined")),
    ("run16", "ours", "ct11_run16_compute4x_emb.npy", own("run16_compute4x")),
    ("run16 seed 2", "ours", "ct11_run16_seed2_emb.npy", own("run16_seed2")),
    ("run17", "ours", "ct11_run17_accum2_emb.npy", own("run17_accum2")),
    ("run18", "ours", "ct11_run18_iter2_emb.npy", own("run18_iter2")),
    ("run19", "ours", "ct11_run19_avesteacher_emb.npy", own("run19_avesteacher")),
    ("run20", "ours", "ct11_run20_fsdonly_emb.npy", own("run20_fsdonly")),
    ("run21", "ours", "ct11_run21_zfonly_emb.npy", own("run21_zfonly")),
    ("DAPT replay x3", "dapt", "ct11_daptreplay3_5e5_step15000_emb.npy", own("daptreplay3_5e5_step15000")),
] + [
    (lab, "aves", f, (AV[t]["best_11"]["layer"], AV[t]["best_11"]["acc"]))
    for lab, t, f in [
        ("AVES bio", "aves-base-bio", "ct11_aves_emb.npy"),
        ("AVES core", "aves-base-core", "ct11_aves-base-core_emb.npy"),
        ("AVES all", "aves-base-all", "ct11_aves-base-all_emb.npy"),
        ("BirdAVES biox-base", "birdaves-biox-base", "ct11_birdaves-biox-base_emb.npy"),
        ("BirdAVES biox-large", "birdaves-biox-large", "ct11_birdaves-biox-large_emb.npy"),
        ("BirdAVES bioxn-large", "birdaves-bioxn-large", "ct11_birdaves-bioxn-large_emb.npy"),
    ]
]


def main():
    rows = [r for r in collect() if r[3] in KEEP11]
    birds = np.array([r[1].lower() for r in rows])
    tt = np.array([r[3] for r in rows])
    src = np.array([r[4] for r in rows])
    classes = sorted(set(tt))
    y = np.array([classes.index(t) for t in tt])
    assert len(y) == 3412 and len(classes) == 11 and len(set(birds)) == 48
    K = len(classes)

    pred = {}
    out = dict(classes=classes, n_clips=int(len(y)), n_birds=int(len(set(birds))), reference=REF,
               class_counts={c: int((y == i).sum()) for i, c in enumerate(classes)},
               class_birds={c: int(len(set(birds[y == i]))) for i, c in enumerate(classes)},
               class_chick_fraction={c: float((src[y == i] == "ChickVocalizations").mean())
                                     for i, c in enumerate(classes)},
               split="leave-birds-out StratifiedGroupKFold(5), seed 0; StandardScaler + LogisticRegression",
               layer_rule="each model's own reported best layer (the board's protocol)",
               models={})
    for lab, grp, f, (layer, acc_rep) in MODELS:
        t0 = time.time()
        E = np.load(FEAT / f, mmap_mode="r")
        acc, _, P = cv_acc(np.ascontiguousarray(E[:, layer]), y, birds, return_proba=True)
        assert abs(acc - acc_rep) < 1e-9, f"{lab}: reproduced {acc} != reported {acc_rep}"
        p = P.argmax(1); pred[lab] = p
        C = np.zeros((K, K), dtype=int)
        np.add.at(C, (y, p), 1)
        out["models"][lab] = dict(group=grp, layer=int(layer), acc11=float(acc),
                                  recall={c: float(C[i, i] / C[i].sum()) for i, c in enumerate(classes)},
                                  confusion=C.tolist())
        print(f"[{lab}] L{layer} acc {acc:.4f} (reproduces reported) {time.time()-t0:.0f}s", flush=True)

    # per-class recall difference vs run16, cluster bootstrap over birds
    rng = np.random.default_rng(SEED)
    uniq = np.unique(birds)
    idx = {g: np.where(birds == g)[0] for g in uniq}
    sels = [np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq), replace=True)]) for _ in range(N_BOOT)]
    hit = {lab: (p == y) for lab, p in pred.items()}
    for lab in pred:
        if lab == REF:
            continue
        diffs = {}
        for i, c in enumerate(classes):
            m = y == i
            obs = hit[lab][m].mean() - hit[REF][m].mean()
            bs = []
            for s in sels:
                ms = s[m[s]]
                if len(ms):
                    bs.append(hit[lab][ms].mean() - hit[REF][ms].mean())
            lo, hi = np.percentile(bs, [2.5, 97.5])
            diffs[c] = dict(diff=float(obs), lo=float(lo), hi=float(hi),
                            resolved=bool(lo > 0 or hi < 0))
        out["models"][lab]["vs_ref"] = diffs
    OUT.write_text(json.dumps(out, indent=2))

    print("\nper-class recall")
    print(f"{'model':<22}" + "".join(f"{c:>6}" for c in classes) + f"{'all':>8}")
    for lab, d in out["models"].items():
        print(f"{lab:<22}" + "".join(f"{d['recall'][c]:>6.2f}" for c in classes) + f"{d['acc11']:>8.4f}")
    print(f"{'n clips':<22}" + "".join(f"{out['class_counts'][c]:>6}" for c in classes))
    print(f"{'n birds':<22}" + "".join(f"{out['class_birds'][c]:>6}" for c in classes))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

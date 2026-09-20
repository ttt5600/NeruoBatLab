#!/usr/bin/env python
"""Score the random-init control. Same probe, same folds, every variant.

Two probe recipes are reported on purpose:

  raw     no feature scaling, LogisticRegression(C=1) -- the recipe the Savio evaluation used.
          Its only job here is the REPRODUCTION CHECK: pretrained must land on the number already
          reported for this dataset, or the local pipeline is not measuring the same thing.
  scaled  StandardScaler fitted inside each training fold, then C=1. Untrained networks have a
          different activation scale from trained ones, and a fixed L2 penalty on unscaled
          features silently penalises whichever arm has the larger norm. Scaling removes that
          confound, so this is the recipe the trained-vs-random comparison is read from.

kNN-10 is reported alongside because it needs no regularisation constant at all: if the gap
survives a parameter-free geometric classifier, it is not an artefact of how the probe was tuned.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zfeval import metrics as mx, splits as sp                        # noqa: E402

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

TMP = Path.home() / ".claude/jobs/63c218d9/tmp"
OUTDIR = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"


def cvpred(X, y, cv, g, scaled=True, C=1.0):
    est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=C)) if scaled \
        else LogisticRegression(max_iter=4000, C=C)
    return cross_val_predict(est, X, y, groups=g, cv=cv, method="predict_proba")[:, 1]


def knnpred(X, y, cv, g, k=10):
    est = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    return cross_val_predict(est, X, y, groups=g, cv=cv, method="predict_proba")[:, 1]


def reproduction_check(local_X):
    """The local forward pass must reproduce the Savio-extracted features for the same weights.

    Without this the whole comparison could be measuring a local extraction bug rather than the
    difference between trained and untrained weights.
    """
    sav = np.load(ROOT / "runs/run11/windows_soundsep_111021.npz", allow_pickle=True)
    A = sav["X"]
    out = {}
    for l in range(A.shape[1]):
        a, b = A[:, l].ravel(), local_X[:, l].ravel()
        out[f"L{l}"] = dict(cosine=float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
                            max_abs_diff=float(np.abs(a - b).max()))
    worst = min(v["cosine"] for v in out.values())
    print(f"[reproduction] worst layer cosine vs Savio features = {worst:.8f}", flush=True)
    if worst < 0.9999:
        raise RuntimeError("local extraction does not reproduce the reference features")
    return out


def main():
    d = np.load(TMP / "randinit_feats.npz", allow_pickle=True)
    y, grp, starts, en, Xmel = d["y"], d["grp"].astype(str), d["starts"], d["en"], d["Xmel"]
    variants = [str(v) for v in d["variants"]]
    cv, g, cv_desc = sp.choose_cv(grp, starts, n_splits=5, seed=0)
    print(f"split: {cv_desc}\nn={len(y)}  prevalence={y.mean():.4f}\nvariants: {variants}\n", flush=True)
    repro = reproduction_check(d["X_pretrained"])

    res, preds = {}, {}
    for v in variants:
        X, Xc = d[f"X_{v}"], d[f"Xcnn_{v}"]
        row = {}
        for tag, Z in [("cnn", Xc)] + [(f"L{l}", X[:, l]) for l in range(X.shape[1])]:
            p_s = cvpred(Z, y, cv, g, scaled=True)
            # An unscaled L2 probe on untrained activations overflows: their norm is far from the
            # trained scale. Detected, not hidden -- the raw column is only trustworthy where the
            # feature norm is in the range the fixed C=1 penalty was chosen for.
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                p_r = cvpred(Z, y, cv, g, scaled=False)
            raw_ok = bool(np.isfinite(p_r).all())
            p_k = knnpred(Z, y, cv, g)
            row[tag] = dict(auc_scaled=mx.score(y, p_s, cv_desc).auc,
                            auc_raw=mx.score(y, p_r, cv_desc).auc,
                            auc_raw_trustworthy=raw_ok,
                            ap_scaled=mx.score(y, p_s, cv_desc).ap,
                            acc_scaled=mx.score(y, p_s, cv_desc).acc,
                            knn10=mx.score(y, p_k, cv_desc).auc)
            preds[f"{v}|{tag}"] = p_s
            print(f"  {v:16s} {tag:4s} scaled {row[tag]['auc_scaled']:.4f}  "
                  f"raw {row[tag]['auc_raw']:.4f}  knn10 {row[tag]['knn10']:.4f}", flush=True)
        res[v] = row

    # baselines on the identical folds
    base = {}
    for tag, Z, C in [("logmel", Xmel, 0.03), ("logenergy", en[:, None], 1.0)]:
        p = cvpred(Z, y, cv, g, scaled=True, C=C)
        base[tag] = mx.score(y, p, cv_desc).auc
        preds[f"baseline|{tag}"] = p
        print(f"  {'baseline':16s} {tag:9s} scaled {base[tag]:.4f}", flush=True)

    # ---- the comparison. Layer-matched, and averaged over seeds with its spread.
    rands = [v for v in variants if v.startswith("rand_")]
    shufs = [v for v in variants if v.startswith("shuffled_")]
    tags = ["cnn"] + [f"L{l}" for l in range(12)]
    table = {}
    for tag in tags:
        pre = res["pretrained"][tag]["auc_scaled"]
        r = [res[v][tag]["auc_scaled"] for v in rands]
        s = [res[v][tag]["auc_scaled"] for v in shufs]
        table[tag] = dict(pretrained=pre,
                          rand_mean=float(np.mean(r)), rand_sd=float(np.std(r)), rand_all=r,
                          shuf_mean=float(np.mean(s)), shuf_sd=float(np.std(s)), shuf_all=s,
                          gap_vs_rand=pre - float(np.mean(r)),
                          gap_vs_shuf=pre - float(np.mean(s)),
                          knn_pretrained=res["pretrained"][tag]["knn10"],
                          knn_rand_mean=float(np.mean([res[v][tag]["knn10"] for v in rands])))

    # Paired bootstrap at the layer where each arm is strongest AND layer-matched at pretrained's
    # best. Both are reported: picking each arm's own argmax on the test folds flatters whichever
    # arm has more layers to choose from, so it is quoted only next to the matched number.
    BLOCK = 30                              # 1 s windows -> 30 s moving blocks
    best_pre = max(tags, key=lambda t: table[t]["pretrained"])
    boot = {}
    for v in rands + shufs:
        best_v = max(tags, key=lambda t: res[v][t]["auc_scaled"])
        boot[v] = {}
        for label, tp, tv in [("layer_matched", best_pre, best_pre),
                              ("each_own_best", best_pre, best_v)]:
            b = mx.paired_bootstrap(y, preds[f"pretrained|{tp}"], preds[f"{v}|{tv}"],
                                    block=BLOCK, n=2000, seed=0)
            b.update(pretrained_layer=tp, other_layer=tv)
            boot[v][label] = b
            print(f"  boot {v:16s} {label:14s} pre[{tp}] - {v}[{tv}] = "
                  f"{b['delta']:+.4f} [{b['lo']:+.4f}, {b['hi']:+.4f}]  {b['verdict']}", flush=True)

    out = dict(split=cv_desc, n=int(len(y)), prevalence=float(y.mean()),
               variants=variants, per_variant=res, baselines=base,
               layer_table=table, bootstrap=boot, best_pretrained_layer=best_pre,
               reproduction_vs_savio=repro,
               meta=json.loads((TMP / "randinit_feats.meta.json").read_text()))
    (OUTDIR / "randinit_control.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(TMP / "randinit_preds.npz", y=y, starts=starts, en=en,
                        **{k.replace("|", "__"): v for k, v in preds.items()})

    print("\n=== layer-matched summary (scaled probe AUC) ===")
    print(f"{'layer':6s} {'pretrained':>11s} {'rand(3 seeds)':>16s} {'shuffled':>14s} "
          f"{'gap vs rand':>12s} {'gap vs shuf':>12s}")
    for tag in tags:
        t = table[tag]
        print(f"{tag:6s} {t['pretrained']:11.4f} {t['rand_mean']:10.4f}+-{t['rand_sd']:.4f} "
              f"{t['shuf_mean']:8.4f}+-{t['shuf_sd']:.4f} {t['gap_vs_rand']:+12.4f} "
              f"{t['gap_vs_shuf']:+12.4f}")
    print(f"\nbaselines: log-mel {base['logmel']:.4f}   log-energy {base['logenergy']:.4f}")
    print("wrote", OUTDIR / "randinit_control.json")


if __name__ == "__main__":
    main()

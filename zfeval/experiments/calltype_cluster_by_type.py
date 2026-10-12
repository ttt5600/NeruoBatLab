#!/usr/bin/env python
"""Which call types does each encoder's UNSUPERVISED clustering get wrong -- AVES vs ours?

calltype_unsup.py found AVES's space clusters by call type better than run11's (Ward AMI +0.09 at
k=11), much more than the supervised probe gap. That is one number. This breaks it down by call
type, for our models and the AVES family, to see whether the gap is everywhere or in a few types.

  * clustering: calltype_unsup.py's recipe -- StandardScaler, Ward, k = 11 (the number of types,
    the only thing taken from the labels). Layer: each model's best Ward-AMI layer over the full
    depth sweep (picking on AMI uses the labels; the same rule for every model, sweep saved).
  * per type: F1 of its best-matching cluster (1.0 = one cluster holding exactly that type's clips
    and nothing else), the share of its clips in that cluster, that cluster's purity, and the
    other type that contaminates it most.
  * vs run16: observed per-type F1 difference, with (a) a bird bootstrap that keeps the clusterings
    FIXED (sampling variability only, as in calltype_unsup.py -- narrower than the truth), and
    (b) refit stability: Ward refit on 20 random subsets of 80% of birds, and the fraction of
    refits in which the difference has the same sign as the observed one.

  python calltype_cluster_by_type.py   -> analysis/calltype_cluster_by_type.json
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore", category=UserWarning)
import calltype11 as C11                                                  # noqa: E402
from sklearn.preprocessing import StandardScaler                          # noqa: E402
from sklearn.cluster import AgglomerativeClustering                       # noqa: E402
from sklearn.metrics import adjusted_mutual_info_score as AMI             # noqa: E402

ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
OUT = ANA / "calltype_cluster_by_type.json"
REF = "run16"
N_BOOT, N_REFIT, SEED = 2000, 20, 0
MODELS = [("run11", "ct11_run11_emb.npy"), ("run16", "ct11_run16_compute4x_emb.npy"),
          ("run16 seed 2", "ct11_run16_seed2_emb.npy"), ("run19", "ct11_run19_avesteacher_emb.npy"),
          ("AVES bio", "ct11_aves_emb.npy"), ("AVES core", "ct11_aves-base-core_emb.npy"),
          ("AVES all", "ct11_aves-base-all_emb.npy"), ("BirdAVES biox-base", "ct11_birdaves-biox-base_emb.npy")]


def ward(X, k):
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(StandardScaler().fit_transform(X))


def contingency(y, lab, K):
    C = np.zeros((K, lab.max() + 1), dtype=np.int64)
    np.add.at(C, (y, lab), 1)
    return C


def type_f1(C):
    """Per type (row): F1 of its best-matching cluster."""
    tot_t, tot_c = C.sum(1, keepdims=True), C.sum(0, keepdims=True)
    F = 2 * C / np.maximum(tot_t + tot_c, 1)
    return F.max(1)


def main():
    t0 = time.time()
    rows = [r for r in C11.collect() if r[3] in C11.KEEP11]
    birds = np.array([r[1].lower() for r in rows])
    tt = np.array([r[3] for r in rows])
    classes = sorted(set(tt))
    y = np.array([classes.index(t) for t in tt])
    K = len(classes)
    assert len(y) == 3412 and K == 11 and len(set(birds)) == 48

    out = dict(classes=classes, k=K, reference=REF, n_clips=int(len(y)),
               class_counts={c: int((y == i).sum()) for i, c in enumerate(classes)},
               recipe="StandardScaler + Ward, k=11; layer = each model's best Ward-AMI layer", models={})
    E, labs = {}, {}
    for name, f in MODELS:
        E[name] = np.load(C11.FEAT / f, mmap_mode="r")
        assert E[name].shape[0] == len(y), f"{f}: {E[name].shape[0]} rows"
        sweep = [float(AMI(y, ward(np.asarray(E[name][:, l], dtype=np.float32), K)))
                 for l in range(E[name].shape[1])]
        L = int(np.argmax(sweep))
        lab = ward(np.asarray(E[name][:, L], dtype=np.float32), K)
        labs[name] = (L, lab)
        C = contingency(y, lab, K)
        f1 = type_f1(C)
        per = {}
        for i, c in enumerate(classes):
            j = int(np.argmax(2 * C[i] / (C[i].sum() + C.sum(0))))     # best-matching cluster
            others = C[:, j].copy(); others[i] = 0
            per[c] = dict(f1=float(f1[i]), share_in_cluster=float(C[i, j] / C[i].sum()),
                          cluster_purity=float(C[i, j] / C[:, j].sum()),
                          top_contaminant=classes[int(np.argmax(others))] if others.sum() else None,
                          contaminant_share=float(others.max() / C[:, j].sum()))
        out["models"][name] = dict(layer=L, ami=sweep[L], ami_by_layer=sweep, per_type=per,
                                   contingency=C.tolist())
        print(f"[{name}] L{L} AMI {sweep[L]:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- vs run16: bird bootstrap with fixed clusterings
    rng = np.random.default_rng(SEED)
    uniq = np.unique(birds)
    idx = {g: np.where(birds == g)[0] for g in uniq}
    sels = [np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq), replace=True)]) for _ in range(N_BOOT)]
    f1_boot = {n: np.array([type_f1(contingency(y[s], labs[n][1][s], K)) for s in sels]) for n in labs}
    # ---- refit stability: Ward refit on 80% of birds
    subs = [np.isin(birds, rng.choice(uniq, int(0.8 * len(uniq)), replace=False)) for _ in range(N_REFIT)]
    f1_refit = {n: np.array([type_f1(contingency(y[m], ward(np.asarray(E[n][:, labs[n][0]], dtype=np.float32)[m], K), K))
                             for m in subs]) for n in labs}
    print(f"[bootstrap + {N_REFIT} refits] {time.time()-t0:.0f}s", flush=True)
    ref_obs = np.array([out["models"][REF]["per_type"][c]["f1"] for c in classes])
    for n in labs:
        if n == REF:
            continue
        obs = np.array([out["models"][n]["per_type"][c]["f1"] for c in classes]) - ref_obs
        d = f1_boot[n] - f1_boot[REF]
        lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
        dr = f1_refit[n] - f1_refit[REF]
        agree = (np.sign(dr) == np.sign(obs)).mean(0)
        out["models"][n]["vs_ref"] = {c: dict(diff=float(obs[i]), lo=float(lo[i]), hi=float(hi[i]),
                                              resolved=bool(lo[i] > 0 or hi[i] < 0),
                                              refit_sign_agreement=float(agree[i]),
                                              refit_mean_diff=float(dr[:, i].mean()))
                                      for i, c in enumerate(classes)}
    out["n_boot"], out["n_refit"] = N_BOOT, N_REFIT
    OUT.write_text(json.dumps(out, indent=2))

    print("\nper-type F1 of the best-matching cluster (1.0 = a cluster that is exactly that type)")
    print(f"{'model':<20}{'L':>3}{'AMI':>7}" + "".join(f"{c:>6}" for c in classes))
    for n, d in out["models"].items():
        print(f"{n:<20}{d['layer']:>3}{d['ami']:>7.3f}" + "".join(f"{d['per_type'][c]['f1']:>6.2f}" for c in classes))
    print(f"{'n clips':<30}" + "".join(f"{out['class_counts'][c]:>6}" for c in classes))
    print(f"\ndifference vs {REF}  (* = bird-bootstrap interval excludes 0; [n/20] = refits agreeing in sign)")
    for n, d in out["models"].items():
        if n == REF:
            continue
        v = d["vs_ref"]
        print(f"{n:<20}" + " ".join(f"{c}:{v[c]['diff']:+.2f}{'*' if v[c]['resolved'] else ' '}"
                                    f"[{round(v[c]['refit_sign_agreement']*N_REFIT):>2d}]" for c in classes))
    print(f"-> {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

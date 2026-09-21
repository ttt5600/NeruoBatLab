#!/usr/bin/env python
"""Which encoder is better with NO supervised decoder anywhere in the loop?

The probe results have a hole in them. A logistic regression is a trained readout: it can find a
direction that separates classes even when the embedding does not natively group them, so "AVES
decodes better" is not the same claim as "AVES's space is better organised". This script never
fits a classifier. Labels are used ONLY to score structure that was discovered without them.

Four families, all decoder-free:

  cluster    Ward, k-means and HDBSCAN. Scored with AMI (chance-corrected -- NMI and purity both
             inflate with k and would hand the win to whichever run used more clusters) plus
             homogeneity. HDBSCAN additionally chooses its OWN k, so it answers a question the
             others cannot: how many groups are actually there?
  neighbour  leave-one-out kNN agreement. Pure geometry: does a clip's nearest neighbours share
             its call type? No fitting, no parameters beyond k.
  shape      silhouette against the TRUE labels -- are the real classes compact and separated in
             this space -- and, as the confound, silhouette against bird identity.
  reduction  every one of the above recomputed after UMAP to 2-d and 10-d, to test whether the
             ranking is a property of the encoders or of the pipeline we look at them through.

Two things this deliberately does NOT do. It does not report a purely intrinsic score
(silhouette on the discovered clusters, Davies-Bouldin) as evidence of which encoder is better:
those measure whether a space is clumpy, not whether it is clumpy in a way anyone cares about, and
a space perfectly organised by bird identity would win on them while being useless here. And it
does not pick the layer on the metric being reported -- the full depth sweep is shown.
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import calltype11 as C11                                                  # noqa: E402

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans, HDBSCAN
from sklearn.metrics import (adjusted_mutual_info_score as AMI, homogeneity_score,
                             silhouette_score)
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore", category=UserWarning)
OUT = Path(__file__).resolve().parents[2] / "paper" / "figures" / "calltype"
SEED = 0


def cohort():
    rows = [r for r in C11.collect() if r[3] in C11.KEEP11]
    birds = np.array([r[1].lower() for r in rows])
    tt = np.array([r[3] for r in rows])
    classes = sorted(set(tt))
    y = np.array([classes.index(t) for t in tt])
    return y, birds, tt, classes


def knn_agreement(X, y, k=10):
    """Leave-one-out: fraction of a clip's k nearest neighbours sharing its label.

    Geometry only -- nothing is fitted. Self is excluded by asking for k+1 and dropping column 0.
    """
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
    return float((y[idx[:, 1:]] == y[:, None]).mean())


def cluster_scores(X, y, birds, k):
    """Ward / k-means at a fixed k, plus HDBSCAN which picks its own."""
    out = {}
    lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    out["ward"] = dict(ami=float(AMI(y, lab)), hom=float(homogeneity_score(y, lab)),
                       ami_bird=float(AMI(birds, lab)), k=k)
    lab = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(X)
    out["kmeans"] = dict(ami=float(AMI(y, lab)), hom=float(homogeneity_score(y, lab)),
                         ami_bird=float(AMI(birds, lab)), k=k)
    lab = HDBSCAN(min_cluster_size=25).fit_predict(X)
    keep = lab >= 0                      # -1 is HDBSCAN's noise class
    nk = int(len(set(lab[keep])))
    out["hdbscan"] = dict(
        ami=float(AMI(y[keep], lab[keep])) if keep.sum() > 10 and nk > 1 else float("nan"),
        hom=float(homogeneity_score(y[keep], lab[keep])) if keep.sum() > 10 and nk > 1 else float("nan"),
        ami_bird=float(AMI(birds[keep], lab[keep])) if keep.sum() > 10 and nk > 1 else float("nan"),
        k=nk, noise_frac=float((~keep).mean()))
    return out


def bootstrap_ami(y, birds, lab_a, lab_b, n=1000, seed=0):
    """Resample BIRDS and recompute AMI for two FIXED clusterings.

    This captures the sampling variability of the metric, not the variability of the clustering
    itself -- refitting Ward 1000 times is not affordable and would answer a different question.
    Stated rather than hidden, because it makes the interval narrower than a full refit would.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(birds)
    idx = {g: np.where(birds == g)[0] for g in uniq}
    d = []
    for _ in range(n):
        sel = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        d.append(AMI(y[sel], lab_a[sel]) - AMI(y[sel], lab_b[sel]))
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(delta=float(d.mean()), lo=float(lo), hi=float(hi),
                verdict=("a_better" if lo > 0 else "b_better" if hi < 0 else "not_distinguishable"))


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    y, birds, tt, classes = cohort()
    n, K = len(y), len(classes)
    print(f"[cohort] {n} clips, {K} classes, {len(set(birds))} birds", flush=True)

    E = {}
    for nm in ("run11", "aves"):
        p = C11.FEAT / f"ct11_{nm}_emb.npy"
        E[nm] = np.load(p)
        assert E[nm].shape[0] == n, f"{p} has {E[nm].shape[0]} rows, cohort {n}"
    res = {"n_clips": n, "n_classes": K, "n_birds": int(len(set(birds))),
           "classes": classes, "k_used": K, "seed": SEED,
           "note": "no classifier is fitted anywhere; labels only score structure found without them"}

    # ---- 1. full depth sweep, decoder-free -------------------------------
    print("\n[1/3] depth sweep: Ward AMI, kNN-10, silhouette  (no decoder)", flush=True)
    print(f"  {'layer':>5s} {'ward run11':>11s} {'ward aves':>10s} | {'knn run11':>10s} "
          f"{'knn aves':>9s} | {'sil run11':>10s} {'sil aves':>9s}")
    sweep = {nm: {"ward_ami": [], "knn": [], "sil": [], "sil_bird": []} for nm in E}
    for l in range(12):
        line = {}
        for nm in ("run11", "aves"):
            X = StandardScaler().fit_transform(E[nm][:, l])
            lab = AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(X)
            sweep[nm]["ward_ami"].append(float(AMI(y, lab)))
            sweep[nm]["knn"].append(knn_agreement(X, y))
            sweep[nm]["sil"].append(float(silhouette_score(X, y)))
            sweep[nm]["sil_bird"].append(float(silhouette_score(X, birds)))
            line[nm] = (sweep[nm]["ward_ami"][-1], sweep[nm]["knn"][-1], sweep[nm]["sil"][-1])
        print(f"  L{l:<4d} {line['run11'][0]:11.4f} {line['aves'][0]:10.4f} | "
              f"{line['run11'][1]:10.4f} {line['aves'][1]:9.4f} | "
              f"{line['run11'][2]:10.4f} {line['aves'][2]:9.4f}", flush=True)
    res["sweep"] = sweep

    # Pick each encoder's best layer BY THE UNSUPERVISED CRITERION, not by the probe's choice.
    best = {nm: int(np.argmax(sweep[nm]["ward_ami"])) for nm in E}
    res["best_layer_unsup"] = best
    res["best_layer_probe"] = {"run11": 3, "aves": 3}
    print(f"\n  best layer by Ward AMI: run11 L{best['run11']} "
          f"({sweep['run11']['ward_ami'][best['run11']]:.4f}), "
          f"aves L{best['aves']} ({sweep['aves']['ward_ami'][best['aves']]:.4f})", flush=True)

    # ---- 2. algorithm comparison at the unsupervised best layer ----------
    print("\n[2/3] Ward vs k-means vs HDBSCAN at each encoder's unsupervised best layer", flush=True)
    algo, labs = {}, {}
    for nm in ("run11", "aves"):
        X = StandardScaler().fit_transform(E[nm][:, best[nm]])
        algo[nm] = cluster_scores(X, y, birds, K)
        labs[nm] = AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(X)
        for a, v in algo[nm].items():
            extra = f"  found k={v['k']}, noise {v['noise_frac']:.3f}" if a == "hdbscan" else ""
            print(f"  {nm:6s} {a:9s} AMI {v['ami']:.4f}  hom {v['hom']:.4f}  "
                  f"AMI_bird {v['ami_bird']:.4f}{extra}", flush=True)
    res["algorithms"] = algo
    res["bootstrap_ward_aves_vs_run11"] = bootstrap_ami(y, birds, labs["aves"], labs["run11"])
    print(f"  bootstrap AVES-run11 Ward AMI: {res['bootstrap_ward_aves_vs_run11']}", flush=True)

    # ---- 3. does UMAP change the ranking? --------------------------------
    print("\n[3/3] the same measures after UMAP (2-d and 10-d)", flush=True)
    import umap
    red = {}
    for nm in ("run11", "aves"):
        X = StandardScaler().fit_transform(E[nm][:, best[nm]])
        red[nm] = {"full": {"ward_ami": sweep[nm]["ward_ami"][best[nm]],
                            "knn": sweep[nm]["knn"][best[nm]],
                            "sil": sweep[nm]["sil"][best[nm]], "dim": 768}}
        for d_ in (2, 10):
            U = umap.UMAP(n_components=d_, n_neighbors=25, min_dist=0.0 if d_ > 2 else 0.12,
                          random_state=SEED).fit_transform(X)
            lab = AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(U)
            red[nm][f"umap{d_}"] = {"ward_ami": float(AMI(y, lab)),
                                    "knn": knn_agreement(U, y),
                                    "sil": float(silhouette_score(U, y)), "dim": d_}
            print(f"  {nm:6s} umap{d_:<3d} ward {red[nm][f'umap{d_}']['ward_ami']:.4f}  "
                  f"knn {red[nm][f'umap{d_}']['knn']:.4f}  "
                  f"sil {red[nm][f'umap{d_}']['sil']:.4f}", flush=True)
    res["reduction"] = red

    winner = {}
    for space in ("full", "umap2", "umap10"):
        winner[space] = {m: ("aves" if red["aves"][space][m] > red["run11"][space][m] else "run11")
                         for m in ("ward_ami", "knn", "sil")}
    res["winner_by_space"] = winner
    print(f"\n  winner by space/metric: {json.dumps(winner)}", flush=True)

    (OUT / "calltype_unsup.json").write_text(json.dumps(res, indent=2))
    print(f"\n[done] {time.time()-t0:.0f}s -> {OUT/'calltype_unsup.json'}", flush=True)


if __name__ == "__main__":
    main()

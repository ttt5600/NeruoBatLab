"""Unsupervised call-type discovery from run11's 768-d embeddings.

WHAT THIS ADDS OVER cluster_calltype.py
---------------------------------------
cluster_calltype.py sweeps k and reports where AMI-against-the-human-labels peaks (k=16,
AMI 0.5475). That is a legitimate measurement of "how much call-type structure is present",
but it is NOT an unsupervised result: the labels chose k. A coworker who asks "can the model
find the call types on its own?" cannot be handed 0.5475, because on their unlabeled corpus
they would have no way to know to ask for 16 clusters.

So this script separates the two questions that were fused together:

  Q1 (label-free)  How many groups does the embedding actually contain, judged only by its
                   own geometry? Criteria: cluster stability under resampling, silhouette,
                   Calinski-Harabasz, Davies-Bouldin, GMM/BIC, plus HDBSCAN which picks its
                   own count. None of these ever sees y.
  Q2 (scored)      Having picked k WITHOUT labels, how well does that partition line up with
                   the human taxonomy? This is the number that honestly answers the question,
                   and it is a lower bound on 0.5475 by construction.

Why stability is ranked first among the label-free criteria: silhouette, CH and DB all
measure compactness-vs-separation under a geometry assumption (roughly, spherical equal-size
blobs), which is the same assumption k-means already made -- they tend to agree with k-means
because they share its bias, and they slope monotonically toward small k in high dimensions.
Stability (Ben-Hur et al. 2002) asks a different and harder question: cluster two random 80%
subsamples independently and compare their labels on the clips both saw. A k that reflects
real structure reproduces; a k that is slicing a continuum does not. It has its own bias --
it also favours small k, and k=2 is nearly always trivially stable -- so it is read as a
curve, not an argmax, and the whole disagreement between criteria is reported rather than
resolved by picking the flattering one.

Also here, because "how many clusters" is not interesting on its own:
  - CLUSTER ANATOMY: size, dominant call type, purity, and how many birds each cluster draws
    from. A cluster of 40 clips from one bird is a voice, not a call type.
  - SPLIT/MERGE FATE: for each human call type, is it one cluster, several (the model hears
    sub-types the taxonomy lumps), or does it share a cluster with another type (the model
    cannot separate them)? This is where the actual biology question lives.
  - BIRD-IDENTITY CONFOUND at the label-free k, not just at the AMI-optimal k.

Corrections applied (see project_calltype_clustering memory): bird IDs are case-folded
(HPiHPi4748 / HpiHpi4748 are one bird) and the four Unknown* prefixes are dropped, giving
2814 clips / 26 birds rather than the raw 2867 / 31.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

AMI = adjusted_mutual_info_score


# ------------------------------------------------------------------ data

def load(npz_path, layer):
    d = np.load(npz_path, allow_pickle=True)
    emb, y, birds, names = d["emb"], d["y"], d["birds"], d["names"]
    classes = [str(c) for c in d["classes"]]

    keep = ~np.array([str(b).lower().startswith("unknown") for b in birds])
    emb, y, birds, names = emb[keep], y[keep], birds[keep], names[keep]
    birds = np.array([str(b).lower() for b in birds])

    X = emb[:, layer, :].astype(np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n, np.asarray(y), birds, names, classes


# ------------------------------------------------------------------ label-free criteria

def stability(X, k, n_boot, frac, seed, n_init=3):
    """Ben-Hur cluster stability: mean ARI between two independent 80% subsample clusterings,
    measured on the clips that landed in both. High = the partition is a property of the data,
    not of which points happened to be drawn."""
    rng = np.random.default_rng(seed)
    n = len(X)
    m = int(round(frac * n))
    out = []
    for _ in range(n_boot):
        ia = rng.choice(n, m, replace=False)
        ib = rng.choice(n, m, replace=False)
        both = np.intersect1d(ia, ib)
        if len(both) < k * 2:
            continue
        ca = KMeans(k, n_init=n_init, random_state=int(rng.integers(1 << 30))).fit(X[ia])
        cb = KMeans(k, n_init=n_init, random_state=int(rng.integers(1 << 30))).fit(X[ib])
        out.append(adjusted_rand_score(ca.predict(X[both]), cb.predict(X[both])))
    return float(np.mean(out)), float(np.std(out))


def internal(X, c):
    """Geometry-only quality. Never touches y."""
    if len(np.unique(c)) < 2:
        return dict(sil=float("nan"), ch=float("nan"), db=float("nan"))
    return dict(
        sil=float(silhouette_score(X, c)),
        ch=float(calinski_harabasz_score(X, c)),
        db=float(davies_bouldin_score(X, c)),
    )


# ------------------------------------------------------------------ label-based scoring

def within_bird_ami(c, y, birds, min_clips=12, min_types=2):
    """AMI recomputed inside each bird, clip-count weighted. Holds voice constant, so what
    survives is call type rather than individual identity."""
    tot, acc, used = 0, 0.0, 0
    for b in np.unique(birds):
        m = birds == b
        if m.sum() < min_clips or len(np.unique(y[m])) < min_types:
            continue
        acc += AMI(y[m], c[m]) * m.sum()
        tot += int(m.sum())
        used += 1
    return (acc / tot if tot else float("nan")), used


def external(c, y, birds):
    wb, nb = within_bird_ami(c, y, birds)
    return dict(
        ami_calltype=float(AMI(y, c)),
        ami_bird=float(AMI(birds, c)),
        ari_calltype=float(adjusted_rand_score(y, c)),
        hom=float(homogeneity_score(y, c)),
        com=float(completeness_score(y, c)),
        ami_within_bird=float(wb),
        n_birds_used=int(nb),
        n_clusters_used=int(len(np.unique(c))),
    )


# ------------------------------------------------------------------ anatomy

def anatomy(c, y, birds, classes):
    """Per-cluster: who is in it, and is it a call type or a voice?"""
    rows = []
    for cl in sorted(np.unique(c)):
        m = c == cl
        cnt = Counter(classes[i] for i in y[m])
        top, ntop = cnt.most_common(1)[0]
        bc = Counter(birds[m])
        p = np.array(list(bc.values()), dtype=float)
        p /= p.sum()
        rows.append(dict(
            cluster=int(cl), n=int(m.sum()),
            top_type=top, purity=round(ntop / m.sum(), 3),
            n_birds=len(bc),
            bird_frac_top=round(max(bc.values()) / m.sum(), 3),
            bird_entropy_norm=round(
                float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0, 3),
            mix=dict(cnt.most_common(3)),
        ))
    return rows


def fate(c, y, classes, split_thresh=0.15):
    """For each human call type: how is it distributed over clusters?

    n_clusters_15pct  -> how many clusters hold >=15% of the type (>1 means the model SPLITS
                         a category the taxonomy treats as one).
    largest_frac      -> how concentrated it is.
    largest_cluster_dominated_by -> if that is a DIFFERENT type, the model MERGES them.
    """
    out = {}
    dom = {}
    for cl in np.unique(c):
        m = c == cl
        dom[int(cl)] = Counter(classes[i] for i in y[m]).most_common(1)[0][0]
    for ci, name in enumerate(classes):
        m = y == ci
        if not m.any():
            continue
        cnt = Counter(c[m])
        tot = int(m.sum())
        big = cnt.most_common(1)[0][0]
        out[name] = dict(
            n=tot,
            n_clusters_15pct=int(sum(1 for v in cnt.values() if v / tot >= split_thresh)),
            n_clusters_any=len(cnt),
            largest_cluster=int(big),
            largest_frac=round(cnt[big] / tot, 3),
            largest_cluster_dominated_by=dom[int(big)],
        )
    return out


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(Path.home() /
                    "Desktop/vocalizations_lab/release/zf_hubert_run11/data/run11_layersweep.npz"))
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=40)
    ap.add_argument("--n-boot", type=int, default=12)
    ap.add_argument("--frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="unsup_calltype_results.json")
    args = ap.parse_args()

    X, y, birds, names, classes = load(args.npz, args.layer)
    print(f"{len(X)} clips | {len(set(birds))} birds | {len(classes)} call types "
          f"| layer {args.layer} | l2-normalized 768-d\n")

    # Ward is swept alongside k-means because the smoke test showed it winning by a wide
    # margin at small k. k-means assumes isotropic, comparably-sized blobs; call types are
    # neither (DC has 583 clips, Wh 172, and a call type is a curved manifold of renditions,
    # not a ball). Ward only assumes that merging the two clusters that least increase
    # within-cluster variance is a good greedy move, which is a much weaker commitment.
    ks = list(range(args.kmin, args.kmax + 1))
    sweep = []
    print(f"{'k':>3} | {'stab':>6} {'sil':>7} {'CH':>7} {'DB':>6} || "
          f"{'AMI':>6} {'AMIbird':>7} {'hom':>6} {'com':>6} || {'wardAMI':>7} {'wardSil':>7}")
    print("-" * 96)
    for k in ks:
        c = KMeans(k, n_init=10, random_state=args.seed).fit_predict(X)
        st, sd = stability(X, k, args.n_boot, args.frac, args.seed + k)
        w = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        w_sc = external(w, y, birds)
        w_in = internal(X, w)
        row = dict(k=k, stability=st, stability_sd=sd, **internal(X, c), **external(c, y, birds),
                   ward={**w_sc, **w_in})
        sweep.append(row)
        print(f"{k:>3} | {st:>6.3f} {row['sil']:>7.4f} {row['ch']:>7.1f} {row['db']:>6.3f} || "
              f"{row['ami_calltype']:>6.4f} {row['ami_bird']:>7.4f} "
              f"{row['hom']:>6.3f} {row['com']:>6.3f} || "
              f"{w_sc['ami_calltype']:>7.4f} {w_in['sil']:>7.4f}", flush=True)

    # ---- GMM / BIC on PCA (full 768-d BIC is dominated by the parameter count, not the fit)
    pca = PCA(n_components=30, random_state=args.seed).fit(X)
    Xp = pca.transform(X)
    bic = []
    for k in ks:
        g = GaussianMixture(k, covariance_type="diag", n_init=2,
                            random_state=args.seed).fit(Xp)
        bic.append(dict(k=k, bic=float(g.bic(Xp))))
    k_bic = min(bic, key=lambda r: r["bic"])["k"]

    # ---- label-free picks
    picks = {
        "stability":         max(sweep, key=lambda r: r["stability"])["k"],
        "stability_k>=5":    max([r for r in sweep if r["k"] >= 5],
                                 key=lambda r: r["stability"])["k"],
        "silhouette":        max(sweep, key=lambda r: r["sil"])["k"],
        "calinski_harabasz": max(sweep, key=lambda r: r["ch"])["k"],
        "davies_bouldin":    min(sweep, key=lambda r: r["db"])["k"],
        "gmm_bic_pca30":     k_bic,
    }
    picks["ward_silhouette"] = max(sweep, key=lambda r: r["ward"]["sil"])["k"]
    k_ami = max(sweep, key=lambda r: r["ami_calltype"])["k"]
    k_ami_ward = max(sweep, key=lambda r: r["ward"]["ami_calltype"])["k"]

    by_k = {r["k"]: r for r in sweep}
    print("\n=== label-free k selection (labels never used) ===")
    print(f"{'criterion':<20} {'k':>4} | {'kmeans AMI':>11} {'ward AMI':>9}")
    for name, k in picks.items():
        r = by_k.get(k)
        km = r["ami_calltype"] if r else float("nan")
        wd = r["ward"]["ami_calltype"] if r else float("nan")
        print(f"{name:<20} {k:>4} | {km:>11.4f} {wd:>9.4f}")
    print("-" * 48)
    print(f"{'AMI-optimal (uses y)':<20} {k_ami:>4} | {by_k[k_ami]['ami_calltype']:>11.4f} "
          f"{by_k[k_ami]['ward']['ami_calltype']:>9.4f}")
    print(f"{'ward-AMI-opt (uses y)':<20} {k_ami_ward:>4} | "
          f"{by_k[k_ami_ward]['ami_calltype']:>11.4f} "
          f"{by_k[k_ami_ward]['ward']['ami_calltype']:>9.4f}")

    # ---- HDBSCAN: chooses its own cluster count, and may refuse to assign
    hdb = []
    for mcs in (10, 20, 30, 50):
        h = HDBSCAN(min_cluster_size=mcs).fit_predict(X)
        noise = h == -1
        rec = dict(min_cluster_size=mcs, n_clusters=int(len(set(h[~noise]))),
                   noise_frac=round(float(noise.mean()), 3),
                   ami_all_noise_as_cluster=float(AMI(y, h)),
                   ami_assigned_only=float("nan"))
        if (~noise).sum() > 10 and len(set(h[~noise])) > 1:
            rec["ami_assigned_only"] = float(AMI(y[~noise], h[~noise]))
        hdb.append(rec)
    print("\n=== HDBSCAN (picks its own cluster count) ===")
    for r in hdb:
        print(f"  min_cluster_size={r['min_cluster_size']:>3}  "
              f"n_clusters={r['n_clusters']:>3}  noise={r['noise_frac']:>6.1%}  "
              f"AMI(assigned)={r['ami_assigned_only']:.4f}")

    # ---- Ward, as a non-spherical alternative at the label-free k
    k_free = picks["stability_k>=5"]
    ward = AgglomerativeClustering(n_clusters=k_free, linkage="ward").fit_predict(X)
    ward_sc = external(ward, y, birds)
    print(f"\n=== Ward agglomerative at k={k_free} ===")
    print(f"  AMI={ward_sc['ami_calltype']:.4f}  (kmeans at same k: "
          f"{by_k[k_free]['ami_calltype']:.4f})")

    # ---- anatomy at the label-free k and at k=8 (one cluster per human type)
    detail = {}
    for tag, k in (("label_free_k", k_free), ("k8_taxonomy", 8), ("ami_optimal_k", k_ami)):
        c = KMeans(k, n_init=10, random_state=args.seed).fit_predict(X)
        detail[tag] = dict(k=k, scores=external(c, y, birds),
                           clusters=anatomy(c, y, birds, classes),
                           fate=fate(c, y, classes))

    for tag in ("label_free_k", "ami_optimal_k"):
        d = detail[tag]
        print(f"\n=== cluster anatomy: {tag} (k={d['k']}) ===")
        print(f"{'cl':>3} {'n':>5} {'top':>4} {'pur':>5} {'birds':>6} "
              f"{'topbird':>8} {'Hbird':>6}  mix")
        for r in sorted(d["clusters"], key=lambda r: -r["n"]):
            print(f"{r['cluster']:>3} {r['n']:>5} {r['top_type']:>4} {r['purity']:>5.2f} "
                  f"{r['n_birds']:>6} {r['bird_frac_top']:>8.2f} {r['bird_entropy_norm']:>6.2f}  "
                  + ", ".join(f"{a}:{b}" for a, b in r["mix"].items()))
        print(f"  --- call-type fate at k={d['k']} ---")
        print(f"{'type':>5} {'n':>5} {'#cl>=15%':>9} {'largest':>8} {'dom_by':>7}")
        for t, f in d["fate"].items():
            print(f"{t:>5} {f['n']:>5} {f['n_clusters_15pct']:>9} "
                  f"{f['largest_frac']:>8.2f} {f['largest_cluster_dominated_by']:>7}")

    out = dict(
        config=vars(args), n_clips=len(X), n_birds=len(set(birds)), classes=classes,
        sweep=sweep, bic=bic, picks=picks, k_ami_optimal=k_ami, k_ami_optimal_ward=k_ami_ward,
        ami_at_picks={n: dict(k=k, kmeans=by_k[k]["ami_calltype"],
                              ward=by_k[k]["ward"]["ami_calltype"])
                      for n, k in picks.items() if k in by_k},
        hdbscan=hdb, ward_at_label_free_k=dict(k=k_free, **ward_sc), detail=detail,
    )
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

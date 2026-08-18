"""Is Ward's win over k-means real, or did k-means get an unlucky seed / too few restarts?

Ward is deterministic; the swept k-means used a single random_state with n_init=10. Comparing
a deterministic method against one draw of a stochastic one is not a fair fight. So: 20 seeds
at n_init=10, plus one very-high-restart run (n_init=200) to see whether more search closes
the gap. If Ward still beats the BEST of 20 seeds and the high-restart run, the difference is
about the clustering model, not optimization luck.

The correlation between inertia and AMI is the diagnostic that separates two very different
explanations. If it is ~0, k-means' own objective is not aligned with recovering call types
and better fits buy nothing -- interesting, and an argument for a different objective. If it
is negative (lower inertia -> higher AMI), the objective IS aligned, the optimizer is doing
its job, and any remaining gap to Ward is the cluster-shape assumption failing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_mutual_info_score as AMI

from unsup_calltype import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(Path.home() /
                    "Desktop/vocalizations_lab/release/zf_hubert_run11/data/run11_layersweep.npz"))
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 9, 16])
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--n-init-high", type=int, default=200)
    ap.add_argument("--out", default="unsup_kmeans_vs_ward.json")
    args = ap.parse_args()

    X, y, birds, names, classes = load(args.npz, args.layer)
    res = {}
    for k in args.ks:
        amis, inertias = [], []
        for s in range(args.n_seeds):
            km = KMeans(k, n_init=10, random_state=s).fit(X)
            amis.append(float(AMI(y, km.labels_)))
            inertias.append(float(km.inertia_))
        hi = KMeans(k, n_init=args.n_init_high, random_state=0).fit(X)
        w = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)

        best_fit = int(np.argmin(inertias))
        rec = dict(
            kmeans_seed_mean=float(np.mean(amis)), kmeans_seed_sd=float(np.std(amis)),
            kmeans_seed_min=min(amis), kmeans_seed_max=max(amis),
            kmeans_high_restart=float(AMI(y, hi.labels_)),
            kmeans_lowest_inertia_ami=amis[best_fit],
            corr_inertia_ami=float(np.corrcoef(inertias, amis)[0, 1]),
            ward=float(AMI(y, w)),
        )
        res[str(k)] = rec
        print(f"\nk={k}")
        print(f"  kmeans {args.n_seeds} seeds  : AMI mean {rec['kmeans_seed_mean']:.4f} "
              f"sd {rec['kmeans_seed_sd']:.4f} min {rec['kmeans_seed_min']:.4f} "
              f"max {rec['kmeans_seed_max']:.4f}")
        print(f"  kmeans n_init={args.n_init_high:<4}: AMI {rec['kmeans_high_restart']:.4f}")
        print(f"  lowest-inertia seed  : AMI {rec['kmeans_lowest_inertia_ami']:.4f} | "
              f"corr(inertia, AMI) {rec['corr_inertia_ami']:+.3f}")
        print(f"  ward                 : AMI {rec['ward']:.4f}  "
              f"(beats best k-means seed by {rec['ward'] - rec['kmeans_seed_max']:+.4f})")

    Path(args.out).write_text(json.dumps(dict(config=vars(args), results=res), indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

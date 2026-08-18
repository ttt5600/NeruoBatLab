"""Per-cluster anatomy and call-type split/merge fate for WARD clusterings.

unsup_calltype.py prints anatomy for k-means, because that is what the original result used.
Ward turned out to be the better clusterer (see UNSUPERVISED_CALLTYPES.md §1), so the tables
that describe "what did it actually find" need to describe the Ward partition, not the
k-means one. Reporting k-means anatomy alongside a Ward headline number would be quietly
mismatched -- the clusters in the table would not be the clusters the number came from.

Defaults are k=9 (Ward's AMI peak) and k=13 (the k that label-free stability selects).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.cluster import AgglomerativeClustering

from unsup_calltype import anatomy, external, fate, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(Path.home() /
                    "Desktop/vocalizations_lab/release/zf_hubert_run11/data/run11_layersweep.npz"))
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--ks", type=int, nargs="+", default=[9, 13])
    ap.add_argument("--out", default="results/ward_anatomy.json")
    args = ap.parse_args()

    X, y, birds, names, classes = load(args.npz, args.layer)
    out = {}
    for k in args.ks:
        c = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        sc, an, ft = external(c, y, birds), anatomy(c, y, birds, classes), fate(c, y, classes)
        out[str(k)] = dict(scores=sc, clusters=an, fate=ft)

        print(f"\n=== WARD k={k} | AMI {sc['ami_calltype']:.4f} "
              f"bird {sc['ami_bird']:.4f} within-bird {sc['ami_within_bird']:.4f} ===")
        print(f"{'cl':>3} {'n':>5} {'top':>4} {'pur':>5} {'birds':>6} {'topbird':>8}  mix")
        for r in sorted(an, key=lambda r: -r["n"]):
            print(f"{r['cluster']:>3} {r['n']:>5} {r['top_type']:>4} {r['purity']:>5.2f} "
                  f"{r['n_birds']:>6} {r['bird_frac_top']:>8.2f}  "
                  + ", ".join(f"{a}:{b}" for a, b in r["mix"].items()))
        print(f"  {'type':>5} {'n':>5} {'#cl>=15%':>9} {'largest':>8} {'dom_by':>7}")
        for t, f in ft.items():
            print(f"  {t:>5} {f['n']:>5} {f['n_clusters_15pct']:>9} "
                  f"{f['largest_frac']:>8.2f} {f['largest_cluster_dominated_by']:>7}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

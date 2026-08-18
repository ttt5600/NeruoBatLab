"""What is HDBSCAN's confident core, and is its high AMI earned or an artifact of easy clips?

HDBSCAN at min_cluster_size=20 finds 9 clusters, declares 60% of clips noise, and scores
AMI 0.774 on the 40% it does assign -- far above anything k-means or Ward reach on the full
set. That comparison is NOT apples-to-apples: AMI on a self-selected subset is measured on
whatever the method found easy, so a high number is partly a statement about the subset.

The check that decides whether it means anything: if the assigned core is mostly Song (which
every method separates trivially, being 10x longer than any other call), the 0.774 is hollow.
If the core spans all eight types at roughly their corpus proportions, then there really is a
dense, cleanly-separated region of the embedding covering the whole taxonomy, and the diffuse
60% is where the ambiguity lives -- which is directly actionable: auto-label the core, route
the halo to a human.

Also reported: a like-for-like control -- Ward restricted to the SAME assigned clips. If Ward
scores similarly there, the core is just an easy subset and HDBSCAN's contribution is only
that it identified it. If HDBSCAN still leads, its density model is genuinely doing more.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN, AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score as AMI

from unsup_calltype import anatomy, external, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(Path.home() /
                    "Desktop/vocalizations_lab/release/zf_hubert_run11/data/run11_layersweep.npz"))
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--min-cluster-size", type=int, default=20)
    ap.add_argument("--out", default="unsup_hdbscan_core.json")
    args = ap.parse_args()

    X, y, birds, names, classes = load(args.npz, args.layer)
    h = HDBSCAN(min_cluster_size=args.min_cluster_size).fit_predict(X)
    core = h != -1
    print(f"{len(X)} clips | core {core.sum()} ({core.mean():.1%}) | "
          f"noise {(~core).sum()} ({1 - core.mean():.1%}) | "
          f"{len(set(h[core]))} clusters")

    print(f"\n=== is the core representative, or just Song? ===")
    print(f"{'type':>5} {'corpus':>7} {'core':>6} {'noise':>6} {'core rate':>10}")
    comp = {}
    for ci, cn in enumerate(classes):
        m = y == ci
        n_core = int((m & core).sum())
        comp[cn] = dict(corpus=int(m.sum()), core=n_core,
                        core_rate=round(n_core / max(int(m.sum()), 1), 3))
        print(f"{cn:>5} {int(m.sum()):>7} {n_core:>6} {int((m & ~core).sum()):>6} "
              f"{comp[cn]['core_rate']:>10.1%}")

    # AMI on the core, with and without Song, to see how much of it Song is carrying.
    so = classes.index("So") if "So" in classes else None
    res = dict(n_core=int(core.sum()), noise_frac=float(1 - core.mean()),
               n_clusters=int(len(set(h[core]))),
               ami_core=float(AMI(y[core], h[core])), composition=comp)
    if so is not None:
        keep = core & (y != so)
        res["ami_core_excl_song"] = float(AMI(y[keep], h[keep]))
        res["n_core_excl_song"] = int(keep.sum())
        print(f"\nAMI on core                {res['ami_core']:.4f}  (n={core.sum()})")
        print(f"AMI on core excluding Song {res['ami_core_excl_song']:.4f}  (n={keep.sum()})")

    # Like-for-like: Ward on the same clips, same cluster count.
    k = len(set(h[core]))
    w = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X[core])
    res["ward_same_clips"] = external(w, y[core], birds[core])
    print(f"\nWard on the SAME {core.sum()} clips at k={k}: "
          f"AMI {res['ward_same_clips']['ami_calltype']:.4f} "
          f"(HDBSCAN {res['ami_core']:.4f})")

    res["clusters"] = anatomy(h[core], y[core], birds[core], classes)
    print(f"\n=== HDBSCAN core clusters ===")
    print(f"{'cl':>3} {'n':>5} {'top':>4} {'pur':>5} {'birds':>6} {'topbird':>8}  mix")
    for r in sorted(res["clusters"], key=lambda r: -r["n"]):
        print(f"{r['cluster']:>3} {r['n']:>5} {r['top_type']:>4} {r['purity']:>5.2f} "
              f"{r['n_birds']:>6} {r['bird_frac_top']:>8.2f}  "
              + ", ".join(f"{a}:{b}" for a, b in r["mix"].items()))

    # Which call types does the noise halo over-represent? That is where a human is needed.
    print(f"\n=== hardest types (lowest core rate = most often refused) ===")
    for cn, v in sorted(comp.items(), key=lambda kv: kv[1]["core_rate"]):
        print(f"  {cn:>3}  core rate {v['core_rate']:>6.1%}  ({v['core']}/{v['corpus']})")

    Path(args.out).write_text(json.dumps(dict(config=vars(args), **res), indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

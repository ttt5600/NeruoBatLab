"""Which layer should an unsupervised user tap, and could they tell without labels?

`layer 3` is the published choice, but it was chosen by a supervised probe (leave-birds-out
accuracy 0.811, spread across layers only 0.022 -- smaller than the 0.034 fold sd, so it was
already a weak preference). Someone clustering an unlabeled corpus cannot run that probe.
So there are two distinct questions:

  1. Does the layer that maximizes unsupervised agreement (AMI) match the layer the
     supervised probe liked? If clustering peaks somewhere else, "layer 3" is advice about
     probes, not about representations.
  2. Would a label-free criterion (silhouette, stability) have identified the right layer?
     If the label-free curve is flat or peaks in the wrong place, then layer choice is
     something an unsupervised user simply cannot get right on their own, and that limitation
     belongs in the README rather than being papered over.

Run after unsup_calltype.py; it reuses that module's loaders and metrics so the two results
are directly comparable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans

from unsup_calltype import external, internal, load, stability


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(Path.home() /
                    "Desktop/vocalizations_lab/release/zf_hubert_run11/data/run11_layersweep.npz"))
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--n-boot", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="unsup_layer_sweep.json")
    args = ap.parse_args()

    rows = []
    for layer in range(12):
        X, y, birds, names, classes = load(args.npz, layer)
        rec = dict(layer=layer, per_k={})
        for k in args.ks:
            c = KMeans(k, n_init=10, random_state=args.seed).fit_predict(X)
            w = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
            st, _ = stability(X, k, args.n_boot, 0.8, args.seed + layer)
            rec["per_k"][str(k)] = dict(
                stability=st,
                kmeans={**external(c, y, birds), **internal(X, c)},
                ward={**external(w, y, birds), **internal(X, w)},
            )
        rows.append(rec)
        r8 = rec["per_k"][str(args.ks[0])]
        print(f"layer {layer:>2} | k={args.ks[0]} stab={r8['stability']:.3f} "
              f"sil={r8['kmeans']['sil']:.4f} | kmAMI={r8['kmeans']['ami_calltype']:.4f} "
              f"wardAMI={r8['ward']['ami_calltype']:.4f} "
              f"AMIbird={r8['kmeans']['ami_bird']:.4f}", flush=True)

    print("\n=== per-k layer picks ===")
    summary = {}
    for k in args.ks:
        sk = str(k)
        best_ami_km = max(rows, key=lambda r: r["per_k"][sk]["kmeans"]["ami_calltype"])
        best_ami_wd = max(rows, key=lambda r: r["per_k"][sk]["ward"]["ami_calltype"])
        best_sil = max(rows, key=lambda r: r["per_k"][sk]["kmeans"]["sil"])
        best_stab = max(rows, key=lambda r: r["per_k"][sk]["stability"])
        amis = [r["per_k"][sk]["kmeans"]["ami_calltype"] for r in rows]
        summary[sk] = dict(
            best_layer_ami_kmeans=best_ami_km["layer"],
            best_layer_ami_ward=best_ami_wd["layer"],
            best_layer_silhouette=best_sil["layer"],
            best_layer_stability=best_stab["layer"],
            ami_spread_across_layers=round(max(amis) - min(amis), 4),
            ami_at_layer3=round(rows[3]["per_k"][sk]["kmeans"]["ami_calltype"], 4),
            ami_best=round(max(amis), 4),
        )
        s = summary[sk]
        print(f"k={k}: AMI-best layer (kmeans) = {s['best_layer_ami_kmeans']}, "
              f"(ward) = {s['best_layer_ami_ward']}; "
              f"silhouette-best = {s['best_layer_silhouette']}, "
              f"stability-best = {s['best_layer_stability']}; "
              f"layer-3 AMI {s['ami_at_layer3']} vs best {s['ami_best']} "
              f"(spread {s['ami_spread_across_layers']})")

    Path(args.out).write_text(json.dumps(dict(config=vars(args), layers=rows,
                                              summary=summary), indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

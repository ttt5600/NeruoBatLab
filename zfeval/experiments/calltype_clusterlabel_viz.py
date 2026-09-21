#!/usr/bin/env python
"""Accuracy of cluster-then-vote against k, with the silhouette choice marked."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2] / "paper" / "figures" / "calltype"
d = json.loads((OUT / "calltype_clusterlabel.json").read_text())
ks = d["ks"]; C_R, C_A = "#C2410C", "#1D4ED8"
probe, maj = d["probe_acc_reference"], d["majority"]

fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.6), sharey=True)
for ax, algo in zip(axes, ("ward", "kmeans")):
    for nm, c, lbl in (("run11", C_R, "run11"), ("aves", C_A, "AVES")):
        acc = [d[nm][algo]["acc_by_k"][str(k)] for k in ks]
        ax.plot(ks, acc, "o-", color=c, lw=2, ms=5, label=lbl)
        ax.axhline(probe[nm], color=c, ls=":", lw=1.4, alpha=.85)
        # where silhouette actually landed, and what it cost
        s = d[nm][algo]["acc_silhouette"]
        kc = d[nm][algo]["k_chosen"]
        ax.scatter([np.median(kc)], [s], s=210, marker="X", color=c, zorder=6,
                   edgecolors="white", linewidths=1.5)
    ax.axhline(maj, color="#888", ls="--", lw=1.2)
    ax.set_xscale("log"); ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks], fontsize=9)
    ax.set_xlabel("number of clusters k", fontsize=11)
    ax.set_title(algo.replace("kmeans", "k-means").title() if algo == "ward" else "k-means",
                 fontsize=13, pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.22, lw=.6); ax.set_axisbelow(True)
axes[0].set_ylabel("test accuracy, 11 classes", fontsize=11)
axes[0].set_ylim(0.15, 0.90)
axes[0].text(4.1, maj + .012, f"majority floor {maj:.3f}", fontsize=9, color="#666")
axes[0].text(4.1, probe["aves"] + .012, "dotted = supervised probe", fontsize=9, color="#666")
axes[0].legend(frameon=False, fontsize=10.5, loc="lower right")
axes[1].text(0.5, 0.06, "X  =  where silhouette chose k", transform=axes[1].transAxes,
             ha="center", fontsize=10, color="#444",
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#ddd"))
fig.suptitle("Cluster without labels, give each cluster its majority label, predict on held-out "
             "birds.\nAVES is ahead at every matched k — the apparent run11 win is entirely "
             "silhouette choosing k=4 for AVES.", fontsize=12.5, y=1.04)
fig.tight_layout(); fig.savefig(OUT / "10_cluster_vote.png", dpi=140, bbox_inches="tight")
print("wrote 10_cluster_vote.png")

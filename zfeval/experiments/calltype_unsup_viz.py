#!/usr/bin/env python
"""Figures for the decoder-free comparison. Numbers come from calltype_unsup.json; nothing is recomputed."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2] / "paper" / "figures" / "calltype"
d = json.loads((OUT / "calltype_unsup.json").read_text())
C_R, C_A = "#C2410C", "#1D4ED8"
sw, L = d["sweep"], np.arange(12)


def fig_sweep(path):
    """The headline: three decoder-free measures against depth. Two of them disagree."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.1))
    panels = [("ward_ami", "Ward clustering AMI", "global structure:\ndo the groups match call type?"),
              ("knn", "kNN-10 agreement", "local structure:\ndo neighbours share a call type?"),
              ("sil", "silhouette vs true labels", "compactness:\nare the real classes tight and apart?")]
    for ax, (key, title, sub) in zip(axes, panels):
        r, a = sw["run11"][key], sw["aves"][key]
        ax.plot(L, r, "o-", color=C_R, lw=2, ms=5.5, label="run11")
        ax.plot(L, a, "o-", color=C_A, lw=2, ms=5.5, label="AVES")
        if key == "sil":
            ax.axhline(0, color="#999", lw=1, ls="--")
        win = "AVES" if np.mean(a) > np.mean(r) else "run11"
        n_a = int(sum(1 for i in range(12) if a[i] > r[i]))
        ax.set_title(title, fontsize=12.5, pad=8)
        ax.text(0.5, 1.10, sub, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=9.5, color="#666", linespacing=1.3)
        ax.set_xlabel("layer", fontsize=10.5)
        ax.set_xticks(L[::2])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.22, lw=0.6); ax.set_axisbelow(True)
        ax.text(0.97, 0.04 if key != "knn" else 0.92, f"AVES ahead at {n_a}/12",
                transform=ax.transAxes, ha="right", fontsize=9.5,
                color=C_A if n_a > 6 else C_R,
                bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#ddd", lw=0.8))
    axes[0].legend(frameon=False, fontsize=10.5, loc="lower left")
    fig.suptitle("Three decoder-free measures. Clustering and compactness favour AVES at nearly "
                 "every depth; nearest-neighbour agreement favours run11 at every depth.",
                 fontsize=12.5, y=1.06)
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("  wrote", path.name)


def fig_algos(path):
    """Does the verdict depend on which clustering algorithm we chose? No."""
    al = d["algorithms"]
    names = ["ward", "kmeans", "hdbscan"]
    disp = ["Ward", "k-means", f"HDBSCAN\n(chose k={al['run11']['hdbscan']['k']})"]
    x = np.arange(3); w = 0.37
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.2),
                                  gridspec_kw={"width_ratios": [1.25, 1]})
    r = [al["run11"][n]["ami"] for n in names]
    a = [al["aves"][n]["ami"] for n in names]
    ax.bar(x - w/2, r, w, color=C_R, label="run11")
    ax.bar(x + w/2, a, w, color=C_A, label="AVES")
    for i in range(3):
        ax.text(x[i]-w/2, r[i]+.012, f"{r[i]:.3f}", ha="center", fontsize=9.5, color=C_R)
        ax.text(x[i]+w/2, a[i]+.012, f"{a[i]:.3f}", ha="center", fontsize=9.5, color=C_A)
    ax.set_xticks(x); ax.set_xticklabels(disp, fontsize=10.5)
    ax.set_ylabel("AMI vs call type", fontsize=11); ax.set_ylim(0, 0.78)
    ax.legend(frameon=False, fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.22, lw=0.6); ax.set_axisbelow(True)
    ax.set_title("AVES leads whichever algorithm does the clustering", fontsize=12.5, pad=10)

    # the confound, same three algorithms
    rb = [al["run11"][n]["ami_bird"] for n in names]
    ab = [al["aves"][n]["ami_bird"] for n in names]
    ax2.bar(x - w/2, rb, w, color=C_R, alpha=.62, label="run11")
    ax2.bar(x + w/2, ab, w, color=C_A, alpha=.62, label="AVES")
    ax2.set_xticks(x); ax2.set_xticklabels([s.split("\n")[0] for s in disp], fontsize=10.5)
    ax2.set_ylabel("AMI vs BIRD IDENTITY", fontsize=11); ax2.set_ylim(0, 0.78)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", alpha=0.22, lw=0.6); ax2.set_axisbelow(True)
    ax2.set_title("…and is less contaminated by who is calling", fontsize=12.5, pad=10)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    print("  wrote", path.name)


def fig_reduction(path):
    """Does UMAP change the answer? It changes the SIZE of the gap, not its sign -- except for kNN."""
    red = d["reduction"]
    spaces = ["full", "umap10", "umap2"]
    disp = ["full 768-d", "UMAP 10-d", "UMAP 2-d"]
    metrics = [("ward_ami", "Ward AMI"), ("knn", "kNN-10"), ("sil", "silhouette")]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    x = np.arange(3); w = 0.37
    for ax, (m, title) in zip(axes, metrics):
        r = [red["run11"][s][m] for s in spaces]
        a = [red["aves"][s][m] for s in spaces]
        ax.bar(x - w/2, r, w, color=C_R, label="run11")
        ax.bar(x + w/2, a, w, color=C_A, label="AVES")
        for i in range(3):
            gap = a[i] - r[i]
            ax.text(x[i], max(r[i], a[i]) + (0.02 if m != "knn" else 0.006),
                    f"{gap:+.3f}", ha="center", fontsize=9.5,
                    color=C_A if gap > 0 else C_R, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(disp, fontsize=10.5)
        ax.set_title(title, fontsize=12.5, pad=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.22, lw=0.6); ax.set_axisbelow(True)
        if m == "knn":
            ax.set_ylim(0.80, 0.885)
    axes[0].legend(frameon=False, fontsize=10.5, loc="upper left")
    axes[0].set_ylabel("score", fontsize=11)
    fig.suptitle("Recomputed after UMAP. The gap label is AVES − run11: reduction shrinks the "
                 "clustering gap and flips kNN at 10-d, but never hands the win to run11.",
                 fontsize=12.5, y=1.02)
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("  wrote", path.name)


if __name__ == "__main__":
    fig_sweep(OUT / "07_unsup_sweep.png")
    fig_algos(OUT / "08_unsup_algos.png")
    fig_reduction(OUT / "09_unsup_reduction.png")
    print("done")

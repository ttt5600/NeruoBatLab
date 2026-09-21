#!/usr/bin/env python
"""Figures 11-13: can a label-free criterion pick the clustering?"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

OUT = Path(__file__).resolve().parents[2] / "paper" / "figures" / "calltype"
B = json.loads((OUT / "calltype_blindselect.json").read_text())
K = json.loads((OUT / "calltype_kceiling.json").read_text())
REAL = [e for e in B["encoders"] if e != "NULL-shuffled"]
NULL = "NULL-shuffled"
KS = B["ks"]
COL = {"run11": "#C2410C", "aves": "#1D4ED8", "aves-base-all": "#0F766E",
       "aves-base-core": "#7C3AED", "birdaves-biox-base": "#DB2777", NULL: "#94A3B8"}
LBL = {"run11": "run11 (ZF)", "aves": "AVES", "aves-base-all": "AVES-base-all",
       "aves-base-core": "AVES-base-core", "birdaves-biox-base": "birdAVES-biox",
       NULL: "shuffled null"}


def fig11():
    """Accuracy vs k runs all the way to the 1-NN ceiling -- there is no interior optimum."""
    fig, ax = plt.subplots(figsize=(10.6, 6.2))
    for e in REAL + [NULL]:
        d = K["encoders"][e]["vote_acc"]
        ks = sorted(int(k) for k in d)
        ax.plot(ks, [d[str(k)] for k in ks], "o-", color=COL[e], lw=2.1, ms=4.5,
                label=LBL[e], alpha=.95 if e != NULL else .8,
                ls="--" if e == NULL else "-")
        nn = K["encoders"][e]["knn1"]
        ax.plot([ks[-1] * 1.15], [nn], marker="<", color=COL[e], ms=9, clip_on=False)
        ax.hlines(nn, ks[-1], ks[-1] * 1.15, color=COL[e], lw=1.1, ls=":")
    ax.axhline(K["majority"], color="#64748B", ls="--", lw=1.2)
    ax.text(4.4, K["majority"] + .012, "majority class", fontsize=9, color="#475569")
    ax.text(1750, 0.815, "1-NN\nceiling", fontsize=9, color="#334155", ha="center")
    ax.set_xscale("log")
    ax.set_xticks([4, 8, 16, 30, 70, 140, 300, 650, 1600])
    ax.set_xticklabels(["4", "8", "16", "30", "70", "140", "300", "650", "1600"], fontsize=9.5)
    ax.set_xlabel("number of clusters  k   (log scale)", fontsize=11.5)
    ax.set_ylabel("cluster-vote accuracy, 11 classes", fontsize=11.5)
    ax.set_title("No interior optimum: cluster-then-vote climbs to nearest-neighbour\n"
                 "the null is the only curve that peaks and falls back",
                 fontsize=13, pad=10)
    ax.legend(fontsize=9.5, frameon=False, loc="upper left", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.22, lw=.6); ax.set_axisbelow(True)
    ax.set_ylim(0.08, 0.86)
    fig.tight_layout(); fig.savefig(OUT / "11_k_ceiling.png", dpi=155); plt.close(fig)


def fig12():
    """Spearman rho between each blind criterion and true accuracy, across the 5 encoders."""
    crits = [("stab_bird", "bird-held-out stability", "#0F766E"),
             ("stab_clip", "bootstrap stability", "#14B8A6"),
             ("prediction_strength", "prediction strength", "#84CC16"),
             ("gap", "gap statistic", "#A3A3A3"),
             ("neg_davies_bouldin", "Davies-Bouldin", "#F59E0B"),
             ("silhouette", "silhouette", "#DC2626"),
             ("calinski_harabasz", "Calinski-Harabasz", "#991B1B")]
    fig, axes = plt.subplots(1, 2, figsize=(15.4, 5.9),
                             gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    for c, lbl, col in crits:
        r = []
        for k in KS:
            v = np.array([B["encoders"][e]["criteria"][c][str(k)] for e in REAL])
            a = np.array([B["encoders"][e]["truth"]["vote_acc"][str(k)] for e in REAL])
            r.append(spearmanr(v, a).statistic)
        ax.plot(KS, r, "o-", color=col, lw=2.1, ms=4.5, label=lbl,
                alpha=.95, ls="-" if "stab" in c else "--")
    am = [spearmanr(np.array([B["encoders"][e]["truth"]["ami"][str(k)] for e in REAL]),
                    np.array([B["encoders"][e]["truth"]["vote_acc"][str(k)] for e in REAL])
                    ).statistic for k in KS]
    ax.plot(KS, am, "-", color="#111827", lw=1.6, alpha=.55, label="AMI (uses labels)")
    ax.axhline(0, color="#111827", lw=1.1)
    ax.fill_between([3.6, 78], -1.05, 0, color="#DC2626", alpha=.045)
    ax.text(4.1, -0.09, "below this line the criterion ranks encoders BACKWARDS",
            fontsize=9.3, color="#B91C1C", va="top")
    ax.set_xscale("log"); ax.set_xticks(KS)
    ax.set_xticklabels([str(k) for k in KS], fontsize=9)
    ax.set_xlim(3.6, 78); ax.set_ylim(-1.06, 1.06)
    ax.set_xlabel("number of clusters  k", fontsize=11.5)
    ax.set_ylabel(r"Spearman $\rho$  vs true accuracy   (5 encoders)", fontsize=11.5)
    ax.set_title("Only reproducibility criteria rank the encoders correctly", fontsize=13, pad=9)
    ax.legend(fontsize=8.8, frameon=False, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, -0.32), columnspacing=1.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.22, lw=.6); ax.set_axisbelow(True)

    ax = axes[1]
    means, labels, cols = [], [], []
    for c, lbl, col in crits:
        rr = []
        for k in KS:
            v = np.array([B["encoders"][e]["criteria"][c][str(k)] for e in REAL])
            a = np.array([B["encoders"][e]["truth"]["vote_acc"][str(k)] for e in REAL])
            rr.append(spearmanr(v, a).statistic)
        means.append(np.mean(rr)); labels.append(lbl); cols.append(col)
    means.append(np.mean(am)); labels.append("AMI (uses labels)"); cols.append("#111827")
    order = np.argsort(means)
    ax.barh([labels[i] for i in order], [means[i] for i in order],
            color=[cols[i] for i in order], height=.66)
    for i, o in enumerate(order):
        v = means[o]
        ax.text(v + (.03 if v >= 0 else -.03), i, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9.5)
    ax.axvline(0, color="#111827", lw=1.1)
    ax.set_xlim(-0.95, 1.0)
    ax.set_xlabel(r"mean $\rho$ over the 12 values of k", fontsize=11.5)
    ax.set_title("Averaged over k", fontsize=13, pad=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=9.5)
    ax.grid(axis="x", alpha=.22, lw=.6); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(OUT / "12_rho_profile.png", dpi=155); plt.close(fig)


def fig13():
    """Which criteria notice that the embedding has been destroyed?"""
    crits = [("stab_clip", "bootstrap stability"), ("stab_bird", "bird-held-out stability"),
             ("calinski_harabasz", "Calinski-Harabasz"), ("prediction_strength", "prediction strength"),
             ("silhouette", "silhouette"), ("neg_davies_bouldin", "Davies-Bouldin"),
             ("gap", "gap statistic")]
    kfix = 30
    rows = []
    for c, lbl in crits:
        r = np.median([B["encoders"][e]["criteria"][c][str(kfix)] for e in REAL])
        n = B["encoders"][NULL]["criteria"][c][str(kfix)]
        span = abs(r - n) / (abs(r) + abs(n) + 1e-12)
        rows.append((lbl, r, n, span))
    rows.sort(key=lambda x: x[3])
    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    y = np.arange(len(rows))
    for i, (lbl, r, n, span) in enumerate(rows):
        col = "#0F766E" if span > .5 else "#DC2626"
        ax.barh(i, span, color=col, height=.62)
        ax.text(span + .015, i, f"real {r:.3g}  vs  null {n:.3g}",
                va="center", fontsize=9.3, color="#334155")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=10)
    ax.set_xlim(0, 1.42)
    ax.set_xticks([0, .25, .5, .75, 1.0])
    ax.axvline(.5, color="#94A3B8", ls="--", lw=1.2)
    ax.set_xlabel(f"separation from a column-shuffled embedding at k={kfix}\n"
                  r"$|real-null| \, / \, (|real|+|null|)$   —   1.0 means the null scores zero",
                  fontsize=11)
    ax.set_title("Can the criterion tell a real embedding from destroyed one?\n"
                 "The gap statistic cannot", fontsize=13, pad=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=.22, lw=.6); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(OUT / "13_null_separation.png", dpi=155); plt.close(fig)


if __name__ == "__main__":
    fig11(); fig12(); fig13()
    print("wrote 11_k_ceiling.png, 12_rho_profile.png, 13_null_separation.png")

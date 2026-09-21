#!/usr/bin/env python
"""What the call-type space actually looks like, for run11 and for AVES.

calltype11.py already established the headline: leave-birds-out, 11 classes, AVES L3 beats
run11 L3 by -0.0338 [-0.0609, -0.0069] and run11 loses at all 12 layers. That is one number.
It does not say WHICH calls are confused, whether the two encoders fail on the SAME calls, or
whether either space is organised by call type at all rather than by which bird is calling.

Five views:

  1 confusion      out-of-fold confusion matrices, row-normalised, at each encoder's best layer.
  2 per_class      per-class recall side by side, ordered by support, with the class counts --
                   a 4-clip class and a 600-clip class should not be read the same way.
  3 layers         accuracy against depth for both encoders. The shape matters: if both decline
                   with depth, the useful representation is early and the deep blocks are
                   specialising for the pretraining objective, not for this task.
  4 umap_type      UMAP coloured by call type, both encoders, same treatment.
  5 umap_bird      THE CONFOUND. Identical coordinates, coloured by bird identity. If the space
                   is organised by who is calling rather than by what was called, view 4 is
                   showing identity wearing a call-type costume.

  + clustering     Ward AMI against call type and against identity, swept over k. Label-free:
                   a probe can carve structure that is not natively there, clustering cannot.

UMAP is for looking, never for measuring. Every number here is computed in the full 768-d
space; the 2-d embedding is only ever plotted.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import calltype11 as C11                                                  # noqa: E402
from aves_calltype import cv_acc, bird_bootstrap                          # noqa: E402

from sklearn.metrics import confusion_matrix, adjusted_mutual_info_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parents[2] / "paper" / "figures" / "calltype"
FEAT = C11.FEAT

# Encoder identity is carried by colour consistently across every figure.
C_RUN11, C_AVES = "#C2410C", "#1D4ED8"


def cohort():
    """Rebuild calltype11's cohort EXACTLY -- the cached embeddings are positional."""
    rows = [r for r in C11.collect() if r[3] in C11.KEEP11]
    birds = np.array([r[1].lower() for r in rows])
    tt = np.array([r[3] for r in rows])
    src = np.array([r[4] for r in rows])
    classes = sorted(set(tt))
    y = np.array([classes.index(t) for t in tt])
    return y, birds, tt, src, classes


def load(name, n_expected):
    p = FEAT / f"ct11_{name}_emb.npy"
    if not p.exists():
        sys.exit(f"FATAL: {p} missing -- run calltype11.py first")
    E = np.load(p)
    if E.shape[0] != n_expected:
        sys.exit(f"FATAL: {p} has {E.shape[0]} rows, cohort has {n_expected}. The cache is "
                 "positional; a mismatch means it was built from a different clip set.")
    return E


def fig_confusion(cm_r, cm_a, classes, counts, accs, path):
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.2))
    for ax, cm, nm, acc, col in ((axes[0], cm_r, "run11", accs[0], C_RUN11),
                                 (axes[1], cm_a, "AVES", accs[1], C_AVES)):
        im = ax.imshow(cm, cmap="magma_r", vmin=0, vmax=1, aspect="equal")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels([f"{c}  ({n})" for c, n in zip(classes, counts)], fontsize=10)
        ax.set_xlabel("predicted", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("true  (n clips)", fontsize=11)
        ax.set_title(f"{nm}   L{acc[0]}   acc {acc[1]:.4f}", fontsize=13, color=col, pad=10)
        for i in range(len(classes)):
            for j in range(len(classes)):
                v = cm[i, j]
                if v >= 0.005:
                    ax.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center",
                            fontsize=8, color="white" if v > 0.5 else "#333")
        for s in ax.spines.values():
            s.set_visible(False)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="fraction of true class")
    fig.suptitle("Out-of-fold confusion, leave-birds-out. Rows sum to 1; the diagonal is recall.",
                 fontsize=12.5, y=0.98)
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("  wrote", path.name, flush=True)


def fig_per_class(rec_r, rec_a, classes, counts, path):
    order = np.argsort(-np.array(counts))
    x = np.arange(len(classes)); w = 0.38
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    ax.bar(x - w/2, rec_r[order], w, label="run11", color=C_RUN11)
    ax.bar(x + w/2, rec_a[order], w, label="AVES", color=C_AVES)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{classes[i]}\nn={counts[i]}" for i in order], fontsize=9.5)
    ax.set_ylabel("out-of-fold recall", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.axhline(0, color="#333", lw=0.8)
    ax.legend(frameon=False, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Per-class recall, ordered by support. Rare classes carry almost no weight in "
                 "the headline accuracy.", fontsize=12.5, pad=12)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    print("  wrote", path.name, flush=True)


def fig_layers(ra, aa, path):
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    L = np.arange(12)
    ax.plot(L, ra, "o-", color=C_RUN11, lw=2, ms=6, label="run11 (ZF-specific, 116 h)")
    ax.plot(L, aa, "o-", color=C_AVES, lw=2, ms=6, label="AVES (generic animal audio, 360 h)")
    for a, c in ((ra, C_RUN11), (aa, C_AVES)):
        b = int(np.argmax(a))
        ax.scatter([b], [a[b]], s=190, facecolors="none", edgecolors=c, lw=2, zorder=5)
    ax.set_xticks(L); ax.set_xlabel("transformer layer", fontsize=11)
    ax.set_ylabel("out-of-fold accuracy (11 classes)", fontsize=11)
    ax.legend(frameon=False, fontsize=10.5, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6); ax.set_axisbelow(True)
    ax.set_title("AVES is above run11 at every depth. Circles mark each encoder's best layer.",
                 fontsize=12.5, pad=12)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    print("  wrote", path.name, flush=True)


def _umap(X, seed=0):
    import umap
    Z = StandardScaler().fit_transform(X)
    return umap.UMAP(n_neighbors=25, min_dist=0.12, metric="euclidean",
                     random_state=seed).fit_transform(Z)


def fig_umap(U_r, U_a, colour, names, title, path, legend_title, max_legend=11):
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.4))
    uniq = list(dict.fromkeys(names))
    cmap = plt.get_cmap("tab20" if len(uniq) > 10 else "tab10")
    cols = {u: cmap(i % cmap.N) for i, u in enumerate(uniq)}
    for ax, U, nm, c in ((axes[0], U_r, "run11", C_RUN11), (axes[1], U_a, "AVES", C_AVES)):
        for u in uniq:
            m = colour == u
            ax.scatter(U[m, 0], U[m, 1], s=7, alpha=0.72, color=cols[u], linewidths=0,
                       label=u if len(uniq) <= max_legend else None)
        ax.set_title(nm, fontsize=13, color=c, pad=8)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#ddd")
    if len(uniq) <= max_legend:
        axes[1].legend(frameon=False, fontsize=9.5, markerscale=2.4,
                       loc="center left", bbox_to_anchor=(1.01, 0.5), title=legend_title)
    fig.suptitle(title, fontsize=12.5, y=0.97)
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("  wrote", path.name, flush=True)


def fig_clustering(res, ks, path):
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    for nm, c, ls in (("run11", C_RUN11, "-"), ("aves", C_AVES, "-")):
        ax.plot(ks, res[nm]["ami_type"], ls, marker="o", color=c, lw=2, ms=5,
                label=f"{'run11' if nm=='run11' else 'AVES'} — vs call type")
        ax.plot(ks, res[nm]["ami_bird"], "--", marker="x", color=c, lw=1.6, ms=5, alpha=0.75,
                label=f"{'run11' if nm=='run11' else 'AVES'} — vs bird identity")
    ax.axvline(11, color="#888", lw=1, ls=":")
    ax.text(11, ax.get_ylim()[1]*0.97, " k = 11 classes", fontsize=9, color="#666", va="top")
    ax.set_xlabel("number of Ward clusters (k)", fontsize=11)
    ax.set_ylabel("adjusted mutual information", fontsize=11)
    ax.legend(frameon=False, fontsize=9.5, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6); ax.set_axisbelow(True)
    ax.set_title("Label-free Ward clustering. Solid = recovers call type, dashed = recovers "
                 "identity.\nIdentity well below type means the space is not just bird-coding.",
                 fontsize=12.5, pad=12)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    print("  wrote", path.name, flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    y, birds, tt, src, classes = cohort()
    n = len(y)
    counts = [int((tt == c).sum()) for c in classes]
    print(f"[cohort] {n} clips, {len(classes)} classes, {len(set(birds))} birds, "
          f"majority {np.bincount(y).max()/n:.4f}", flush=True)

    E = {nm: load(nm, n) for nm in ("run11", "aves")}
    prev = json.loads((C11.ANA / "calltype11.json").read_text())
    ra = prev["run11"]["per_layer_acc"]; aa = prev["aves"]["per_layer_acc"]
    rb = prev["run11"]["best"]["layer"]; ab = prev["aves"]["best"]["layer"]
    print(f"[prior] run11 best L{rb} {ra[rb]:.4f} | AVES best L{ab} {aa[ab]:.4f}", flush=True)

    out = {"n_clips": n, "n_birds": int(len(set(birds))), "classes": classes,
           "class_counts": dict(zip(classes, counts)),
           "majority": float(np.bincount(y).max() / n),
           "best_layer": {"run11": rb, "aves": ab},
           "acc": {"run11": ra[rb], "aves": aa[ab]},
           "split": "leave-birds-out StratifiedGroupKFold(5)"}

    print("[1/6] out-of-fold predictions at the best layers", flush=True)
    _, _, Pr = cv_acc(E["run11"][:, rb], y, birds, return_proba=True)
    _, _, Pa = cv_acc(E["aves"][:, ab], y, birds, return_proba=True)
    pr, pa = Pr.argmax(1), Pa.argmax(1)
    out["bootstrap_run11_vs_aves"] = bird_bootstrap(y, Pr, Pa, birds)
    print(f"      bootstrap {out['bootstrap_run11_vs_aves']}", flush=True)

    # Where do the two encoders fail on the SAME clip? A shared failure is a property of the
    # task or the labels; a divergent one is a property of the encoder.
    er, ea = pr != y, pa != y
    out["errors"] = {"run11_wrong": int(er.sum()), "aves_wrong": int(ea.sum()),
                     "both_wrong": int((er & ea).sum()),
                     "only_run11_wrong": int((er & ~ea).sum()),
                     "only_aves_wrong": int((ea & ~er).sum()),
                     "jaccard": float((er & ea).sum() / max((er | ea).sum(), 1))}
    print(f"      errors {out['errors']}", flush=True)

    cm_r = confusion_matrix(y, pr, labels=range(len(classes)), normalize="true")
    cm_a = confusion_matrix(y, pa, labels=range(len(classes)), normalize="true")
    out["recall"] = {"run11": np.diag(cm_r).tolist(), "aves": np.diag(cm_a).tolist()}

    print("[2/6] confusion matrices", flush=True)
    fig_confusion(cm_r, cm_a, classes, counts, [(rb, ra[rb]), (ab, aa[ab])],
                  OUT / "01_confusion.png")
    print("[3/6] per-class recall", flush=True)
    fig_per_class(np.diag(cm_r), np.diag(cm_a), classes, counts, OUT / "02_per_class.png")
    print("[4/6] accuracy vs depth", flush=True)
    fig_layers(ra, aa, OUT / "03_layers.png")

    print("[5/6] UMAP (this is the slow one)", flush=True)
    U_r = _umap(E["run11"][:, rb]); U_a = _umap(E["aves"][:, ab])
    fig_umap(U_r, U_a, tt, list(classes),
             "The call-type space, coloured by call type. "
             f"run11 L{rb} vs AVES L{ab}.", OUT / "04_umap_calltype.png", "call type")
    # Identity confound: show the birds with the most clips, everything else grey.
    top = [b for b, _ in sorted(((b, int((birds == b).sum())) for b in set(birds)),
                                key=lambda kv: -kv[1])[:10]]
    bshow = np.where(np.isin(birds, top), birds, "other")
    fig_umap(U_r, U_a, bshow, list(dict.fromkeys(["other"] + top)),
             "THE CONFOUND: the same coordinates, coloured by bird identity "
             "(10 largest birds; rest grey).", OUT / "05_umap_bird.png", "bird")

    print("[6/6] label-free Ward clustering", flush=True)
    ks = [4, 6, 8, 11, 13, 16, 20, 25]
    res = {}
    for nm, l in (("run11", rb), ("aves", ab)):
        Z = StandardScaler().fit_transform(E[nm][:, l])
        at, ab_ = [], []
        for k in ks:
            lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)
            at.append(float(adjusted_mutual_info_score(y, lab)))
            ab_.append(float(adjusted_mutual_info_score(birds, lab)))
            print(f"      {nm} k={k:<3d} AMI_type {at[-1]:.4f}  AMI_bird {ab_[-1]:.4f}", flush=True)
        res[nm] = {"ks": ks, "ami_type": at, "ami_bird": ab_}
    out["clustering"] = res
    fig_clustering(res, ks, OUT / "06_clustering.png")

    (OUT / "calltype_viz.json").write_text(json.dumps(out, indent=2))
    print(f"\n[done] {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

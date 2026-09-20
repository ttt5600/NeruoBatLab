#!/usr/bin/env python
"""What the 768-d layers actually look like, layer by layer.

Six views, each answering a question the AUC table cannot:

  1  layer_umap_labels     Does call/noise structure appear, sharpen, or wash out with depth?
  2  layer_umap_loudness   Same coordinates, coloured by dB. If the loudness picture looks like
                           the label picture, the probe may be reading loudness.
  3  layer_geometry        Linear AUC, kNN-10, silhouette, k-means/Ward AMI, PCA effective dim,
                           and the correlation of PC0 with dB -- all against depth.
  4  cluster_spectrograms  Label-free Ward clusters at the chosen k, each shown as its MEAN
                           spectrogram with its label composition. This is the "what did it group"
                           view; a cluster that is 95% call is a discovered call detector.
  5  error_map             UMAP coloured by TP/TN/FP/FN with the high-confidence FPs ringed, so
                           errors can be located in the representation rather than just counted.
  6  trained_vs_random     Pretrained and random-init UMAP side by side at a matched layer.

UMAP is for looking, never for measuring: every number in view 3 is computed in the full 768-d
space, never on the 2-d embedding.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zfeval import splits as sp                                        # noqa: E402

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (adjusted_mutual_info_score, silhouette_score, roc_auc_score,
                             homogeneity_score)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict

TMP = Path.home() / ".claude/jobs/63c218d9/tmp"
OUT = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis/viz"
AUDIO = Path.home() / "zf_labelset/audio/111021-000.wav"
SR, WIN = 16000, 16000
CALL, NOISE = "#d94801", "#2171b5"


def umap2(X, seed=0, n_neighbors=30, min_dist=0.1):
    import umap
    Z = StandardScaler().fit_transform(X)
    return umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                     random_state=seed, metric="euclidean").fit_transform(Z)


def spec(x, n_fft=512, hop=128):
    from scipy.signal import spectrogram
    f, t, S = spectrogram(x, SR, nperseg=n_fft, noverlap=n_fft - hop, mode="psd")
    return f, t, 10 * np.log10(S + 1e-12)


# ------------------------------------------------------------------ views
def view_umap_grid(Xl, y, en, layers, path, color="label", title=""):
    n = len(layers)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4.0 * ((n + 1) // 2), 8.0), squeeze=False)
    for ax, l in zip(axes.ravel(), layers):
        Z = umap2(Xl[l])
        if color == "label":
            for v, c, nm in [(0, NOISE, "no call"), (1, CALL, "call")]:
                m = y == v
                ax.scatter(Z[m, 0], Z[m, 1], s=3, c=c, alpha=.55, linewidths=0, label=nm)
        else:
            s = ax.scatter(Z[:, 0], Z[:, 1], s=3, c=en, cmap="viridis", alpha=.7, linewidths=0)
            plt.colorbar(s, ax=ax, fraction=.046, label="window dB")
        ax.set_title(f"layer {l}" if isinstance(l, int) else str(l), fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    if color == "label":
        axes[0, 0].legend(markerscale=4, fontsize=9, loc="best")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=135); plt.close(fig)
    print("  wrote", path, flush=True)


def view_geometry(Xl, y, en, grp, starts, layers, path):
    cv, g, desc = sp.choose_cv(grp, starts, n_splits=5, seed=0)
    rows = {}
    for l in layers:
        X = Xl[l]
        Zs = StandardScaler().fit_transform(X)
        p_lin = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000)),
                                  X, y, groups=g, cv=cv, method="predict_proba")[:, 1]
        p_knn = cross_val_predict(make_pipeline(StandardScaler(), KNeighborsClassifier(10)),
                                  X, y, groups=g, cv=cv, method="predict_proba")[:, 1]
        pca = PCA(n_components=min(120, X.shape[1])).fit(Zs)
        cum = np.cumsum(pca.explained_variance_ratio_)
        pc0 = pca.transform(Zs)[:, 0]
        km = KMeans(n_clusters=8, n_init=10, random_state=0).fit_predict(Zs)
        wd = AgglomerativeClustering(n_clusters=8, linkage="ward").fit_predict(Zs)
        rows[str(l)] = dict(
            linear_auc=float(roc_auc_score(y, p_lin)), knn10_auc=float(roc_auc_score(y, p_knn)),
            silhouette_label=float(silhouette_score(Zs, y, sample_size=1500, random_state=0)),
            kmeans_ami=float(adjusted_mutual_info_score(y, km)),
            ward_ami=float(adjusted_mutual_info_score(y, wd)),
            ward_homog=float(homogeneity_score(y, wd)),
            pca_dim_90=int(np.searchsorted(cum, .90) + 1),
            pc0_var=float(pca.explained_variance_ratio_[0]),
            pc0_db_corr=float(abs(np.corrcoef(pc0, en)[0, 1])))
        print(f"  {str(l):>4s} lin {rows[str(l)]['linear_auc']:.4f} knn {rows[str(l)]['knn10_auc']:.4f} "
              f"sil {rows[str(l)]['silhouette_label']:+.3f} wardAMI {rows[str(l)]['ward_ami']:.3f} "
              f"dim90 {rows[str(l)]['pca_dim_90']:3d} |r(PC0,dB)| {rows[str(l)]['pc0_db_corr']:.3f}",
              flush=True)

    xs = list(range(len(layers)))
    lab = [str(l) for l in layers]
    fig, ax = plt.subplots(2, 3, figsize=(15, 7.5))
    panels = [("linear_auc", "linear probe AUC", 0), ("knn10_auc", "kNN-10 AUC", 0),
              ("silhouette_label", "silhouette (by label)", 0),
              ("ward_ami", "Ward AMI vs label (k=8)", 0),
              ("pca_dim_90", "dims for 90% variance", 0),
              ("pc0_db_corr", "|corr(PC0, dB)|", 0)]
    for a, (key, ttl, _) in zip(ax.ravel(), panels):
        v = [rows[l][key] for l in lab]
        a.plot(xs, v, "o-", color="#333")
        a.set_xticks(xs); a.set_xticklabels(lab, fontsize=8)
        a.set_title(ttl, fontsize=11); a.grid(alpha=.3)
        a.set_xlabel("layer")
    fig.suptitle(f"Embedding geometry vs depth -- {desc}", fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=135); plt.close(fig)
    print("  wrote", path, flush=True)
    return rows, desc


def view_clusters(X, y, en, starts, k, path, rec=None):
    """Label-free Ward clusters, three rows per cluster.

    An earlier version showed the MEAN spectrogram and auto-scaled each label bar chart
    separately. Both were misleading: a zebra finch call is ~80 ms inside a 1 s window, so
    averaging 40 windows smears every call into broadband mush, and independently scaled bars made
    a 66-window cluster look the same size as a 453-window one. Now: a real medoid window (row 1),
    the mean frequency profile against the corpus mean (row 2, where the spectral differences
    actually show), and label composition on ONE shared axis (row 3).
    """
    Zs = StandardScaler().fit_transform(X)
    cl = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Zs)
    ami = adjusted_mutual_info_score(y, cl)
    order = sorted(range(k), key=lambda c: -y[cl == c].mean())

    prof, med_idx = {}, {}
    f0 = None
    for c in range(k):
        idx = np.where(cl == c)[0]
        ctr = Zs[idx].mean(0)
        med_idx[c] = int(idx[np.argmin(((Zs[idx] - ctr) ** 2).sum(1))])
        P = []
        for i in idx[np.linspace(0, len(idx) - 1, min(60, len(idx))).astype(int)]:
            s0 = int(starts[i])
            f0, _, S = spec(rec[s0:s0 + WIN])
            P.append(S.mean(1))
        prof[c] = np.mean(P, axis=0)
    globalp = np.mean([prof[c] for c in range(k)], axis=0)

    fig, axes = plt.subplots(3, k, figsize=(2.15 * k, 7.4),
                             gridspec_kw=dict(height_ratios=[2.5, 1.9, 1.5]))
    ymax = max((cl == c).sum() for c in range(k))
    for j, c in enumerate(order):
        m = cl == c
        s0 = int(starts[med_idx[c]])
        fq, t, S = spec(rec[s0:s0 + WIN])
        ax = axes[0, j]
        ax.imshow(S, origin="lower", aspect="auto", cmap="magma",
                  extent=[0, 1, fq[0] / 1000, fq[-1] / 1000],
                  vmin=np.percentile(S, 30), vmax=np.percentile(S, 99.7))
        ax.set_ylim(0, 8); ax.set_xticks([])
        ax.set_title(f"c{c}   n={m.sum()}\n{y[m].mean()*100:.0f}% call", fontsize=9)
        if j == 0: ax.set_ylabel("kHz", fontsize=8)
        else: ax.set_yticks([])

        a1 = axes[1, j]
        a1.plot(globalp, fq / 1000, color="#bbb", lw=1.0, label="all windows")
        a1.plot(prof[c], fq / 1000, color="#111", lw=1.4, label="this cluster")
        a1.set_ylim(0, 8)
        a1.set_xlim(min(globalp.min(), min(p.min() for p in prof.values())) - 2,
                    max(globalp.max(), max(p.max() for p in prof.values())) + 2)
        a1.set_xticks([]); a1.grid(alpha=.25)
        if j == 0:
            a1.set_ylabel("kHz", fontsize=8); a1.legend(fontsize=6.5, loc="upper right", frameon=False)
        else:
            a1.set_yticks([])

        a2 = axes[2, j]
        a2.bar([0, 1], [int((y[m] == 0).sum()), int((y[m] == 1).sum())],
               color=[NOISE, CALL], width=.66)
        a2.set_ylim(0, ymax * 1.06)
        a2.set_xticks([0, 1]); a2.set_xticklabels(["no call", "call"], fontsize=7)
        a2.set_title(f"med {np.median(en[m]):.0f} dB", fontsize=8)
        if j: a2.set_yticks([])
        else: a2.set_ylabel("windows", fontsize=8)

    fig.suptitle(f"Ward clusters, k={k}, computed WITHOUT labels -- AMI vs label = {ami:.3f}\n"
                 f"row 1: a real medoid window   row 2: mean spectrum vs the corpus mean   "
                 f"row 3: label composition (shared scale)", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=135); plt.close(fig)
    print("  wrote", path, flush=True)
    return dict(k=k, ami=float(ami), sizes={int(c): int((cl == c).sum()) for c in range(k)},
                call_frac={int(c): float(y[cl == c].mean()) for c in range(k)},
                median_db={int(c): float(np.median(en[cl == c])) for c in range(k)},
                medoid_start={int(c): int(starts[med_idx[c]]) for c in range(k)}), cl


def view_error_map(X, y, p, en, path, hi_conf=0.944):
    Z = umap2(X)
    pred = p > .5
    tp = (pred == 1) & (y == 1); tn = (pred == 0) & (y == 0)
    fp = (pred == 1) & (y == 0); fn = (pred == 0) & (y == 1)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.4))
    for a, (title, sets) in zip(ax, [
            ("all four outcomes", [("TN", tn, "#c6dbef", 3), ("TP", tp, "#fdd0a2", 3),
                                   ("FP", fp, "#e6550d", 16), ("FN", fn, "#08519c", 16)]),
            (f"high-confidence FP (p>{hi_conf}) ringed", [("correct", tp | tn, "#dddddd", 3),
                                                          ("other errors", fp | fn, "#999999", 8)])]):
        for nm, m, c, s in sets:
            a.scatter(Z[m, 0], Z[m, 1], s=s, c=c, alpha=.8, linewidths=0, label=f"{nm} ({m.sum()})")
        a.set_title(title, fontsize=12); a.set_xticks([]); a.set_yticks([])
        a.legend(markerscale=2.2, fontsize=9)
    hc = fp & (p > hi_conf)
    ax[1].scatter(Z[hc, 0], Z[hc, 1], s=78, facecolors="none", edgecolors="#d7191c", linewidths=1.6,
                  label=f"high-conf FP ({hc.sum()})")
    ax[1].legend(markerscale=1.2, fontsize=9)
    fig.suptitle("Where the errors live in the representation", fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=135); plt.close(fig)
    print("  wrote", path, flush=True)
    return dict(n_tp=int(tp.sum()), n_tn=int(tn.sum()), n_fp=int(fp.sum()), n_fn=int(fn.sum()),
                n_high_conf_fp=int(hc.sum()))


def view_trained_vs_random(Xp, Xr, y, path, layer):
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.2))
    for a, (X, ttl) in zip(ax, [(Xp, f"pretrained (run11) layer {layer}"),
                                (Xr, f"random init, untrained, layer {layer}")]):
        Z = umap2(X)
        for v, c, nm in [(0, NOISE, "no call"), (1, CALL, "call")]:
            m = y == v
            a.scatter(Z[m, 0], Z[m, 1], s=3.4, c=c, alpha=.6, linewidths=0, label=nm)
        a.set_title(ttl, fontsize=12); a.set_xticks([]); a.set_yticks([])
    ax[0].legend(markerscale=4, fontsize=10)
    fig.suptitle("Same architecture, same audio, same UMAP settings -- "
                 "only the weights differ", fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=135); plt.close(fig)
    print("  wrote", path, flush=True)


def linear_cka(A, B):
    """Centred kernel alignment. 1.0 = the two layers carry the same geometry.

    float64 throughout and a finite check at the end: A.T @ A on 1801x768 float32 activations
    overflows, and the overflow is silent apart from a RuntimeWarning -- the returned value can
    still look like a plausible similarity.
    """
    A = np.asarray(A, dtype=np.float64); B = np.asarray(B, dtype=np.float64)
    A = A - A.mean(0); B = B - B.mean(0)
    num = np.linalg.norm(A.T @ B, "fro") ** 2
    den = np.linalg.norm(A.T @ A, "fro") * np.linalg.norm(B.T @ B, "fro")
    v = float(num / den)
    if not np.isfinite(v):
        raise RuntimeError("CKA overflowed even in float64")
    return v
    # NOTE: numpy on Apple Accelerate emits spurious divide-by-zero/overflow RuntimeWarnings from
    # matmul here even though every input is finite and |x| < 16. Checked by recomputing the same
    # quantity in Gram space (K=AA', L=BB', <K,L>/sqrt(<K,K><L,L>)), which dispatches different
    # BLAS shapes: the two routes agree to 7e-16. The warnings are flags, not a corrupted result.


def view_cka(Xl_pre, Xl_rand, layers, path):
    def mat(Xl):
        n = len(layers)
        M = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                M[i, j] = M[j, i] = linear_cka(Xl[layers[i]], Xl[layers[j]])
        return M
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
    out = {}
    for a, (Xl, ttl, key) in zip(ax, [(Xl_pre, "pretrained (run11)", "pretrained"),
                                      (Xl_rand, "random init", "random")]):
        M = mat(Xl); out[key] = M.tolist()
        im = a.imshow(M, cmap="cividis", vmin=0, vmax=1)
        a.set_xticks(range(len(layers))); a.set_xticklabels([str(l) for l in layers], fontsize=8)
        a.set_yticks(range(len(layers))); a.set_yticklabels([str(l) for l in layers], fontsize=8)
        a.set_title(ttl, fontsize=12); a.set_xlabel("layer"); a.set_ylabel("layer")
        plt.colorbar(im, ax=a, fraction=.046, label="linear CKA")
    fig.suptitle("Layer-to-layer representational similarity.  A trained stack transforms its "
                 "input; an untrained one mostly passes it through.", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=135); plt.close(fig)
    print("  wrote", path, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 2, 4, 6, 9, 11])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    rec, _ = sf.read(AUDIO, dtype="float32")

    d = np.load(TMP / "randinit_feats.npz", allow_pickle=True)
    y, grp, starts, en = d["y"], d["grp"].astype(str), d["starts"], d["en"]
    Xpre, Xrnd = d["X_pretrained"], d["X_rand_seed0"]
    Xl_pre = {l: Xpre[:, l] for l in range(12)}
    Xl_pre["cnn"] = d["Xcnn_pretrained"]
    Xl_rnd = {l: Xrnd[:, l] for l in range(12)}
    Xl_rnd["cnn"] = d["Xcnn_rand_seed0"]
    report = {}

    print("[1/6] layer UMAP, coloured by label", flush=True)
    view_umap_grid(Xl_pre, y, en, a.layers, OUT / "01_layer_umap_labels.png",
                   "label", "run11 embeddings by depth, coloured by hand label")
    print("[2/6] layer UMAP, coloured by loudness", flush=True)
    view_umap_grid(Xl_pre, y, en, a.layers, OUT / "02_layer_umap_loudness.png",
                   "db", "the same embeddings, coloured by window loudness")
    print("[3/6] geometry vs depth", flush=True)
    report["geometry"], desc = view_geometry(Xl_pre, y, en, grp, starts,
                                            ["cnn"] + list(range(12)),
                                            OUT / "03_layer_geometry.png")
    report["split"] = desc
    print(f"[4/6] Ward clusters at k={a.k} with mean spectrograms", flush=True)
    report["clusters"], _ = view_clusters(Xl_pre[6], y, en, starts, a.k,
                                          OUT / "04_cluster_spectrograms.png", rec=rec)
    print("[5/6] error map", flush=True)
    pr = np.load(TMP / "randinit_preds.npz")
    best = json.loads((Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
                       / "randinit_control.json").read_text())["best_pretrained_layer"]
    p = pr[f"pretrained__{best}"]
    report["errors"] = view_error_map(Xl_pre[6], y, p, en, OUT / "05_error_map.png")
    report["errors"]["probe_layer"] = best
    print("[6/6] trained vs random, and layer CKA", flush=True)
    view_trained_vs_random(Xl_pre[6], Xl_rnd[6], y, OUT / "06_trained_vs_random.png", 6)
    report["cka"] = view_cka(Xl_pre, Xl_rnd, ["cnn"] + list(range(12)),
                             OUT / "07_layer_cka.png")
    (OUT / "viz_report.json").write_text(json.dumps(report, indent=2))
    print("wrote", OUT / "viz_report.json")


if __name__ == "__main__":
    main()

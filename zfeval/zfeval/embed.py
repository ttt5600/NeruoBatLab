"""Embedding geometry, clustering and UMAP.

Linear probe AUC is one number per layer and hides the structure underneath: on run11 it is flat
across layers 0-7 while kNN neighbourhood purity rises 0.883 -> 0.906 and PCA dimensionality grows
15 -> 51. Different metrics favour different layers, which is exactly why AUC picks L0 and accuracy
picks L5.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (roc_auc_score, silhouette_score, adjusted_mutual_info_score,
                             homogeneity_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def layer_geometry(X, y, groups, cv, layers=None, silhouette_n=3000, seed=0):
    """Per-layer: linear AUC, kNN purity, silhouette, Fisher ratio, PCA dim, k-means AMI."""
    layers = range(X.shape[1]) if layers is None else layers
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(y), min(silhouette_n, len(y)), replace=False)
    out = {}
    for l in layers:
        M = X[:, l].astype(np.float32)
        p = cross_val_predict(LogisticRegression(max_iter=2000), M, y, groups=groups, cv=cv,
                              method="predict_proba")[:, 1]
        knn = cross_val_predict(KNeighborsClassifier(10), M, y, groups=groups, cv=cv)
        mu0, mu1 = M[y == 0].mean(0), M[y == 1].mean(0)
        sw = M[y == 0].var(0).sum() + M[y == 1].var(0).sum()
        km = KMeans(8, n_init=10, random_state=seed).fit_predict(M)
        out[f"L{l}"] = dict(
            auc=float(roc_auc_score(y, p)), knn10_acc=float((knn == y).mean()),
            silhouette=float(silhouette_score(M[sub], y[sub], metric="cosine")),
            fisher_ratio=float(((mu1 - mu0) ** 2).sum() / (sw + 1e-12)),
            pca_dim_90=int(np.searchsorted(np.cumsum(PCA().fit(M).explained_variance_ratio_), 0.90) + 1),
            kmeans_ami_k8=float(adjusted_mutual_info_score(y, km)),
            homogeneity_k8=float(homogeneity_score(y, km)))
    return out


def dominant_structure(X_layer, targets: dict, ks=(8, 32, 64), seed=0):
    """What does unsupervised structure actually track? Run k-means, score AMI against each
    candidate (label, recording, bird, loudness bin).

    On run11 the answer is loudness (AMI 0.335 at k=8) ahead of the label (0.169) and recording
    identity (0.043) -- so the main axis of the UMAP is dB, not vocalization. AMI, not NMI or
    purity: those inflate with k and would pick the largest k for free.
    """
    M = X_layer.astype(np.float32)
    out = {}
    for k in ks:
        km = KMeans(k, n_init=10, random_state=seed).fit_predict(M)
        out[str(k)] = {name: float(adjusted_mutual_info_score(v, km)) for name, v in targets.items()}
    return out


def loudness_dependence(X_layer, y, energy_db, groups, cv, n_pcs=64):
    """How much of the signal lives in the loudness-correlated direction?

    On run11 the top PC is 52.8% of variance and correlates 0.724 with log-energy; dropping it
    costs 0.194 AUC while PCA with nothing dropped costs 0.004. But PC0 alone (0.851) beats the
    scalar it correlates with (0.791), so it is not a loudness detector -- it carries spectral
    structure the scalar throws away. Report the decomposition, do not 'fix' it.
    """
    from sklearn.linear_model import LinearRegression
    M = X_layer.astype(np.float32)
    en = np.asarray(energy_db).reshape(-1, 1)

    def auc(F):
        p = cross_val_predict(LogisticRegression(max_iter=4000), F, y, groups=groups, cv=cv,
                              method="predict_proba")[:, 1]
        return float(roc_auc_score(y, p))

    pca = PCA(n_pcs).fit(M)
    Z = pca.transform(M)
    cors = np.array([abs(np.corrcoef(Z[:, i], en[:, 0])[0, 1]) for i in range(Z.shape[1])])
    top = int(np.argmax(cors))
    resid = Z[:, [top]] - LinearRegression().fit(en, Z[:, [top]]).predict(en)
    keep = np.argsort(-cors)[1:]
    return dict(top_pc=top, top_pc_corr_energy=float(cors[top]),
                top_pc_var_explained=float(pca.explained_variance_ratio_[top]),
                auc_full=auc(M), auc_pca_all=auc(Z), auc_drop_top_pc=auc(Z[:, keep]),
                auc_top_pc_only=auc(Z[:, [top]]), auc_energy_only=auc(en),
                auc_top_pc_residual=auc(resid))


def umap_embed(X_layer, seed=0, n_neighbors=30, min_dist=0.1):
    import umap
    return umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine",
                     random_state=seed).fit_transform(X_layer)

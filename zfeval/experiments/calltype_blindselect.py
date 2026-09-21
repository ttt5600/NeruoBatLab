#!/usr/bin/env python
"""Can you pick the clustering WITHOUT ground truth?

Part III chose k two ways: by silhouette (label-free, and wrong) and by test accuracy
(oracle, and unavailable on a species you have no labels for). Neither is what you would
actually do on bat or chick recordings. This asks the question properly.

The move is not to pretend. It is to CALIBRATE the blind protocol on the one corpus where
ground truth exists, then carry the validated protocol to corpora where it does not.

Three questions, all answered by running label-free criteria and then -- only afterwards --
looking at the labels to see whether they were right:

  Q1  CHOOSE k.        Each criterion picks k with no labels. How much cluster-vote accuracy
                       does that choice cost against the oracle k?
  Q2  CHOOSE ENCODER.  Five encoders. Does the criterion rank them the way ground truth does?
                       (Spearman rho across encoders, each scored at its OWN chosen k.)
  Q3  DETECT NOTHING.  A criterion must be able to say "no structure here". Run the same
                       pipeline on column-shuffled embeddings -- identical marginals, joint
                       structure destroyed -- and see which criteria notice.

Criteria (all strictly label-free):
  silhouette, Calinski-Harabasz, Davies-Bouldin, gap statistic   -- COMPACTNESS
  bootstrap stability, bird-held-out stability, prediction strength -- REPRODUCIBILITY

The distinction is the point. Compactness asks "are these blobs tight?". Reproducibility asks
"would I find these same groups again in birds I have not seen?" -- which is the question a
field recording actually poses.
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import calltype11 as C11                                                  # noqa: E402

from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_mutual_info_score as AMI,
                             adjusted_rand_score as ARI)
from sklearn.model_selection import StratifiedGroupKFold
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[2] / "paper" / "figures" / "calltype"
KS = [4, 6, 8, 11, 13, 16, 20, 25, 30, 40, 55, 70]
ENCODERS = ["run11", "aves", "aves-base-all", "aves-base-core", "birdaves-biox-base"]
LAYER, NFOLD, SEED = 3, 5, 0
B_STAB, B_GAP, PS_THRESH = 15, 5, 0.8


# ---------------------------------------------------------------- data ----
def cohort():
    rows = [r for r in C11.collect() if r[3] in C11.KEEP11]
    birds = np.array([r[1].lower() for r in rows])
    tt = np.array([r[3] for r in rows])
    classes = sorted(set(tt))
    return np.array([classes.index(t) for t in tt]), birds, classes


def ward_all_k(X):
    """One linkage tree serves every k -- the whole reason this is affordable."""
    Z = linkage(X, method="ward")
    return {k: fcluster(Z, t=k, criterion="maxclust") - 1 for k in KS}


def centroids(X, lab, k):
    return np.vstack([X[lab == c].mean(0) if (lab == c).any() else np.full(X.shape[1], 1e9)
                      for c in range(k)])


def assign(X, C):
    return np.stack([((X - c) ** 2).sum(1) for c in C], 1).argmin(1)


# ------------------------------------------------------- label-free ------
def compactness(X, labs):
    """silhouette / CH / -DB. All want tight, well-separated, convex blobs."""
    sil, ch, db = {}, {}, {}
    for k, lab in labs.items():
        sil[k] = float(silhouette_score(X, lab))
        ch[k] = float(calinski_harabasz_score(X, lab))
        db[k] = float(-davies_bouldin_score(X, lab))      # negate: higher is better
    return sil, ch, db


def wss(X, lab, k):
    return float(sum(((X[lab == c] - X[lab == c].mean(0)) ** 2).sum()
                     for c in range(k) if (lab == c).any()))


def gap_stat(X, labs, rng):
    """Tibshirani 2001, reference = uniform over the PCA-aligned bounding box."""
    logW = {k: np.log(wss(X, labs[k], k)) for k in KS}
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Xp = Xc @ Vt.T
    lo, hi = Xp.min(0), Xp.max(0)
    ref = {k: [] for k in KS}
    for _ in range(B_GAP):
        Zp = rng.uniform(lo, hi, size=Xp.shape)
        Zr = Zp @ Vt
        rl = ward_all_k(Zr)
        for k in KS:
            ref[k].append(np.log(wss(Zr, rl[k], k)))
    return {k: float(np.mean(ref[k]) - logW[k]) for k in KS}


def _transfer_agreement(Xa, Xb, rng, metric):
    """Cluster A, cluster B, carry A's partition onto B's points, compare with B's own."""
    la, lb = ward_all_k(Xa), ward_all_k(Xb)
    out = {}
    for k in KS:
        pred = assign(Xb, centroids(Xa, la[k], k))
        out[k] = float(metric(lb[k], pred))
    return out


def stability(X, groups, rng, by_group):
    """Reproducibility of the partition under resampling.

    by_group=False  -> split clips at random   (classic bootstrap stability)
    by_group=True   -> split BIRDS at random   (would these groups reappear in new individuals?)

    ARI is chance-corrected, so a larger k gets no free credit.
    """
    accum = {k: [] for k in KS}
    units = np.unique(groups) if by_group else np.arange(len(X))
    for _ in range(B_STAB):
        perm = rng.permutation(len(units))
        h = len(units) // 2
        ua, ub = units[perm[:h]], units[perm[h:]]
        ia = np.isin(groups, ua) if by_group else np.isin(np.arange(len(X)), ua)
        Xa, Xb = X[ia], X[~ia]
        if min(len(Xa), len(Xb)) < max(KS) + 5:
            continue
        r = _transfer_agreement(Xa, Xb, rng, ARI)
        for k in KS:
            accum[k].append(r[k])
    return {k: float(np.mean(accum[k])) for k in KS}, {k: float(np.std(accum[k])) for k in KS}


def prediction_strength(X, rng):
    """Tibshirani & Walther 2005: worst cluster's co-membership survival rate.

    Rule of thumb: take the LARGEST k whose PS still clears 0.8. Unlike the others this is a
    threshold rule, not an argmax -- it is built to be conservative about k.
    """
    accum = {k: [] for k in KS}
    n = len(X)
    for _ in range(B_STAB):
        perm = rng.permutation(n); h = n // 2
        Xa, Xb = X[perm[:h]], X[perm[h:]]
        la, lb = ward_all_k(Xa), ward_all_k(Xb)
        for k in KS:
            C = centroids(Xa, la[k], k)
            pred = assign(Xb, C)
            worst = 1.0
            for c in range(k):
                m = lb[k] == c
                nc = int(m.sum())
                if nc < 2:
                    continue
                p = pred[m]
                same = (p[:, None] == p[None, :]).sum() - nc      # ordered pairs, drop diagonal
                worst = min(worst, same / (nc * (nc - 1)))
            accum[k].append(worst)
    return {k: float(np.mean(accum[k])) for k in KS}


# ------------------------------------------------------ ground truth -----
def truth_ami(X, labs, y):
    return {k: float(AMI(y, labs[k])) for k in KS}


def truth_vote_acc(E, y, birds):
    """Leave-birds-out cluster-then-vote, exactly the Part III protocol."""
    cv = StratifiedGroupKFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
    correct = {k: np.zeros(len(y), bool) for k in KS}
    nclass = len(set(y))
    for tr, te in cv.split(E, y, birds):
        sc = StandardScaler().fit(E[tr])
        Xtr, Xte = sc.transform(E[tr]), sc.transform(E[te])
        labs = ward_all_k(Xtr)
        for k in KS:
            lab = labs[k]
            C = centroids(Xtr, lab, k)
            maj = np.array([np.bincount(y[tr][lab == c], minlength=nclass).argmax()
                            if (lab == c).any() else np.bincount(y[tr]).argmax()
                            for c in range(k)])
            correct[k][te] = (maj[assign(Xte, C)] == y[te])
    return {k: float(correct[k].mean()) for k in KS}


# ------------------------------------------------------------- main -----
def evaluate(name, E, y, birds):
    t = time.time()
    rng = np.random.default_rng(SEED)
    X = StandardScaler().fit_transform(E)
    labs = ward_all_k(X)
    sil, ch, db = compactness(X, labs)
    gap = gap_stat(X, labs, rng)
    stc, _ = stability(X, birds, rng, by_group=False)
    stb, stb_sd = stability(X, birds, rng, by_group=True)
    ps = prediction_strength(X, rng)
    res = dict(
        criteria=dict(silhouette=sil, calinski_harabasz=ch, neg_davies_bouldin=db,
                      gap=gap, stab_clip=stc, stab_bird=stb, stab_bird_sd=stb_sd,
                      prediction_strength=ps),
        truth=dict(ami=truth_ami(X, labs, y), vote_acc=truth_vote_acc(E, y, birds)),
        seconds=round(time.time() - t, 1))
    print(f"[{name}] {res['seconds']}s  "
          f"sil_k={max(sil,key=sil.get)} gap_k={max(gap,key=gap.get)} "
          f"stabclip_k={max(stc,key=stc.get)} stabbird_k={max(stb,key=stb.get)} "
          f"| ami_k={max(res['truth']['ami'],key=res['truth']['ami'].get)} "
          f"acc_k={max(res['truth']['vote_acc'],key=res['truth']['vote_acc'].get)}", flush=True)
    return res


def main():
    t0 = time.time()
    y, birds, classes = cohort()
    print(f"[cohort] {len(y)} clips, {len(classes)} classes, {len(set(birds))} birds, "
          f"layer {LAYER}, ward, KS={KS}", flush=True)
    out = {"ks": KS, "layer": LAYER, "n_clips": len(y), "classes": classes,
           "n_birds": int(len(set(birds))), "algo": "ward",
           "b_stability": B_STAB, "b_gap": B_GAP, "ps_threshold": PS_THRESH,
           "encoders": {}}
    for nm in ENCODERS:
        E = np.load(C11.FEAT / f"ct11_{nm}_emb.npy")[:, LAYER].astype(np.float64)
        out["encoders"][nm] = evaluate(nm, E, y, birds)
    # Q3 null: identical per-dimension marginals, joint structure destroyed
    rng = np.random.default_rng(1234)
    E = np.load(C11.FEAT / "ct11_aves_emb.npy")[:, LAYER].astype(np.float64)
    Esh = np.column_stack([rng.permutation(E[:, j]) for j in range(E.shape[1])])
    out["encoders"]["NULL-shuffled"] = evaluate("NULL-shuffled", Esh, y, birds)
    (OUT / "calltype_blindselect.json").write_text(json.dumps(out, indent=2))
    print(f"[done] {time.time()-t0:.0f}s -> {OUT/'calltype_blindselect.json'}")


if __name__ == "__main__":
    main()

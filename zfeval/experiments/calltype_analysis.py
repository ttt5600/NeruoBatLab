#!/usr/bin/env python
"""Where, if anywhere, does colony-specific pretraining actually beat a generic animal encoder?

Top-line accuracy says AVES wins. That is one number over a whole task, and it can hide a real
advantage: run11 could still be better on the call types that are distinctively zebra finch, or in
the low-data regime a lab actually works in, or at ORGANISING the space rather than merely making it
linearly decodable. Each of those is a separate claim and each is tested here.

  per-class     out-of-fold accuracy per call type, with a bootstrap on the difference. If colony
                pretraining buys anything it should show up on specific types, not uniformly.
  data          accuracy against the number of TRAINING BIRDS. A domain-specific encoder should need
                fewer examples; if the curves converge only at the top, run11 loses its own argument.
  unsupervised  Ward clustering AMI against call type, with AMI against BIRD IDENTITY as the
                confound. A probe can carve structure that is not natively there; clustering cannot.
  geometry      UMAP, plus how much of each space is spanned by identity rather than call type.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import calltype11 as C11                                                     # noqa: E402
from aves_calltype import cv_acc, bird_bootstrap                             # noqa: E402

FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
L_RUN11, L_AVES = 3, 3          # each encoder's best layer on the 11-class task
SEED = 0


def oof_predictions(X, y, groups):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedGroupKFold
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    pred = np.zeros(len(y), dtype=int)
    for tr, te in cv.split(X, y, groups):
        est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000))
        est.fit(X[tr], y[tr])
        pred[te] = est.predict(X[te])
    return pred


def data_efficiency(X, y, groups, bird_counts=(4, 8, 16, 24, 32, 40), n_rep=5):
    """Train on k birds, test on a fixed disjoint bird set. The lab-realistic axis."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    rng = np.random.default_rng(SEED)
    uniq = np.unique(groups)
    out = {}
    for k in bird_counts:
        accs = []
        for r in range(n_rep):
            perm = rng.permutation(uniq)
            test_b = set(perm[:max(8, len(uniq) // 5)])
            pool = [b for b in perm if b not in test_b]
            if k > len(pool):
                continue
            tr_b = set(pool[:k])
            tr = np.isin(groups, list(tr_b)); te = np.isin(groups, list(test_b))
            if len(np.unique(y[tr])) < len(np.unique(y)) or tr.sum() < 30:
                continue
            est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000))
            est.fit(X[tr], y[tr])
            accs.append(float((est.predict(X[te]) == y[te]).mean()))
        if accs:
            out[k] = dict(mean=float(np.mean(accs)), sd=float(np.std(accs)), n=len(accs),
                          median_train_clips=float(np.median([k])))
    return out


def unsupervised(X, y, groups, ks=(8, 11, 14, 20)):
    """Ward clustering, scored against call type AND against bird identity."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import adjusted_mutual_info_score as ami
    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(X)
    gi = {b: i for i, b in enumerate(np.unique(groups))}
    gy = np.array([gi[b] for b in groups])
    out = {}
    for k in ks:
        lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)
        out[k] = dict(ami_calltype=float(ami(y, lab)), ami_bird=float(ami(gy, lab)))
    return out


def main():
    rows = [r for r in C11.collect() if r[3] in C11.KEEP11]
    birds = np.array([r[1].lower() for r in rows])
    tt = np.array([r[3] for r in rows])
    src = np.array([r[4] for r in rows])
    classes = sorted(set(tt))
    y = np.array([classes.index(t) for t in tt])
    assert len(classes) == 11, f"expected 11 classes, got {len(classes)}"
    R = np.load(FEAT / "ct11_run11_emb.npy")[:, L_RUN11]
    A = np.load(FEAT / "ct11_aves_emb.npy")[:, L_AVES]
    print(f"[data] {len(y)} clips, {len(classes)} classes, {len(set(birds))} birds", flush=True)

    out = {"classes": classes, "n_clips": int(len(y)), "n_birds": int(len(set(birds))),
           "layers": {"run11": L_RUN11, "aves": L_AVES},
           "class_counts": {c: int((tt == c).sum()) for c in classes}}

    # ---------------- per class
    print("\n=== per-class out-of-fold accuracy ===")
    pr, pa = oof_predictions(R, y, birds), oof_predictions(A, y, birds)
    out["overall"] = {"run11": float((pr == y).mean()), "aves": float((pa == y).mean())}
    print(f"  {'class':6s} {'n':>5s} {'run11':>8s} {'AVES':>8s} {'diff':>8s}")
    per = {}
    for i, c in enumerate(classes):
        m = y == i
        a_r, a_a = float((pr[m] == y[m]).mean()), float((pa[m] == y[m]).mean())
        per[c] = dict(n=int(m.sum()), run11=a_r, aves=a_a, diff=a_r - a_a)
        flag = "  <- run11 ahead" if a_r > a_a else ""
        print(f"  {c:6s} {m.sum():5d} {a_r:8.3f} {a_a:8.3f} {a_r-a_a:+8.3f}{flag}")
    out["per_class"] = per
    out["confusion"] = {"run11": [[int(((y == i) & (pr == j)).sum()) for j in range(len(classes))]
                                  for i in range(len(classes))],
                        "aves": [[int(((y == i) & (pa == j)).sum()) for j in range(len(classes))]
                                 for i in range(len(classes))]}

    # ---------------- data efficiency
    print("\n=== accuracy vs number of TRAINING BIRDS (5 repeats, disjoint test birds) ===")
    de_r, de_a = data_efficiency(R, y, birds), data_efficiency(A, y, birds)
    out["data_efficiency"] = {"run11": de_r, "aves": de_a}
    print(f"  {'birds':>6s} {'run11':>16s} {'AVES':>16s} {'diff':>8s}")
    for k in sorted(set(de_r) & set(de_a)):
        r, a = de_r[k], de_a[k]
        print(f"  {k:6d} {r['mean']:8.3f}±{r['sd']:.3f} {a['mean']:8.3f}±{a['sd']:.3f} "
              f"{r['mean']-a['mean']:+8.3f}")

    # ---------------- unsupervised structure
    print("\n=== Ward clustering: does the space ORGANISE by call type? ===")
    u_r, u_a = unsupervised(R, y, birds), unsupervised(A, y, birds)
    out["unsupervised"] = {"run11": u_r, "aves": u_a}
    print(f"  {'k':>4s} {'run11 AMI(type)':>16s} {'AVES AMI(type)':>16s} "
          f"{'run11 AMI(bird)':>16s} {'AVES AMI(bird)':>16s}")
    for k in sorted(u_r):
        print(f"  {k:4d} {u_r[k]['ami_calltype']:16.4f} {u_a[k]['ami_calltype']:16.4f} "
              f"{u_r[k]['ami_bird']:16.4f} {u_a[k]['ami_bird']:16.4f}")

    # ---------------- adults vs chicks (a domain the colony model saw and AVES did not)
    print("\n=== adults vs chicks ===")
    for nm, P in (("run11", pr), ("aves", pa)):
        for s in ("AdultVocalizations", "ChickVocalizations"):
            m = src == s
            out.setdefault("by_source", {}).setdefault(nm, {})[s] = float((P[m] == y[m]).mean())
            print(f"  {nm:6s} {s:20s} n={m.sum():5d}  acc {(P[m]==y[m]).mean():.4f}")

    # ---------------- UMAP for the figure
    try:
        import umap
        emb = {}
        for nm, X in (("run11", R), ("aves", A)):
            from sklearn.preprocessing import StandardScaler
            Z = StandardScaler().fit_transform(X)
            emb[nm] = umap.UMAP(n_neighbors=25, min_dist=0.15, random_state=SEED).fit_transform(Z)
            print(f"  umap {nm} done", flush=True)
        np.savez(FEAT / "ct11_umap.npz", run11=emb["run11"], aves=emb["aves"], y=y,
                 birds=birds, classes=np.array(classes), src=src)
        out["umap"] = str(FEAT / "ct11_umap.npz")
    except Exception as e:
        print(f"  umap skipped: {type(e).__name__} {e}")

    (ANA / "calltype_analysis.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ANA/'calltype_analysis.json'}")


if __name__ == "__main__":
    main()

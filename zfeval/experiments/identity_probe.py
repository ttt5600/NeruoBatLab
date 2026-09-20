#!/usr/bin/env python
"""If run11 encodes WHO is calling more strongly, that should be measurable — and it would be the
one thing colony-specific pretraining demonstrably buys.

The clustering result points straight here. Ward clusters of the run11 space track bird identity
MORE than AVES's do (AMI 0.248-0.329 vs 0.201-0.291) while tracking call type LESS. For call-type
work that is a liability. For individual recognition it would be an asset, and it is exactly what
120 recordings of a fixed set of colony birds should produce.

The split has to be chosen carefully. Identity cannot be evaluated leave-birds-out, since the bird IS
the label. A random split is worse than useless: clips from one recording session share channel,
distance and background, so a random split lets the probe recognise the SESSION rather than the bird.
So train and test are split by DATE within bird — every test clip comes from a recording session the
probe never trained on. Birds with only one date are dropped, because for them no such split exists.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import calltype11 as C11                                                     # noqa: E402

FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
SEED = 0


def leave_session_out(X, y, sess, n_splits=5):
    """Grouped by recording date: a test clip's session never appears in training."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupKFold
    cv = GroupKFold(n_splits=n_splits)
    pred = np.full(len(y), -1, dtype=int)
    for tr, te in cv.split(X, y, sess):
        seen = np.isin(y[te], np.unique(y[tr]))          # only score classes present in training
        est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000))
        est.fit(X[tr], y[tr])
        pred[te[seen]] = est.predict(X[te[seen]])
    m = pred >= 0
    return float((pred[m] == y[m]).mean()), m.sum(), pred


def boot(y, pa, pb, sess, n=2000, seed=0):
    """Cluster bootstrap over recording sessions."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(sess)
    idx = {g: np.where(sess == g)[0] for g in uniq}
    d = []
    for _ in range(n):
        sel = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        sel = sel[(pa[sel] >= 0) & (pb[sel] >= 0)]
        if len(sel) < 50:
            continue
        d.append((pa[sel] == y[sel]).mean() - (pb[sel] == y[sel]).mean())
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(delta=float(d.mean()), lo=float(lo), hi=float(hi),
                verdict=("a_better" if lo > 0 else "b_better" if hi < 0 else "not_distinguishable"))


def main():
    rows = [r for r in C11.collect() if r[3] in C11.KEEP11]
    birds = np.array([r[1].lower() for r in rows])
    dates = np.array([r[2] for r in rows])
    R_all = np.load(FEAT / "ct11_run11_emb.npy")
    A_all = np.load(FEAT / "ct11_aves_emb.npy")

    # keep birds recorded on at least two dates, so a session-disjoint split exists
    keep = np.array([len(set(dates[birds == b])) >= 2 for b in birds])
    birds, dates = birds[keep], dates[keep]
    R_all, A_all = R_all[keep], A_all[keep]
    ub = sorted(set(birds))
    y = np.array([ub.index(b) for b in birds])
    sess = np.array([f"{b}|{d}" for b, d in zip(birds, dates)])
    print(f"[data] {len(y)} clips, {len(ub)} birds with >=2 sessions, "
          f"{len(set(sess))} sessions, majority {np.bincount(y).max()/len(y):.4f}", flush=True)

    out = {"n_clips": int(len(y)), "n_birds": len(ub), "n_sessions": int(len(set(sess))),
           "majority": float(np.bincount(y).max() / len(y)),
           "split": "GroupKFold(5) over recording sessions (bird|date)", "per_layer": {}}

    print("\n=== bird identity, leave-session-out ===")
    print(f"  {'layer':6s} {'run11':>8s} {'AVES':>8s} {'diff':>8s}")
    best = {"run11": (-1, -1), "aves": (-1, -1)}
    P = {}
    for l in range(12):
        ar, nr, pr = leave_session_out(R_all[:, l], y, sess)
        aa, na, pa = leave_session_out(A_all[:, l], y, sess)
        out["per_layer"][f"L{l}"] = dict(run11=ar, aves=aa, n_scored=int(nr))
        P[("run11", l)], P[("aves", l)] = pr, pa
        if ar > best["run11"][1]:
            best["run11"] = (l, ar)
        if aa > best["aves"][1]:
            best["aves"] = (l, aa)
        flag = "  <- run11 ahead" if ar > aa else ""
        print(f"  L{l:<5d} {ar:8.4f} {aa:8.4f} {ar-aa:+8.4f}{flag}", flush=True)

    lr, la = best["run11"][0], best["aves"][0]
    out["best"] = {"run11": dict(layer=lr, acc=best["run11"][1]),
                   "aves": dict(layer=la, acc=best["aves"][1])}
    print(f"\n  best run11 L{lr} {best['run11'][1]:.4f}   best AVES L{la} {best['aves'][1]:.4f}   "
          f"run11-AVES {best['run11'][1]-best['aves'][1]:+.4f}")
    r = boot(y, P[("run11", lr)], P[("aves", la)], sess)
    out["bootstrap_run11_vs_aves"] = r
    print(f"  bootstrap over {len(set(sess))} sessions: {r['delta']:+.4f} "
          f"[{r['lo']:+.4f}, {r['hi']:+.4f}]  {r['verdict']}")

    (ANA / "identity_probe.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ANA/'identity_probe.json'}")


if __name__ == "__main__":
    main()

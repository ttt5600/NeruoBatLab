"""Controls. Each one declares the null it expects and FAILS if the null is wrong.

That rule exists because a broken control is worse than no control: the first energy-matched pair
matcher scanned upward and systematically paired positives with quieter negatives, giving energy a
0.926 win rate on pairs that were supposed to be equal-loudness. It was caught only because the
printed label said 'near chance by construction' and the number was not.
"""
from __future__ import annotations
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score


class ControlFailed(Exception):
    pass


def shuffled_label(X, y, groups, cv, n_perm=5, seed=0, tol=0.05, clf=None):
    """Permute labels. Expected null: AUC ~ 0.5. Anything higher means the plumbing leaks."""
    clf = clf or LogisticRegression(max_iter=4000)
    aucs = []
    for s in range(n_perm):
        ys = np.random.default_rng(seed + s).permutation(y)
        p = cross_val_predict(clone(clf), X, ys, groups=groups, cv=cv,
                              method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(ys, p))
    m = float(np.mean(aucs))
    ok = abs(m - 0.5) < tol
    out = dict(mean_auc=m, std=float(np.std(aucs)), per_perm=[float(a) for a in aucs],
               expected_null=0.5, passed=bool(ok))
    if not ok:
        raise ControlFailed(f"shuffled-label control returned AUC {m:.4f}, expected ~0.5 "
                            f"-- the split or prediction path leaks")
    return out


def per_group(y, p, groups, min_n=12):
    """Per-recording (or per-bird) AUC. A high pooled score can hide a few carrying files."""
    groups = np.asarray(groups)
    rows = []
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < min_n or len(np.unique(y[m])) < 2:
            continue
        rows.append(dict(group=str(g), n=int(m.sum()), auc=float(roc_auc_score(y[m], p[m]))))
    a = np.array([r["auc"] for r in rows])
    return dict(groups=rows, n_groups=len(rows), median=float(np.median(a)), mean=float(a.mean()),
                q25=float(np.percentile(a, 25)), q75=float(np.percentile(a, 75)),
                min=float(a.min()), max=float(a.max()),
                n_below_0p80=int((a < 0.80).sum()))


def loudness_stratified(y, p, energy_db, n_bands=5):
    """AUC within narrow dB bands, so no comparison crosses loudness.

    Expected null: the energy baseline collapses toward 0.5 inside a band. If it does not, the
    bands are too wide to have removed loudness and the model number from them means little.
    """
    y, p, e = np.asarray(y).astype(int), np.asarray(p), np.asarray(energy_db)
    edges = np.percentile(e, np.linspace(0, 100, n_bands + 1))
    rows, w = [], []
    for i in range(n_bands):
        m = (e >= edges[i]) & (e <= edges[i + 1])
        if len(np.unique(y[m])) < 2:
            continue
        rows.append(dict(band=f"{edges[i]:.1f}..{edges[i+1]:.1f} dB", n=int(m.sum()),
                         model=float(roc_auc_score(y[m], p[m])),
                         energy=float(roc_auc_score(y[m], e[m]))))
        w.append(m.sum())
    w = np.array(w, float); w /= w.sum()
    mm = float(np.sum(w * [r["model"] for r in rows]))
    me = float(np.sum(w * [r["energy"] for r in rows]))
    ok = abs(me - 0.5) < 0.12
    return dict(bands=rows, weighted_model_auc=mm, weighted_energy_auc=me,
                energy_null_ok=bool(ok),
                note=("energy is near chance within bands, so loudness is controlled" if ok else
                      "WARNING: energy is NOT near chance within bands -- bands too wide, "
                      "loudness is not actually controlled"))


def loudness_matched_pairs(y, p, energy_db, tol_db=0.5, seed=0, window=200):
    """1:1 pairs with (nearly) identical dB. Expected null: energy wins ~50% of pairs.

    Takes the NEAREST unused negative, not the first one found while scanning -- scanning in one
    direction biases the pairing and inflates the energy null.
    """
    y, p, e = np.asarray(y).astype(int), np.asarray(p), np.asarray(energy_db)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    order = np.argsort(e[neg]); negs = neg[order]; en = e[negs]
    used = np.zeros(len(negs), bool)
    pairs = []
    for i in np.random.default_rng(seed).permutation(pos):
        j = np.searchsorted(en, e[i])
        best, bd = None, tol_db + 1
        for k in range(max(0, j - window), min(len(negs), j + window)):
            if used[k]:
                continue
            dd = abs(en[k] - e[i])
            if dd < bd:
                bd, best = dd, k
        if best is not None and bd <= tol_db:
            used[best] = True
            pairs.append((i, negs[best]))
    if not pairs:
        raise ControlFailed(f"no pairs matched within {tol_db} dB")
    pr = np.array(pairs)
    win = lambda v: float((v[pr[:, 0]] > v[pr[:, 1]]).mean() + 0.5 * (v[pr[:, 0]] == v[pr[:, 1]]).mean())
    we, wm = win(e), win(p)
    ok = abs(we - 0.5) < 0.06
    out = dict(n_pairs=len(pr), mean_signed_db_gap=float((e[pr[:, 0]] - e[pr[:, 1]]).mean()),
               max_abs_db_gap=float(np.abs(e[pr[:, 0]] - e[pr[:, 1]]).max()),
               model_win_rate=wm, energy_win_rate=we, expected_energy_null=0.5,
               matching_unbiased=bool(ok))
    if not ok:
        raise ControlFailed(f"pair matching is biased: energy wins {we:.4f} of equal-loudness "
                            f"pairs, expected ~0.5. Do not use the model number from this run.")
    return out

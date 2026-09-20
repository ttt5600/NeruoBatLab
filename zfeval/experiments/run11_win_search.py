#!/usr/bin/env python
"""A disciplined search for regimes where colony-specific pretraining beats generic pretraining.

The frozen single-layer linear probe is one protocol, and run11 loses under it on call type. That
does not settle whether the colony model is worse at everything, but the honest way to ask is to
enumerate the axes IN ADVANCE, test them all, and report the whole list -- including the losses --
so the reader can see how many chances were taken. Reporting only the axis that won would be a
garden of forking paths.

Axes tested here (all CPU, all on cached 12-layer embeddings):
  A  multi-layer concatenation, with the layer subset chosen OUT OF FOLD so the selection cannot
     leak; this is where a model whose information is spread across depth would gain
  B  bird identity, the task the clustering result predicts run11 should win, with the layer choice
     also made out of fold
  C  per-call-type, restricted to the types run11 led on in the exploratory pass (Be, Ag), now with
     a bootstrap so we can tell a real effect from the 2-in-11 we would expect by chance
  D  chick vs adult, since chick calls are heavily represented in the colony corpus and rare in
     generic animal audio

Every axis reports a cluster bootstrap over the correct independent unit, and the script prints a
running count of how many axes were tested so the multiplicity is visible in the output itself.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import calltype11 as C11                                                     # noqa: E402
from aves_calltype import bird_bootstrap                                     # noqa: E402

FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
SEED = 0


def pipe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000))


def oof_with_inner_layer_choice(E, y, groups, layer_sets, n_splits=5, session=None):
    """Out-of-fold predictions where the LAYER SET is chosen inside each training fold.

    Picking the best layer on the same data you report is the single easiest way to manufacture a
    win, and with 12 layers plus subsets there are a lot of chances. Here each outer fold runs its
    own inner CV over the candidate sets and only then touches its test rows.
    """
    from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
    outer = (GroupKFold(n_splits=n_splits) if session is not None
             else StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED))
    g_outer = session if session is not None else groups
    pred = np.full(len(y), -1, dtype=int)
    chosen = []
    for tr, te in outer.split(E[0], y, g_outer):
        best, best_acc = None, -1.0
        inner = (GroupKFold(n_splits=3) if session is not None
                 else StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED))
        gi = g_outer[tr]
        for name, ls in layer_sets.items():
            X = np.hstack([E[l] for l in ls])[tr]
            acc, n = 0.0, 0
            for itr, ite in inner.split(X, y[tr], gi):
                seen = np.isin(y[tr][ite], np.unique(y[tr][itr]))
                if seen.sum() == 0:
                    continue
                e = pipe().fit(X[itr], y[tr][itr])
                acc += float((e.predict(X[ite][seen]) == y[tr][ite][seen]).mean()); n += 1
            if n and acc / n > best_acc:
                best, best_acc = name, acc / n
        chosen.append(best)
        Xf = np.hstack([E[l] for l in layer_sets[best]])
        seen = np.isin(y[te], np.unique(y[tr]))
        e = pipe().fit(Xf[tr], y[tr])
        pred[te[seen]] = e.predict(Xf[te[seen]])
    m = pred >= 0
    return float((pred[m] == y[m]).mean()), pred, chosen


def main():
    rows = [r for r in C11.collect() if r[3] in C11.KEEP11]
    birds = np.array([r[1].lower() for r in rows])
    dates = np.array([r[2] for r in rows])
    tt = np.array([r[3] for r in rows])
    src = np.array([r[4] for r in rows])
    classes = sorted(set(tt))
    y = np.array([classes.index(t) for t in tt])
    assert len(classes) == 11 and len(y) == 3412, f"cohort drift: {len(y)} clips, {len(classes)} cls"
    R = np.load(FEAT / "ct11_run11_emb.npy")
    A = np.load(FEAT / "ct11_aves_emb.npy")
    E_r = {l: R[:, l] for l in range(12)}
    E_a = {l: A[:, l] for l in range(12)}
    print(f"[data] {len(y)} clips, {len(classes)} classes, {len(set(birds))} birds", flush=True)

    LSETS = {**{f"L{l}": [l] for l in range(12)},
             "L0-2": [0, 1, 2], "L0-3": [0, 1, 2, 3], "L2-5": [2, 3, 4, 5],
             "L0+3+6": [0, 3, 6], "L0+2+4+6": [0, 2, 4, 6], "all12": list(range(12))}
    out = {"n_clips": int(len(y)), "classes": classes, "layer_sets": {k: v for k, v in LSETS.items()},
           "axes": {}}
    n_axis = 0

    # ---------------- A: call type, multi-layer, layer set chosen out of fold
    n_axis += 1
    print(f"\n=== AXIS {n_axis}: call type, multi-layer, OUT-OF-FOLD layer selection ===", flush=True)
    ar, pr, cr = oof_with_inner_layer_choice(E_r, y, birds, LSETS)
    aa, pa, ca = oof_with_inner_layer_choice(E_a, y, birds, LSETS)
    b = bird_bootstrap(y, (np.eye(len(classes))[pr]), (np.eye(len(classes))[pa]), birds)
    out["axes"]["A_calltype_multilayer"] = dict(run11=ar, aves=aa, diff=ar - aa,
                                                run11_chosen=cr, aves_chosen=ca, bootstrap=b)
    print(f"  run11 {ar:.4f} (chose {cr})")
    print(f"  AVES  {aa:.4f} (chose {ca})")
    print(f"  diff {ar-aa:+.4f}  bootstrap {b['delta']:+.4f} [{b['lo']:+.4f}, {b['hi']:+.4f}] "
          f"{b['verdict']}", flush=True)

    # ---------------- B: identity, multi-layer, out-of-fold layer choice, session-disjoint
    n_axis += 1
    print(f"\n=== AXIS {n_axis}: bird identity, multi-layer, leave-SESSION-out ===", flush=True)
    keep = np.array([len(set(dates[birds == bb])) >= 2 for bb in birds])
    yb_all = np.array([sorted(set(birds[keep])).index(bb) for bb in birds[keep]])
    sess = np.array([f"{bb}|{d}" for bb, d in zip(birds[keep], dates[keep])])
    Er2 = {l: R[keep][:, l] for l in range(12)}
    Ea2 = {l: A[keep][:, l] for l in range(12)}
    ir, pir, cir = oof_with_inner_layer_choice(Er2, yb_all, None, LSETS, session=sess)
    ia, pia, cia = oof_with_inner_layer_choice(Ea2, yb_all, None, LSETS, session=sess)

    def sess_boot(yy, p1, p2, ss, n=2000):
        rng = np.random.default_rng(SEED)
        uq = np.unique(ss); idx = {g: np.where(ss == g)[0] for g in uq}
        d = []
        for _ in range(n):
            sel = np.concatenate([idx[g] for g in rng.choice(uq, len(uq), replace=True)])
            sel = sel[(p1[sel] >= 0) & (p2[sel] >= 0)]
            if len(sel) < 50:
                continue
            d.append((p1[sel] == yy[sel]).mean() - (p2[sel] == yy[sel]).mean())
        d = np.array(d); lo, hi = np.percentile(d, [2.5, 97.5])
        return dict(delta=float(d.mean()), lo=float(lo), hi=float(hi),
                    verdict=("a_better" if lo > 0 else "b_better" if hi < 0 else
                             "not_distinguishable"))
    bb_ = sess_boot(yb_all, pir, pia, sess)
    out["axes"]["B_identity_multilayer"] = dict(run11=ir, aves=ia, diff=ir - ia,
                                                n_birds=int(len(set(yb_all))),
                                                n_sessions=int(len(set(sess))),
                                                run11_chosen=cir, aves_chosen=cia, bootstrap=bb_)
    print(f"  run11 {ir:.4f} (chose {cir})")
    print(f"  AVES  {ia:.4f} (chose {cia})")
    print(f"  diff {ir-ia:+.4f}  bootstrap {bb_['delta']:+.4f} [{bb_['lo']:+.4f}, {bb_['hi']:+.4f}] "
          f"{bb_['verdict']}", flush=True)

    # ---------------- C: the two call types run11 led on, now with a bootstrap
    n_axis += 1
    print(f"\n=== AXIS {n_axis}: per-class, the types run11 led on exploratorily ===", flush=True)
    from sklearn.model_selection import StratifiedGroupKFold
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    L = 3
    pr3 = np.zeros(len(y), int); pa3 = np.zeros(len(y), int)
    for tr, te in cv.split(R[:, L], y, birds):
        pr3[te] = pipe().fit(R[tr, L], y[tr]).predict(R[te, L])
        pa3[te] = pipe().fit(A[tr, L], y[tr]).predict(A[te, L])
    per = {}
    for c in ("Be", "Ag"):
        i = classes.index(c); m = y == i
        rng = np.random.default_rng(SEED)
        ub = np.unique(birds[m]); idx = {g: np.where(birds[m] == g)[0] for g in ub}
        rr = (pr3[m] == y[m]); aaa = (pa3[m] == y[m])
        d = []
        for _ in range(2000):
            sel = np.concatenate([idx[g] for g in rng.choice(ub, len(ub), replace=True)])
            d.append(rr[sel].mean() - aaa[sel].mean())
        d = np.array(d); lo, hi = np.percentile(d, [2.5, 97.5])
        per[c] = dict(n=int(m.sum()), run11=float(rr.mean()), aves=float(aaa.mean()),
                      delta=float(d.mean()), lo=float(lo), hi=float(hi),
                      verdict=("run11_better" if lo > 0 else "aves_better" if hi < 0 else
                               "not_distinguishable"))
        print(f"  {c}: run11 {rr.mean():.3f} AVES {aaa.mean():.3f}  "
              f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  {per[c]['verdict']}", flush=True)
    out["axes"]["C_per_class_followup"] = per

    # ---------------- D: chicks vs adults
    n_axis += 1
    print(f"\n=== AXIS {n_axis}: chick vs adult clips ===", flush=True)
    d4 = {}
    for s in ("ChickVocalizations", "AdultVocalizations"):
        m = src == s
        d4[s] = dict(n=int(m.sum()), run11=float((pr3[m] == y[m]).mean()),
                     aves=float((pa3[m] == y[m]).mean()))
        d4[s]["diff"] = d4[s]["run11"] - d4[s]["aves"]
        print(f"  {s:20s} n={m.sum():5d}  run11 {d4[s]['run11']:.4f}  AVES {d4[s]['aves']:.4f}  "
              f"{d4[s]['diff']:+.4f}")
    out["axes"]["D_chick_vs_adult"] = d4

    out["n_axes_tested"] = n_axis
    wins = [k for k, v in out["axes"].items()
            if isinstance(v, dict) and v.get("bootstrap", {}).get("verdict") == "a_better"]
    out["axes_where_run11_significantly_ahead"] = wins
    print(f"\n=== {n_axis} axes tested; run11 significantly ahead on: {wins or 'none'} ===")
    (ANA / "run11_win_search.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {ANA/'run11_win_search.json'}")


if __name__ == "__main__":
    main()

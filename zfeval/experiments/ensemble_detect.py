#!/usr/bin/env python
"""If two encoders pretrained on disjoint audio make different mistakes, combining them should beat
either. This tests that, and it is the one model improvement the AVES result actually motivates.

The setup so far: run11 (ZF colony, 120 recordings) and AVES (generic animal sound) tie
in-distribution, and on the BirdPark holdout AVES is better at every layer while run11 sits barely
above a log-energy detector. Finding 013 showed that concatenating LAYERS of one encoder buys
+0.005 at best, which is unsurprising -- adjacent layers are nearly the same representation
(CKA 0.85 across the whole stack). Two encoders with disjoint pretraining corpora are a much larger
perturbation, so this is a different bet from 013, not a rerun of it.

Three combiners, cheapest first, because a gain that needs the expensive one is a weaker result:
  CONCAT   features side by side, one linear probe        (1536-d)
  MEAN     average the two probes' probabilities          (no new parameters)
  STACK    logistic regression on the two probabilities   (2 parameters)

Both regimes are scored, because they answer different questions: in-distribution is what a lab
working in this colony would get, and ZF->BirdPark is whether the combination generalises.

The BirdPark bootstrap is reported at two block lengths. 118.5 s of audio is only about four
independent 30 s blocks, which is too few for an interval to mean much; the 10 s blocks give ~12.
Neither is generous, so a per-block win count is reported alongside -- with n this small a
consistent sign across blocks is more informative than a confidence interval on the mean.
"""
from __future__ import annotations
import gc, json, sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zfeval import metrics as mx, splits as sp                              # noqa: E402
from aves_holdout import frame_pass, load_run11                             # noqa: E402
from aves_baseline import load_aves                                         # noqa: E402

SR, HOP, RF = 16000, 320, 400
BP = Path.home() / "zf_labelset/external/birdpark"
FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
# Layer choice is the trap here. run11's best ZF layer is L0 and AVES's best ZF layer is L6
# (0.9674 vs L3's 0.9656) -- but L3 is AVES's best layer on the BIRDPARK test set. Picking L3 and
# then reporting a BirdPark win would be selection on the test set, so L6 is the pre-committed
# default and L3 is available only as an explicitly post-hoc sensitivity check.
#   --aves-layer 6   pre-committed (chosen on ZF)          <- the number to report
#   --aves-layer 3   post-hoc (chosen on BirdPark)         <- sensitivity only
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--aves-layer", type=int, default=6, choices=[0, 3, 6, 9])
_ap.add_argument("--tag", default=None)
_A = _ap.parse_args()
L_RUN11, L_AVES = 0, _A.aves_layer
LAYER_CHOICE = "pre-committed (best AVES layer on ZF)" if L_AVES == 6 else \
               "POST-HOC (best AVES layer on the BirdPark test set) -- sensitivity check only"


def blocks_of(n, size):
    return [np.arange(i, min(i + size, n)) for i in range(0, n, size)]


def block_wins(y, pa, pb, size, metric):
    """How many contiguous blocks does A win? Nonparametric, and honest when n_blocks is tiny."""
    w = t = 0
    for ix in blocks_of(len(y), size):
        if len(np.unique(y[ix])) < 2:
            continue
        t += 1
        w += int(metric(y[ix], pa[ix]) > metric(y[ix], pb[ix]))
    return w, t


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score, average_precision_score

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[device] {device}", flush=True)

    ZF = np.load(FEAT / "frames_30min.npz", allow_pickle=True)
    AV = np.load(FEAT / "aves_frames_matched.npz", allow_pickle=True)
    y_zf, centers = ZF["y"], ZF["centers"]
    Xr = ZF[f"F{L_RUN11}"].astype(np.float32)
    Xa = AV[f"F{L_AVES}"].astype(np.float32)
    print(f"[zf] {len(y_zf)} frames, prevalence {y_zf.mean():.4f}, "
          f"run11 L{L_RUN11} + AVES L{L_AVES}\n[layers] {LAYER_CHOICE}", flush=True)

    grp = np.zeros(len(y_zf), dtype="<U3")
    cv, g, desc = sp.choose_cv(grp, centers, n_splits=5, seed=0)
    out = {"split_in_distribution": desc, "layers": {"run11": L_RUN11, "aves": L_AVES},
           "layer_choice": LAYER_CHOICE,
           "in_distribution": {}, "zf_to_bp": {}, "bootstrap": {}, "block_wins": {}}

    def pipe():
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=1.0))

    # ---------------- in-distribution, out-of-fold
    print("\n=== in-distribution (out-of-fold, contiguous 60 s blocks) ===")
    P = {}
    for nm, X in (("run11", Xr), ("aves", Xa), ("concat", np.hstack([Xr, Xa]))):
        P[nm] = cross_val_predict(pipe(), X, y_zf, groups=g, cv=cv, method="predict_proba")[:, 1]
        s = mx.score(y_zf, P[nm], desc)
        out["in_distribution"][nm] = dict(auc=s.auc, ap=s.ap)
        print(f"  {nm:8s} AUC {s.auc:.4f}  AP {s.ap:.4f}", flush=True)
        gc.collect()
    P["mean"] = (P["run11"] + P["aves"]) / 2
    s = mx.score(y_zf, P["mean"], desc)
    out["in_distribution"]["mean"] = dict(auc=s.auc, ap=s.ap)
    print(f"  {'mean':8s} AUC {s.auc:.4f}  AP {s.ap:.4f}")
    # stacking needs its own out-of-fold pass or it reads its inputs' labels
    P["stack"] = cross_val_predict(pipe(), np.c_[P["run11"], P["aves"]], y_zf, groups=g, cv=cv,
                                   method="predict_proba")[:, 1]
    s = mx.score(y_zf, P["stack"], desc)
    out["in_distribution"]["stack"] = dict(auc=s.auc, ap=s.ap)
    print(f"  {'stack':8s} AUC {s.auc:.4f}  AP {s.ap:.4f}")

    # how correlated are the two encoders' errors? a ceiling on what any combiner can buy
    er, ea = (P["run11"] - y_zf), (P["aves"] - y_zf)
    out["error_correlation"] = float(np.corrcoef(er, ea)[0, 1])
    print(f"\n  error correlation run11 vs AVES: {out['error_correlation']:.4f}"
          f"   (1.0 would mean no combiner can help)")

    print("\n=== paired block bootstrap, in-distribution (1500-frame blocks) ===")
    for nm, f in (("auc", roc_auc_score), ("ap", average_precision_score)):
        for a, b in (("concat", "run11"), ("concat", "aves"), ("mean", "run11"), ("aves", "run11")):
            r = mx.paired_bootstrap(y_zf, P[a], P[b], block=1500, n=2000, seed=0, metric=f)
            out["bootstrap"][f"zf_{a}_vs_{b}_{nm}"] = r
            print(f"  {nm.upper():3s} {a:6s} - {b:6s}: {r['delta']:+.4f} "
                  f"[{r['lo']:+.4f}, {r['hi']:+.4f}]  {r['verdict']}", flush=True)

    # ---------------- ZF -> BirdPark
    bp, sr = sf.read(BP / "birdpark_16k.wav", dtype="float32")
    lab = np.load(BP / "birdpark_labels.npz", allow_pickle=True)
    iv = lab["merged"]
    nF = (len(bp) - RF) // HOP + 1
    t = (np.arange(nF) * HOP + RF / 2) / SR
    y_bp = np.zeros(nF, dtype=int)
    for a, b in iv:
        y_bp[(t >= a) & (t <= b)] = 1
    keep = t <= float(iv.max())
    y_bp = y_bp[keep]

    m = load_run11(device); Fr, _ = frame_pass(m, bp, device, layers=[L_RUN11]); del m; gc.collect()
    m = load_aves(device);  Fa, _ = frame_pass(m, bp, device, layers=[L_AVES]);  del m; gc.collect()
    Br, Ba = Fr[L_RUN11][keep], Fa[L_AVES][keep]

    print(f"\n=== ZF -> BirdPark ({len(y_bp)} frames, prevalence {y_bp.mean():.4f}) ===")
    Q, fitted = {}, {}
    for nm, Xtr, Xte in (("run11", Xr, Br), ("aves", Xa, Ba),
                         ("concat", np.hstack([Xr, Xa]), np.hstack([Br, Ba]))):
        e = pipe().fit(Xtr, y_zf)
        fitted[nm] = e
        Q[nm] = e.predict_proba(Xte)[:, 1]
        s = mx.score(y_bp, Q[nm], "ZF->BP")
        out["zf_to_bp"][nm] = dict(auc=s.auc, ap=s.ap)
        print(f"  {nm:8s} AUC {s.auc:.4f}  AP {s.ap:.4f}", flush=True)
        gc.collect()
    Q["mean"] = (Q["run11"] + Q["aves"]) / 2
    s = mx.score(y_bp, Q["mean"], "ZF->BP")
    out["zf_to_bp"]["mean"] = dict(auc=s.auc, ap=s.ap)
    print(f"  {'mean':8s} AUC {s.auc:.4f}  AP {s.ap:.4f}")
    en = np.array([20 * np.log10(np.sqrt((bp[i * HOP:i * HOP + RF] ** 2).mean()) + 1e-12)
                   for i in range(nF)])[keep]
    Q["logenergy"] = en
    s = mx.score(y_bp, en, "ZF->BP")
    out["zf_to_bp"]["logenergy"] = dict(auc=s.auc, ap=s.ap)
    print(f"  {'energy':8s} AUC {s.auc:.4f}  AP {s.ap:.4f}")

    print("\n=== BirdPark: bootstrap at two block lengths + per-block win counts ===")
    for size, lbl in ((1500, "30 s"), (500, "10 s")):
        for nm, f in (("auc", roc_auc_score), ("ap", average_precision_score)):
            for a, b in (("aves", "run11"), ("concat", "run11"), ("mean", "run11"),
                         ("aves", "logenergy"), ("run11", "logenergy")):
                r = mx.paired_bootstrap(y_bp, Q[a], Q[b], block=size, n=2000, seed=0, metric=f)
                w, tt = block_wins(y_bp, Q[a], Q[b], size, f)
                out["bootstrap"][f"bp{size}_{a}_vs_{b}_{nm}"] = r
                out["block_wins"][f"bp{size}_{a}_vs_{b}_{nm}"] = [w, tt]
                print(f"  [{lbl}] {nm.upper():3s} {a:9s} - {b:9s}: {r['delta']:+.4f} "
                      f"[{r['lo']:+.4f}, {r['hi']:+.4f}]  {r['verdict']:20s} "
                      f"wins {w}/{tt}", flush=True)

    tag = _A.tag or ("" if L_AVES == 6 else f"_avesL{L_AVES}")
    (ANA / f"ensemble_detect{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ANA/f'ensemble_detect{tag}.json'}")


if __name__ == "__main__":
    main()

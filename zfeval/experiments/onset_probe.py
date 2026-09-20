#!/usr/bin/env python
"""Detect BOUNDARIES directly, instead of reading them off a region detector.

Every onset number so far comes from thresholding a "is this frame inside a call" curve and taking
where it crosses. That curve is trained to be flat inside a call -- it is optimised for the region,
and its edges are a by-product. A probe trained on "is this frame within +-1 frame of an onset"
is optimised for exactly the thing being measured.

Two decoders are compared on the same features and the same folds:

  region + crossing   threshold the detection curve, interpolate the crossing   (the incumbent)
  boundary + peak     peak-pick the onset curve, parabolic-interpolate the peak (the challenger)

Parabolic interpolation on three samples around a peak is the standard sub-sample peak estimator;
it is the peak-picking analogue of interpolating a threshold crossing, so neither arm is given a
sub-frame advantage the other lacks.

Both layers are run, because frame-level detection prefers L0 (0.9687) over L6 (0.9602) and there
is no reason to assume boundary timing prefers the same one.
"""
from __future__ import annotations
import argparse, gc, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zfeval import metrics as mx, splits as sp                           # noqa: E402
import subframe_onset as SO                                              # noqa: E402

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict

SR, HOP, RF = 16000, 320, 400
FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"


def cvpred(X, y, cv, g):
    est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000))
    return cross_val_predict(est, X, y, groups=g, cv=cv, method="predict_proba")[:, 1]


def peak_pick(t, p, thr, min_sep_s, parabolic=True):
    """Local maxima above thr, at least min_sep_s apart, strongest first."""
    loc = np.where((p[1:-1] >= p[:-2]) & (p[1:-1] > p[2:]) & (p[1:-1] > thr))[0] + 1
    if len(loc) == 0:
        return np.array([])
    loc = loc[np.argsort(-p[loc])]
    dt = float(np.median(np.diff(t)))
    keep, taken = [], []
    for i in loc:
        if all(abs(t[i] - t[j]) >= min_sep_s for j in taken):
            taken.append(i); keep.append(i)
    keep = np.sort(np.array(keep))
    if not parabolic:
        return t[keep]
    a, b, c = p[keep - 1], p[keep], p[keep + 1]
    denom = (a - 2 * b + c)
    delta = np.where(np.abs(denom) < 1e-12, 0.0, 0.5 * (a - c) / np.where(np.abs(denom) < 1e-12, 1, denom))
    return t[keep] + np.clip(delta, -0.5, 0.5) * dt


def onset_scores(pred_on, true_on, tols=(0.005, 0.010, 0.020, 0.050, 0.100)):
    out = {}
    for tol in tols:
        tp, errs = SO.match_onsets(np.sort(pred_on), true_on, tol)
        P = tp / max(len(pred_on), 1); R = tp / len(true_on)
        out[f"{int(tol*1000)}ms"] = dict(P=P, R=R, F1=(2*P*R/(P+R) if P+R else 0.0))
    tp, errs = SO.match_onsets(np.sort(pred_on), true_on, 0.05)
    out["n_pred"] = int(len(pred_on))
    out["med_abs_err_ms"] = float(np.median(np.abs(errs)) * 1000) if len(errs) else float("nan")
    out["p90_abs_err_ms"] = float(np.percentile(np.abs(errs), 90) * 1000) if len(errs) else float("nan")
    out["bias_ms"] = float(np.mean(errs) * 1000) if len(errs) else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--halfwidth", type=int, default=1, help="frames each side of an onset counted positive")
    a = ap.parse_args()

    FR = np.load(FEAT / "frames_30min.npz", allow_pickle=True)
    yreg, centers, iv = FR["y"], FR["centers"], FR["intervals"]
    layers = [int(l) for l in FR["layers"]]
    nF = len(yreg)
    t = (np.arange(nF) * HOP + RF / 2) / SR
    onsets_s = np.sort(iv[:, 0] / SR)

    # boundary target: frames within +-halfwidth of a true onset
    onset_frame = np.clip(np.round((iv[:, 0] - RF / 2) / HOP).astype(int), 0, nF - 1)
    ybnd = np.zeros(nF, dtype=int)
    for o in onset_frame:
        ybnd[max(0, o - a.halfwidth): o + a.halfwidth + 1] = 1
    print(f"{nF} frames | region prevalence {yreg.mean():.4f} | "
          f"boundary prevalence {ybnd.mean():.4f} (+-{a.halfwidth} frame(s) = "
          f"+-{a.halfwidth*HOP/SR*1000:.0f} ms)\n", flush=True)

    grp = np.zeros(nF, dtype="<U3")
    cv, g, desc = sp.choose_cv(grp, centers, n_splits=5, seed=0)
    out = {"split": desc, "halfwidth_frames": a.halfwidth, "n_true": int(len(onsets_s)),
           "boundary_prevalence": float(ybnd.mean()), "layers": {}}

    for l in layers:
        X = FR[f"F{l}"].astype(np.float32)
        p_reg = cvpred(X, yreg, cv, g)
        p_bnd = cvpred(X, ybnd, cv, g)
        del X; gc.collect()
        sb = mx.score(ybnd, p_bnd, desc)
        print(f"=== layer {l} ===")
        print(f"  boundary probe itself: AUC {sb.auc:.4f}  AP {sb.ap:.4f}")

        res = {"boundary_frame_auc": sb.auc, "boundary_frame_ap": sb.ap, "decoders": {}}

        # incumbent: region curve, threshold crossing with interpolation
        best, bf1 = None, -1
        for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
            for md in (0.02, 0.04):
                for mg in (0.0, 0.02):
                    ivp = SO.events_from_curve(t, p_reg, thr, md, mg, 1, interp=True)
                    if len(ivp) == 0:
                        continue
                    f1 = onset_scores(ivp[:, 0], onsets_s)["20ms"]["F1"]
                    if f1 > bf1:
                        bf1, best = f1, dict(thr=thr, min_dur=md, merge_gap=mg)
        ivp = SO.events_from_curve(t, p_reg, best["thr"], best["min_dur"], best["merge_gap"], 1, True)
        res["decoders"]["region_crossing_interp"] = dict(params=best, **onset_scores(ivp[:, 0], onsets_s))

        # challenger: boundary curve, parabolic peak picking
        best2, bf2 = None, -1
        for thr in (0.1, 0.2, 0.3, 0.4, 0.5):
            for sep in (0.03, 0.05, 0.08):
                pk = peak_pick(t, p_bnd, thr, sep)
                if len(pk) == 0:
                    continue
                f1 = onset_scores(pk, onsets_s)["20ms"]["F1"]
                if f1 > bf2:
                    bf2, best2 = f1, dict(thr=thr, min_sep=sep)
        pk = peak_pick(t, p_bnd, best2["thr"], best2["min_sep"])
        res["decoders"]["boundary_peak_parabolic"] = dict(params=best2, **onset_scores(pk, onsets_s))
        pk_np = peak_pick(t, p_bnd, best2["thr"], best2["min_sep"], parabolic=False)
        res["decoders"]["boundary_peak_nearest"] = dict(params=best2, **onset_scores(pk_np, onsets_s))

        for nm, r in res["decoders"].items():
            print(f"  {nm:26s} n={r['n_pred']:5d}  med {r['med_abs_err_ms']:5.1f} ms  "
                  f"bias {r['bias_ms']:+5.1f}  F1@10 {r['10ms']['F1']:.3f}  "
                  f"F1@20 {r['20ms']['F1']:.3f}  F1@50 {r['50ms']['F1']:.3f}")
        out["layers"][f"L{l}"] = res
        np.save(FEAT / f"onset_p_L{l}.npy", p_bnd)
        print()

    (ANA / "onset_probe.json").write_text(json.dumps(out, indent=2))
    print("wrote", ANA / "onset_probe.json")


if __name__ == "__main__":
    main()

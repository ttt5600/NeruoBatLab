#!/usr/bin/env python
"""Can onset timing beat HuBERT's 20 ms frame grid?

The encoder emits one frame every 320 samples, so an onset read off the frame grid carries a
quantisation error uniform on +-10 ms (mean |err| 5 ms) before the model is even wrong. The
measured median onset error is 20 ms -- larger than quantisation -- so the grid may not be the
binding constraint. This tests it three ways, cheapest first:

  nearest   read the onset at the frame where the probability crosses the threshold. The baseline.
  interp    linearly interpolate the crossing BETWEEN frames. Costs nothing: no extra forward
            passes, no retraining, just arithmetic on the curve already computed.
  shifted   run the encoder K times on the input shifted by 320/K samples and interleave the
            probabilities into an effective 20/K ms grid. Costs K forward passes.

If `interp` matches `shifted`, sub-frame onsets are free and nobody should pay for K passes. If
both plateau above 20/K ms, the limit is the model's temporal acuity -- the 400-sample (25 ms)
receptive field -- not the grid, and the recommendation changes from "sample finer" to "the
architecture caps you here".

The probe is fitted ONCE per fold on the unshifted pass and applied unchanged to every shifted
pass, so nothing about the comparison depends on refitting.
"""
from __future__ import annotations
import argparse, gc, json, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zfeval import splits as sp                                          # noqa: E402
import resolution_extract as RX                                          # noqa: E402

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.ndimage import median_filter

SR, HOP, RF = 16000, 320, 400
FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
AUDIO = Path.home() / "zf_labelset/audio/111021-000.wav"


def crossings(t, p, thr, mode):
    """Onset (upward) or offset (downward) threshold crossings.

    mode='nearest' returns the grid time of the first frame past the threshold; 'interp' returns
    the linearly interpolated crossing between the bracketing frames.
    """
    a = p[:-1]; b = p[1:]
    up = (a <= thr) & (b > thr) if mode_is_onset(mode) else (a > thr) & (b <= thr)
    i = np.where(up)[0]
    if len(i) == 0:
        return np.array([])
    if mode.endswith("nearest"):
        return t[i + 1] if mode_is_onset(mode) else t[i]
    f = (thr - a[i]) / np.where(np.abs(b[i] - a[i]) < 1e-12, 1e-12, b[i] - a[i])
    return t[i] + f * (t[i + 1] - t[i])


def mode_is_onset(mode):
    return mode.startswith("on")


def events_from_curve(t, p, thr, min_dur_s, merge_gap_s, smooth, interp):
    """Threshold the curve into intervals, with sub-frame edges when interp=True."""
    q = median_filter(p, size=int(smooth), mode="nearest") if smooth > 1 else p
    m = q > thr
    d = np.diff(np.concatenate(([0], m.view(np.int8), [0])))
    lo = np.where(d == 1)[0]
    hi = np.where(d == -1)[0] - 1
    if len(lo) == 0:
        return np.zeros((0, 2))
    if not interp:
        iv = np.stack([t[lo], t[np.minimum(hi + 1, len(t) - 1)]], 1)
    else:
        on, off = [], []
        for a, b in zip(lo, hi):
            if a > 0:
                f = (thr - q[a - 1]) / max(q[a] - q[a - 1], 1e-12)
                on.append(t[a - 1] + np.clip(f, 0, 1) * (t[a] - t[a - 1]))
            else:
                on.append(t[a])
            if b + 1 < len(t):
                f = (q[b] - thr) / max(q[b] - q[b + 1], 1e-12)
                off.append(t[b] + np.clip(f, 0, 1) * (t[b + 1] - t[b]))
            else:
                off.append(t[b])
        iv = np.stack([np.array(on), np.array(off)], 1)
    # bridge short gaps, then drop short events
    out = [iv[0].tolist()]
    for s, e in iv[1:]:
        if s - out[-1][1] <= merge_gap_s:
            out[-1][1] = e
        else:
            out.append([s, e])
    iv = np.array(out)
    return iv[(iv[:, 1] - iv[:, 0]) >= min_dur_s]


def match_onsets(pred_on, true_on, tol_s):
    """Greedy nearest matching within tolerance; returns matched pairs and the signed errors."""
    used = np.zeros(len(true_on), bool)
    errs, tp = [], 0
    for x in pred_on:
        j = np.searchsorted(true_on, x)
        best, bd = -1, np.inf
        for jj in (j - 1, j, j + 1):
            if 0 <= jj < len(true_on) and not used[jj]:
                dd = abs(true_on[jj] - x)
                if dd < bd:
                    best, bd = jj, dd
        if best >= 0 and bd <= tol_s:
            used[best] = True; tp += 1; errs.append(x - true_on[best])
    return tp, np.array(errs)


def score(iv_pred, iv_true, tol_s):
    if len(iv_pred) == 0:
        return dict(P=0.0, R=0.0, F1=0.0, n_pred=0, med_abs_err_ms=float("nan"),
                    p90_abs_err_ms=float("nan"), bias_ms=float("nan"))
    tp, errs = match_onsets(iv_pred[:, 0], iv_true[:, 0], tol_s)
    P = tp / len(iv_pred); R = tp / len(iv_true)
    return dict(P=P, R=R, F1=(2 * P * R / (P + R) if P + R else 0.0), n_pred=int(len(iv_pred)),
                med_abs_err_ms=float(np.median(np.abs(errs)) * 1000) if len(errs) else float("nan"),
                p90_abs_err_ms=float(np.percentile(np.abs(errs), 90) * 1000) if len(errs) else float("nan"),
                bias_ms=float(np.mean(errs) * 1000) if len(errs) else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4, help="shifted passes; effective grid is 20/K ms")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    FR = np.load(FEAT / "frames_30min.npz", allow_pickle=True)
    y, centers, iv_true = FR["y"], FR["centers"], FR["intervals"]
    X0 = FR[f"F{a.layer}"].astype(np.float32)
    nF = len(y)
    t0 = (np.arange(nF) * HOP + RF / 2) / SR
    print(f"{nF} frames, layer {a.layer}, {len(iv_true)} true calls", flush=True)

    grp = np.zeros(nF, dtype="<U3")
    cv, g, desc = sp.choose_cv(grp, centers, n_splits=5, seed=0)
    print(f"split: {desc}", flush=True)
    folds = list(cv.split(X0, y, groups=g))

    # Only the annotated span is scored; reading all 80.55 min pushed an earlier run into swap.
    need = min(int(RX.SPAN + 20 * SR), sf.info(AUDIO).frames)
    rec, _ = sf.read(AUDIO, dtype="float32", frames=need)
    model = RX.load_model(torch.device(a.device))

    # Probe per fold on the UNSHIFTED pass; reused unchanged for every shifted pass.
    probes = []
    p_base = np.zeros(nF)
    for tr, te in folds:
        est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000))
        est.fit(X0[tr], y[tr])
        probes.append((est, te))
        p_base[te] = est.predict_proba(X0[te])[:, 1]

    # Shifted passes: features at offset d, scored by the fold whose test block covers each frame.
    P = {0: (t0, p_base)}
    for k in range(1, a.K):
        d = int(round(k * HOP / a.K))
        print(f"\n[shifted pass {k}/{a.K-1}] offset {d} samples = {d/SR*1000:.1f} ms", flush=True)
        Fk, nk = RX.frame_pass(model, rec, torch.device(a.device), offset=d, layers=[a.layer])
        Xk = Fk[a.layer].astype(np.float32)
        tk = (d + np.arange(nk) * HOP + RF / 2) / SR
        pk = np.zeros(nk)
        ck = d + np.arange(nk) * HOP                      # sample position of each shifted frame
        # Each shifted frame is scored by the fold that OWNS its time block, so no shifted frame is
        # ever scored by a probe that trained on audio overlapping it.
        for est, te in probes:
            blocks = np.unique((centers[te] / (60.0 * SR)).astype(int))
            sel = np.isin((ck / (60.0 * SR)).astype(int), blocks)
            if sel.any():
                pk[sel] = est.predict_proba(Xk[sel])[:, 1]
        P[k] = (tk, pk)
        del Fk, Xk
        gc.collect()

    # Interleave every pass into one fine grid.
    tf = np.concatenate([P[k][0] for k in range(a.K)])
    pf = np.concatenate([P[k][1] for k in range(a.K)])
    o = np.argsort(tf); tf, pf = tf[o], pf[o]
    iv_s = iv_true / SR
    print(f"\nfine grid: {len(tf)} points, spacing {np.median(np.diff(tf))*1000:.2f} ms", flush=True)

    # Decoder parameters are tuned ONCE on the coarse/nearest arm and then held FIXED across arms,
    # so a difference between arms is the arm, not a different threshold.
    best, bestf1 = None, -1
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        for smooth in (1, 3, 5):
            for md in (0.02, 0.04):
                for mg in (0.0, 0.02, 0.04):
                    iv = events_from_curve(t0, p_base, thr, md, mg, smooth, interp=False)
                    f1 = score(iv, iv_s, 0.05)["F1"]
                    if f1 > bestf1:
                        bestf1, best = f1, dict(thr=thr, smooth=smooth, min_dur=md, merge_gap=mg)
    print(f"decoder tuned on coarse/nearest: {best}  (collar-50 F1 {bestf1:.3f})\n", flush=True)

    TOL = [0.005, 0.010, 0.020, 0.050, 0.100]
    arms = {
        "coarse_nearest": (t0, p_base, False),
        "coarse_interp":  (t0, p_base, True),
        f"shifted_K{a.K}_nearest": (tf, pf, False),
        f"shifted_K{a.K}_interp":  (tf, pf, True),
    }
    out = {"split": desc, "layer": a.layer, "K": a.K, "decoder": best,
           "n_true": int(len(iv_s)), "grid_ms": {"coarse": HOP / SR * 1000,
                                                 "fine": HOP / SR * 1000 / a.K},
           "arms": {}}
    hdr = f"{'arm':24s}{'n_pred':>8s}{'medErr':>9s}{'p90Err':>9s}{'bias':>8s}" + \
          "".join(f"{int(t*1000):>7d}ms" for t in TOL)
    print(hdr); print("-" * len(hdr))
    for nm, (t, p, itp) in arms.items():
        iv = events_from_curve(t, p, best["thr"], best["min_dur"], best["merge_gap"],
                               best["smooth"], interp=itp)
        r = score(iv, iv_s, 0.05)
        r["F1_by_tol"] = {f"{int(tt*1000)}ms": score(iv, iv_s, tt)["F1"] for tt in TOL}
        out["arms"][nm] = r
        print(f"{nm:24s}{r['n_pred']:8d}{r['med_abs_err_ms']:9.1f}{r['p90_abs_err_ms']:9.1f}"
              f"{r['bias_ms']:8.1f}" + "".join(f"{r['F1_by_tol'][f'{int(t*1000)}ms']:9.3f}" for t in TOL))

    (ANA / "subframe_onset.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(FEAT / "subframe_curves.npz", t_coarse=t0, p_coarse=p_base,
                        t_fine=tf, p_fine=pf, intervals_s=iv_s)
    print("\nwrote", ANA / "subframe_onset.json")


if __name__ == "__main__":
    main()

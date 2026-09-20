#!/usr/bin/env python
"""How detection behaves as the analysis segment shrinks -- in both inference regimes.

Read the table by COLUMN, not by row. Each window size is a DIFFERENT task with a different
prevalence, so absolute AUCs are not comparable across rows. What is comparable is the GAP to the
baselines within a row: whether the learned representation matters more or less as resolution
increases.
"""
from __future__ import annotations
import gc, json, sys
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zfeval import metrics as mx, splits as sp, features as fxx        # noqa: E402

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict

SR, HOP, RF = 16000, 320, 400
SPAN_MAX = 79434253 * SR / 44100 + 20 * SR
FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
AUDIO = Path.home() / "zf_labelset/audio/111021-000.wav"
LABEL_DEFS = ["center", "overlap", "half"]


def cvpred(X, y, cv, g, C=1.0):
    est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=C))
    return cross_val_predict(est, X, y, groups=g, cv=cv, method="predict_proba")[:, 1]


def pool_frames(Fl, starts, win):
    # Fl stays float16 in memory (138 MB/layer instead of 277); each slice is cast before the
    # mean, so precision is float32 where it matters and the machine does not swap.
    """Continuous regime: mean of the frames whose CENTRES fall inside each segment.

    A segment shorter than the 20 ms hop can contain no frame centre at all; those rows fall back
    to the single nearest frame, and the count is reported, because silently emitting a zero vector
    would look like a feature rather than a gap.
    """
    centers = np.arange(len(Fl)) * HOP + RF // 2
    lo = np.searchsorted(centers, starts)
    hi = np.searchsorted(centers, starts + win)
    out = np.zeros((len(starts), Fl.shape[1]), dtype=np.float32)
    empty = 0
    for k, (a, b) in enumerate(zip(lo, hi)):
        if b > a:
            out[k] = Fl[a:b].astype(np.float32).mean(0)
        else:
            out[k] = Fl[min(max(a, 0), len(Fl) - 1)].astype(np.float32)
            empty += 1
    return out, empty


def main():
    files = sorted(FEAT.glob("windowed_w*.npz"),
                   key=lambda p: int(p.stem.split("w")[-1]), reverse=True)
    sizes = [int(p.stem.split("w")[-1]) for p in files]
    FR = np.load(FEAT / "frames_30min.npz", allow_pickle=True)
    layers = [int(l) for l in FR["layers"]]
    need = int(round(SPAN_MAX))
    rec, _ = sf.read(AUDIO, dtype="float32", frames=need)
    Fl = {l: FR[f"F{l}"] for l in layers}          # kept float16 on purpose, see pool_frames
    print(f"window sizes {sizes} ms | layers {layers} | audio {len(rec)/SR/60:.2f} min\n")

    report = {}
    for w_ms, fpath in zip(sizes, files):
        Wd = np.load(fpath, allow_pickle=True)
        k = f"w{w_ms}"
        starts, win = Wd["starts"], int(Wd["win"])
        cov = Wd["coverage"]
        grp = np.zeros(len(starts), dtype="<U3")
        cv, g, desc = sp.choose_cv(grp, starts, n_splits=5, seed=0)

        # baselines recomputed at THIS resolution, from audio
        mel = np.stack([fxx.mel_features(rec[int(s):int(s) + win]) for s in starts])
        en = np.array([10 * np.log10(np.mean(rec[int(s):int(s) + win] ** 2) + 1e-10)
                       for s in starts])[:, None]

        row = {"n": len(starts), "n_all": int(Wd["n_all"]), "split": desc,
               "mean_coverage": float(cov.mean()), "frames_per_window": win / HOP}
        print(f"=== {w_ms} ms  ({len(starts)} of {row['n_all']} windows, "
              f"{win/HOP:.2f} frames/window, {desc.split('(')[0].strip()}) ===")
        for ld in LABEL_DEFS:
            y = Wd[ld]
            if y.sum() < 40 or (1 - y).sum() < 40:
                print(f"  [{ld}] skipped: {int(y.sum())} positives of {len(y)}")
                continue
            r = {"prevalence": float(y.mean())}
            for l in layers:
                r[f"windowed_L{l}"] = mx.score(
                    y, cvpred(Wd[f"X{l}"].astype(np.float32), y, cv, g), desc).auc
                Xc, empty = pool_frames(Fl[l], starts, win)
                r[f"continuous_L{l}"] = mx.score(y, cvpred(Xc, y, cv, g), desc).auc
                r["n_empty_pool"] = int(empty)
            r["logmel"] = mx.score(y, cvpred(mel, y, cv, g, C=0.03), desc).auc
            r["logenergy"] = mx.score(y, cvpred(en, y, cv, g), desc).auc
            best_w = max(r[f"windowed_L{l}"] for l in layers)
            best_c = max(r[f"continuous_L{l}"] for l in layers)
            r["gap_windowed_vs_mel"] = best_w - r["logmel"]
            r["gap_continuous_vs_mel"] = best_c - r["logmel"]
            r["context_benefit"] = best_c - best_w
            row[ld] = r
            print(f"  [{ld:7s}] prev {r['prevalence']:.3f} | "
                  f"windowed {best_w:.4f}  continuous {best_c:.4f}  "
                  f"mel {r['logmel']:.4f}  energy {r['logenergy']:.4f} | "
                  f"context {r['context_benefit']:+.4f}  vs-mel {r['gap_continuous_vs_mel']:+.4f}")
        report[k] = row
        del Wd
        gc.collect()
        print()

    # ---- native frame resolution, the floor of this sweep
    yf = FR["y"]
    centers = FR["centers"]
    grp = np.zeros(len(yf), dtype="<U3")
    cvf, gf, descf = sp.choose_cv(grp, centers, n_splits=5, seed=0)
    print(f"=== 20 ms native frames ({len(yf)} frames, {descf.split('(')[0].strip()}) ===")
    fr = {"n": int(len(yf)), "prevalence": float(yf.mean()), "split": descf}
    enf = FR["energy"][:, None]
    for l in layers:
        p = cvpred(Fl[l].astype(np.float32), yf, cvf, gf)
        fr[f"L{l}"] = mx.score(yf, p, descf).auc
        fr[f"L{l}_ap"] = mx.score(yf, p, descf).ap
        np.save(FEAT / f"frame_p_L{l}.npy", p)
    fr["logenergy"] = mx.score(yf, cvpred(enf, yf, cvf, gf), descf).auc
    fr["logenergy_ap"] = mx.score(yf, cvpred(enf, yf, cvf, gf), descf).ap
    for l in layers:
        print(f"  L{l}: AUC {fr[f'L{l}']:.4f}  AP {fr[f'L{l}_ap']:.4f}")
    print(f"  energy: AUC {fr['logenergy']:.4f}  AP {fr['logenergy_ap']:.4f}")
    report["frames_20ms"] = fr

    (ANA / "resolution_sweep.json").write_text(json.dumps(report, indent=2))
    print("\nwrote", ANA / "resolution_sweep.json")


if __name__ == "__main__":
    main()

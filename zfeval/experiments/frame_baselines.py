#!/usr/bin/env python
"""Two things the resolution sweep left unfinished, both of which could overturn its reading.

1. THE MISSING BASELINE. The sweep compares HuBERT to log-mel at every window size except the
   native 20 ms frame grid, where only energy was run. Energy is a straw man at frame level. If
   log-mel is close to HuBERT there too, then the honest claim is "the learned representation wins
   at detecting PRESENCE, not at LOCALISING", and that changes the recommendation.
   Mel is given a 100 ms context per frame -- four times HuBERT's 400-sample receptive field --
   deliberately, so the comparison is conservative for HuBERT.

2. THE SAMPLE-SIZE CONFOUND. The 20 ms arm trains on all 90,061 frames; the 40 ms arm was capped
   at 6,000 windows. A 15x data advantage is not a resolution effect. Both are rerun at n=6,000.
"""
from __future__ import annotations
import gc, json, sys
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zfeval import metrics as mx, splits as sp, features as fxx          # noqa: E402

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict

SR, HOP, RF = 16000, 320, 400
SPAN_MAX = int(79434253 * SR / 44100 + 20 * SR)
FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
AUDIO = Path.home() / "zf_labelset/audio/111021-000.wav"


def cvpred(X, y, cv, g, C=1.0):
    est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=C))
    return cross_val_predict(est, X, y, groups=g, cv=cv, method="predict_proba")[:, 1]


def main():
    FR = np.load(FEAT / "frames_30min.npz", allow_pickle=True)
    y, centers, en = FR["y"], FR["centers"], FR["energy"][:, None]
    layers = [int(l) for l in FR["layers"]]
    nF = len(y)
    rec, _ = sf.read(AUDIO, dtype="float32", frames=min(SPAN_MAX, sf.info(AUDIO).frames))
    print(f"{nF} frames, prevalence {y.mean():.4f}\n", flush=True)

    # ---- frame-level mel, at two context widths
    mels = {}
    for ctx_ms in (25, 100):
        half = int(ctx_ms / 1000 * SR / 2)
        M = np.zeros((nF, 384), dtype=np.float32)
        for i in range(nF):
            c = int(centers[i])
            lo = max(0, c - half); hi = min(len(rec), lo + 2 * half); lo = max(0, hi - 2 * half)
            M[i] = fxx.mel_features(rec[lo:hi])
            if i % 20000 == 0:
                print(f"  mel{ctx_ms}: {i}/{nF}", flush=True)
        mels[ctx_ms] = M
    del rec; gc.collect()

    grp = np.zeros(nF, dtype="<U3")
    cv, g, desc = sp.choose_cv(grp, centers, n_splits=5, seed=0)
    out = {"split": desc, "n_frames": int(nF), "prevalence": float(y.mean()), "full": {}, "matched": {}}

    print(f"\n=== 20 ms frames, ALL {nF} ({desc.split('(')[0].strip()}) ===")
    for nm, X, C in ([(f"hubert_L{l}", FR[f"F{l}"].astype(np.float32), 1.0) for l in layers]
                     + [("logmel_25ms", mels[25], 0.03), ("logmel_100ms", mels[100], 0.03),
                        ("logenergy", en, 1.0)]):
        p = cvpred(X, y, cv, g, C=C)
        s = mx.score(y, p, desc)
        out["full"][nm] = dict(auc=s.auc, ap=s.ap)
        print(f"  {nm:14s} AUC {s.auc:.4f}  AP {s.ap:.4f}")
        del X; gc.collect()

    # ---- matched n, so the 20 ms row can be compared with the 40 ms row
    rng = np.random.default_rng(0)
    sub = np.sort(rng.choice(nF, 6000, replace=False))
    cv2, g2, desc2 = sp.choose_cv(grp[sub], centers[sub], n_splits=5, seed=0)
    print(f"\n=== 20 ms frames, SUBSAMPLED to {len(sub)} to match the windowed arms ===")
    for nm, X, C in ([(f"hubert_L{l}", FR[f"F{l}"][sub].astype(np.float32), 1.0) for l in layers]
                     + [("logmel_100ms", mels[100][sub], 0.03), ("logenergy", en[sub], 1.0)]):
        p = cvpred(X, y[sub], cv2, g2, C=C)
        s = mx.score(y[sub], p, desc2)
        out["matched"][nm] = dict(auc=s.auc, ap=s.ap)
        print(f"  {nm:14s} AUC {s.auc:.4f}  AP {s.ap:.4f}")

    best_h = max(out["full"][f"hubert_L{l}"]["auc"] for l in layers)
    out["hubert_minus_mel100_auc"] = best_h - out["full"]["logmel_100ms"]["auc"]
    out["hubert_minus_mel100_ap"] = (max(out["full"][f"hubert_L{l}"]["ap"] for l in layers)
                                     - out["full"]["logmel_100ms"]["ap"])
    print(f"\nHuBERT - log-mel(100 ms) at frame level:  "
          f"AUC {out['hubert_minus_mel100_auc']:+.4f}   AP {out['hubert_minus_mel100_ap']:+.4f}")
    (ANA / "frame_baselines.json").write_text(json.dumps(out, indent=2))
    print("wrote", ANA / "frame_baselines.json")


if __name__ == "__main__":
    main()

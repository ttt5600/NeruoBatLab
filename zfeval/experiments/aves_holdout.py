#!/usr/bin/env python
"""Does zebra-finch-specific pretraining buy anything a generic animal-sound encoder does not?

aves_baseline.py answered the in-distribution version and the answer was essentially no:
run11 0.9687 vs AVES 0.9674 AUC on the same 30 min of colony audio. That is not yet a verdict.
In-distribution both encoders are read out by a probe fitted on the very recording being scored, so
a representation only has to be linearly separable THERE. The claim that colony-specific
pretraining is worth making has always been a claim about NEW audio.

So this runs the same two encoders through the same probe on the one genuine encoder-level holdout
available: BirdPark (a different lab, different birds, different rig, never in either pretraining
corpus). Direction is ZF -> BP, the clean one per finding 019: fit on 30 min of ZF, test on 118.5 s
of BirdPark. If run11 beats AVES here, colony-specific pretraining buys generalisation and the
in-distribution tie is a ceiling effect. If it does not, the release model's value is not its
weights.

Layer selection is made on ZF and then frozen, never on the BirdPark test set: run11 uses L0 and
AVES uses L6 because those are each encoder's best ZF layer. Every other layer is still printed so
the choice is auditable, but the headline comparison is the pre-committed one.
"""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zfeval import metrics as mx, splits as sp                              # noqa: E402
from aves_baseline import load_aves, LAYERS as AVES_LAYERS                  # noqa: E402

SR, HOP, RF = 16000, 320, 400
BP = Path.home() / "zf_labelset/external/birdpark"
WEIGHTS = ROOT.parent / "release/zf_hubert_run11/weights/zf_hubert_run11_encoder.pt"
FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
BLOCK = 1500                      # 30 s of frames; neighbouring frames are heavily correlated
HEADLINE = {"run11": 0, "aves": 6}


def frame_pass(model, x, device, layers, normalize=True, chunk_sec=20.0):
    nF = (len(x) - RF) // HOP + 1
    F = {l: np.zeros((nF, 768), dtype=np.float32) for l in layers}
    C = int(chunk_sec * SR)
    filled = 0
    for c in range((len(x) + C - 1) // C):
        lo = c * C
        seg = x[lo:min(len(x), lo + C + RF - HOP)]
        if len(seg) < RF:
            break
        if normalize:
            mu, var = float(seg.mean()), float(seg.var())
            xin = ((seg - mu) / np.sqrt(var + 1e-5)).astype(np.float32)
        else:
            xin = seg.astype(np.float32)
        with torch.no_grad():
            feats, _ = model.extract_features(torch.from_numpy(xin).unsqueeze(0).to(device), None)
        i0 = lo // HOP
        take = min(feats[0].shape[1], C // HOP, nF - i0)
        for l in layers:
            F[l][i0:i0 + take] = feats[l][0, :take].cpu().numpy()
        filled += take
    if filled < nF - 2:
        raise RuntimeError(f"only filled {filled}/{nF}")
    return F, nF


def load_run11(device):
    from torchaudio.models import hubert_base
    m = hubert_base()
    m.load_state_dict(torch.load(WEIGHTS, map_location="cpu",
                                 weights_only=False)["state_dict"], strict=True)
    return m.eval().to(device)


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_predict

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[device] {device}", flush=True)

    # ---------------- BirdPark audio + frame labels
    bp, sr = sf.read(BP / "birdpark_16k.wav", dtype="float32")
    assert sr == SR and bp.ndim == 1
    lab = np.load(BP / "birdpark_labels.npz", allow_pickle=True)
    iv_bp = lab["merged"]                                  # seconds
    ann_end = float(iv_bp.max())
    nF_bp = (len(bp) - RF) // HOP + 1
    t_bp = (np.arange(nF_bp) * HOP + RF / 2) / SR
    y_bp = np.zeros(nF_bp, dtype=int)
    for a, b in iv_bp:
        y_bp[(t_bp >= a) & (t_bp <= b)] = 1
    keep = t_bp <= ann_end                                 # never score past the annotated span
    print(f"[birdpark] {len(bp)/SR:.1f} s audio, {len(iv_bp)} events, annotated to {ann_end:.1f} s")
    print(f"[birdpark] {keep.sum()} scorable frames, prevalence {y_bp[keep].mean():.4f}", flush=True)

    # ---------------- encode BirdPark with both encoders
    bpf = {}
    m = load_run11(device)
    F, _ = frame_pass(m, bp, device, layers=[0, 6])
    bpf["run11"] = F
    del m; gc.collect()
    m = load_aves(device)
    F, _ = frame_pass(m, bp, device, layers=AVES_LAYERS)
    bpf["aves"] = F
    del m; gc.collect()
    print("[encoded] BirdPark through both encoders", flush=True)

    # ---------------- ZF training features
    ZF = np.load(FEAT / "frames_30min.npz", allow_pickle=True)
    y_zf, centers = ZF["y"], ZF["centers"]
    AV = np.load(FEAT / "aves_frames_matched.npz", allow_pickle=True)
    zff = {"run11": {0: ZF["F0"], 6: ZF["F6"]}, "aves": {l: AV[f"F{l}"] for l in AVES_LAYERS}}

    out = {"birdpark": dict(n_frames_scored=int(keep.sum()), prevalence=float(y_bp[keep].mean()),
                            n_events=int(len(iv_bp)), annotated_sec=ann_end),
           "zf_train": dict(n_frames=int(len(y_zf)), prevalence=float(y_zf.mean())),
           "block_frames": BLOCK, "headline_layers": HEADLINE,
           "zf_to_bp": {}, "bp_internal": {}, "zf_in_distribution": {}}

    # ---------------- ZF -> BP transfer
    print("\n=== ZF -> BirdPark  (fit on 30 min ZF, test on unseen BirdPark) ===")
    preds_bp = {}
    for enc, layers in (("run11", [0, 6]), ("aves", AVES_LAYERS)):
        for l in layers:
            est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=1.0))
            est.fit(zff[enc][l].astype(np.float32), y_zf)
            p = est.predict_proba(bpf[enc][l])[:, 1][keep]
            s = mx.score(y_bp[keep], p, "ZF->BP")
            out["zf_to_bp"][f"{enc}_L{l}"] = dict(auc=s.auc, ap=s.ap)
            preds_bp[f"{enc}_L{l}"] = p
            star = "  <- headline" if HEADLINE.get(enc) == l else ""
            print(f"  {enc}_L{l:<2d}  AUC {s.auc:.4f}  AP {s.ap:.4f}{star}", flush=True)
            gc.collect()
    # energy floor on BirdPark, same frame grid
    en = np.array([20 * np.log10(np.sqrt((bp[i * HOP:i * HOP + RF] ** 2).mean()) + 1e-12)
                   for i in range(nF_bp)])[keep]
    s = mx.score(y_bp[keep], en, "ZF->BP")
    out["zf_to_bp"]["logenergy"] = dict(auc=s.auc, ap=s.ap)
    preds_bp["logenergy"] = en
    print(f"  {'logenergy':<9s}  AUC {s.auc:.4f}  AP {s.ap:.4f}")

    # ---------------- within-BirdPark reference (how hard is BP at all?)
    print("\n=== within-BirdPark 5-fold (reference only, not a holdout) ===")
    grp = np.zeros(int(keep.sum()), dtype="<U3")
    cv, g, desc = sp.choose_cv(grp, (t_bp[keep] * SR).astype(int), n_splits=5, seed=0)
    for enc, l in (("run11", 0), ("run11", 6), ("aves", 6)):
        est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=1.0))
        p = cross_val_predict(est, bpf[enc][l][keep], y_bp[keep], groups=g, cv=cv,
                              method="predict_proba")[:, 1]
        s = mx.score(y_bp[keep], p, desc)
        out["bp_internal"][f"{enc}_L{l}"] = dict(auc=s.auc, ap=s.ap)
        print(f"  {enc}_L{l}  AUC {s.auc:.4f}  AP {s.ap:.4f}")

    # ---------------- bootstraps
    hr, ha = f"run11_L{HEADLINE['run11']}", f"aves_L{HEADLINE['aves']}"
    print(f"\n=== paired block bootstrap on BirdPark ({BLOCK} frames = {BLOCK*HOP/SR:.0f} s blocks) ===")
    for metric, nm in ((None, "auc"), ("ap", "ap")):
        from sklearn.metrics import average_precision_score, roc_auc_score
        f = roc_auc_score if nm == "auc" else average_precision_score
        for a, b in ((hr, ha), (hr, "logenergy"), (ha, "logenergy")):
            r = mx.paired_bootstrap(y_bp[keep], preds_bp[a], preds_bp[b], block=BLOCK,
                                    n=2000, seed=0, metric=f)
            out["zf_to_bp"].setdefault("bootstrap", {})[f"{a}_vs_{b}_{nm}"] = r
            print(f"  {nm.upper():3s} {a} - {b}: {r['delta']:+.4f} "
                  f"[{r['lo']:+.4f}, {r['hi']:+.4f}]  {r['verdict']}", flush=True)

    # ---------------- in-distribution bootstrap, the number aves_baseline could not qualify
    print("\n=== ZF in-distribution, out-of-fold, paired block bootstrap ===")
    grpz = np.zeros(len(y_zf), dtype="<U3")
    cvz, gz, descz = sp.choose_cv(grpz, centers, n_splits=5, seed=0)
    pz = {}
    for enc, l in (("run11", 0), ("aves", 6)):
        est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=1.0))
        pz[f"{enc}_L{l}"] = cross_val_predict(est, zff[enc][l].astype(np.float32), y_zf,
                                              groups=gz, cv=cvz, method="predict_proba")[:, 1]
        gc.collect()
    from sklearn.metrics import average_precision_score, roc_auc_score
    for nm, f in (("auc", roc_auc_score), ("ap", average_precision_score)):
        r = mx.paired_bootstrap(y_zf, pz[hr], pz[ha], block=BLOCK,
                                n=2000, seed=0, metric=f)
        out["zf_in_distribution"][f"{hr}_vs_{ha}_{nm}"] = r
        print(f"  {nm.upper():3s} {hr} - {ha}: {r['delta']:+.4f} "
              f"[{r['lo']:+.4f}, {r['hi']:+.4f}]  {r['verdict']}", flush=True)

    (ANA / "aves_holdout.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ANA/'aves_holdout.json'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""The comparison a reviewer asks for first: does ZF-specific pretraining beat an off-the-shelf
bioacoustic encoder?

AVES (Hagiwara 2023) is HuBERT-base self-supervised on the animal-sound subset of AudioSet/VGGSound.
Same architecture as run11 down to the 320-sample hop, so it is comparable frame for frame and the
ONLY variable is which audio the encoder was pretrained on. If AVES matches run11, the contribution
is "a recipe that works", not "a model worth releasing"; if run11 wins, colony-specific pretraining
is doing something a general animal-sound encoder cannot.

Normalisation is a confound worth spending a second pass on. run11 consumes
(x - mu) / sqrt(var + 1e-5) over the 20 s chunk. AVES was trained on un-normalised waveforms
(group_norm extractor). Running AVES only our way would hobble it; running it only its way would
change two variables at once. Both are run and both are reported.
"""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zfeval import metrics as mx, splits as sp                            # noqa: E402

SR, HOP, RF = 16000, 320, 400
SPAN = int(round(79434253 * SR / 44100))
AUDIO = Path.home() / "zf_labelset/audio/111021-000.wav"
AVES_W = Path.home() / "zf_labelset/external/aves/aves-base-bio.torchaudio.pt"
AVES_C = Path.home() / "zf_labelset/external/aves/aves-base-bio.torchaudio.model_config.json"
FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
LAYERS = [0, 3, 6, 9]


def load_aves(device):
    from torchaudio.models import wav2vec2_model
    cfg = json.load(open(AVES_C))
    cfg["encoder_layer_drop"] = 0.0          # eval-time determinism; layerdrop is a training-only trick
    m = wav2vec2_model(**cfg, aux_num_out=None)
    sd = torch.load(AVES_W, map_location="cpu", weights_only=True)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    # the released checkpoint has no masked-prediction head, which we never use; nothing else may differ
    bad = [k for k in missing if not k.startswith("aux")]
    if bad or unexpected:
        raise RuntimeError(f"AVES state_dict mismatch\n  missing={bad}\n  unexpected={unexpected}")
    print(f"[aves] loaded, {sum(p.numel() for p in m.parameters())/1e6:.1f}M params, "
          f"missing={list(missing)} unexpected={list(unexpected)}")
    return m.eval().to(device)


def frame_pass(model, rec, device, normalize, layers=LAYERS, chunk_sec=20.0):
    """Identical grid to resolution_extract.frame_pass: frame i covers [i*HOP, i*HOP+RF)."""
    nF = (SPAN - RF) // HOP + 1
    F = {l: np.zeros((nF, 768), dtype=np.float16) for l in layers}
    C = int(chunk_sec * SR)
    filled, t0 = 0, time.time()
    for c in range((SPAN + C - 1) // C):
        lo = c * C
        x = rec[lo:min(SPAN, lo + C + RF - HOP)]
        if len(x) < RF:
            break
        if normalize:
            mu, var = float(x.mean()), float(x.var())
            xin = ((x - mu) / np.sqrt(var + 1e-5)).astype(np.float32)
        else:
            xin = x.astype(np.float32)
        with torch.no_grad():
            feats, _ = model.extract_features(torch.from_numpy(xin).unsqueeze(0).to(device), None)
        i0 = lo // HOP
        take = min(feats[0].shape[1], C // HOP, nF - i0)
        for l in layers:
            F[l][i0:i0 + take] = feats[l][0, :take].cpu().numpy().astype(np.float16)
        filled += take
        if c % 20 == 0:
            print(f"    chunk {c}  {filled}/{nF}  {time.time()-t0:.0f}s", flush=True)
    if filled < nF - 2:
        raise RuntimeError(f"only filled {filled}/{nF} frames -- grid misaligned")
    return F, nF


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_predict

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[device] {device}", flush=True)

    FR = np.load(FEAT / "frames_30min.npz", allow_pickle=True)
    y, centers = FR["y"], FR["centers"]
    nF_ref = len(y)

    need = min(SPAN + 20 * SR, sf.info(AUDIO).frames)
    rec, _ = sf.read(AUDIO, dtype="float32", frames=need)
    model = load_aves(device)

    feats_out = {}
    for tag, norm in (("matched", True), ("native", False)):
        p = FEAT / f"aves_frames_{tag}.npz"
        if p.exists():
            print(f"[skip] {p.name} exists")
            feats_out[tag] = p
            continue
        print(f"\n=== AVES frame pass, normalisation={'run11-matched' if norm else 'AVES-native raw'} ===",
              flush=True)
        F, nF = frame_pass(model, rec, device, normalize=norm)
        if nF != nF_ref:
            raise RuntimeError(f"frame count {nF} != reference {nF_ref}")
        np.savez(p, layers=np.array(LAYERS), **{f"F{l}": F[l] for l in LAYERS})
        feats_out[tag] = p
        del F; gc.collect()
    del rec, model; gc.collect()

    # ---------------- score through the identical probe + CV as frame_baselines.py
    grp = np.zeros(nF_ref, dtype="<U3")
    cv, g, desc = sp.choose_cv(grp, centers, n_splits=5, seed=0)
    out = {"split": desc, "n_frames": int(nF_ref), "prevalence": float(y.mean()),
           "aves_layers": LAYERS, "results": {}}

    def run(nm, X):
        est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=1.0))
        pr = cross_val_predict(est, X, y, groups=g, cv=cv, method="predict_proba")[:, 1]
        s = mx.score(y, pr, desc)
        out["results"][nm] = dict(auc=s.auc, ap=s.ap)
        print(f"  {nm:26s} AUC {s.auc:.4f}  AP {s.ap:.4f}", flush=True)
        return pr

    print(f"\n=== frame-level detection, {nF_ref} frames, prevalence {y.mean():.4f} ===")
    print("  --- run11 (ZF-pretrained), from the existing feature file ---")
    for l in [int(v) for v in FR["layers"]]:
        run(f"run11_L{l}", FR[f"F{l}"].astype(np.float32))
    for tag in ("matched", "native"):
        print(f"  --- AVES ({'run11-matched norm' if tag=='matched' else 'AVES-native raw'}) ---")
        A = np.load(feats_out[tag], allow_pickle=True)
        for l in LAYERS:
            run(f"aves_{tag}_L{l}", A[f"F{l}"].astype(np.float32))
            gc.collect()

    best_run11 = max((v["auc"], k) for k, v in out["results"].items() if k.startswith("run11"))
    best_aves = max((v["auc"], k) for k, v in out["results"].items() if k.startswith("aves"))
    out["best_run11"] = {"name": best_run11[1], **out["results"][best_run11[1]]}
    out["best_aves"] = {"name": best_aves[1], **out["results"][best_aves[1]]}
    out["run11_minus_aves_auc"] = best_run11[0] - best_aves[0]
    out["run11_minus_aves_ap"] = (out["results"][best_run11[1]]["ap"]
                                  - out["results"][best_aves[1]]["ap"])
    print(f"\nbest run11 {best_run11[1]} AUC {best_run11[0]:.4f}")
    print(f"best AVES  {best_aves[1]} AUC {best_aves[0]:.4f}")
    print(f"run11 - AVES: AUC {out['run11_minus_aves_auc']:+.4f}  AP {out['run11_minus_aves_ap']:+.4f}")
    (ANA / "aves_baseline.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ANA/'aves_baseline.json'}")


if __name__ == "__main__":
    main()

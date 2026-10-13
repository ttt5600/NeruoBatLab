#!/usr/bin/env python
"""E25: same-size speech encoders from Hugging Face on the 11-class call-type probe.

    ~/zf_labelset/envs/e25/bin/python e25_hf_encoders.py            # every model below
    ~/zf_labelset/envs/e25/bin/python e25_hf_encoders.py --only hf-hubert-base-ls960

All four are ~95M-parameter, 12-layer, 768-d transformers on a wav2vec2-style conv front end, the same
size as run16 and AVES-base. They change the training data (960 h LibriSpeech speech, or 94k h for
WavLM+) and, for three of them, the objective:
  hf-hubert-base-ls960     HuBERT, the same objective AVES uses, on speech (Meta's full 2-iteration recipe)
  hf-wav2vec2-base         contrastive (wav2vec 2.0) on speech
  hf-wavlm-base-plus       HuBERT-style masked prediction + denoising, 94k h speech
  hf-data2vec-audio-base   regression onto teacher layer averages, on speech
Protocol is aves_variants_calltype.py's: calltype11 cohort (3412 clips, 11 classes, 48 birds), each clip
encoded alone at its own length (never batched: finding on group-norm padding), mean-pooled over real
frames, the 12 transformer-layer outputs (hidden_states[1:]), the board probe (aves_calltype.cv_acc,
StandardScaler + LogisticRegression, StratifiedGroupKFold(5) by bird, seed 0) under sklearn 1.7.2.
Input goes through each model's own feature extractor, so normalisation is native to each model (the
setting is recorded). Writes features/ct11_<tag>_emb.npy and analysis/<tag>_calltype.json in the format
calltype_pair.py reads.
"""
from __future__ import annotations
import argparse, json, sys, time, warnings
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
from calltype11 import collect, KEEP11                                     # noqa: E402
from aves_calltype import load_audio, cv_acc                               # noqa: E402

FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
MODELS = {
    "hf-hubert-base-ls960": "facebook/hubert-base-ls960",
    "hf-wav2vec2-base": "facebook/wav2vec2-base",
    "hf-wavlm-base-plus": "microsoft/wavlm-base-plus",
    "hf-data2vec-audio-base": "facebook/data2vec-audio-base",
}


def embed(tag, hf_id, paths, device):
    from transformers import AutoFeatureExtractor, AutoModel
    fe = AutoFeatureExtractor.from_pretrained(hf_id)
    m = AutoModel.from_pretrained(hf_id, output_hidden_states=True).to(device).eval()
    cfg = m.config
    nl, dim = int(cfg.num_hidden_layers), int(cfg.hidden_size)
    params = int(sum(p.numel() for p in m.parameters()))
    out = np.zeros((len(paths), nl, dim), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i, p in enumerate(paths):
            w = load_audio(p).squeeze(0).numpy()
            if w.shape[-1] < 400:
                w = np.pad(w, (0, 400 - w.shape[-1]))
            x = fe(w, sampling_rate=16000, return_tensors="pt").input_values.to(device)
            hs = m(x).hidden_states
            if len(hs) != nl + 1:
                raise RuntimeError(f"{tag}: expected {nl + 1} hidden states, got {len(hs)}")
            for l in range(nl):
                h = hs[l + 1].squeeze(0)
                if h.shape[-1] != dim:
                    raise RuntimeError(f"{tag}: dim {h.shape[-1]} != {dim}")
                out[i, l] = h.mean(0).float().cpu().numpy()
            if (i + 1) % 500 == 0:
                print(f"    {tag} {i + 1}/{len(paths)}  {time.time() - t0:.0f}s", flush=True)
    info = dict(hf_id=hf_id, n_layers=nl, dim=dim, params=params,
                do_normalize=bool(getattr(fe, "do_normalize", False)), model_type=cfg.model_type)
    print(f"    {tag}: {params / 1e6:.2f}M params, {nl} layers x {dim}, do_normalize={info['do_normalize']}, "
          f"{time.time() - t0:.0f}s", flush=True)
    del m
    return out, info


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only"); a = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rows = [r for r in collect() if r[3] in KEEP11]
    paths = [r[0] for r in rows]
    birds = np.array([r[1].lower() for r in rows])
    classes = sorted(set(r[3] for r in rows))
    y = np.array([classes.index(r[3]) for r in rows])
    assert len(y) == 3412 and len(classes) == 11 and len(set(birds)) == 48, "cohort changed"
    for tag, hf_id in MODELS.items():
        if a.only and tag != a.only:
            continue
        print(f"=== {tag} ({hf_id})", flush=True)
        cp = FEAT / f"ct11_{tag}_emb.npy"
        info_p = ANA / f"{tag}_calltype.json"
        if cp.exists() and info_p.exists():
            E = np.load(cp); info = {k: v for k, v in json.load(open(info_p)).items() if k in
                                     ("hf_id", "n_layers", "dim", "params", "do_normalize", "model_type")}
            print(f"    [cache] {cp.name} {E.shape}", flush=True)
        else:
            E, info = embed(tag, hf_id, paths, device)
            np.save(cp, E)
        accs = [cv_acc(np.ascontiguousarray(E[:, l]), y, birds)[0] for l in range(E.shape[1])]
        b = int(np.argmax(accs))
        rec = dict(tag=tag, **info, best_layer11=b, acc11=float(accs[b]), per_layer_acc11=[float(x) for x in accs],
                   split="leave-birds-out StratifiedGroupKFold(5), seed 0",
                   note="E25: Hugging Face checkpoint, frozen; hidden_states[1:], mean-pooled, own feature extractor")
        if info_p.exists():
            old = json.load(open(info_p))
            assert abs(old["acc11"] - rec["acc11"]) < 1e-9, f"{tag}: acc11 not reproduced"
        info_p.write_text(json.dumps(rec, indent=2))
        print(f"    {tag}: best L{b} {accs[b]:.4f}  L3 {accs[3]:.4f}  per-layer {[round(x, 4) for x in accs]}", flush=True)


if __name__ == "__main__":
    main()

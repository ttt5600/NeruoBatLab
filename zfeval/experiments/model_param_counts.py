#!/usr/bin/env python
"""Parameter counts and shapes read straight from each encoder's weights -- no fallbacks, no published figures.

Counts every floating-point tensor in the encoder state_dict (the same weights the call-type board embeds with):
  ours / DAPT   ~/zf_labelset/external/dapt/<tag>.pt (export_weights.py output) and release/zf_hubert_run11
  AVES family   ~/zf_labelset/external/aves/<tag>.torchaudio.pt (+ its model_config.json for heads / FFN width)
  HuBERT base   torchaudio's hubert_fairseq_base_ls960.pth and HF facebook/hubert-base-ls960 (HF adds one
                768-d masked_spec_embed, hence +768)
  Whisper       HF openai/whisper-*; the board embeds the ENCODER only, so both encoder and full-model counts

Needs torch + transformers (the e25 venv):
  HF_HUB_OFFLINE=1 ~/zf_labelset/envs/e25/bin/python -I zfeval/experiments/model_param_counts.py
  -> analysis/model_param_counts.json
"""
from __future__ import annotations
import json, re
from pathlib import Path
import torch

E = Path.home() / "zf_labelset/external"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
REL = Path(__file__).resolve().parents[2] / "release/zf_hubert_run11/weights/zf_hubert_run11_encoder.pt"
TA_HUBERT = Path.home() / ".cache/torch/hub/checkpoints/hubert_fairseq_base_ls960.pth"
OURS = ["run11", "run15_combined", "run16_compute4x", "run16_seed2", "run16_seed3", "run17_accum2", "run18_iter2",
        "run19_avesteacher", "run19_seed2", "run20_fsdonly", "run21_zfonly", "daptreplay3_5e5_step15000"]
AVES = ["aves-base-bio", "aves-base-core", "aves-base-all", "aves-base-nonbio", "birdaves-biox-base",
        "birdaves-biox-large", "birdaves-bioxn-large"]


def load(p):
    sd = torch.load(p, map_location="cpu", weights_only=False)
    for k in ("state_dict", "model"):
        if isinstance(sd, dict) and isinstance(sd.get(k), dict):
            sd = sd[k]
    return sd


# old torch names weight-norm tensors weight_g / weight_v, new torch parametrizations.weight.original0 / 1
WN = {"weight_g": "parametrizations.weight.original0", "weight_v": "parametrizations.weight.original1"}


def norm_key(k):
    head, _, last = k.rpartition(".")
    return f"{head}.{WN[last]}" if last in WN else k


def describe(sd):
    floats = {k: v for k, v in sd.items() if torch.is_tensor(v) and torch.is_floating_point(v)}
    layers = {int(m.group(1)) for k in floats for m in [re.search(r"transformer\.layers\.(\d+)\.", k)] if m}
    proj = next(tuple(v.shape) for k, v in floats.items() if k.endswith("feature_projection.projection.weight"))
    return dict(params=sum(v.numel() for v in floats.values()), tensors=len(floats), layers=len(layers),
                width=proj[0], cnn_channels=proj[1],
                keys=sorted((norm_key(k), tuple(v.shape)) for k, v in floats.items()))


def main():
    out, keysets = {}, {}
    rel_tag = {"run11": REL}
    for t in OURS:
        raw = torch.load(rel_tag.get(t, E / f"dapt/{t}.pt"), map_location="cpu", weights_only=False)
        d = describe(load(rel_tag.get(t, E / f"dapt/{t}.pt")))
        keysets[t] = d.pop("keys")
        cfg, meta = raw.get("encoder_config") or {}, raw.get("meta") or {}
        if cfg:   # export_weights.py records the encoder config and the source checkpoint's step
            d.update(heads=cfg["encoder_num_heads"], ffn=cfg["encoder_ff_interm_features"])
        if "global_step" in meta:
            d.update(global_step=meta["global_step"], source_checkpoint=meta.get("source_checkpoint"))
        out[t] = d
    for t in AVES:
        d = describe(load(E / f"aves/{t}.torchaudio.pt"))
        keysets[t] = d.pop("keys")
        cfg = json.load(open(E / f"aves/{t}.torchaudio.model_config.json"))
        d.update(heads=cfg["encoder_num_heads"], ffn=cfg["encoder_ff_interm_features"],
                 cnn_layers=len(cfg["extractor_conv_layer_config"]))
        assert d["layers"] == cfg["encoder_num_layers"] and d["width"] == cfg["encoder_embed_dim"], t
        out[t] = d
    d = describe(load(TA_HUBERT)); keysets["hubert-base-torchaudio"] = d.pop("keys"); out["hubert-base-torchaudio"] = d

    from transformers import AutoModel, WhisperModel
    m = AutoModel.from_pretrained("facebook/hubert-base-ls960")
    out["hf-hubert-base-ls960"] = dict(params=sum(p.numel() for p in m.parameters()),
                                       layers=m.config.num_hidden_layers, width=m.config.hidden_size,
                                       heads=m.config.num_attention_heads, ffn=m.config.intermediate_size)
    for sz in ["tiny", "base", "small"]:
        w = WhisperModel.from_pretrained(f"openai/whisper-{sz}")
        out[f"hf-whisper-{sz}-enc"] = dict(params=sum(p.numel() for p in w.encoder.parameters()),
                                           params_full_model=sum(p.numel() for p in w.parameters()),
                                           layers=w.config.encoder_layers, width=w.config.d_model,
                                           heads=w.config.encoder_attention_heads, ffn=w.config.encoder_ffn_dim)

    ref = keysets["run16_compute4x"]   # (name, shape) pairs, weight-norm names normalised
    same = sorted(t for t, k in keysets.items() if k == ref)
    res = dict(method="sum of numel over floating-point tensors of the encoder state_dict; Whisper: encoder only "
                      "(what the board embeds), full model alongside",
               same_tensors_as_run16=same, models=out)
    (ANA / "model_param_counts.json").write_text(json.dumps(res, indent=1))
    for t, d in out.items():
        print(f"{t:28s} {d['params']:>12,d}  {d['layers']:>2} x {d['width']}")
    print("same tensor names and shapes as run16:", same)


if __name__ == "__main__":
    main()

"""Hugging Face waveform SSL encoders (E25 batch 1) behind the torchaudio extract_features() interface,
for detection_variants.frame_pass.

These are the same conv front end as ours (kernels 10,3,3,3,3,2,2, strides 5,2,2,2,2,2,2), so frame i covers
samples [320 i, 320 i + 400) -- detection_variants' grid, no shift. The waveform goes straight to the model:
frame_pass already gives both input conventions ('matched' = chunk zero-mean unit-variance, which is what the
feature extractor's do_normalize does; 'native' = raw), so the feature extractor is not used here.
Layers = hidden_states[1:] (the 12 transformer block outputs), as in e25_hf_encoders.
Needs transformers: run detection_variants.py from ~/zf_labelset/envs/e25/bin/python for these models.
"""
from __future__ import annotations
import torch

HF_IDS = {"hf-hubert-base-ls960": "facebook/hubert-base-ls960",
          "hf-wav2vec2-base": "facebook/wav2vec2-base",
          "hf-wavlm-base-plus": "microsoft/wavlm-base-plus",
          "hf-data2vec-audio-base": "facebook/data2vec-audio-base"}


def hf_spec(name: str) -> dict:
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(HF_IDS[name])
    return dict(kind="hf-ssl", cfg=None, weights=None, hf_id=HF_IDS[name],
                n_layers=int(c.num_hidden_layers), dim=int(c.hidden_size))


class HFFrames(torch.nn.Module):
    def __init__(self, hf_id: str, device: str):
        super().__init__()
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained(hf_id).eval().to(device)

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor, lengths=None):
        hs = self.m(x, output_hidden_states=True).hidden_states[1:]
        return list(hs), None

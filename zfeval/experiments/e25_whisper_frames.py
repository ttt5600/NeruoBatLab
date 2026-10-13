"""Whisper encoder behind the torchaudio extract_features() interface, for detection_variants.frame_pass.

frame_pass feeds chunks of up to ~20 s of 16 kHz audio and reads per-layer (1, T, dim) frame features on a 20 ms grid
(frame i covers samples [320 i, 320 i + 400)). Whisper's encoder takes a 30 s log-mel window (10 ms hop, padded)
and its stride-2 conv gives one position per 20 ms, so position i is centred at sample 320 i, 12.5 ms before frame i's centre (320 i + 200). Only
positions over real audio are returned (the rest of the 30 s window is padding), as in e25_hf_encoders.
Needs transformers: run detection_variants.py from ~/zf_labelset/envs/e25/bin/python for these models.
"""
from __future__ import annotations
import numpy as np
import torch

HF_IDS = {"hf-whisper-tiny-enc": "openai/whisper-tiny", "hf-whisper-base-enc": "openai/whisper-base",
          "hf-whisper-small-enc": "openai/whisper-small"}
MAX_SAMPLES = 30 * 16000


def whisper_spec(name: str) -> dict:
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(HF_IDS[name])
    return dict(kind="hf-whisper", cfg=None, weights=None, hf_id=HF_IDS[name],
                n_layers=int(c.encoder_layers), dim=int(c.d_model))


class WhisperFrames(torch.nn.Module):
    def __init__(self, hf_id: str, device: str):
        super().__init__()
        from transformers import AutoFeatureExtractor, WhisperModel
        self.fe = AutoFeatureExtractor.from_pretrained(hf_id)
        self.enc = WhisperModel.from_pretrained(hf_id).get_encoder().eval().to(device)
        self.device = device

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor, lengths=None):
        w = x.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if w.shape[-1] > MAX_SAMPLES:
            raise ValueError(f"chunk of {w.shape[-1]} samples exceeds Whisper's 30 s window")
        feats = self.fe(w, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
        hs = self.enc(feats, output_hidden_states=True).hidden_states[1:]
        n = (len(w) - 400) // 320 + 1                      # frames on detection_variants' grid
        n = min(n, hs[0].shape[1])
        return [h[:, :n] for h in hs], None

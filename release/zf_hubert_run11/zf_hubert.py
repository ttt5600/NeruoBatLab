"""Load the zebra-finch HuBERT encoder and turn audio into embeddings.

Dependencies: torch, torchaudio, numpy. That is the whole list -- no pytorch_lightning, no
fairseq, no code from the training repo.

    from zf_hubert import load_encoder, embed_file
    model = load_encoder("weights/zf_hubert_run11_encoder.pt")
    emb = embed_file(model, "examples/BlaBla0506_110302-DC-01.wav")   # (12, 768)

The model returns 12 layer outputs, one per transformer block. Which one you want depends on
the task -- see `LAYER_NOTES` at the bottom of this file and the README. There is no single
"best layer" for everything, which is exactly why all 12 are exposed.
"""
from pathlib import Path

import numpy as np
import torch
import torchaudio

SAMPLE_RATE = 16000  # the model was pretrained at 16 kHz; anything else is resampled
N_LAYERS = 12
EMBED_DIM = 768
FRAME_STRIDE_MS = 20  # one embedding frame per 20 ms of audio

_resamplers = {}


def load_encoder(path, device="cpu"):
    """Rebuild the encoder from the exported weights file.

    The file carries its own constructor kwargs, so there is no config to keep in sync on your
    end: whatever architecture the weights were trained with is what gets built.
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)
    model = torchaudio.models.wav2vec2_model(aux_num_out=None, **obj["encoder_config"])
    model.load_state_dict(obj["state_dict"], strict=True)
    model.eval().to(device)
    model.zf_meta = obj["meta"]
    return model


def load_audio(path, sample_rate=SAMPLE_RATE):
    """Read a file as mono float32 at the model's sample rate.

    Mono is taken as the channel mean, matching how every evaluation in the README was run.
    If your recordings are multi-channel with a real spatial layout you probably want to pick
    a channel instead -- averaging two mics of the same bird is fine, averaging two different
    birds is not.
    """
    wav, sr = torchaudio.load(str(path))
    if sr != sample_rate:
        if (sr, sample_rate) not in _resamplers:
            _resamplers[(sr, sample_rate)] = torchaudio.transforms.Resample(sr, sample_rate)
        wav = _resamplers[(sr, sample_rate)](wav)
    return wav.mean(0, keepdim=True)


@torch.no_grad()
def embed_frames(model, wav, device=None):
    """All 12 layers at frame resolution: returns (12, T, 768) float32.

    T is roughly len(wav)/16000/0.02. Use this when you care about *when* something happened;
    use embed_file / mean-pooling when you only care about what the clip is.
    """
    device = device or next(model.parameters()).device
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    feats, _ = model.extract_features(wav.to(device), None)  # list of 12 x (1, T, 768)
    return torch.stack([f.squeeze(0) for f in feats]).cpu().numpy()


@torch.no_grad()
def embed_file(model, path, layer=None, device=None):
    """Mean-pooled clip embedding: (12, 768), or (768,) if `layer` is given.

    Mean-pooling over time is what every probe result in the README used. It throws away
    temporal structure, which is the right trade for short single-call clips and the wrong one
    for anything with internal sequence (song motifs, for instance) -- there, use embed_frames.
    """
    out = embed_frames(model, load_audio(path), device=device).mean(axis=1)  # (12, 768)
    return out if layer is None else out[layer]


@torch.no_grad()
def embed_files(model, paths, layer=None, device=None, progress_every=200):
    """Batch version of embed_file. Clips are run one at a time (they are short and of uneven
    length, so padding into batches would need a mask to avoid contaminating the mean)."""
    rows = []
    for i, p in enumerate(paths):
        rows.append(embed_file(model, p, layer=layer, device=device))
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  {i+1}/{len(paths)}")
    return np.stack(rows)


LAYER_NOTES = """\
Which layer to use (measured on this corpus -- see README for the full tables):

  layer 0   vocalization vs. background detection. AUC 0.921, and it declines monotonically
            with depth (0.883 by layer 11). Shallow layers keep the acoustic detail that
            separates "sound happened" from "silence".

  layer 3   8-way call-type classification. Accuracy 0.811 leave-birds-out (2814 clips, 26
            birds). But the profile is nearly flat across all 12 layers (0.789-0.811) and the
            fold-to-fold sd is 0.034, so layer 3 is a weak peak, not a
            sharp optimum. Any of layers 0-5 will perform about the same.

  all 12    If you are doing something new, extract everything (it costs one forward pass) and
            sweep. That is how both numbers above were found.
"""

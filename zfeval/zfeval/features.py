"""Feature extraction: the model under test, plus the baselines it must beat."""
from __future__ import annotations
import hashlib
import numpy as np

SR = 16000
HOP, RF = 320, 400          # HuBERT base: 320-sample stride, 400-sample receptive field


def n_frames(n_samples: int) -> int:
    return max(0, (n_samples - RF) // HOP + 1)


def load_encoder(ckpt_path, device, num_classes, hubert_dir=None):
    """Load a pretraining checkpoint and REFUSE to continue if the encoder did not fully load.

    strict=False is needed because the final projection has k outputs that differ per run and is
    never used -- but it would equally tolerate a wholesale encoder mismatch and hand back a
    randomly initialised model whose numbers look plausible. So every non-final_proj key is
    checked, and a fingerprint of the encoder weights goes into the report for provenance.
    """
    import sys, torch
    if hubert_dir:
        sys.path.insert(0, str(hubert_dir))
    from lightning_modules import HuBERTPreTrainModule
    module = HuBERTPreTrainModule(
        model_name="hubert_pretrain_base", feature_grad_mult=0.1, num_classes=num_classes,
        dataset="x", dataset_path="x", feature_type="spectrogram",
        seconds_per_batch=87.5, learning_rate=1e-4, betas=(0.9, 0.98),
        eps=1e-6, weight_decay=0.01, clip_norm=1.0,
        warmup_updates=0, max_updates=1, extractor_mode="group_norm")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    res = module.load_state_dict(sd, strict=False)
    miss = [k for k in res.missing_keys if not k.startswith("model.final_proj")]
    unex = [k for k in res.unexpected_keys if not k.startswith("model.final_proj")]
    if miss or unex:
        raise RuntimeError(f"encoder did not load cleanly: {len(miss)} missing, "
                           f"{len(unex)} unexpected (e.g. {miss[:3]} {unex[:3]})")
    h = hashlib.md5()
    for k in sorted(sd):
        if ".encoder." in k or "feature_extractor" in k:
            h.update(k.encode()); h.update(sd[k].float().numpy().tobytes())
    meta = dict(path=str(ckpt_path), epoch=ckpt.get("epoch"), global_step=ckpt.get("global_step"),
                n_tensors=len(sd), encoder_fingerprint=h.hexdigest()[:16], num_classes=num_classes)
    return module.model.to(device).eval(), meta


def normalize_segment(rec, start, win, scope="context", ctx_samples=20 * SR):
    """Waveform normalization. 'context' (20 s) reproduces the training chunk statistic."""
    seg = rec[start:start + win].astype(np.float32)
    if scope == "none":
        return seg
    ref = seg if scope == "window" else \
        rec[max(0, start + win // 2 - ctx_samples // 2):
            min(len(rec), max(0, start + win // 2 - ctx_samples // 2) + ctx_samples)]
    m, v = float(np.mean(ref)), float(np.var(ref))
    out = ((seg - m) / np.sqrt(v + 1e-5)).astype(np.float32)
    return np.zeros_like(out) if not np.isfinite(out).all() else out


def window_features(model, rec, starts, win, device, scope="context", ctx_sec=20.0, layers=None):
    """Mean-pooled per-layer features for a list of window starts."""
    import torch
    ctx = int(ctx_sec * SR)
    X, en = [], []
    for s in starts:
        s = int(np.clip(s, 0, len(rec) - win))
        xin = normalize_segment(rec, s, win, scope, ctx)
        with torch.no_grad():
            feats, _ = model.wav2vec2.extract_features(
                torch.from_numpy(xin).float().unsqueeze(0).to(device), None)
        sel = range(len(feats)) if layers is None else layers
        X.append(np.stack([feats[l].mean(1).squeeze(0).cpu().numpy() for l in sel]))
        en.append(10 * np.log10(np.mean(rec[s:s + win] ** 2) + 1e-10))
    return np.stack(X), np.array(en)


def frame_features(model, rec, device, layers, chunk_sec=20.0):
    """Per-frame (50 Hz) features over a whole recording, chunked on the time axis.

    The chunk read extends by RF-HOP so the frame grid lines up with the global one. Without that
    tail each 20 s chunk yields 999 frames instead of 1000 and silently leaves all-zero rows --
    90 of them across a 30-minute file.
    """
    import torch
    span = len(rec)
    nF = n_frames(span)
    C = int(chunk_sec * SR)
    F = {l: np.zeros((nF, 768), dtype=np.float16) for l in layers}
    energy = np.zeros(nF, dtype=np.float32)
    filled = 0
    for c in range((span + C - 1) // C):
        lo = c * C
        x = rec[lo:min(span, lo + C + RF - HOP)]
        if len(x) < RF:
            break
        mu, var = float(x.mean()), float(x.var())
        xin = ((x - mu) / np.sqrt(var + 1e-5)).astype(np.float32)
        with torch.no_grad():
            feats, _ = model.wav2vec2.extract_features(
                torch.from_numpy(xin).unsqueeze(0).to(device), None)
        i0 = lo // HOP
        take = min(feats[0].shape[1], C // HOP, nF - i0)
        for l in layers:
            F[l][i0:i0 + take] = feats[l][0, :take].cpu().numpy().astype(np.float16)
        for k in range(take):
            s = (i0 + k) * HOP
            energy[i0 + k] = 10 * np.log10(np.mean(rec[s:s + RF] ** 2) + 1e-10)
        filled += take
    if filled < nF - 2:
        raise RuntimeError(f"only filled {filled}/{nF} frames -- chunk grid is misaligned")
    return F, energy, nF


# ---------------------------------------------------------------- baselines
def _mel_filterbank(n_fft, n_mels=64, fmin=50, fmax=8000, sr=SR):
    f = np.linspace(0, sr / 2, n_fft // 2 + 1)
    mel = lambda h: 2595 * np.log10(1 + h / 700)
    imel = lambda x: 700 * (10 ** (x / 2595) - 1)
    pts = imel(np.linspace(mel(fmin), mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(f)))
    for i in range(n_mels):
        l, c, r = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.clip(np.minimum((f - l) / (c - l + 1e-9), (r - f) / (r - c + 1e-9)), 0, None)
    return fb


_FB = None


def mel_features(x, n_fft=400, hop=160):
    """384-d log-mel summary. The baseline the model must beat to justify pretraining at all.

    Built to be strong: 64 bands summarized by mean/std/max/p90 plus delta mean/std, and NOT
    normalized, so it keeps absolute loudness that context normalization partly removes from the
    model's own input. Handicapping the model, not the baseline.
    """
    from scipy.signal import spectrogram
    global _FB
    if _FB is None or _FB.shape[1] != n_fft // 2 + 1:
        _FB = _mel_filterbank(n_fft)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    _, _, S = spectrogram(x, SR, nperseg=n_fft, noverlap=n_fft - hop, mode="psd")
    M = np.log(_FB @ S + 1e-10)
    D = np.diff(M, axis=1) if M.shape[1] > 1 else np.zeros((M.shape[0], 1))
    return np.concatenate([M.mean(1), M.std(1), M.max(1), np.percentile(M, 90, axis=1),
                           D.mean(1), D.std(1)])


def energy_features(x):
    """log-energy, the weakest baseline. Reported so the mel gap is visible in context."""
    return np.array([10 * np.log10(np.mean(x ** 2) + 1e-10)])


def frame_energy_features(energy_db):
    """log-energy plus two smoothed versions, for frame-level baselines."""
    return np.stack([energy_db,
                     np.convolve(energy_db, np.ones(5) / 5, "same"),
                     np.convolve(energy_db, np.ones(25) / 25, "same")], 1)

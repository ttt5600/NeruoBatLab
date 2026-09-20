#!/usr/bin/env python
"""Random-init encoder control: does PRETRAINING do the work, or does the ARCHITECTURE?

A linear probe on layer L of a trained HuBERT reaches AUC ~0.96 on vocalization detection. But a
randomly initialised CNN + transformer is still a nonlinear function of the waveform, and random
projections of spectral input preserve a lot of spectral information. So "0.96" only means
"pretraining learned something" if an UNTRAINED network of the same shape scores materially lower.

Three model variants, identical architecture, identical probe, identical folds:

  pretrained   run11 release weights.
  rand_seedN   torchaudio hubert_base() fresh init, no training. Tests the architecture.
  shuffled_seedN
               the pretrained tensors with each tensor's values randomly permuted in place. This
               is the sharper control: the weight DISTRIBUTION is preserved exactly (same mean,
               variance, kurtosis, same per-layer scale), so any gap cannot be explained by
               initialisation scale differing from trained scale. Only the learned STRUCTURE is
               destroyed.

Also extracts the pre-transformer CNN output ("layer -1"), so the trained-vs-random comparison
covers the convolutional front end as well as the 12 transformer blocks.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zfeval import features as fx                                   # noqa: E402

SR, WIN = 16000, 16000
WEIGHTS = Path.home() / "Desktop/vocalizations_lab/release/zf_hubert_run11/weights/zf_hubert_run11_encoder.pt"
AUDIO = Path.home() / "zf_labelset/audio/111021-000.wav"
FEATS = Path.home() / "Desktop/vocalizations_lab/zfeval/runs/run11/windows_soundsep_111021.npz"


def build(variant: str, seed: int):
    """Return (model, meta). Every variant goes through the same constructor."""
    from torchaudio.models import hubert_base
    torch.manual_seed(seed)
    m = hubert_base()
    ref = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    sd_pre = ref["state_dict"]
    # Architecture identity is asserted, not assumed: if the release weights did not exactly fill
    # this constructor's state_dict, the "random" and "pretrained" arms would not be the same shape
    # of network and the comparison would be meaningless.
    own = m.state_dict()
    if sorted(own) != sorted(sd_pre):
        raise RuntimeError("architecture mismatch between hubert_base() and release weights")
    for k in own:
        if own[k].shape != sd_pre[k].shape:
            raise RuntimeError(f"shape mismatch at {k}: {own[k].shape} vs {sd_pre[k].shape}")

    if variant == "pretrained":
        m.load_state_dict(sd_pre, strict=True)
    elif variant == "rand":
        pass                                            # fresh init from manual_seed above
    elif variant == "shuffled":
        g = torch.Generator().manual_seed(seed)
        sd = {}
        for k, v in sd_pre.items():
            flat = v.reshape(-1).clone()
            sd[k] = flat[torch.randperm(flat.numel(), generator=g)].reshape(v.shape)
        m.load_state_dict(sd, strict=True)
        # the permutation must preserve the value multiset exactly, per tensor
        for k in sd:
            a = torch.sort(sd[k].reshape(-1).float()).values
            b = torch.sort(sd_pre[k].reshape(-1).float()).values
            if not torch.allclose(a, b):
                raise RuntimeError(f"shuffle changed the values of {k}")
    else:
        raise ValueError(variant)

    h = 0
    for k in sorted(own):
        h = hash((h, float(m.state_dict()[k].float().sum())))
    return m.eval(), dict(variant=variant, seed=seed, fingerprint=f"{h & 0xffffffffffff:012x}")


def extract(model, rec, starts, device, scope="context"):
    """12 transformer layers + the CNN front end, mean-pooled over the window.

    The CNN output is 512-d (feature_projection widens it to 768 before block 0), so it cannot
    share an array with the transformer layers and is returned separately.
    """
    model = model.to(device)
    X = np.zeros((len(starts), 12, 768), dtype=np.float32)
    Xc = np.zeros((len(starts), 512), dtype=np.float32)
    t0 = time.time()
    for i, s in enumerate(starts):
        s = int(s)
        if s + WIN > len(rec):
            raise RuntimeError(f"window {i} at {s} runs past the {len(rec)}-sample recording")
        xin = fx.normalize_segment(rec, s, WIN, scope)
        xt = torch.from_numpy(xin).float().unsqueeze(0).to(device)
        with torch.no_grad():
            cnn, _ = model.feature_extractor(xt, None)
            feats, _ = model.extract_features(xt, None)
        Xc[i] = cnn.mean(1).squeeze(0).cpu().numpy()
        for l in range(12):
            X[i, l] = feats[l].mean(1).squeeze(0).cpu().numpy()
        if i % 300 == 0:
            print(f"    {i}/{len(starts)}  {time.time()-t0:.0f}s", flush=True)
    return X, Xc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / ".claude/jobs/63c218d9/tmp/randinit_feats.npz"))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()

    d = np.load(FEATS, allow_pickle=True)
    starts, y, grp, ids = d["starts"], d["y"], d["grp"].astype(str), d["ids"]
    Xen_ref = d["Xen"][:, 0]
    rec, sr = sf.read(AUDIO, dtype="float32")
    assert sr == SR, sr
    print(f"{len(starts)} windows, recording {len(rec)/SR/60:.1f} min", flush=True)

    # The stored Savio energies must reproduce locally, or the local audio is not the audio the
    # reference features came from and no comparison against them is valid.
    en = np.array([10 * np.log10(np.mean(rec[int(s):int(s) + WIN] ** 2) + 1e-10) for s in starts])
    md = float(np.median(np.abs(en - Xen_ref)))
    print(f"[align] median |local dB - Savio dB| = {md:.6f}", flush=True)
    if md > 0.01:
        raise RuntimeError("local audio does not match the audio the reference features came from")

    device = torch.device(a.device if a.device == "cpu" or torch.backends.mps.is_available() else "cpu")
    print(f"device {device}", flush=True)

    jobs = [("pretrained", 0)] + [("rand", s) for s in a.seeds] + [("shuffled", s) for s in a.seeds[:2]]
    out, cnn, metas = {}, {}, {}
    for variant, seed in jobs:
        name = variant if variant == "pretrained" else f"{variant}_seed{seed}"
        print(f"[{name}]", flush=True)
        m, meta = build(variant, seed)
        assert not m.training
        out[name], cnn[name] = extract(m, rec, starts, device)
        metas[name] = meta
        del m
        if device.type == "mps":
            torch.mps.empty_cache()

    np.savez_compressed(a.out, y=y, grp=grp, ids=ids, starts=starts, en=en,
                        Xmel=d["Xmel"], variants=np.array(list(out)),
                        **{f"X_{k}": v for k, v in out.items()},
                        **{f"Xcnn_{k}": v for k, v in cnn.items()})
    Path(a.out).with_suffix(".meta.json").write_text(json.dumps(metas, indent=2))
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()

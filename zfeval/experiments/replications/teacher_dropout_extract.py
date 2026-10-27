"""Independent replication: AVES-bio teacher, layer-6 features, train-mode x2 + eval-mode x1, minutes 40-60.
Built from scratch (no import of the project's build_teacher / diagnostic). CPU only."""
import json, sys, time, wave
from pathlib import Path
import numpy as np
import torch
import torchaudio

OUT = Path("/Users/jonathanwang/.claude/jobs/63c218d9/tmp")
WAV = Path.home() / "zf_labelset/audio/111021-000.wav"
CK = Path.home() / "zf_labelset/external/aves/aves-base-bio.torchaudio.pt"
CFG = Path.home() / "zf_labelset/external/aves/aves-base-bio.torchaudio.model_config.json"
SR, WIN, LOOK, LAYER = 16000, 320000, 80, 6
START_MIN, END_MIN = 40, 60
N_WIN = (END_MIN - START_MIN) * 60 * SR // WIN          # 60 windows

torch.set_num_threads(8)
assert not torch.backends.mps.is_built() or True     # we never move anything to mps

# ---- audio: own loader via stdlib wave, int16 -> float32 / 32768
with wave.open(str(WAV), "rb") as w:
    assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2
    n_total = w.getnframes()
    s0 = START_MIN * 60 * SR
    need = N_WIN * WIN + LOOK
    assert s0 + need <= n_total, (s0, need, n_total)
    w.setpos(s0)
    raw = w.readframes(need)
x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
print(f"total {n_total} samples = {n_total/SR/60:.2f} min; stretch {s0/SR/60:.1f}-{(s0+need)/SR/60:.3f} min, "
      f"{len(x)} samples, rms {np.sqrt(np.mean(x**2)):.5f}, min {x.min():.4f} max {x.max():.4f}", flush=True)

# ---- teacher: hubert_pretrain_base (Savio relabel's container) with the bare encoder in .wav2vec2
sd = torch.load(str(CK), map_location="cpu", weights_only=True)
print("state_dict keys", len(sd), "e.g.", list(sd)[:2], flush=True)
hub = torchaudio.models.hubert_pretrain_base()
missing, unexpected = hub.wav2vec2.load_state_dict(sd, strict=True), None
enc = hub.wav2vec2
# dropout probabilities actually present in the module tree
ps = sorted({(type(m).__name__, getattr(m, "p", None)) for m in enc.modules() if "Dropout" in type(m).__name__})
attn_p = sorted({m.dropout for m in enc.modules() if type(m).__name__ == "SelfAttention"})
print("dropout modules:", ps, "attention dropout:", attn_p, "layer_drop:", enc.encoder.transformer.layer_drop, flush=True)

# cross-check: same weights in a wav2vec2_model built from the model_config.json -> identical eval outputs
alt = torchaudio.models.wav2vec2_model(**json.loads(CFG.read_text()), aux_num_out=None)
alt.load_state_dict(sd, strict=True)
alt.eval(); enc.eval()
with torch.inference_mode():
    seg = torch.from_numpy(x[:WIN + LOOK]).view(1, -1)
    f1 = enc.extract_features(seg, num_layers=LAYER)[0][-1][0]
    f2 = alt.extract_features(seg, num_layers=LAYER)[0][-1][0]
    f3 = enc.extract_features(seg, num_layers=LAYER)[0][-1][0]
print("frames/window", f1.shape, "hubert_base vs config-built max|diff|", float((f1 - f2).abs().max()),
      "eval repeat max|diff|", float((f1 - f3).abs().max()), flush=True)
del alt


def run(train: bool, seed: int):
    enc.train(train)
    torch.manual_seed(seed)
    feats = []
    with torch.inference_mode():
        for i in range(N_WIN):
            seg = torch.from_numpy(x[i * WIN: i * WIN + WIN + LOOK]).view(1, -1)
            hs, _ = enc.extract_features(seg, num_layers=LAYER)
            assert len(hs) == LAYER
            feats.append(hs[LAYER - 1][0].numpy().copy())
    return np.concatenate(feats)


for name, train, seed in [("A_train_s11", True, 11), ("B_train_s22", True, 22), ("E_eval", False, 0)]:
    t0 = time.time()
    F = run(train, seed)
    np.save(OUT / f"{name}.npy", F)
    print(name, F.shape, F.dtype, f"mean {F.mean():.4f} std {F.std():.4f}  {time.time()-t0:.0f}s", flush=True)
print("done")

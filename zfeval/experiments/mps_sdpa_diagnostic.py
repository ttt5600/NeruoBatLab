#!/usr/bin/env python
"""Record the MPS attention bug that corrupted the Mac call-type embeddings (finding 076).

Three measurements, all on aves-base-bio (the same torchaudio wav2vec2 module every model here uses):
  1. per sequence length T (frames): transformer layer 0's attention and feed-forward blocks run on random
     input of shape (1, T, 768) on MPS and on CPU; max abs difference
  2. 10 real call-type clips: MPS fp32 and CPU fp32 against a CPU fp64 reference, per layer
  3. per real clip length: which clips the MPS cache gets wrong (from calltype_cpu_rescore's CPU embeddings)

    python zfeval/experiments/mps_sdpa_diagnostic.py   -> analysis/mps_sdpa_diagnostic.json
"""
import json, sys, platform
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from calltype11 import collect, KEEP11                                   # noqa: E402
from detection_variants import spec, load_model                          # noqa: E402
from aves_calltype import load_audio                                     # noqa: E402

FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
OUT = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis/mps_sdpa_diagnostic.json"
res = dict(torch=torch.__version__, platform=platform.platform(), mps=torch.backends.mps.is_available())

mc, _ = load_model("aves-base-bio", spec("aves-base-bio"), "cpu")
mm, _ = load_model("aves-base-bio", spec("aves-base-bio"), "mps")
Lc, Lm = mc.encoder.transformer.layers[0], mm.encoder.transformer.layers[0]
torch.manual_seed(0)
sweep = {}
with torch.no_grad():
    for T in list(range(1, 17)) + [20, 24, 32, 64]:
        x = torch.randn(1, T, 768)
        a = Lc.attention(x)[0]; b = Lm.attention(x.to("mps"))[0].cpu()
        f = Lc.feed_forward(x); g = Lm.feed_forward(x.to("mps")).cpu()
        sweep[T] = dict(attention=float((a - b).abs().max()), feed_forward=float((f - g).abs().max()))
res["layer0_by_T"] = sweep

rows = [r for r in collect() if r[3] in KEEP11]
paths = [r[0] for r in rows[:10]]


def enc(model, dev, dtype):
    model = model.to("cpu").to(dtype).to(dev).eval(); E = []; n = []
    with torch.no_grad():
        for p in paths:
            w = load_audio(p); n.append(int(w.shape[-1]))
            if w.shape[-1] < 400:
                w = torch.nn.functional.pad(w, (0, 400 - w.shape[-1]))
            f, _ = model.extract_features(w.to(dtype).to(dev), None)
            E.append(np.stack([x.squeeze(0).mean(0).cpu().double().numpy() for x in f]))
    return np.array(E), n


g64, n = enc(load_model("aves-base-bio", spec("aves-base-bio"), "cpu")[0], "cpu", torch.float64)
c32, _ = enc(load_model("aves-base-bio", spec("aves-base-bio"), "cpu")[0], "cpu", torch.float32)
m32, _ = enc(load_model("aves-base-bio", spec("aves-base-bio"), "cpu")[0], "mps", torch.float32)
frames = [(max(k, 400) - 400) // 320 + 1 for k in n]
res["ten_clips"] = dict(lengths_samples=n, frames=frames,
                        cpu32_vs_fp64_by_layer=[float(np.abs(c32[:, l] - g64[:, l]).max()) for l in range(12)],
                        mps32_vs_fp64_by_layer=[float(np.abs(m32[:, l] - g64[:, l]).max()) for l in range(12)],
                        mps32_vs_fp64_by_clip_L3=[float(np.abs(m32[i, 3] - g64[i, 3]).max()) for i in range(len(n))])

# all 3,412 clips: which ones the MPS cache gets wrong, by frame count
cpu = np.load(FEAT / "ct11_aves-base-bio_cpuemb.npy", mmap_mode="r")
mps = np.load(FEAT / "ct11_aves_emb.npy", mmap_mode="r")
lens = np.array([int(load_audio(r[0]).shape[-1]) for r in rows])
fr = (np.maximum(lens, 400) - 400) // 320 + 1
bad = np.abs(np.asarray(cpu) - np.asarray(mps)).max(axis=(1, 2)) > 1e-2
res["all_clips"] = dict(n=int(len(rows)), n_changed=int(bad.sum()),
                        by_frames={int(t): dict(n=int((fr == t).sum()), changed=int((bad & (fr == t)).sum()))
                                   for t in sorted(set(fr.tolist())) if t <= 20},
                        changed_with_frames_over_8=int((bad & (fr > 8)).sum()),
                        changed_with_frames_1=int((bad & (fr == 1)).sum()))
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps({k: res[k] for k in ("torch", "mps")}), "->", OUT)
print("attention diff by T:", {t: f"{v['attention']:.1e}" for t, v in sweep.items()})
print("all clips:", res["all_clips"]["n_changed"], "changed; frames>8 changed:", res["all_clips"]["changed_with_frames_over_8"],
      "; T=1 changed:", res["all_clips"]["changed_with_frames_1"])

#!/usr/bin/env python
"""Call-type floors: how much of a pretrained model's 11-class call-type score needs pretraining at all?

The board's models span ~0.81-0.86 and most recipe changes move them by <0.02 (MFCC vs spectrogram labels: +0.0067,
finding 082). Before reading those differences as "targets do not matter", measure what the probe gets with NO
pretraining, on exactly the board protocol (11-class clips, each encoded alone on CPU, mean over frames per layer,
StandardScaler + logistic regression, leave-birds-out StratifiedGroupKFold(5) seed 0):

  randinit_s{0,1,2}   hubert-base architecture (aves-base-bio's config, 12 x 768), random weights, eval mode
  logmel              no network: each clip's 80-bin log-mel (25 ms / 10 ms), mean and sd over time (160-d)

Each is compared with run16_compute4x at its CPU best layer with the 95% bird bootstrap (2,000 resamples of the 48
birds, seed 0, the same resamples as seed_pairs_cpu.py); observed differences.

    python zfeval/experiments/calltype_floors.py      -> analysis/calltype_floors.json
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")
from calltype11 import collect, KEEP11                                     # noqa: E402
from aves_calltype import cv_acc, load_audio                               # noqa: E402
from aves_variants_calltype import embed_all                               # noqa: E402

ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
CFG = Path.home() / "zf_labelset/external/aves/aves-base-bio.torchaudio.model_config.json"


def main():
    torch.set_num_threads(8)
    rows = [r for r in collect() if r[3] in KEEP11]
    paths = [r[0] for r in rows]
    birds = np.array([r[1].lower() for r in rows])
    classes = sorted({r[3] for r in rows}); y = np.array([classes.index(r[3]) for r in rows])
    rng = np.random.default_rng(0); ub = np.unique(birds); bidx = {b: np.where(birds == b)[0] for b in ub}
    BOOT = [np.concatenate([bidx[b] for b in rng.choice(ub, len(ub))]) for _ in range(2000)]
    cpu = json.loads((ANA / "calltype_cpu_rescore.json").read_text())["models"]

    def hits(X):
        a, _, P = cv_acc(np.ascontiguousarray(X), y, birds, return_proba=True)
        return a, (P.argmax(1) == y).astype(float)

    L16 = int(cpu["run16_compute4x"]["best_layer_cpu"])
    a16, h16 = hits(np.load(FEAT / "ct11_run16_compute4x_cpuemb.npy", mmap_mode="r")[:, L16])
    assert abs(a16 - cpu["run16_compute4x"]["acc11_cpu"]) < 1e-9, (a16, cpu["run16_compute4x"]["acc11_cpu"])

    def vs16(h, a):
        bs = np.array([h[s].mean() - h16[s].mean() for s in BOOT]); lo, hi = np.percentile(bs, [2.5, 97.5])
        return dict(diff=float(a - a16), lo=float(lo), hi=float(hi), resolved=bool(lo > 0 or hi < 0))

    out = dict(protocol="board: 11-class, CPU, mean over frames, cv_acc leave-birds-out SGKF(5) seed 0; bird bootstrap "
                        "2000 seed 0 vs run16_compute4x at its best layer", n_clips=len(rows),
               run16=dict(layer=L16, acc=a16), floors={})
    from torchaudio.models import wav2vec2_model
    cfg = json.loads(CFG.read_text()); cfg["encoder_layer_drop"] = 0.0
    for seed in (0, 1, 2):
        t0 = time.time()
        torch.manual_seed(seed)
        m = wav2vec2_model(**cfg, aux_num_out=None).eval()
        E = embed_all(m, paths, "cpu", 12, 768, tag=f"randinit_s{seed}")
        accs = [hits(E[:, l])[0] for l in range(12)]
        L = int(np.argmax(accs)); a, h = hits(E[:, L])
        out["floors"][f"randinit_s{seed}"] = dict(per_layer=accs, best_layer=L, acc=a, layer3=accs[3], vs_run16=vs16(h, a))
        print(f"randinit seed {seed}: best L{L} {a:.4f}, L3 {accs[3]:.4f}, vs run16 {a - a16:+.4f}  ({time.time()-t0:.0f}s)", flush=True)

    import torchaudio
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=80)
    X = np.zeros((len(paths), 160), dtype=np.float32)
    with torch.no_grad():
        for i, p in enumerate(paths):
            w = load_audio(p)
            if w.shape[-1] < 400:
                w = torch.nn.functional.pad(w, (0, 400 - w.shape[-1]))
            s = torch.log(mel(w)[0] + 1e-6)                                   # (80, T)
            X[i] = torch.cat([s.mean(1), s.std(1, unbiased=False)]).numpy()
    a, h = hits(X)
    out["floors"]["logmel"] = dict(acc=a, vs_run16=vs16(h, a))
    print(f"log-mel mean+sd (no network): {a:.4f}, vs run16 {a - a16:+.4f}", flush=True)
    p = ANA / "calltype_floors.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()

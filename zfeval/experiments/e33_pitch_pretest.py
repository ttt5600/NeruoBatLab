#!/usr/bin/env python
"""E33 pre-test: is pitch an axis of individual voice that call type does not need?

E33 proposes voice-invariant DAPT: pitch-shift the input while keeping the targets. Before any GPU
time, this measures on the Mac, with frozen encoders, what a small pitch shift does to
  (a) call type: the board's probe (leave-birds-out folds) trained on unshifted clips and tested on
      the same test-fold clips pitch-shifted by s semitones -> accuracy, prediction flip rate, per type
  (b) bird identity: a 48-way bird probe (StratifiedKFold(5) over clips, seed 0) trained on unshifted
      clips and tested on shifted ones -> accuracy
If (b) collapses while (a) holds, pitch carries voice and not call type, and E33's augmentation targets
the right thing. If (a) collapses too, pitch is part of the call-type code and E33 should not shift it.

Shift: torchaudio.functional.pitch_shift on the 16 kHz waveform (phase vocoder + resample; duration
preserved). Very short clips get some phase-vocoder smearing; this is a diagnostic, not a score.
The unshifted path is checked against the cached board embeddings before anything else.

    python zfeval/experiments/e33_pitch_pretest.py   -> analysis/e33_pitch_pretest.json
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
import numpy as np, torch, torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
from calltype11 import collect, KEEP11                                      # noqa: E402
from aves_calltype import load_audio, SR                                    # noqa: E402
from detection_variants import spec, load_model                             # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold   # noqa: E402
from sklearn.pipeline import make_pipeline                                  # noqa: E402
from sklearn.preprocessing import StandardScaler                            # noqa: E402
from sklearn.linear_model import LogisticRegression                         # noqa: E402

FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
OUT = ANA / "e33_pitch_pretest.json"
MODELS = {"aves-base-bio": ("ct11_aves_emb.npy", 3), "daptv4two_2e4s2_step15000": ("ct11_daptv4two_2e4s2_step15000_emb.npy", 1)}
SHIFTS = [-2, -1, 0, 1, 2]


@torch.no_grad()
def encode(model, paths, layer, device, shift):
    out = np.zeros((len(paths), 768), np.float32)
    for i, p in enumerate(paths):
        w = load_audio(p)
        if shift:
            w = torchaudio.functional.pitch_shift(w, SR, shift, n_fft=512, hop_length=128)
        if w.shape[-1] < 400:
            w = torch.nn.functional.pad(w, (0, 400 - w.shape[-1]))
        feats, _ = model.extract_features(w.to(device), None)
        out[i] = feats[layer].squeeze(0).mean(0).cpu().numpy()
    return out


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rows = [r for r in collect() if r[3] in KEEP11]
    paths = [r[0] for r in rows]
    birds = np.array([r[1].lower() for r in rows])
    tt = np.array([r[3] for r in rows]); classes = sorted(set(tt))
    y = np.array([classes.index(t) for t in tt])
    ub = sorted(set(birds)); yb = np.array([ub.index(b) for b in birds])
    ct_folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(paths, y, birds))
    # bird probe: drop birds with <5 clips so every fold can stratify
    cnt = np.bincount(yb); okb = cnt[yb] >= 5
    bid_idx = np.where(okb)[0]
    bid_folds = [(bid_idx[tr], bid_idx[te]) for tr, te in
                 StratifiedKFold(5, shuffle=True, random_state=0).split(bid_idx, yb[bid_idx])]
    res = dict(shifts=SHIFTS, classes=classes, n_clips=len(y), n_birds_bird_probe=int(len(set(yb[okb]))),
               note=__doc__.split("\n\n")[1], models={})
    for tag, (cache, layer) in MODELS.items():
        t0 = time.time()
        s = spec(tag); model, _ = load_model(tag, s, device); model.eval()
        ref = np.load(FEAT / cache, mmap_mode="r")[:20, layer]
        chk = encode(model, paths[:20], layer, device, 0)
        diff = float(np.abs(chk - ref).max())
        print(f"[{tag}] unshifted path vs cached board embeddings: max abs diff {diff:.2e}", flush=True)
        assert diff < 1e-2, f"{tag}: encoding path does not reproduce the cache ({diff})"
        E = {sh: (np.ascontiguousarray(np.load(FEAT / cache, mmap_mode="r")[:, layer]) if sh == 0
                  else encode(model, paths, layer, device, sh)) for sh in SHIFTS}
        print(f"[{tag}] encoded {len(SHIFTS) - 1} shifts in {time.time() - t0:.0f}s", flush=True)
        pred = {sh: np.zeros(len(y), int) for sh in SHIFTS}
        for tr, te in ct_folds:
            est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000)).fit(E[0][tr], y[tr])
            for sh in SHIFTS:
                pred[sh][te] = est.predict(E[sh][te])
        bpred = {sh: np.zeros(len(y), int) - 1 for sh in SHIFTS}
        for tr, te in bid_folds:
            est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000)).fit(E[0][tr], yb[tr])
            for sh in SHIFTS:
                bpred[sh][te] = est.predict(E[sh][te])
        m = dict(layer=layer, check_max_abs_diff=diff, calltype={}, bird={})
        for sh in SHIFTS:
            m["calltype"][str(sh)] = dict(
                acc=float((pred[sh] == y).mean()), flip_vs_unshifted=float((pred[sh] != pred[0]).mean()),
                recall={c: float((pred[sh][y == k] == k).mean()) for k, c in enumerate(classes)},
                flip_by_type={c: float((pred[sh][y == k] != pred[0][y == k]).mean()) for k, c in enumerate(classes)})
            m["bird"][str(sh)] = dict(acc=float((bpred[sh][okb] == yb[okb]).mean()))
        res["models"][tag] = m
        print(f"[{tag}] call type acc by shift " + " ".join(f"{sh:+d}:{m['calltype'][str(sh)]['acc']:.4f}" for sh in SHIFTS))
        print(f"[{tag}] call type flips     " + " ".join(f"{sh:+d}:{m['calltype'][str(sh)]['flip_vs_unshifted']:.3f}" for sh in SHIFTS))
        print(f"[{tag}] bird-ID acc by shift " + " ".join(f"{sh:+d}:{m['bird'][str(sh)]['acc']:.4f}" for sh in SHIFTS), flush=True)
        del model
    OUT.write_text(json.dumps(res, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

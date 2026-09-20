#!/usr/bin/env python
"""Reproduce the 2023 notebook's AVES number, then remove its defects one at a time.

Asserting that the old ~60% was a preprocessing artifact is cheap. Reproducing it is not, so this
rebuilds the April-2023 pipeline exactly and walks it back to the current protocol one change at a
time. If arm 1 lands near 60 and arm 5 lands near 84, the ladder itself is the argument and nobody
has to take my word for which step mattered.

  1 notebook      half-wave rectify + pad every clip to 250,606 samples + mean over all ~782 frames
                  + nn.Linear trained by SGD(lr=0.01), batch 1, 5 epochs + random 80/20 split
  2 -rectify      drop wav[wav < 0] = 0
  3 -padding      encode each clip at its own length, mean over real frames only
  4 -weak head    converged logistic regression instead of the 5-epoch SGD head
  5 -random split leave-birds-out instead of a random 80/20 split

Same checkpoint throughout: the 2022 fairseq file and the torchaudio port were verified
tensor-identical, so nothing here is a version difference.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aves_baseline import load_aves                                          # noqa: E402
from aves_calltype import load_audio, cv_acc, bird_bootstrap                 # noqa: E402
import calltype11 as C11                                                     # noqa: E402

FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
PAD = 250606           # the notebook's padded length, from out.size() == [3433, 250606]
N_SUB = 800            # padded encoding is ~78x the audio of per-clip; subsample to keep it hours-free
SEED = 0


def notebook_audio(path, rectify):
    """The notebook's load path: mono, lowpass at 8 kHz, optional rectify, resample to 16 k."""
    import soundfile as sf
    from scipy.signal import butter, sosfilt, resample_poly
    x, sr = sf.read(str(path), dtype="float64", always_2d=True)
    x = x.mean(1)
    sos = butter(4, 8000, btype="low", fs=sr, output="sos")
    x = sosfilt(sos, x)
    if rectify:
        x[x < 0] = 0                       # the line that mangles the waveform
    g = np.gcd(int(sr), 16000)
    return resample_poly(x, 16000 // g, int(sr) // g).astype(np.float32)


@torch.no_grad()
def embed_padded(model, paths, device, rectify, layer=None, tag=""):
    """Pad every clip to PAD, then mean-pool over ALL frames -- padding included, as in 2023."""
    n_layers = 12
    out = np.zeros((len(paths), n_layers, 768), dtype=np.float32)
    t0 = time.time()
    for i, p in enumerate(paths):
        w = notebook_audio(p, rectify)
        buf = np.zeros(PAD, dtype=np.float32)
        buf[:min(len(w), PAD)] = w[:PAD]
        t = torch.from_numpy(buf).unsqueeze(0).to(device)
        feats, _ = model.extract_features(t, None)
        for l in range(n_layers):
            out[i, l] = feats[l].squeeze(0).mean(0).cpu().numpy()
        if (i + 1) % 50 == 0:
            print(f"    {tag} {i+1}/{len(paths)}  {time.time()-t0:.0f}s", flush=True)
    return out


def sgd_head_acc(X, y, n_classes, seed=SEED, epochs=5, lr=0.01):
    """The notebook's head: one Linear, SGD, batch size 1, 5 epochs, random 80/20 split."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(y), generator=g).numpy()
    Xs, ys = X[perm], y[perm]
    cut = 4 * len(ys) // 5
    Xtr, ytr, Xte, yte = Xs[:cut], ys[:cut], Xs[cut:], ys[cut:]
    torch.manual_seed(seed)
    head = nn.Linear(768, n_classes)
    opt = torch.optim.SGD(head.parameters(), lr=lr)
    lf = nn.CrossEntropyLoss()
    Xtr_t = torch.from_numpy(Xtr).float(); ytr_t = torch.from_numpy(ytr).long()
    for _ in range(epochs):
        for i in range(len(ytr_t)):
            opt.zero_grad()
            loss = lf(head(Xtr_t[i:i+1]), ytr_t[i:i+1])
            loss.backward(); opt.step()
    with torch.no_grad():
        pred = head(torch.from_numpy(Xte).float()).argmax(1).numpy()
    return float((pred == yte).mean())


def logreg_random_split(X, y, seed=SEED):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    cut = 4 * len(y) // 5
    tr, te = perm[:cut], perm[cut:]
    est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000))
    est.fit(X[tr], y[tr])
    return float((est.predict(X[te]) == y[te]).mean())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sub", type=int, default=N_SUB,
                    help="clips in the ablation subsample; pass a large number to use all of them")
    A = ap.parse_args()
    n_sub = A.n_sub
    sfx = "" if n_sub <= N_SUB else f"_n{n_sub}"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rows = [r for r in C11.collect() if r[3] in C11.KEEP11]
    paths_all = [r[0] for r in rows]
    birds_all = np.array([r[1].lower() for r in rows])
    tt = np.array([r[3] for r in rows])
    classes = sorted(set(tt))
    y_all = np.array([classes.index(t) for t in tt])
    assert len(classes) == 11, f"expected 11 classes, got {len(classes)}: {classes}"

    rng = np.random.default_rng(SEED)
    idx = np.sort(np.concatenate([
        rng.choice(np.where(y_all == c)[0],
                   size=min((y_all == c).sum(), max(6, int(round(n_sub * (y_all == c).mean())))),
                   replace=False) for c in range(len(classes))]))
    paths = [paths_all[i] for i in idx]
    y, birds = y_all[idx], birds_all[idx]
    print(f"[device] {device}")
    print(f"[cohort] full {len(y_all)} clips / {len(classes)} classes; ablation subsample "
          f"{len(y)} clips, {len(set(birds))} birds, majority {np.bincount(y).max()/len(y):.4f}",
          flush=True)

    model = load_aves(device)
    E = {}
    for tag, rect in (("pad_rect", True), ("pad_only", False)):
        cp = FEAT / f"abl_{tag}{sfx}.npy"
        if cp.exists():
            E[tag] = np.load(cp); print(f"[skip] cached {cp.name}")
        else:
            print(f"=== padded encode ({'rectified' if rect else 'clean'} audio), "
                  f"{len(paths)} clips x {PAD/16000:.1f}s ===", flush=True)
            E[tag] = embed_padded(model, paths, device, rect, tag=tag)
            np.save(cp, E[tag])
    # per-clip embeddings: reuse the calltype11 cache, subset to the same clips
    E["clip"] = np.load(FEAT / "ct11_aves_emb.npy")[idx]
    print(f"[clip] per-clip embeddings {E['clip'].shape}", flush=True)

    L = 3                        # AVES's best layer on this task, fixed across all arms
    out = {"n_clips": int(len(y)), "n_classes": len(classes), "n_birds": int(len(set(birds))),
           "majority": float(np.bincount(y).max() / len(y)), "layer": L, "pad_samples": PAD,
           "arms": {}}

    print(f"\n=== ablation ladder (layer {L}, {len(y)} clips, "
          f"majority {out['majority']:.3f}) ===", flush=True)
    a1 = sgd_head_acc(E["pad_rect"][:, L], y, len(classes))
    out["arms"]["1_notebook_full"] = a1
    print(f"  1  notebook as written (rectify + pad + weak head + random split)   {a1:.4f}")

    a2 = sgd_head_acc(E["pad_only"][:, L], y, len(classes))
    out["arms"]["2_minus_rectify"] = a2
    print(f"  2  - half-wave rectification                                        {a2:.4f}")

    a3 = sgd_head_acc(E["clip"][:, L], y, len(classes))
    out["arms"]["3_minus_padding"] = a3
    print(f"  3  - zero padding (encode each clip at its own length)              {a3:.4f}")

    a4 = logreg_random_split(E["clip"][:, L], y)
    out["arms"]["4_minus_weak_head"] = a4
    print(f"  4  - weak head (converged logistic regression)                      {a4:.4f}")

    a5, _ = cv_acc(E["clip"][:, L], y, birds)
    out["arms"]["5_leave_birds_out"] = a5
    print(f"  5  - random split (leave-birds-out)  <- current protocol            {a5:.4f}")

    # how much of the padded embedding is actually the clip?
    durs = np.array([len(notebook_audio(p, False)) / 16000 for p in paths[:200]])
    out["median_clip_sec"] = float(np.median(durs))
    out["signal_fraction_median"] = float(np.median(durs) / (PAD / 16000))
    print(f"\n  median clip {np.median(durs):.3f}s in a {PAD/16000:.2f}s buffer -> "
          f"{100*out['signal_fraction_median']:.1f}% signal")
    # and how collapsed the padded embeddings are
    for tag in ("pad_rect", "pad_only", "clip"):
        X = E[tag][:, L]
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        cs = Xn @ Xn.T
        m = float(cs[np.triu_indices(len(cs), 1)].mean())
        out.setdefault("mean_pairwise_cosine", {})[tag] = m
        print(f"  mean pairwise cosine between clip embeddings, {tag:9s}: {m:.4f}")

    (ANA / f"aves_2023_ablation{sfx}.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ANA/f'aves_2023_ablation{sfx}.json'}")


if __name__ == "__main__":
    main()

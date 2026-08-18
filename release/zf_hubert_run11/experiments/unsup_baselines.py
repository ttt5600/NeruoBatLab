"""Baselines for unsupervised call-type discovery: is HuBERT earning its keep?

An AMI of 0.6 against the human taxonomy sounds good until you ask what a dumb method gets.
Two controls, both computed on exactly the same clips as the HuBERT run:

  DURATION (1 scalar).  Call types differ systematically in length -- a Distance call is not
    a Tet. A clustering that merely sorted clips by duration would post a respectable AMI
    while encoding nothing about spectral content. This is the cheapest way to be embarrassed,
    so it runs first. It also tells us how much of HuBERT's AMI is available for free.

  SPECTROGRAM (log-spaced band energies, mean+std pooled).  The raw acoustics. HuBERT's whole
    claim is that self-supervised pretraining buys structure beyond what is visible in the
    spectrogram; if a mean-pooled spectrogram clusters just as well, the pretraining did not
    pay for itself on this task. Note this is deliberately NOT MFCCs -- log band energies with
    no cepstral transform, because MFCCs have already been tested on this corpus and do not
    work on bioacoustic data.

Not included: a random-init HuBERT control, which is the sharpest test of whether PRETRAINING
(rather than the conv stack's spectral bias) did the work. It needs torch, which the local
analysis env lacks; it belongs on Savio.

The number to watch is not just AMI but AMI-within-bird, for the same reason as everywhere
else in this project: a baseline can score globally by tracking who is calling.
"""
from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np
from scipy.signal import spectrogram as scipy_spec
from sklearn.cluster import AgglomerativeClustering, KMeans

from unsup_calltype import AMI, external, load


def read_mono(path):
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
        width = w.getsampwidth()
        ch = w.getnchannels()
    dt = {1: np.int8, 2: np.int16, 4: np.int32}[width]
    x = np.frombuffer(raw, dtype=dt).astype(np.float64)
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    peak = float(np.abs(x).max()) or 1.0
    return x / peak, sr


def log_bands(n_bands, fmin, fmax, freqs):
    """Triangular filterbank on a log-frequency axis -- a mel-like warp without the cepstral
    step that MFCCs add. Returns (n_bands, n_freqs)."""
    edges = np.geomspace(fmin, fmax, n_bands + 2)
    fb = np.zeros((n_bands, len(freqs)))
    for i in range(n_bands):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        up = (freqs - lo) / max(mid - lo, 1e-9)
        dn = (hi - freqs) / max(hi - mid, 1e-9)
        fb[i] = np.clip(np.minimum(up, dn), 0, None)
        s = fb[i].sum()
        if s > 0:
            fb[i] /= s
    return fb


def spec_features(path, n_bands=64, fmin=250.0, fmax=15000.0):
    x, sr = read_mono(path)
    if len(x) < 256:
        x = np.pad(x, (0, 256 - len(x)))
    nper = min(512, len(x))
    f, t, S = scipy_spec(x, fs=sr, nperseg=nper, noverlap=nper // 2, scaling="spectrum")
    fb = log_bands(n_bands, fmin, min(fmax, sr / 2 * 0.99), f)
    B = np.log(fb @ S + 1e-10)                       # (n_bands, n_frames)
    # mean captures the average spectral shape; std captures how much it moves over the clip,
    # which is what separates a flat tonal call from a swept or modulated one.
    return np.concatenate([B.mean(1), B.std(1)]), len(x) / sr


def cluster_and_score(X, y, birds, ks, seed, l2=True):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if l2 and X.shape[1] > 1:
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n[n == 0] = 1.0
        X = X / n
    out = {}
    for k in ks:
        km = KMeans(k, n_init=10, random_state=seed).fit_predict(X)
        wd = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        out[str(k)] = dict(kmeans=external(km, y, birds), ward=external(wd, y, birds))
    return out, X


def main():
    ap = argparse.ArgumentParser()
    root = Path.home() / "Desktop/vocalizations_lab"
    ap.add_argument("--npz", default=str(root / "release/zf_hubert_run11/data/run11_layersweep.npz"))
    ap.add_argument("--wav-dir", default=str(root / "datasets/11905533/AdultVocalizations"))
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 9, 16])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="unsup_baselines.json")
    args = ap.parse_args()

    X, y, birds, names, classes = load(args.npz, args.layer)
    wav_dir = Path(args.wav_dir)

    feats, durs, missing = [], [], []
    for nm in names:
        p = wav_dir / str(nm)
        if not p.exists():
            missing.append(str(nm))
            feats.append(None)
            durs.append(np.nan)
            continue
        f, d = spec_features(p)
        feats.append(f)
        durs.append(d)
    ok = np.array([f is not None for f in feats])
    print(f"{ok.sum()}/{len(names)} clips read"
          + (f" | {len(missing)} MISSING, excluded" if missing else ""))
    if missing[:5]:
        print("  e.g.", missing[:5])

    S = np.stack([f for f in feats if f is not None])
    D = np.array(durs)[ok]
    Xh, yh, bh = X[ok], y[ok], birds[ok]

    print(f"\nduration: mean {D.mean():.3f}s  median {np.median(D):.3f}s  "
          f"range {D.min():.3f}-{D.max():.3f}s")
    print("per call type (mean duration, s):")
    for ci, cn in enumerate(classes):
        m = yh == ci
        if m.any():
            print(f"  {cn:>3} n={m.sum():>4}  {D[m].mean():.3f} +/- {D[m].std():.3f}")

    res = {}
    res["duration"], _ = cluster_and_score(D, yh, bh, args.ks, args.seed, l2=False)
    res["spectrogram"], Sn = cluster_and_score(S, yh, bh, args.ks, args.seed)
    res["hubert_l%d" % args.layer], _ = cluster_and_score(Xh, yh, bh, args.ks, args.seed)

    print(f"\n=== AMI vs human call types (n={ok.sum()}) ===")
    hdr = " | ".join(f"k={k}".center(17) for k in args.ks)
    print(f"{'representation':<16} | {hdr}")
    print(f"{'':<16} | " + " | ".join("  kmeans    ward ".center(17) for _ in args.ks))
    print("-" * (18 + 20 * len(args.ks)))
    for name, r in res.items():
        cells = " | ".join(f"{r[str(k)]['kmeans']['ami_calltype']:>8.4f} "
                           f"{r[str(k)]['ward']['ami_calltype']:>8.4f}" for k in args.ks)
        print(f"{name:<16} | {cells}")

    print(f"\n=== AMI WITHIN bird (voice held constant) ===")
    print(f"{'representation':<16} | {hdr}")
    print("-" * (18 + 20 * len(args.ks)))
    for name, r in res.items():
        cells = " | ".join(f"{r[str(k)]['kmeans']['ami_within_bird']:>8.4f} "
                           f"{r[str(k)]['ward']['ami_within_bird']:>8.4f}" for k in args.ks)
        print(f"{name:<16} | {cells}")

    # How much of HuBERT's partition is just duration? Compare the two partitions directly.
    k0 = args.ks[0]
    d_c = KMeans(k0, n_init=10, random_state=args.seed).fit_predict(D[:, None])
    h_c = KMeans(k0, n_init=10, random_state=args.seed).fit_predict(
        Xh / np.linalg.norm(Xh, axis=1, keepdims=True))
    s_c = KMeans(k0, n_init=10, random_state=args.seed).fit_predict(Sn)
    overlap = dict(hubert_vs_duration=float(AMI(h_c, d_c)),
                   hubert_vs_spectrogram=float(AMI(h_c, s_c)),
                   spectrogram_vs_duration=float(AMI(s_c, d_c)))
    print(f"\n=== partition overlap at k={k0} (AMI between clusterings, not vs labels) ===")
    for a, b in overlap.items():
        print(f"  {a:<26} {b:.4f}")

    Path(args.out).write_text(json.dumps(dict(
        config=vars(args), n=int(ok.sum()), missing=missing, classes=classes,
        duration_stats=dict(mean=float(D.mean()), median=float(np.median(D)),
                            per_class={cn: float(D[yh == ci].mean())
                                       for ci, cn in enumerate(classes) if (yh == ci).any()}),
        results=res, partition_overlap=overlap), indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""What did run11's SPECTROGRAM-instead-of-MFCC choice actually buy?

This settles a question that has been asked as an architecture question and is not one.
`--feature-type {spectrogram,hubert}` (train.py:119) selects the feature the k-means PSEUDO-LABELS
are built from. It never reaches the network: the model is `torchaudio.models.hubert_pretrain_base`
(lightning_modules.py:467), raw waveform in, and the only place `feature_type == "mfcc"` survives in
the codebase is `label = label[::2]` (dataset/hubert_dataset.py:527), a frame-rate halving of the
LABEL sequence because Kaldi MFCC runs at 10 ms and HuBERT frames at 20 ms.

So run11 and AVES have bit-identical architectures, and "spectrogram instead of MFCC" is a
TARGET-QUALITY difference. That difference is real and large, and it is measurable without training:

    run11 iter-1     soundsig Gaussian STFT, 1 ms resolution, 50 Hz spacing, log10 power,
                     25 ms windows flattened -> 4000-d, k-means k=100
    AVES  iter-1     39-d Kaldi MFCC (13 ceps + delta + deltadelta), k-means k=200
                     (fairseq stock, examples/hubert/simple_kmeans/dump_mfcc_feature.py)

Both are scored as CLUSTERINGS on identical frames, so the feature axis and the k axis separate:

    feature in {soundsig4000, mfcc39, logmel384}  x  k in {100, 200}

References, as in target_quality.py:
    voc/noise   does a cluster know it is inside a call at all
    call id     do clusters respect CALL BOUNDARIES (the phone-purity analogue used to validate
                speech HuBERT targets, and the finer question)

AMI, not NMI or purity: the latter two rise monotonically with k and would hand k=200 a free win.

Frame alignment is exact, not approximate, and is asserted at runtime:
    HuBERT frame i  -> samples [i*320, i*320+400), centre i*320+200
    soundsig window i (stride 20 ms, kernel 25 ms) -> ms [20i, 20i+25) -> the same samples
    Kaldi MFCC frame 2i (shift 10 ms, length 25 ms) -> ms [20i, 20i+25) -> the same samples
"""
from __future__ import annotations
import csv, gc, json, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HUBERT_SRC = Path(__file__).resolve().parents[2] / "pytorchAudio/examples/hubert"
sys.path.insert(0, str(HUBERT_SRC))

SR, HOP, RF = 16000, 320, 400
STRIDE_MS, KERNEL_MS = 20, 25
AUDIO = Path.home() / "zf_labelset/audio/111021-000.wav"
SEGMENTS = Path.home() / "Downloads/Segments Data.csv"
FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
TMP = Path("/Users/jonathanwang/.claude/jobs/63c218d9/tmp")
SEED = 0


def standardise(X, name):
    """Z-score, dropping columns with no variance.

    run11's 4000-d soundsig feature contains constant columns: the extractor takes
    log10(power + 1e-8), so any frequency bin that is identically zero across the corpus pins at
    -8.0 exactly. StandardScaler divides those by a zero sigma and emits inf, which then poisons
    k-means with nan centroids -- silently, because MiniBatchKMeans still returns labels. Dropped
    here and counted, because the count is itself worth reporting.
    """
    X = np.asarray(X, dtype=np.float64)
    mu, sd = X.mean(0), X.std(0)
    keep = sd > 1e-8
    Z = ((X[:, keep] - mu[keep]) / sd[keep]).astype(np.float32)
    assert np.isfinite(Z).all(), f"{name}: non-finite after standardisation"
    n_drop = int((~keep).sum())
    if n_drop:
        print(f"[{name}] dropped {n_drop}/{X.shape[1]} zero-variance columns", flush=True)
    return Z, n_drop


def call_index_reference(centers, span):
    """Per frame: index of the annotated call it falls inside, else -1. Same as target_quality.py."""
    rows = list(csv.DictReader(open(SEGMENTS)))
    iv = np.array([[float(r["StartIndex"]), float(r["StopIndex"])] for r in rows])
    iv = iv[~np.isnan(iv).any(1)]
    iv = iv[iv[:, 1] > iv[:, 0]] * SR / 44100
    iv = np.clip(iv, 0, span)
    iv = iv[np.argsort(iv[:, 0])]
    ref = np.full(len(centers), -1, dtype=int)
    for i, (a, b) in enumerate(iv):
        ref[(centers >= a) & (centers <= b)] = i
    return ref


def soundsig_4000d(n_frames):
    """run11's EXACT iteration-1 target feature, via the production extractor.

    Calls `utils.feature_utils.extract_feature_spectrogram` -- the same function that produced the
    features run11's k-means was fit on -- rather than a re-implementation, so this is the recipe and
    not an approximation of it. The extractor consumes a whole file, so the annotated span is written
    to a temp wav first to bound memory at 90k windows instead of the file's 241k.
    """
    cache = FEAT / "soundsig4000_30min.npy"
    if cache.exists():
        X = np.load(cache, mmap_mode="r")
        print(f"[skip] cached soundsig {X.shape}")
        return np.asarray(X[:n_frames])
    from utils.feature_utils import extract_feature_spectrogram

    need = (n_frames - 1) * HOP + RF
    tmpwav = TMP / "span_16k.wav"
    if not tmpwav.exists():
        rec, sr = sf.read(AUDIO, dtype="float32", frames=need + SR, start=0)
        assert sr == SR, sr
        sf.write(tmpwav, rec, SR, subtype="FLOAT")
        del rec; gc.collect()
    t0 = time.time()
    F = extract_feature_spectrogram(str(tmpwav), torch.device("cpu"), SR, KERNEL_MS, STRIDE_MS)
    F = F.numpy().astype(np.float32)
    print(f"[soundsig] {F.shape} in {time.time()-t0:.0f}s", flush=True)
    assert F.shape[0] >= n_frames, f"only {F.shape[0]} windows for {n_frames} frames"
    F = np.ascontiguousarray(F[:n_frames])
    np.save(cache, F)
    return F


def mfcc_39d(n_frames):
    """AVES's iteration-1 target feature: fairseq's stock 13 MFCC + delta + deltadelta.

    Mirrors fairseq/examples/hubert/simple_kmeans/dump_mfcc_feature.py, which is what AVES ran for
    iteration 1. Kaldi's 10 ms shift means MFCC frame 2i aligns to HuBERT frame i -- the same 2:1 the
    production collate encodes as `label[::2]`. Computed in chunks whose starts are multiples of the
    10 ms shift so concatenation is seamless, then verified against a contiguous reference window.
    """
    cache = FEAT / "mfcc39_30min.npy"
    if cache.exists():
        X = np.load(cache, mmap_mode="r")
        print(f"[skip] cached mfcc {X.shape}")
        return np.asarray(X[:n_frames])
    import torchaudio.compliance.kaldi as kaldi

    shift = SR // 100                                     # 160 samples = 10 ms
    need_mfcc = 2 * n_frames                              # MFCC frame 2i <-> HuBERT frame i
    need_samples = (need_mfcc - 1) * shift + RF
    rec, sr = sf.read(AUDIO, dtype="float32", frames=need_samples + SR, start=0)
    assert sr == SR, sr
    wav = torch.from_numpy(rec).unsqueeze(0)
    del rec; gc.collect()

    def ceps(x):                                          # (1, samples) -> (frames, 13)
        return kaldi.mfcc(x, num_ceps=13, num_mel_bins=40, frame_length=float(KERNEL_MS),
                          frame_shift=10.0, sample_frequency=float(SR), snip_edges=True)

    per = 60000                                           # frames per chunk (10 min of audio)
    out, start_f, t0 = [], 0, time.time()
    while start_f < need_mfcc:
        take = min(per, need_mfcc - start_f)
        lo = start_f * shift
        hi = min(wav.shape[1], lo + (take - 1) * shift + RF)
        block = ceps(wav[:, lo:hi])
        assert block.shape[0] >= take, f"chunk gave {block.shape[0]} < {take}"
        out.append(block[:take])
        start_f += take
        print(f"    mfcc {start_f}/{need_mfcc}  {time.time()-t0:.0f}s", flush=True)
    C = torch.cat(out, 0)

    # brute-force cross-check: the chunked path must equal one contiguous call on a window that
    # straddles a chunk boundary. Guards the alignment arithmetic above.
    b = per * shift
    ref = ceps(wav[:, b - 50 * shift: b + 50 * shift + RF])
    err = (ref[:90] - C[per - 50: per + 40]).abs().max().item()
    print(f"[mfcc] chunk-boundary max|diff| vs contiguous = {err:.3e}", flush=True)
    assert err < 1e-3, f"chunked MFCC disagrees with contiguous at the boundary: {err}"

    d1 = torchaudio.functional.compute_deltas(C.T.unsqueeze(0)).squeeze(0)
    d2 = torchaudio.functional.compute_deltas(d1.unsqueeze(0)).squeeze(0)
    F = torch.cat([C.T, d1, d2], 0).T.contiguous().numpy().astype(np.float32)   # (frames, 39)
    F = np.ascontiguousarray(F[::2][:n_frames])           # 10 ms -> 20 ms, the label[::2] step
    print(f"[mfcc] {F.shape}", flush=True)
    np.save(cache, F)
    return F


def main():
    import torchaudio  # noqa: F401  (compute_deltas is reached through the module in mfcc_39d)
    globals()["torchaudio"] = torchaudio
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import adjusted_mutual_info_score as ami

    FR = np.load(FEAT / "frames_30min.npz", allow_pickle=True)
    y, centers = FR["y"], FR["centers"]
    nF = len(y)
    assert np.array_equal(centers, np.arange(nF) * HOP + RF // 2), "frame centres are not the " \
        "canonical i*320+200; the soundsig/MFCC alignment below would be wrong"
    span = int(centers[-1]) + RF // 2
    print(f"[frames] {nF:,}  voc fraction {y.mean():.4f}  span {span/SR:.1f}s", flush=True)

    ref_call = call_index_reference(centers, span)
    n_calls = len(set(ref_call[ref_call >= 0]))
    print(f"[reference] call-id: {n_calls} calls, {(ref_call>=0).mean():.4f} of frames inside a call",
          flush=True)

    feats = {
        "soundsig4000": (soundsig_4000d(nF),
                         "run11's ACTUAL iteration-1 feature: soundsig Gaussian STFT, 50 Hz spacing, "
                         "25 ms window flattened"),
        "mfcc39":       (mfcc_39d(nF),
                         "AVES's iteration-1 feature: fairseq stock 13 Kaldi MFCC + delta + ddelta"),
        "logmel384":    (np.load(FEAT / "mel_frames_30min.npy"),
                         "384-d log-mel summary, the stand-in used in target_quality.py; kept so the "
                         "two experiments are comparable"),
    }
    for k, (X, _) in feats.items():
        assert X.shape[0] == nF, (k, X.shape)
        print(f"[feature] {k:14s} {X.shape}", flush=True)

    out = {"n_frames": int(nF), "voc_fraction": float(y.mean()), "n_calls": int(n_calls),
           "seed": SEED,
           "note": "feature x k grid on identical frames. AMI is chance-corrected; NMI/purity would "
                   "favour k=200 for free. 'architecture' is NOT a factor here: run11 and AVES are "
                   "both torchaudio hubert_pretrain_base, 94,370,944 params, bit-identical keys.",
           "grid": {}}

    print(f"\n{'feature':14s} {'k':>5s} {'AMI(voc/noise)':>15s} {'AMI(call id)':>14s} {'used':>6s}", flush=True)
    for fname, (X, note) in feats.items():
        Z, n_drop = standardise(X, fname)
        for k in (100, 200):
            km = MiniBatchKMeans(n_clusters=k, random_state=SEED, n_init=5, batch_size=4096,
                                 max_iter=300).fit(Z)
            lab = km.labels_
            a_voc, a_call = float(ami(y, lab)), float(ami(ref_call, lab))
            out["grid"][f"{fname}_k{k}"] = dict(feature=fname, k=k, dim=int(X.shape[1]),
                                                ami_voc=a_voc, ami_call=a_call,
                                                used_clusters=int(len(set(lab))),
                                                inertia=float(km.inertia_),
                                                dropped_constant_cols=n_drop, note=note)
            print(f"{fname:14s} {k:5d} {a_voc:15.4f} {a_call:14.4f} {len(set(lab)):6d}", flush=True)
            del km, lab; gc.collect()
        del Z; gc.collect()

    g = out["grid"]
    eff = {}
    for k in (100, 200):
        eff[f"spectrogram_minus_mfcc_at_k{k}_ami_call"] = \
            g[f"soundsig4000_k{k}"]["ami_call"] - g[f"mfcc39_k{k}"]["ami_call"]
        eff[f"spectrogram_minus_mfcc_at_k{k}_ami_voc"] = \
            g[f"soundsig4000_k{k}"]["ami_voc"] - g[f"mfcc39_k{k}"]["ami_voc"]
    eff["run11_actual_minus_aves_iter1_ami_call"] = \
        g["soundsig4000_k100"]["ami_call"] - g["mfcc39_k200"]["ami_call"]
    out["effects"] = eff
    print("\neffect of run11's spectrogram choice over AVES's iteration-1 MFCC, AMI(call id):")
    for k in (100, 200):
        print(f"  at k={k:<4d}: {eff[f'spectrogram_minus_mfcc_at_k{k}_ami_call']:+.4f}")
    print(f"  run11 as shipped (spec,k=100) vs AVES iter-1 as shipped (mfcc,k=200): "
          f"{eff['run11_actual_minus_aves_iter1_ami_call']:+.4f}")

    (ANA / "target_features.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ANA/'target_features.json'}")


if __name__ == "__main__":
    main()

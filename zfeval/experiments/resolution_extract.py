#!/usr/bin/env python
"""Extract features for the temporal-resolution study, in two regimes.

The question is what happens as the analysis segment shrinks. There are two different ways to
shrink it, and they are not the same experiment:

  WINDOWED   the encoder sees ONLY the segment. 49 frames of self-attention context at 1 s, 1 frame
             at 40 ms. This is what a short-clip classifier does.
  CONTINUOUS the encoder runs over the whole recording (20 s chunks) and frames are pooled into
             segments AFTERWARDS. Every segment keeps full context regardless of its length. This
             is what a detector sweeping a recording does.

Both use the same 20 s normalisation statistic, so the only variable between them is how much
audio the transformer attends over. If WINDOWED collapses at small segments while CONTINUOUS holds,
the model's value is in context, and the deployment recommendation follows directly.

Ground truth is the 2541 SoundSep intervals, so a segment of any length can be labelled exactly.
"""
from __future__ import annotations
import argparse, csv, gc, json, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zfeval import features as fx                                        # noqa: E402

SR, SRC_SR = 16000, 44100
HOP, RF = 320, 400
SPAN_SRC = 79434253
SPAN = int(round(SPAN_SRC * SR / SRC_SR))
AUDIO = Path.home() / "zf_labelset/audio/111021-000.wav"
WEIGHTS = ROOT.parent / "release/zf_hubert_run11/weights/zf_hubert_run11_encoder.pt"
SEGMENTS = Path.home() / "Downloads/Segments Data.csv"
LAYERS = [0, 6]


def load_intervals():
    """The 2541 annotated calls, at 16 kHz, sorted, merged, clipped to the annotated span."""
    rows = list(csv.DictReader(open(SEGMENTS)))
    iv = np.array([[float(r["StartIndex"]), float(r["StopIndex"])] for r in rows])
    iv = iv[~np.isnan(iv).any(1)]
    iv = iv[iv[:, 1] > iv[:, 0]]
    iv = iv * SR / SRC_SR
    iv = np.clip(iv, 0, SPAN)
    iv = iv[np.argsort(iv[:, 0])]
    out = [iv[0]]
    for a, b in iv[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)      # one annotated pair overlaps by 84 ms
        else:
            out.append(np.array([a, b]))
    return np.array(out)


def load_model(device):
    from torchaudio.models import hubert_base
    m = hubert_base()
    ref = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    m.load_state_dict(ref["state_dict"], strict=True)
    return m.eval().to(device)


# ------------------------------------------------------------------ continuous pass
def frame_pass(model, rec, device, offset=0, chunk_sec=20.0, layers=LAYERS):
    """Per-frame features over [offset, SPAN), on the grid frame i -> [offset+i*HOP, +RF).

    Chunks are read with a RF-HOP tail so the local frame grid lines up with the global one;
    without it each chunk yields one frame fewer and leaves silent all-zero rows.
    """
    span = SPAN
    nF = max(0, (span - offset - RF) // HOP + 1)
    F = {l: np.zeros((nF, 768), dtype=np.float16) for l in layers}
    C = int(chunk_sec * SR)
    filled, t0 = 0, time.time()
    for c in range((span - offset + C - 1) // C):
        lo = offset + c * C
        x = rec[lo:min(span, lo + C + RF - HOP)]
        if len(x) < RF:
            break
        mu, var = float(x.mean()), float(x.var())
        xin = ((x - mu) / np.sqrt(var + 1e-5)).astype(np.float32)
        with torch.no_grad():
            feats, _ = model.extract_features(torch.from_numpy(xin).unsqueeze(0).to(device), None)
        i0 = (lo - offset) // HOP
        take = min(feats[0].shape[1], C // HOP, nF - i0)
        for l in layers:
            F[l][i0:i0 + take] = feats[l][0, :take].cpu().numpy().astype(np.float16)
        filled += take
        if c % 20 == 0:
            print(f"    chunk {c}  {filled}/{nF} frames  {time.time()-t0:.0f}s", flush=True)
    if filled < nF - 2:
        raise RuntimeError(f"only filled {filled}/{nF} frames -- chunk grid is misaligned")
    return F, nF


# ------------------------------------------------------------------ windowed pass
def windowed_pass(model, rec, starts, win, device, batch=32, ctx_sec=20.0, layers=LAYERS):
    """Mean-pooled features where the encoder sees ONLY each window. Batched: the windows are all
    the same length, so they stack, which is what makes the small-window arms affordable."""
    ctx = int(ctx_sec * SR)
    X = {l: np.zeros((len(starts), 768), dtype=np.float32) for l in layers}
    t0 = time.time()
    for b0 in range(0, len(starts), batch):
        sel = starts[b0:b0 + batch]
        xs = np.stack([fx.normalize_segment(rec, int(s), win, "context", ctx) for s in sel])
        with torch.no_grad():
            feats, _ = model.extract_features(torch.from_numpy(xs).float().to(device), None)
        for l in layers:
            X[l][b0:b0 + len(sel)] = feats[l].mean(1).cpu().numpy()
        if b0 % (batch * 40) == 0:
            print(f"    {b0}/{len(starts)}  {time.time()-t0:.0f}s", flush=True)
    return X


def label_windows(starts, win, iv):
    """Three label definitions, because the choice is not innocent at short windows.

    center   the window's midpoint falls inside a call. Length-neutral: a 40 ms window and a 1 s
             window are asked the same question about their centre.
    overlap  any overlap at all. At 1 s this marks a window positive for an 80 ms call that
             happens to clip its edge -- 92% of that window is background.
    half     at least half the window is inside a call. Almost nothing qualifies at 1 s.
    """
    mid = starts + win // 2
    j = np.searchsorted(iv[:, 0], mid, "right") - 1
    center = (j >= 0) & (iv[np.clip(j, 0, len(iv) - 1), 1] > mid)

    ends = starts + win
    lo = np.searchsorted(iv[:, 1], starts, "right")
    overlap = np.zeros(len(starts), bool)
    cov = np.zeros(len(starts))
    for k, (s, e) in enumerate(zip(starts, ends)):
        i = lo[k]
        tot = 0.0
        while i < len(iv) and iv[i, 0] < e:
            tot += max(0.0, min(e, iv[i, 1]) - max(s, iv[i, 0]))
            i += 1
        cov[k] = tot / win
        overlap[k] = tot > 0
    return dict(center=center.astype(int), overlap=overlap.astype(int),
                half=(cov >= 0.5).astype(int), coverage=cov)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--windows-ms", type=float, nargs="+",
                    default=[1000, 500, 250, 125, 80, 40])
    ap.add_argument("--max-windows", type=int, default=6000,
                    help="cap per window size; short windows would otherwise reach 45k segments")
    ap.add_argument("--out", default=str(Path.home() / "zf_labelset/zf_detection_dataset_v1/features"))
    ap.add_argument("--skip-frames", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    iv = load_intervals()
    # Only the annotated span is ever scored, and a 20 s normalisation context needs a little
    # past it. Reading all 80.55 min costs 309 MB against an 8 GB machine and sent the first
    # attempt into swap, where it ran 34x slower.
    need = min(SPAN + 20 * SR, sf.info(AUDIO).frames)
    rec, sr = sf.read(AUDIO, dtype="float32", frames=need)
    assert sr == SR
    print(f"loaded {len(rec)/SR/60:.2f} min of audio ({rec.nbytes/1e6:.0f} MB)", flush=True)
    dev = torch.device(a.device)
    model = load_model(dev)
    print(f"{len(iv)} merged intervals, span {SPAN/SR/60:.3f} min, "
          f"voiced {np.diff(iv,axis=1).sum()/SPAN:.4f}", flush=True)

    if not a.skip_frames:
        print("\n[continuous] frame pass over the annotated span", flush=True)
        F, nF = frame_pass(model, rec, dev)
        centers = np.arange(nF) * HOP + RF // 2
        j = np.searchsorted(iv[:, 0], centers, "right") - 1
        yf = ((j >= 0) & (iv[np.clip(j, 0, len(iv) - 1), 1] > centers)).astype(int)
        en = np.array([10*np.log10(np.mean(rec[i*HOP:i*HOP+RF]**2) + 1e-10) for i in range(nF)],
                      dtype=np.float32)
        np.savez_compressed(out / "frames_30min.npz", y=yf, centers=centers, energy=en,
                            layers=np.array(LAYERS), intervals=iv,
                            **{f"F{l}": F[l] for l in LAYERS})
        print(f"  {nF} frames, voiced-by-centre {yf.mean():.4f} -> {out/'frames_30min.npz'}",
              flush=True)
        del F
        gc.collect()

    for w_ms in a.windows_ms:
        win = int(round(w_ms / 1000 * SR))
        n_all = (SPAN - win) // win + 1
        allst = np.arange(n_all) * win
        if n_all > a.max_windows:
            rng = np.random.default_rng(0)
            st = np.sort(rng.choice(allst, a.max_windows, replace=False))
        else:
            st = allst
        lab = label_windows(st, win, iv)
        print(f"\n[windowed] {w_ms:g} ms: {len(st)} of {n_all} windows, "
              f"centre-positive {lab['center'].mean():.4f}", flush=True)
        X = windowed_pass(model, rec, st, win, dev)
        f = out / f"windowed_w{int(w_ms)}.npz"
        np.savez_compressed(f, starts=st, win=win, n_all=int(n_all), intervals=iv,
                            layers=np.array(LAYERS), **lab,
                            **{f"X{l}": X[l].astype(np.float16) for l in LAYERS})
        print(f"  -> {f.name}", flush=True)
        del X
        gc.collect()

    (out / "windowed_sizes.json").write_text(json.dumps([int(w) for w in a.windows_ms]))
    print("\nwrote", out / "windowed_sizes.json")


if __name__ == "__main__":
    main()

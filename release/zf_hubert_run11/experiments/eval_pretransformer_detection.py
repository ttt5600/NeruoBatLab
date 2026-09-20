#!/usr/bin/env python
"""Does the PRE-transformer representation beat transformer layer 0 at vocalization detection?

Motivation: detection AUC peaks at layer 0 (0.921) and declines monotonically with depth
(0.883 at layer 11). torchaudio's `extract_features` only returns the 12 transformer block
outputs, so the CNN feature-extractor output -- the representation *before* block 1 -- has
never been evaluated. Extrapolating the depth trend backwards predicts it should win.

This adds that 13th representation and scores all 13 under one protocol.

Usage
-----
  python eval_pretransformer_detection.py --weights WEIGHTS.pt --manifest windows.csv
  python eval_pretransformer_detection.py --weights WEIGHTS.pt --selftest    # verify plumbing

Manifest columns: path,start,end,label,recording
  start/end in seconds; label 1=vocalization 0=background; recording = group for CV.
IMPORTANT: point this at the SAME window list the 0.921 number came from, or the comparison
against the published layer sweep is not apples-to-apples.
"""
import argparse, csv, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch, torchaudio
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

SR = 16000


def load_encoder(path, device):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    m = torchaudio.models.wav2vec2_model(aux_num_out=None, **obj["encoder_config"])
    m.load_state_dict(obj["state_dict"], strict=True)
    return m.eval().to(device)


@torch.no_grad()
def represent(model, wav, device):
    """(13, 768) mean-pooled: index 0 = pre-transformer, 1..12 = transformer blocks 0..11."""
    wav = wav.to(device)
    feat, _ = model.feature_extractor(wav, None)
    pre = model.encoder.feature_projection(feat).squeeze(0).mean(0)     # (768,)
    feats, _ = model.extract_features(wav, None)
    blocks = torch.stack([f.squeeze(0).mean(0) for f in feats])         # (12, 768)
    return torch.cat([pre.unsqueeze(0), blocks]).cpu().numpy()


def read_window(path, start, end):
    """Windowed read -- never loads a whole recording (they can be hours long)."""
    off, num = int(start * SR), max(1, int((end - start) * SR))
    wav, sr = torchaudio.load(str(path), frame_offset=off, num_frames=num)
    if sr != SR:
        wav = torchaudio.transforms.Resample(sr, SR)(wav)
    return wav.mean(0, keepdim=True)


def build_selftest(tmp):
    """Synthesize a manifest so the plumbing can be verified without the real corpus."""
    ex = sorted((Path(__file__).resolve().parent.parent / "examples").glob("*.wav"))
    if not ex:
        sys.exit("selftest needs the bundled examples/ directory")
    rows, rng = [], np.random.default_rng(0)
    bg = tmp / "selftest_background.wav"
    torchaudio.save(str(bg), torch.from_numpy(
        (rng.normal(0, 0.01, SR * 40)).astype(np.float32)).unsqueeze(0), SR)
    for i, p in enumerate(ex):
        info = torchaudio.info(str(p))
        dur = info.num_frames / info.sample_rate
        rows.append(dict(path=str(p), start=0, end=min(dur, 1.0), label=1, recording=f"r{i%4}"))
        rows.append(dict(path=str(bg), start=i * 2.0, end=i * 2.0 + 1.0, label=0,
                         recording=f"r{i%4}"))
    mf = tmp / "selftest_manifest.csv"
    with open(mf, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "start", "end", "label", "recording"])
        w.writeheader(); w.writerows(rows)
    print(f"[selftest] wrote {len(rows)} synthetic windows -> {mf}")
    return mf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default="pretransformer_detection.npz")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    tmp = Path(a.out).resolve().parent
    manifest = Path(build_selftest(tmp)) if a.selftest else Path(a.manifest)
    if not a.selftest and not manifest:
        sys.exit("need --manifest or --selftest")

    rows = list(csv.DictReader(open(manifest)))
    print(f"{len(rows)} windows | device {a.device}")
    model = load_encoder(a.weights, a.device)

    # group windows by file so each recording is opened once
    by_file = defaultdict(list)
    for i, r in enumerate(rows):
        by_file[r["path"]].append(i)

    X = np.zeros((len(rows), 13, 768), np.float32)
    energy = np.zeros(len(rows), np.float32)
    y = np.array([int(r["label"]) for r in rows])
    grp = np.array([r["recording"] for r in rows])
    t0, done, bad = time.time(), 0, []
    for path, idxs in by_file.items():
        for i in idxs:
            try:
                w = read_window(path, float(rows[i]["start"]), float(rows[i]["end"]))
                if w.shape[1] < int(0.05 * SR):
                    bad.append(i); continue
                X[i] = represent(model, w, a.device)
                energy[i] = float(torch.log10(w.pow(2).mean().sqrt() + 1e-8))
            except Exception as ex:                       # noqa: BLE001
                bad.append(i); print(f"  SKIP {path} @{rows[i]['start']}: {ex}")
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(rows)}  ({time.time()-t0:.0f}s)")
    ok = np.setdiff1d(np.arange(len(rows)), bad)
    if len(bad):
        print(f"dropped {len(bad)} unreadable/too-short windows")
    X, y, grp, energy = X[ok], y[ok], grp[ok], energy[ok]
    print(f"{len(y)} usable windows | {y.sum()} voc / {(1-y).sum()} background "
          f"| {len(np.unique(grp))} recordings\n")

    n_splits = min(a.folds, len(np.unique(grp)))
    cv = GroupKFold(n_splits=n_splits)
    names = ["pre-transformer"] + [f"layer {i}" for i in range(12)]

    p = cross_val_predict(LogisticRegression(max_iter=1000), energy.reshape(-1, 1), y,
                          cv=cv, groups=grp, method="predict_proba")[:, 1]
    print(f"{'representation':16s} {'AUC':>7s} {'acc':>7s} {'F1':>7s}")
    print(f"{'energy baseline':16s} {roc_auc_score(y,p):7.3f} "
          f"{accuracy_score(y,p>.5):7.3f} {f1_score(y,p>.5):7.3f}")

    res, probs = [], np.zeros((len(y), len(names)), np.float32)
    for j, nm in enumerate(names):
        Z = StandardScaler().fit_transform(X[:, j, :])
        p = cross_val_predict(LogisticRegression(max_iter=1000), Z, y, cv=cv, groups=grp,
                              method="predict_proba")[:, 1]
        probs[:, j] = p
        auc = roc_auc_score(y, p)
        res.append((nm, auc, accuracy_score(y, p > .5), f1_score(y, p > .5)))
        print(f"{nm:16s} {auc:7.3f} {res[-1][2]:7.3f} {res[-1][3]:7.3f}")

    # NOTE: acc/F1 above are at threshold 0.5, which nobody chose for a reason. Save the
    # out-of-fold probabilities so the operating point can be tuned afterwards without
    # re-running the whole probe -- the earlier artifact omitted these, which is why the
    # published recall (0.734) was stuck at an arbitrary cutoff.
    aucs = np.array([r[1] for r in res])
    print(f"\npre-transformer - layer 0 : {aucs[0]-aucs[1]:+.3f}")
    print(f"layer 0 - layer 11        : {aucs[1]-aucs[12]:+.3f}")
    print("\nHYPOTHESIS: pre-transformer > layer 0. Confirmed only if the first line is",
          "clearly positive\nrelative to fold-to-fold noise -- rerun with different",
          "--folds before believing a small gap.")
    keep = np.array([[rows[i][k] for k in ("path", "start", "end", "recording")] for i in ok],
                    dtype=object)
    np.savez(a.out, aucs=aucs, names=np.array(names), y=y, groups=grp,
             probs=probs, energy=energy, windows=keep)
    print(f"\nwrote {a.out}  (now includes out-of-fold probs + window index)")


if __name__ == "__main__":
    main()

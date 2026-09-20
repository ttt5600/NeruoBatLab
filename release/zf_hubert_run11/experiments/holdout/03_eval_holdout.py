#!/usr/bin/env python
"""The comparison the whole holdout exists to make.

Same probe protocol both times: train an 8-way logistic regression on the 25 other birds,
test on LblRed0613's 241 clips. The only difference is whether the ENCODER heard that bird
during pretraining.

    run11    heard LblRed0613 in pretraining   -> baseline, measured 0.867 (layer 8)
    holdout  never heard it                    -> this script

A drop measures how much of the published number came from pretraining exposure.

READ THIS BEFORE INTERPRETING A DROP. The holdout corpus is 100 recordings, not 120, so a
drop confounds two things: never hearing the bird, and 16.7% less pretraining audio. If the
drop is large, run the control -- retrain dropping 20 RANDOM recordings that do NOT contain
LblRed0613, and compare holdout against that instead of against run11. Only the
holdout-vs-control gap isolates the exposure effect.

Usage:
    python 03_eval_holdout.py --ckpt <holdout_ckpt> --clip-dir <adultvoc_16k> \
        --labels calltype_labels.csv --out holdout_result.json
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio

HOLD = "lblred0613"


def load_encoder_from_ckpt(path, device, num_classes=100):
    """Lightning checkpoint -> bare wav2vec2 encoder (same path eval_layer_sweep.py uses)."""
    import sys
    sys.path.insert(0, "/global/home/users/jonathanswang/pytorchAudio/examples/hubert")
    from lightning_modules import HuBERTPreTrainModule
    module = HuBERTPreTrainModule(
        model_name="hubert_pretrain_base", feature_grad_mult=0.1, num_classes=num_classes,
        dataset="x", dataset_path="x", feature_type="spectrogram",
        seconds_per_batch=87.5, learning_rate=1e-4, betas=(0.9, 0.98),
        eps=1e-6, weight_decay=0.01, clip_norm=1.0,
        warmup_updates=0, max_updates=1, extractor_mode="group_norm",
    )
    ckpt = torch.load(path, map_location="cpu")
    module.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    return module.model.wav2vec2.to(device).eval()


@torch.no_grad()
def embed(model, path, device):
    wav, sr = torchaudio.load(str(path))
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    wav = wav.mean(0, keepdim=True).to(device)
    feats, _ = model.extract_features(wav, None)
    return torch.stack([f.squeeze(0).mean(0) for f in feats]).cpu().numpy()   # (12, 768)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--clip-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="holdout_result.json")
    ap.add_argument("--num-classes", type=int, default=100)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_encoder_from_ckpt(a.ckpt, device, a.num_classes)

    lab = {r["fname"]: r["label"] for r in csv.DictReader(open(a.labels))}
    classes = np.array(sorted(set(lab.values())))
    cix = {c: i for i, c in enumerate(classes)}

    clip_dir = Path(a.clip_dir)
    X, y, birds = [], [], []
    for i, (fn, lb) in enumerate(sorted(lab.items())):
        p = clip_dir / fn
        if not p.exists():
            continue
        b = fn.split("_")[0].lower()
        if b.startswith("unknown"):          # catch-alls, not individuals
            continue
        X.append(embed(model, p, device)); y.append(cix[lb]); birds.append(b)
        if len(X) % 250 == 0:
            print(f"  {len(X)} clips", flush=True)
    X = np.stack(X); y = np.array(y); birds = np.array(birds)
    print(f"\n{len(y)} clips, {len(np.unique(birds))} birds")

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    te, tr = birds == HOLD, birds != HOLD
    assert te.sum() > 0, f"{HOLD} not found in clip dir"
    cnt = np.bincount(y[te], minlength=len(classes))
    print(f"test = {HOLD}: {te.sum()} clips, majority {cnt.max()/cnt.sum():.3f}")
    print(f"train = {len(np.unique(birds[tr]))} other birds, {tr.sum()} clips\n")

    res = {}
    for L in range(X.shape[1]):
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr][:, L, :], y[tr])
        pred = clf.predict(X[te][:, L, :])
        res[L] = dict(acc=float(accuracy_score(y[te], pred)),
                      macro_f1=float(f1_score(y[te], pred, average="macro")))
        print(f"  layer {L:2d}  acc {res[L]['acc']:.3f}  macroF1 {res[L]['macro_f1']:.3f}")

    best = max(res, key=lambda L: res[L]["acc"])
    RUN11 = {3: 0.846, 8: 0.867}
    print(f"\nHOLDOUT best: layer {best}, acc {res[best]['acc']:.3f}")
    print(f"run11 baseline (heard this bird): layer 8 = {RUN11[8]:.3f}, layer 3 = {RUN11[3]:.3f}")
    print(f"delta at layer 8: {res[8]['acc'] - RUN11[8]:+.3f}")
    print(f"delta at layer 3: {res[3]['acc'] - RUN11[3]:+.3f}")
    print("\nA delta near 0 means pretraining exposure was NOT inflating the published number.")
    print("A large negative delta needs the random-20-dropped control before it can be read")
    print("as an exposure effect rather than a data-quantity effect.")

    json.dump({"per_layer": res, "best_layer": int(best),
               "run11_baseline": RUN11, "n_test": int(te.sum()),
               "test_majority": float(cnt.max()/cnt.sum())}, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

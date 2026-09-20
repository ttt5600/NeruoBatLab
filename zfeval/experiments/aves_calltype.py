#!/usr/bin/env python
"""The last place colony-specific pretraining could still win: call-type classification.

Detection turned out not to need it -- AVES, pretrained on generic animal sound, ties run11
in-distribution and beats it on the BirdPark holdout (finding 030). But "is a call here" is not a
zebra-finch-specific judgement, so that result does not settle much. "Which of the 8 call types is
this" is species-specific in a way detection is not, and it is the task where a colony-trained
encoder should have an advantage if it has one anywhere.

Setup is run11's own published call-type evaluation, unchanged, so the numbers are comparable to the
released 0.8109: 2814 curated clips (the Unknown* placeholder birds dropped, leaving 26 real birds),
8 classes, majority 0.207, leave-birds-out 5-fold. run11's embeddings are read from the released
layersweep file rather than recomputed; AVES is put through the identical path -- raw audio (no
waveform normalisation, which is what zf_hubert.embed_file does), mono channel mean, 16 kHz,
mean-pooled over time, all 12 layers.

A reproduction guard runs first: if this script cannot recover the released per-layer accuracies from
the released embeddings, the probe or the split does not match the published one and no comparison
built on it means anything.
"""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "release/zf_hubert_run11"))
from aves_baseline import load_aves                                          # noqa: E402

REL = ROOT.parent / "release/zf_hubert_run11"
SWEEP = REL / "data/run11_layersweep.npz"
METRICS = REL / "data/corrected_metrics.json"
AUDIO_DIRS = [ROOT.parent / "datasets/11905533/AdultVocalizations",
              Path.home() / "Desktop/excess/savioFolders/Zebra Finch Vocal Repertoires/AdultVocalizations"]
FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
SR = 16000
NFOLD, SEED = 5, 0


def audio_dir():
    for d in AUDIO_DIRS:
        if d.exists():
            return d
    raise RuntimeError(f"no curated-clip directory found; looked in {AUDIO_DIRS}")


_resamplers = {}


def load_audio(path):
    """Exactly zf_hubert.load_audio: mono channel mean, resampled, NO waveform normalisation.

    The resampler is cached exactly as zf_hubert does. Every clip here is 44.1 kHz, and building a
    fresh Resample per clip rebuilds its filter kernel 2814 times -- which is most of the runtime.
    """
    import torchaudio
    wav, sr = torchaudio.load(str(path))
    if sr != SR:
        if sr not in _resamplers:
            _resamplers[sr] = torchaudio.transforms.Resample(sr, SR)
        wav = _resamplers[sr](wav)
    return wav.mean(0, keepdim=True)


@torch.no_grad()
def embed_all(model, paths, device, n_layers=12):
    """(N, n_layers, 768) mean-pooled over time, clips run one at a time (uneven lengths)."""
    out = np.zeros((len(paths), n_layers, 768), dtype=np.float32)
    t0 = time.time()
    for i, p in enumerate(paths):
        w = load_audio(p)
        feats, _ = model.extract_features(w.to(device), None)
        for l in range(n_layers):
            out[i, l] = feats[l].squeeze(0).mean(0).cpu().numpy()
        if (i + 1) % 400 == 0:
            print(f"    {i+1}/{len(paths)}  {time.time()-t0:.0f}s", flush=True)
    return out


def cv_acc(X, y, groups, return_proba=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedGroupKFold
    cv = StratifiedGroupKFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
    P = np.zeros((len(y), len(np.unique(y))), dtype=float)
    folds = []
    for tr, te in cv.split(X, y, groups):
        est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000))
        est.fit(X[tr], y[tr])
        P[te] = est.predict_proba(X[te])
        folds.append(float((P[te].argmax(1) == y[te]).mean()))
    acc = float((P.argmax(1) == y).mean())
    return (acc, folds, P) if return_proba else (acc, folds)


def bird_bootstrap(y, Pa, Pb, groups, n=2000, seed=0):
    """Cluster bootstrap over BIRDS -- the unit that is actually independent here."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    d = []
    for _ in range(n):
        sel = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        d.append((Pa[sel].argmax(1) == y[sel]).mean() - (Pb[sel].argmax(1) == y[sel]).mean())
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(delta=float(d.mean()), lo=float(lo), hi=float(hi),
                verdict=("a_better" if lo > 0 else "b_better" if hi < 0 else "not_distinguishable"))


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[device] {device}", flush=True)

    d = np.load(SWEEP, allow_pickle=True)
    emb, y, birds, names = d["emb"], d["y"], d["birds"], d["names"]
    classes = [str(c) for c in d["classes"]]
    # the released metrics drop the Unknown* placeholder "birds"; this reproduces 2814 / 26 exactly
    keep = np.array([not b.lower().startswith("unknown") for b in birds])
    emb, y, birds, names = emb[keep], y[keep], birds[keep], names[keep]
    groups = np.array([b.lower() for b in birds])          # HPiHPi4748 / HpiHpi4748 are one bird
    ref = json.loads(METRICS.read_text())
    print(f"[data] {len(y)} clips, {len(np.unique(groups))} birds, {len(classes)} classes, "
          f"majority {np.bincount(y).max()/len(y):.4f}")
    assert len(y) == ref["n_clips"] and len(np.unique(groups)) == ref["n_birds"], \
        f"filtering does not reproduce the released cohort: {len(y)} vs {ref['n_clips']}"

    out = {"n_clips": int(len(y)), "n_birds": int(len(np.unique(groups))), "classes": classes,
           "majority": float(np.bincount(y).max() / len(y)),
           "split": f"leave-birds-out StratifiedGroupKFold({NFOLD}), seed {SEED}",
           "run11": {}, "aves": {}, "reproduction": {}}

    # ---------------- reproduction guard
    print("\n=== reproduction check against the released per-layer accuracies ===")
    run11_acc, run11_P = [], {}
    for l in range(12):
        a, f, P = cv_acc(emb[:, l], y, groups, return_proba=True)
        run11_acc.append(a); run11_P[l] = P
        pub = ref["probe"]["per_layer_acc"][l]
        print(f"  L{l:<2d}  ours {a:.4f}   published {pub:.4f}   diff {a-pub:+.4f}", flush=True)
    diffs = np.array(run11_acc) - np.array(ref["probe"]["per_layer_acc"])
    out["reproduction"] = dict(max_abs_diff=float(np.abs(diffs).max()),
                               mean_diff=float(diffs.mean()),
                               published=ref["probe"]["per_layer_acc"], ours=run11_acc)
    print(f"  max |diff| {np.abs(diffs).max():.4f}")
    if np.abs(diffs).max() > 0.02:
        print("  WARNING: the probe/split does not match the published one within 0.02; "
              "the comparison below is still internally fair (same pipeline both encoders) "
              "but is NOT comparable to the released 0.8109.")
    out["run11"]["per_layer_acc"] = run11_acc

    # ---------------- AVES on the same clips
    ad = audio_dir()
    paths = [ad / str(n) for n in names]
    missing = [p for p in paths[:50] if not p.exists()]
    if missing:
        raise RuntimeError(f"clips not found in {ad}, e.g. {missing[0]}")
    print(f"\n[audio] {ad}")
    cache = FEAT / "aves_calltype_emb.npy"
    if cache.exists():
        A = np.load(cache)
        print(f"[skip] loaded cached AVES embeddings {A.shape}")
    else:
        m = load_aves(device)
        print(f"=== embedding {len(paths)} clips with AVES (raw audio, mean-pooled) ===", flush=True)
        A = embed_all(m, paths, device)
        np.save(cache, A)
        del m; gc.collect()
    assert A.shape[0] == len(y)

    print("\n=== call-type accuracy, leave-birds-out ===")
    print(f"  {'layer':6s} {'run11':>8s} {'AVES':>8s} {'diff':>8s}")
    aves_acc, aves_P = [], {}
    for l in range(12):
        a, f, P = cv_acc(A[:, l], y, groups, return_proba=True)
        aves_acc.append(a); aves_P[l] = P
        print(f"  L{l:<5d} {run11_acc[l]:8.4f} {a:8.4f} {run11_acc[l]-a:+8.4f}", flush=True)
    out["aves"]["per_layer_acc"] = aves_acc

    br = int(np.argmax(run11_acc)); ba = int(np.argmax(aves_acc))
    out["run11"]["best"] = dict(layer=br, acc=run11_acc[br])
    out["aves"]["best"] = dict(layer=ba, acc=aves_acc[ba])
    print(f"\n  best run11: L{br} {run11_acc[br]:.4f}   (published best L{ref['probe']['best_layer']} "
          f"{ref['probe']['best_acc']:.4f})")
    print(f"  best AVES : L{ba} {aves_acc[ba]:.4f}")
    print(f"  run11 - AVES: {run11_acc[br]-aves_acc[ba]:+.4f}")

    # ---------------- bootstrap over birds, and the ensemble that worked for detection
    print("\n=== cluster bootstrap over the 26 birds ===")
    r = bird_bootstrap(y, run11_P[br], aves_P[ba], groups)
    out["bootstrap_run11_vs_aves"] = r
    print(f"  run11 L{br} - AVES L{ba}: {r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]  {r['verdict']}")

    Pm = (run11_P[br] + aves_P[ba]) / 2
    acc_m = float((Pm.argmax(1) == y).mean())
    out["ensemble_mean"] = dict(acc=acc_m)
    rm = bird_bootstrap(y, Pm, run11_P[br], groups)
    out["bootstrap_mean_vs_run11"] = rm
    print(f"  mean of the two probes: {acc_m:.4f}  "
          f"(vs run11 {rm['delta']:+.4f} [{rm['lo']:+.4f}, {rm['hi']:+.4f}] {rm['verdict']})")

    # concat, for the same reason it was tested on detection
    ac, _ = cv_acc(np.hstack([emb[:, br], A[:, ba]]), y, groups)
    out["concat"] = dict(acc=ac)
    print(f"  concat 1536-d:          {ac:.4f}")

    (ANA / "aves_calltype.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ANA/'aves_calltype.json'}")


if __name__ == "__main__":
    main()

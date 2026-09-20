#!/usr/bin/env python
"""The 11-class call-type task, both encoders, with the 2023 notebook's preprocessing bugs removed.

The released run11 evaluation uses 8 classes and adults only. The original AVES notebook
(datasets/11905533/AVESZF.ipynb, April 2023) used 11 classes over adults AND chicks, and reported
far lower accuracy. Before treating that as a fact about AVES, note what that notebook's pipeline did
to the audio:

  wav[wav < 0] = 0                                    # half-wave rectification of the waveform
  out = pad_sequence(data, batch_first=True)          # every clip zero-padded to 250,606 samples
  out = self.model.extract_features(x)[0].mean(dim=1) # mean over ~782 frames, ~99% of them padding

A median clip here is 0.142 s. Padded to 15.66 s it is 0.9% signal, and the clip embedding is the
mean over a frame axis that is 99.1% zeros. The head was also a single nn.Linear trained by SGD at
lr 0.01, batch size 1, for 5 epochs, on a RANDOM 80/20 split rather than leave-birds-out.

So that number measures the pipeline, not the encoder. This script runs the same 11-class task with
the audio handled correctly -- each clip encoded at its own length, mean-pooled over real frames only
-- and puts run11 through the identical path, so the two are comparable and both are comparable to
the released 8-class result.
"""
from __future__ import annotations
import gc, json, re, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aves_baseline import load_aves                                          # noqa: E402
from aves_holdout import load_run11                                          # noqa: E402
from aves_calltype import load_audio, cv_acc, bird_bootstrap                 # noqa: E402

DATA = ROOT.parent / "datasets/11905533"
DIRS = [DATA / "AdultVocalizations", DATA / "ChickVocalizations"]
FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
# the 11 the notebook kept; it found a 12th ('WC', 3 files) and skipped it
KEEP11 = ["Ag", "Be", "DC", "Di", "LT", "Ne", "So", "Te", "Th", "Tu", "Wh"]
KEEP8 = ["Ag", "DC", "Ne", "So", "Te", "Th", "Tu", "Wh"]
CANON = {k.lower(): k for k in KEEP11}


def parse(name):
    """bird, date, 2-char call type -- the notebook's own rule, kept so the cohort matches."""
    m = re.match(r"^([A-Za-z]+\d+[A-Za-z]*)[_-](\d{6})[_-]?(.*)$", name)
    if not m:
        return None
    bird, date, rest = m.group(1), m.group(2), m.group(3)
    t = rest[:2]
    return bird, date, t


def collect():
    rows = []
    for d in DIRS:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.wav")):
            p = parse(f.name)
            if p is None:
                continue
            bird, date, t = p
            # canonicalise case only against the known labels: 'DC'/'Dc' and 'LT'/'Lt' are the same
            # type, but blind title-casing would rewrite them to spellings KEEP11 does not contain
            t = CANON.get(t.lower(), t)
            rows.append((f, bird, date, t, d.name))
    return rows


@torch.no_grad()
def embed_all(model, paths, device, n_layers=12, tag=""):
    out = np.zeros((len(paths), n_layers, 768), dtype=np.float32)
    t0 = time.time()
    for i, p in enumerate(paths):
        w = load_audio(p)
        if w.shape[-1] < 400:                       # shorter than one receptive field
            w = torch.nn.functional.pad(w, (0, 400 - w.shape[-1]))
        feats, _ = model.extract_features(w.to(device), None)
        for l in range(n_layers):
            out[i, l] = feats[l].squeeze(0).mean(0).cpu().numpy()
        if (i + 1) % 500 == 0:
            print(f"    {tag} {i+1}/{len(paths)}  {time.time()-t0:.0f}s", flush=True)
    return out


def evaluate(name, E, y, groups, classes, out):
    accs = []
    P = {}
    for l in range(E.shape[1]):
        a, _, p = cv_acc(E[:, l], y, groups, return_proba=True)
        accs.append(a); P[l] = p
    b = int(np.argmax(accs))
    out[name] = dict(per_layer_acc=accs, best=dict(layer=b, acc=accs[b]))
    return accs, P, b


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rows = collect()
    types = {}
    for _, _, _, t, _ in rows:
        types[t] = types.get(t, 0) + 1
    print(f"[device] {device}")
    print(f"[parsed] {len(rows)} clips, call types found: "
          f"{sorted(types.items(), key=lambda kv: -kv[1])}", flush=True)

    sel = [r for r in rows if r[3] in KEEP11]
    paths = [r[0] for r in sel]
    birds = np.array([r[1].lower() for r in sel])
    tt = np.array([r[3] for r in sel])
    src = np.array([r[4] for r in sel])
    classes = sorted(set(tt))
    y = np.array([classes.index(t) for t in tt])
    print(f"[cohort] {len(y)} clips, {len(classes)} classes {classes}, "
          f"{len(set(birds))} birds, majority {np.bincount(y).max()/len(y):.4f}")
    print(f"[source] adults {int((src=='AdultVocalizations').sum())}, "
          f"chicks {int((src=='ChickVocalizations').sum())}", flush=True)

    E = {}
    for nm, loader, cache in (("run11", load_run11, "ct11_run11_emb.npy"),
                              ("aves", load_aves, "ct11_aves_emb.npy")):
        cp = FEAT / cache
        if cp.exists():
            E[nm] = np.load(cp); print(f"[skip] cached {cache} {E[nm].shape}")
            continue
        m = loader(device)
        print(f"=== embedding {len(paths)} clips with {nm} ===", flush=True)
        E[nm] = embed_all(m, paths, device, tag=nm)
        np.save(cp, E[nm]); del m; gc.collect()

    out = {"n_clips": int(len(y)), "n_birds": int(len(set(birds))), "classes": classes,
           "majority": float(np.bincount(y).max() / len(y)),
           "n_adult": int((src == "AdultVocalizations").sum()),
           "n_chick": int((src == "ChickVocalizations").sum()),
           "class_counts": {c: int((tt == c).sum()) for c in classes},
           "split": "leave-birds-out StratifiedGroupKFold(5)"}

    print(f"\n=== 11-class, leave-birds-out ({len(y)} clips, majority {out['majority']:.4f}) ===")
    print(f"  {'layer':6s} {'run11':>8s} {'AVES':>8s} {'diff':>8s}")
    ra, rP, rb = evaluate("run11", E["run11"], y, birds, classes, out)
    aa, aP, ab = evaluate("aves", E["aves"], y, birds, classes, out)
    for l in range(12):
        print(f"  L{l:<5d} {ra[l]:8.4f} {aa[l]:8.4f} {ra[l]-aa[l]:+8.4f}")
    print(f"\n  best run11 L{rb} {ra[rb]:.4f}   best AVES L{ab} {aa[ab]:.4f}   "
          f"run11-AVES {ra[rb]-aa[ab]:+.4f}")
    r = bird_bootstrap(y, rP[rb], aP[ab], birds)
    out["bootstrap_run11_vs_aves"] = r
    print(f"  bootstrap over {len(set(birds))} birds: {r['delta']:+.4f} "
          f"[{r['lo']:+.4f}, {r['hi']:+.4f}]  {r['verdict']}", flush=True)

    # ---- adults-only 8-class, so this run can be tied back to the released number
    m8 = np.array([t in KEEP8 for t in tt]) & (src == "AdultVocalizations")
    c8 = sorted(set(tt[m8])); y8 = np.array([c8.index(t) for t in tt[m8]])
    o8 = {}
    print(f"\n=== adults-only 8-class for continuity ({m8.sum()} clips, "
          f"majority {np.bincount(y8).max()/len(y8):.4f}) ===")
    ra8, rP8, rb8 = evaluate("run11", E["run11"][m8], y8, birds[m8], c8, o8)
    aa8, aP8, ab8 = evaluate("aves", E["aves"][m8], y8, birds[m8], c8, o8)
    print(f"  best run11 L{rb8} {ra8[rb8]:.4f}   best AVES L{ab8} {aa8[ab8]:.4f}   "
          f"run11-AVES {ra8[rb8]-aa8[ab8]:+.4f}")
    r8 = bird_bootstrap(y8, rP8[rb8], aP8[ab8], birds[m8])
    o8["bootstrap_run11_vs_aves"] = r8
    o8["n_clips"] = int(m8.sum()); o8["majority"] = float(np.bincount(y8).max() / len(y8))
    print(f"  bootstrap: {r8['delta']:+.4f} [{r8['lo']:+.4f}, {r8['hi']:+.4f}]  {r8['verdict']}")
    out["adults_8class"] = o8

    (ANA / "calltype11.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ANA/'calltype11.json'}")


if __name__ == "__main__":
    main()

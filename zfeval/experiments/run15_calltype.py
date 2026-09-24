#!/usr/bin/env python
"""Put run15 into the call-type table, next to run11 and every AVES variant.

run15 is run11's EXACT recipe on the combined ZF+FSD50K corpus (224.42 h vs 116.0 h) with a
k=200 vocabulary instead of k=100. Nothing else moved -- same lr, same 93750 updates, same
80.x seconds of audio per step, same HUBERT_NORMALIZE_INPUT=0.

It exists to de-confound. Until now "AVES wins" could have been about the CORPUS or about the
LABEL RECIPE, because those two always moved together. run15 moves exactly those two.

Reuses aves_variants_calltype's own encode + sweep so the number lands on the same footing as
the six AVES rows already in aves_variants_calltype.json -- same cohort, same leave-birds-out
folds, same mean pooling, same per-clip-own-length encoding.

Note on the bar: aves-base-bio (the 'aves' in every headline so far) is tied for the WEAKEST of
the six AVES checkpoints at 0.8453. The honest same-parameter-count bar is birdaves-biox-base
at 0.8520.
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from calltype11 import collect, KEEP11, KEEP8                              # noqa: E402
from aves_variants_calltype import embed_all, sweep, EXPECT, ANA           # noqa: E402
from detection_variants import spec, load_model                            # noqa: E402

FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
# Any checkpoint exported by export_weights.py into external/dapt/ can be scored here:
#   python run15_calltype.py                          -> run15_combined (run15_calltype.json)
#   python run15_calltype.py --tag run16_compute4x    -> run16_compute4x_calltype.json
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--tag", default="run15_combined")
TAG = _ap.parse_args().tag
OUT = ANA / ("run15_calltype.json" if TAG == "run15_combined" else f"{TAG}_calltype.json")
CACHE = FEAT / f"ct11_{TAG}_emb.npy"


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rows = [r for r in collect() if r[3] in KEEP11]
    paths = [r[0] for r in rows]
    birds = np.array([r[1].lower() for r in rows])
    tt = np.array([r[3] for r in rows])
    src = np.array([r[4] for r in rows])
    classes = sorted(set(tt))
    y = np.array([classes.index(t) for t in tt])
    assert len(y) == EXPECT["n_clips"], f"cohort is {len(y)}, expected {EXPECT['n_clips']}"
    assert len(set(birds)) == EXPECT["n_birds"]
    print(f"[cohort] {len(y)} clips, {len(classes)} classes, {len(set(birds))} birds "
          f"| device {device}", flush=True)

    s = spec(TAG)
    print(f"[model] {TAG}: {s['n_layers']} layers, dim {s['dim']}, {s['weights']}", flush=True)

    if CACHE.exists():
        E = np.load(CACHE)
        print(f"[cache] {CACHE.name} {E.shape}", flush=True)
    else:
        model, _ = load_model(TAG, s, device)
        n_par = sum(p.numel() for p in model.parameters())
        print(f"[model] {n_par/1e6:.2f}M params", flush=True)
        t0 = time.time()
        E = embed_all(model, paths, device, s["n_layers"], s["dim"], tag=TAG)
        np.save(CACHE, E)
        print(f"[embed] {E.shape} in {time.time()-t0:.0f}s -> {CACHE.name}", flush=True)

    accs, bl, _ = sweep(E, y, birds, tag=f"{TAG}-11")
    m8 = np.array([t in KEEP8 for t in tt]) & (src == "AdultVocalizations")
    c8 = sorted(set(tt[m8])); y8 = np.array([c8.index(t) for t in tt[m8]])
    accs8, bl8, _ = sweep(E[m8], y8, birds[m8], tag=f"{TAG}-8")

    res = dict(tag=TAG, n_layers=int(s["n_layers"]), dim=int(s["dim"]),
               cohort11=dict(n_clips=int(len(y)), n_classes=len(classes), n_birds=int(len(set(birds)))),
               per_layer_acc11=[float(a) for a in accs], best_layer11=int(bl),
               acc11=float(accs[bl]),
               per_layer_acc8=[float(a) for a in accs8], best_layer8=int(bl8),
               acc8=float(accs8[bl8]),
               split="leave-birds-out StratifiedGroupKFold(5), seed 0",
               note="run11 recipe, combined ZF+FSD50K corpus 224.42 h, k=200 vocabulary")
    OUT.write_text(json.dumps(res, indent=2))

    print()
    print(f"{'checkpoint':<24}{'params':>9}{'dim':>6}{'nL':>4}{'bL11':>6}{'acc11':>9}{'bL8':>5}{'acc8':>9}")
    prev = json.load(open(ANA / "aves_variants_calltype.json"))["table"]
    for r in prev[1:]:
        print("  " + r if isinstance(r, str) else r)
    print(f"{TAG:<24}{'94.37M':>9}{s['dim']:>6}{s['n_layers']:>4}{bl:>6}{accs[bl]:>9.4f}"
          f"{bl8:>5}{accs8[bl8]:>9.4f}")
    print()
    print(f"[per-layer 11-class] " + " ".join(f"L{i}:{a:.4f}" for i, a in enumerate(accs)))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

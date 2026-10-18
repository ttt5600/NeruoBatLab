#!/usr/bin/env python
"""Paired call-type contrasts on the CPU embeddings, for seed replications (harness E8/E9 and later).

Same protocol as dapt_calltype_by_type.py: CPU caches (features/ct11_<tag>_cpuemb.npy from calltype_cpu_rescore.py;
the MPS caches are wrong for 2-8 frame clips, finding 076), each model at its CPU best 11-class layer from
analysis/calltype_cpu_rescore.json (pooled accuracy asserted to reproduce it), board probe cv_acc (leave-birds-out
StratifiedGroupKFold(5) seed 0), 95% bird bootstrap (2,000 resamples of the 48 birds, seed 0). Differences are OBSERVED
(A minus B on the full data), not the resample mean.

    python zfeval/experiments/seed_pairs_cpu.py A:B [A:B ...]   -> appends to analysis/seed_pairs_cpu.json under "A | B"
    python zfeval/experiments/seed_pairs_cpu.py --layer 3 A:B ...  -> both models at a FIXED layer (no selection on the
        board split), pooled accuracy asserted against per_layer_acc11_cpu[3]; written to seed_pairs_cpu_L3.json so the
        board (which reads seed_pairs_cpu.json) never mixes the two
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
from calltype11 import collect, KEEP11                                     # noqa: E402
from aves_calltype import cv_acc                                           # noqa: E402

FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
OUT = ANA / "seed_pairs_cpu.json"
FIXED = None                                   # --layer N: every model at layer N
CPU = json.loads((ANA / "calltype_cpu_rescore.json").read_text())["models"]


def main(pairs):
    rows = [r for r in collect() if r[3] in KEEP11]
    birds = np.array([r[1].lower() for r in rows])
    tt = np.array([r[3] for r in rows]); classes = sorted(set(tt))
    y = np.array([classes.index(t) for t in tt])
    rng = np.random.default_rng(0); ub = np.unique(birds); bidx = {b: np.where(birds == b)[0] for b in ub}
    BOOT = [np.concatenate([bidx[b] for b in rng.choice(ub, len(ub))]) for _ in range(2000)]
    hits = {}

    def hit(tag):
        if tag not in hits:
            L = int(CPU[tag]["best_layer_cpu"]) if FIXED is None else FIXED
            want = CPU[tag]["acc11_cpu"] if FIXED is None else CPU[tag]["per_layer_acc11_cpu"][FIXED]
            E = np.load(FEAT / f"ct11_{tag}_cpuemb.npy", mmap_mode="r")
            a, _, P = cv_acc(np.ascontiguousarray(E[:, L]), y, birds, return_proba=True)
            assert abs(a - want) < 1e-9, (tag, a, want)
            hits[tag] = ((P.argmax(1) == y).astype(float), L, a)
        return hits[tag]

    res = json.loads(OUT.read_text()) if OUT.exists() else dict(
        protocol=f"CPU embeddings; {'each model at its CPU best layer' if FIXED is None else f'every model at fixed layer {FIXED}'}; cv_acc leave-birds-out SGKF(5) seed 0; "
                 "bird bootstrap 2000 seed 0; observed differences", pairs={})
    for a, b in pairs:
        ha, la, aa = hit(a); hb, lb, ab = hit(b)
        bs = np.array([ha[s].mean() - hb[s].mean() for s in BOOT])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        r = dict(a=a, b=b, layer_a=la, layer_b=lb, acc_a=aa, acc_b=ab, diff=float(aa - ab),
                 lo=float(lo), hi=float(hi), resolved=bool(lo > 0 or hi < 0))
        res["pairs"][f"{a} | {b}"] = r
        print(f"{a} L{la} {aa:.4f} - {b} L{lb} {ab:.4f} = {aa - ab:+.4f} [{lo:+.4f}, {hi:+.4f}]"
              f"{' resolved' if r['resolved'] else ''}", flush=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--layer" in args:
        i = args.index("--layer"); FIXED = int(args[i + 1]); del args[i:i + 2]
        OUT = ANA / f"seed_pairs_cpu_L{FIXED}.json"
    main([tuple(p.split(":")) for p in args])

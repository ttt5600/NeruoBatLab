#!/usr/bin/env python
"""Per-call-type recall contrasts between any two models, on the CPU embeddings (finding 076).

Same protocol as dapt_calltype_by_type.py and seed_pairs_cpu.py: CPU caches features/ct11_<tag>_cpuemb.npy, the board
probe cv_acc (leave-birds-out StratifiedGroupKFold(5) seed 0, StandardScaler + LogisticRegression), 95% bird bootstrap
(2,000 resamples of the 48 birds, seed 0, the same resamples for every row). Differences are OBSERVED (A minus B on
the full data). Each pair is read twice: both models at their CPU best layer (pooled accuracy asserted against
calltype_cpu_rescore.json), and both at a fixed layer 3 (no per-model layer choice; CRITIC.md check 4). Per type,
lo/hi are the 95% interval and lo995/hi995 the 99.5% one (Bonferroni over the 11 types; resolved995 uses it).

    python zfeval/experiments/calltype_type_contrasts.py A:B [A:B ...]   -> analysis/calltype_type_contrasts.json
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
OUT = ANA / "calltype_type_contrasts.json"
CPU = json.loads((ANA / "calltype_cpu_rescore.json").read_text())["models"]


def main(pairs):
    rows = [r for r in collect() if r[3] in KEEP11]
    birds = np.array([r[1].lower() for r in rows])
    tt = np.array([r[3] for r in rows]); classes = sorted(set(tt))
    y = np.array([classes.index(t) for t in tt])
    rng = np.random.default_rng(0); ub = np.unique(birds); bidx = {b: np.where(birds == b)[0] for b in ub}
    BOOT = [np.concatenate([bidx[b] for b in rng.choice(ub, len(ub))]) for _ in range(2000)]
    cache = {}

    def pred(tag, layer):
        key = (tag, layer)
        if key not in cache:
            L = int(CPU[tag]["best_layer_cpu"]) if layer == "best" else layer
            want = CPU[tag]["acc11_cpu"] if layer == "best" else CPU[tag]["per_layer_acc11_cpu"][layer]
            E = np.load(FEAT / f"ct11_{tag}_cpuemb.npy", mmap_mode="r")
            a, _, P = cv_acc(np.ascontiguousarray(E[:, L]), y, birds, return_proba=True)
            assert abs(a - want) < 1e-9, (tag, layer, a, want)
            cache[key] = (P.argmax(1), L)
        return cache[key]

    def recall(p, idx):
        return np.array([(p[idx][y[idx] == k] == k).mean() if (y[idx] == k).any() else np.nan
                         for k in range(len(classes))])

    res = json.loads(OUT.read_text()) if OUT.exists() else dict(
        protocol="CPU embeddings; cv_acc leave-birds-out SGKF(5) seed 0; bird bootstrap 2000 seed 0; observed "
                 "differences A minus B; 'best' = each model's CPU best layer, 'L3' = both at layer 3",
        classes=classes, counts={c: int((tt == c).sum()) for c in classes}, pairs={})
    allidx = np.arange(len(y))
    for a, b in pairs:
        for layer in ("best", 3):
            pa, la = pred(a, layer); pb, lb = pred(b, layer)
            obs = recall(pa, allidx) - recall(pb, allidx)
            bs = np.array([recall(pa, s) - recall(pb, s) for s in BOOT])
            lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
            lo9, hi9 = np.nanpercentile(bs, [0.25, 99.75], axis=0)
            hit_a, hit_b = (pa == y).astype(float), (pb == y).astype(float)
            ob = float(hit_a.mean() - hit_b.mean())
            bo = np.array([hit_a[s].mean() - hit_b[s].mean() for s in BOOT]); olo, ohi = np.percentile(bo, [2.5, 97.5])
            key = f"{a} | {b} | {'best' if layer == 'best' else 'L3'}"
            res["pairs"][key] = dict(
                a=a, b=b, layer_a=la, layer_b=lb,
                overall=dict(diff=ob, lo=float(olo), hi=float(ohi), resolved=bool(olo > 0 or ohi < 0)),
                per_type={c: dict(recall_a=float(recall(pa, allidx)[k]), recall_b=float(recall(pb, allidx)[k]),
                                  diff=float(obs[k]), lo=float(lo[k]), hi=float(hi[k]),
                                  resolved=bool(lo[k] > 0 or hi[k] < 0), lo995=float(lo9[k]), hi995=float(hi9[k]),
                                  resolved995=bool(lo9[k] > 0 or hi9[k] < 0)) for k, c in enumerate(classes)})
            print(f"\n{key}  (L{la} vs L{lb})  overall {ob:+.4f} [{olo:+.4f}, {ohi:+.4f}]")
            for k, c in enumerate(classes):
                r = res["pairs"][key]["per_type"][c]
                print(f"   {c:4s} n={int((y == k).sum()):4d}  {r['recall_a']:.3f} vs {r['recall_b']:.3f}  "
                      f"{r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]{'  *' if r['resolved'] else ''}"
                      f"{'  **(99.5%)' if r['resolved995'] else ''}", flush=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main([tuple(p.split(":")) for p in sys.argv[1:]])

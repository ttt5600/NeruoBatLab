#!/usr/bin/env python
"""Paired call-type comparison of any two scored models, plus the "do they know different things" test.

    python3 calltype_pair.py --a daptv3aves_2e4_step15000 --b daptv3aves_5e5_step15000

For each model: its cached clip embeddings (features/ct11_<tag>_emb.npy), the layer its own scoring
reported, and the board's probe (aves_calltype.cv_acc). The reproduced accuracy is ASSERTED equal to
the stored one, so this always runs on the board's split (needs the system python3, sklearn 1.7.2 --
finding 062). Then, with a paired bootstrap over the 48 birds (2000 resamples, seed 0):
  * A minus B: observed difference and 2.5/97.5 percentiles
  * the mean of A's and B's out-of-fold probabilities minus the better of the two. A resolved gain
    means the two models get different clips right (calltype_controls.py: run16 + AVES +0.017,
    two run16 seeds +0.002).
Appends to analysis/calltype_pairs.json under "A | B".
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import calltype11 as C11                                                  # noqa: E402
from aves_calltype import cv_acc                                          # noqa: E402

ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
OUT = ANA / "calltype_pairs.json"
EMB_ALIAS = {"aves-base-bio": "aves"}                       # the first AVES cache predates the tag scheme


CT_ALIAS = {"run15_combined": "run15"}   # tag -> call-type JSON stem where they differ


def resolve(tag):
    emb = C11.FEAT / f"ct11_{EMB_ALIAS.get(tag, tag)}_emb.npy"
    own = ANA / f"{CT_ALIAS.get(tag, tag)}_calltype.json"
    if own.exists():
        d = json.load(open(own)); layer, acc = d["best_layer11"], d["acc11"]
    else:
        av = json.load(open(ANA / "aves_variants_calltype.json"))["models"]
        if tag not in av:
            sys.exit(f"{tag}: no {own.name} and not in aves_variants_calltype.json")
        layer, acc = av[tag]["best_11"]["layer"], av[tag]["best_11"]["acc"]
    if not emb.exists():
        sys.exit(f"{tag}: no embedding cache {emb}")
    return emb, int(layer), float(acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    a = ap.parse_args()
    rows = [r for r in C11.collect() if r[3] in C11.KEEP11]
    birds = np.array([r[1].lower() for r in rows])
    classes = sorted(set(r[3] for r in rows))
    y = np.array([classes.index(r[3]) for r in rows])
    assert len(y) == 3412 and len(classes) == 11

    P, acc = {}, {}
    for tag in (a.a, a.b):
        emb, layer, stored = resolve(tag)
        acc[tag], _, P[tag] = cv_acc(np.ascontiguousarray(np.load(emb, mmap_mode="r")[:, layer]), y, birds,
                                     return_proba=True)
        assert abs(acc[tag] - stored) < 1e-9, f"{tag}: reproduced {acc[tag]} != stored {stored} (wrong env/split?)"
        print(f"[{tag}] L{layer} acc11 {acc[tag]:.4f} (reproduces stored)")

    hit = {t: P[t].argmax(1) == y for t in P}
    hit_e = (P[a.a] + P[a.b]).argmax(1) == y
    best = max(P, key=lambda t: hit[t].mean())
    rng = np.random.default_rng(0)
    ub = np.unique(birds); idx = {g: np.where(birds == g)[0] for g in ub}
    sels = [np.concatenate([idx[g] for g in rng.choice(ub, len(ub))]) for _ in range(2000)]

    def interval(x, z):
        d = np.array([x[s].mean() - z[s].mean() for s in sels])
        lo, hi = np.percentile(d, [2.5, 97.5])
        return dict(diff=float(x.mean() - z.mean()), lo=float(lo), hi=float(hi), resolved=bool(lo > 0 or hi < 0))

    res = dict(acc={t: float(acc[t]) for t in acc}, a_minus_b=interval(hit[a.a], hit[a.b]),
               ensemble=dict(acc=float(hit_e.mean()), better_member=best, **interval(hit_e, hit[best])),
               protocol="board probe (cv_acc, sklearn 1.7.2 split), own best layers; bird bootstrap 2000, seed 0")
    for k in ("a_minus_b", "ensemble"):
        v = res[k]
        print(f"  {k:10s} {v['diff']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]{'  *' if v['resolved'] else ''}"
              + (f"   (ensemble {v['acc']:.4f} vs {best})" if k == "ensemble" else ""))
    allres = json.load(open(OUT)) if OUT.exists() else {}
    allres[f"{a.a} | {a.b}"] = res
    OUT.write_text(json.dumps(allres, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

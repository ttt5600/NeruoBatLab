#!/usr/bin/env python
"""Paired difference of probability-average gains: does model A already hold what R adds, more than B does?

    python3 calltype_gain_diff.py --a daptv4two_2e4_step15000 --b daptv3aves_2e4_step15000 --ref run16_compute4x

gain(X) = accuracy of mean(P_X, P_R) minus accuracy of the better of X and R (calltype_pair.py's ensemble test).
Reports gain(A), gain(B) and gain(A) - gain(B) with one paired bird bootstrap (2000 resamples, seed 0; the
better member is fixed on the full data, as in calltype_pair.py). A negative difference whose interval excludes
zero means A gains less from R than B does, i.e. A already holds more of R's content. Same probe, layers and
reproduction assert as calltype_pair.py (system python3, sklearn 1.7.2 split). Writes calltype_pairs.json under
"gain_diff: A vs B | ref R".
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import calltype11 as C11                                                  # noqa: E402
from aves_calltype import cv_acc                                          # noqa: E402
from calltype_pair import resolve, OUT                                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--ref", required=True)
    a = ap.parse_args()
    rows = [r for r in C11.collect() if r[3] in C11.KEEP11]
    birds = np.array([r[1].lower() for r in rows])
    classes = sorted(set(r[3] for r in rows))
    y = np.array([classes.index(r[3]) for r in rows])
    P = {}
    for tag in (a.a, a.b, a.ref):
        emb, layer, stored = resolve(tag)
        acc, _, P[tag] = cv_acc(np.ascontiguousarray(np.load(emb, mmap_mode="r")[:, layer]), y, birds, return_proba=True)
        assert abs(acc - stored) < 1e-9, f"{tag}: reproduced {acc} != stored {stored}"
    hit = {t: P[t].argmax(1) == y for t in P}

    def gain_hits(x):
        best = max((x, a.ref), key=lambda t: hit[t].mean())
        return ((P[x] + P[a.ref]).argmax(1) == y), hit[best]

    ea, ba = gain_hits(a.a); eb, bb = gain_hits(a.b)
    rng = np.random.default_rng(0)
    ub = np.unique(birds); idx = {g: np.where(birds == g)[0] for g in ub}
    sels = [np.concatenate([idx[g] for g in rng.choice(ub, len(ub))]) for _ in range(2000)]
    ga = np.array([ea[s].mean() - ba[s].mean() for s in sels]); gb = np.array([eb[s].mean() - bb[s].mean() for s in sels])
    obs_a, obs_b = ea.mean() - ba.mean(), eb.mean() - bb.mean()
    res = {}
    for k, obs, bs in (("gain_a", obs_a, ga), ("gain_b", obs_b, gb), ("gain_a_minus_gain_b", obs_a - obs_b, ga - gb)):
        lo, hi = np.percentile(bs, [2.5, 97.5])
        res[k] = dict(diff=float(obs), lo=float(lo), hi=float(hi), resolved=bool(lo > 0 or hi < 0))
        print(f"  {k:22s} {obs:+.4f} [{lo:+.4f}, {hi:+.4f}]{'  *' if lo > 0 or hi < 0 else ''}")
    allres = json.load(open(OUT)) if OUT.exists() else {}
    allres[f"gain_diff: {a.a} vs {a.b} | ref {a.ref}"] = res
    OUT.write_text(json.dumps(allres, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

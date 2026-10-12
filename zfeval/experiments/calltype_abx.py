#!/usr/bin/env python
"""ABX discriminability of call types: how far apart does each encoder put them, with nothing fitted?

ABX (the ZeroSpeech unit-discovery measure): take a clip X, a clip A of the SAME call type and a clip
B of ANOTHER type. The encoder gets the triple right if d(A, X) < d(B, X). The ABX error for a pair
of types is the fraction of triples it gets wrong (ties count half): 0.5 = it cannot tell the two
types apart, 0 = every same-type clip is closer than every other-type clip. No classifier, no
clustering -- only distances in the encoder's own space.

  * distance: cosine on the mean-pooled layer after per-dimension standardisation (label-free).
  * cross-bird (the headline): A and B come from birds other than X's, so a model cannot win by
    recognising the voice. within-bird: A, B and X all from X's bird. cross - within on the same
    (bird, type, type) cells = how much a change of voice costs that model.
  * aggregation: per X, average over all A/B; then over X within a bird; then over birds; a pair's
    error is the mean of both directions; the overall error is the mean over all type pairs.
  * layer: picked OUT OF FOLD -- birds split in two halves, each half scored at the layer with the
    lowest cross-bird error on the other half (picking depth on the reported metric inflated margins
    2.3x before, finding run11-wins-nowhere).
  * vs run16: paired bootstrap over birds on the out-of-fold scores, observed difference + 2.5/97.5
    percentiles. run16 seed 2 is in the table as the seed-noise yardstick.

  python calltype_abx.py   -> analysis/calltype_abx.json
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore", category=RuntimeWarning)               # nanmean of empty cells
import calltype11 as C11                                                  # noqa: E402
from calltype_cluster_by_type import MODELS                               # noqa: E402

ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
OUT = ANA / "calltype_abx.json"
REF = "run16"
N_BOOT, SEED = 2000, 0


def abx_bird_arrays(X, y, birds, ub, K):
    """Returns (cross, within), each (n_birds, K, K): mean over that bird's X of type t1 of
    P(d(A,X) < d(B,X)) against type t2. NaN where undefined."""
    Z = (X - X.mean(0)) / (X.std(0) + 1e-8)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    D = 1 - Z @ Z.T
    n = len(y)
    S = {m: np.full((n, K), np.nan) for m in ("cross", "within")}
    for i in range(n):
        own = birds == birds[i]
        for mode, mask in (("cross", ~own), ("within", own)):
            m = mask.copy(); m[i] = False
            a = np.sort(D[i, m & (y == y[i])])
            if len(a) == 0:
                continue
            dB, yB = D[i, m], y[m]
            # per B: fraction of A with d(A,X) < d(B,X), ties half = P(triple correct)
            p = (np.searchsorted(a, dB, "right") + np.searchsorted(a, dB, "left")) / (2 * len(a))
            s, c = np.bincount(yB, weights=p, minlength=K), np.bincount(yB, minlength=K)
            r = np.where(c > 0, s / np.maximum(c, 1), np.nan); r[y[i]] = np.nan
            S[mode][i] = r
    out = {}
    for mode in S:
        A = np.full((len(ub), K, K), np.nan)
        for bi, b in enumerate(ub):
            for t in range(K):
                k = (birds == b) & (y == t)
                if k.any():
                    A[bi, t] = np.nanmean(S[mode][k], axis=0)
        out[mode] = 1 - A                                 # ERROR = 1 - correct
    return out["cross"], out["within"]


def pair_matrix(A):
    """(n_birds, K, K) -> symmetric (K, K) pair error: mean over birds, then over both directions."""
    M = np.nanmean(A, axis=0)
    return np.nanmean(np.stack([M, M.T]), axis=0)


def overall(A):
    M = pair_matrix(A)
    return float(np.nanmean(M[~np.eye(len(M), dtype=bool)]))


def main():
    t0 = time.time()
    rows = [r for r in C11.collect() if r[3] in C11.KEEP11]
    birds = np.array([r[1].lower() for r in rows])
    tt = np.array([r[3] for r in rows])
    classes = sorted(set(tt))
    y = np.array([classes.index(t) for t in tt])
    K, ub = len(classes), np.unique(birds)
    assert len(y) == 3412 and K == 11 and len(ub) == 48
    half = np.random.default_rng(SEED).permutation(len(ub)) < len(ub) // 2     # bird halves for layer choice

    out = dict(classes=classes, reference=REF, n_boot=N_BOOT, distance="cosine after per-dim standardisation",
               layer_rule="out of fold over two bird halves (cross-bird ABX error)", models={})
    oof = {}
    for name, f in MODELS:
        E = np.load(C11.FEAT / f, mmap_mode="r")
        per_layer = [abx_bird_arrays(np.asarray(E[:, l], dtype=np.float32), y, birds, ub, K)
                     for l in range(E.shape[1])]
        sweep = [overall(c) for c, _ in per_layer]
        L_for = {h: int(np.argmin([overall(c[half != h]) for c, _ in per_layer])) for h in (True, False)}
        cross = np.where(half[:, None, None], per_layer[L_for[True]][0], per_layer[L_for[False]][0])
        within = np.where(half[:, None, None], per_layer[L_for[True]][1], per_layer[L_for[False]][1])
        oof[name] = (cross, within)
        both = ~np.isnan(within) & ~np.isnan(cross)
        Mc, Mw = pair_matrix(cross), pair_matrix(within)
        per_type = {c: dict(cross=float(np.nanmean(np.delete(Mc[i], i))),
                            voice_cost=float(np.nanmean(np.where(both, cross, np.nan)[:, i]) -
                                             np.nanmean(np.where(both, within, np.nan)[:, i])))
                    for i, c in enumerate(classes)}
        out["models"][name] = dict(layers_oof={"half_a": L_for[True], "half_b": L_for[False]},
                                   cross_error=overall(cross),
                                   within_error=overall(np.where(both, within, np.nan)),
                                   cross_error_same_cells=overall(np.where(both, cross, np.nan)),
                                   cross_by_layer=sweep, per_type=per_type,
                                   pair_cross=Mc.round(4).tolist())
        d = out["models"][name]
        print(f"[{name}] layers {L_for[True]}/{L_for[False]}  cross-bird ABX error {d['cross_error']:.4f}  "
              f"within-bird {d['within_error']:.4f} (cross on same cells {d['cross_error_same_cells']:.4f})  "
              f"({time.time()-t0:.0f}s)", flush=True)

    rng = np.random.default_rng(SEED)
    boots = [rng.integers(0, len(ub), len(ub)) for _ in range(N_BOOT)]
    ref_b = np.array([overall(oof[REF][0][b]) for b in boots])
    ref_t = np.array([[np.nanmean(np.delete(pair_matrix(oof[REF][0][b])[i], i)) for i in range(K)] for b in boots])
    for name in oof:
        if name == REF:
            continue
        d = out["models"][name]
        obs = d["cross_error"] - out["models"][REF]["cross_error"]
        bb = np.array([overall(oof[name][0][b]) for b in boots]) - ref_b
        lo, hi = np.percentile(bb, [2.5, 97.5])
        d["vs_ref"] = dict(diff=float(obs), lo=float(lo), hi=float(hi), resolved=bool(lo > 0 or hi < 0))
        tb = np.array([[np.nanmean(np.delete(pair_matrix(oof[name][0][b])[i], i)) for i in range(K)]
                       for b in boots]) - ref_t
        tlo, thi = np.nanpercentile(tb, [2.5, 97.5], axis=0)
        for i, c in enumerate(classes):
            o = d["per_type"][c]["cross"] - out["models"][REF]["per_type"][c]["cross"]
            d["per_type"][c]["vs_ref"] = dict(diff=float(o), lo=float(tlo[i]), hi=float(thi[i]),
                                              resolved=bool(tlo[i] > 0 or thi[i] < 0))
    OUT.write_text(json.dumps(out, indent=2))

    print("\ncross-bird ABX ERROR (lower = types further apart; 0.5 = indistinguishable)")
    print(f"{'model':<20}{'all':>7}{'vs run16':>22}" + "".join(f"{c:>6}" for c in classes))
    for n, d in out["models"].items():
        v = d.get("vs_ref")
        vs = f"{v['diff']:+.4f} [{v['lo']:+.3f},{v['hi']:+.3f}]{'*' if v['resolved'] else ' '}" if v else ""
        print(f"{n:<20}{d['cross_error']:>7.4f}{vs:>22}" + "".join(f"{d['per_type'][c]['cross']:>6.3f}" for c in classes))
    print("\nper type vs run16 (* = interval excludes 0)")
    for n, d in out["models"].items():
        if n != REF:
            print(f"{n:<20}" + " ".join(f"{c}:{d['per_type'][c]['vs_ref']['diff']:+.3f}"
                                        f"{'*' if d['per_type'][c]['vs_ref']['resolved'] else ' '}" for c in classes))
    print("\nvoice cost per type (cross-bird minus within-bird error, same cells)")
    for n, d in out["models"].items():
        print(f"{n:<20}" + "".join(f"{d['per_type'][c]['voice_cost']:>+7.3f}" for c in classes))
    print(f"-> {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

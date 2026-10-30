#!/usr/bin/env python
"""Arm-vs-arm call-type contrasts that average over training seeds (the pass mark E13/E30 pre-register).

seed_pairs_cpu.py compares ONE model with ONE model; its bird bootstrap holds both fixed, so it cannot see seed
variance, and with seeds differing by up to 0.021 (finding 079) a single resolved pair is not a verdict (067, 084).
Here each ARM is a list of seeds: every clip's correct/incorrect (board probe, out-of-fold) is averaged over the arm's
seeds, and the difference of the two arm means is bootstrapped over birds (2,000 resamples of the 48 birds, seed 0, the
same resamples as seed_pairs_cpu.py). That bootstrap still holds the seeds fixed, so the PASS interval adds a seed term:
half-width = 1.96 * sqrt(var_birds + s^2 (1/n_a + 1/n_b)), s^2 = the pooled within-arm variance of the seeds' pooled
accuracies (df = n_a + n_b - 2). With run16's seed SD ~0.011-0.013 the bird-only interval alone would pass a true zero
effect ~9% of the time (E30 critic, 2026-10-27). Both intervals and every seed-by-seed difference are written.
Each model is read at its own CPU best layer, or with --layer N at a fixed layer;
pooled accuracy is asserted to reproduce calltype_cpu_rescore.json. Differences are OBSERVED (arm mean minus arm mean on
the full data), not the resample mean (reference_bootstrap_delta_is_resample_mean).

    python zfeval/experiments/seed_mean_pairs_cpu.py NAME_A=tag1,tag2 NAME_B=tag3,tag4,tag5 [--layer 3]
        -> appends analysis/seed_mean_pairs_cpu[_L3].json under "NAME_A | NAME_B"
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
CPU = json.loads((ANA / "calltype_cpu_rescore.json").read_text())["models"]


def main(arms, fixed=None):
    rows = [r for r in collect() if r[3] in KEEP11]
    birds = np.array([r[1].lower() for r in rows])
    tt = np.array([r[3] for r in rows]); classes = sorted(set(tt))
    y = np.array([classes.index(t) for t in tt])
    rng = np.random.default_rng(0); ub = np.unique(birds); bidx = {b: np.where(birds == b)[0] for b in ub}
    BOOT = [np.concatenate([bidx[b] for b in rng.choice(ub, len(ub))]) for _ in range(2000)]

    def hit(tag):
        L = int(CPU[tag]["best_layer_cpu"]) if fixed is None else fixed
        want = CPU[tag]["acc11_cpu"] if fixed is None else CPU[tag]["per_layer_acc11_cpu"][fixed]
        E = np.load(FEAT / f"ct11_{tag}_cpuemb.npy", mmap_mode="r")
        a, _, P = cv_acc(np.ascontiguousarray(E[:, L]), y, birds, return_proba=True)
        assert abs(a - want) < 1e-9, (tag, a, want)
        return (P.argmax(1) == y).astype(float), L, a

    out_p = ANA / ("seed_mean_pairs_cpu.json" if fixed is None else f"seed_mean_pairs_cpu_L{fixed}.json")
    res = json.loads(out_p.read_text()) if out_p.exists() else dict(
        protocol=("CPU embeddings; " + ("each model at its CPU best layer" if fixed is None else f"fixed layer {fixed}")
                  + "; per-clip correctness averaged over each arm's seeds; cv_acc leave-birds-out SGKF(5) seed 0; "
                  "bird bootstrap 2000 seed 0; observed differences"), pairs={})
    (na, ta), (nb, tb) = arms
    H = {}
    for name, tags in ((na, ta), (nb, tb)):
        hs = [hit(t) for t in tags]
        H[name] = (np.mean([h for h, _, _ in hs], 0), [dict(tag=t, layer=l, acc=a) for t, (_, l, a) in zip(tags, hs)])
    ma, mb = H[na][0], H[nb][0]
    diff = float(ma.mean() - mb.mean())
    bs = np.array([ma[s].mean() - mb[s].mean() for s in BOOT])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    acc_a, acc_b = [x["acc"] for x in H[na][1]], [x["acc"] for x in H[nb][1]]
    # pooled within-arm seed variance; a one-model arm (e.g. a download) contributes no degrees of freedom and is
    # assumed to carry the same seed variance as the arms that have seeds
    df = (len(acc_a) - 1) + (len(acc_b) - 1)
    ss = sum(float(np.var(x, ddof=1)) * (len(x) - 1) for x in (acc_a, acc_b) if len(x) > 1)
    s2 = ss / df if df > 0 else float("nan")
    half = 1.96 * float(np.sqrt(np.var(bs, ddof=1) + s2 * (1 / len(acc_a) + 1 / len(acc_b))))
    plo, phi = diff - half, diff + half
    seedwise = {f"{x['tag']} - {y['tag']}": round(x["acc"] - y["acc"], 4) for x in H[na][1] for y in H[nb][1]}
    r = dict(a=na, b=nb, seeds_a=H[na][1], seeds_b=H[nb][1], mean_a=float(ma.mean()), mean_b=float(mb.mean()),
             diff=diff, lo_birds=float(lo), hi_birds=float(hi), seed_sd_pooled=float(np.sqrt(s2)), seed_df=df,
             lo=float(plo), hi=float(phi), resolved=bool(plo > 0 or phi < 0), seedwise=seedwise)
    res["pairs"][f"{na} | {nb}"] = r
    out_p.write_text(json.dumps(res, indent=1))
    print(f"{na} ({len(ta)} seeds) {ma.mean():.4f} - {nb} ({len(tb)} seeds) {mb.mean():.4f} = {diff:+.4f}  "
          f"birds only [{lo:+.4f}, {hi:+.4f}]  birds + seeds (sd {np.sqrt(s2):.4f}, df {df}) [{plo:+.4f}, {phi:+.4f}]"
          f"{' resolved' if r['resolved'] else ''}  -> {out_p}")
    print("  seed by seed: " + ", ".join(f"{k} {v:+.4f}" for k, v in seedwise.items()))


if __name__ == "__main__":
    args = sys.argv[1:]
    fixed = None
    if "--layer" in args:
        i = args.index("--layer"); fixed = int(args[i + 1]); del args[i:i + 2]
    if len(args) != 2:
        raise SystemExit(__doc__)
    arms = [(a.split("=")[0], a.split("=")[1].split(",")) for a in args]
    main(arms, fixed)

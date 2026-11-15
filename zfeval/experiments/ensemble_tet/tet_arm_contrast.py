#!/usr/bin/env python
"""Part 2 (EXPLORATORY, not pre-registered): per-call-type recall, arm vs arm, seeds averaged.

Arm metric = mean over the arm's seeds of each seed's recall (identical to recall of per-clip correctness averaged over
seeds, as in seed_mean_pairs_cpu.py). Interval = bird bootstrap of the arm difference (2000, seed 0, seeds held fixed)
plus seed_mean_pairs_cpu.py's seed term: half = z * sqrt(var_birds + s^2 (1/n_a + 1/n_b)), s^2 = pooled within-arm
variance of the seeds' metric (df = n_a + n_b - 2; a one-model arm adds no df). Reported for all 11 types (so Tet can
be read against the other types as a built-in placebo), plus TTT = recall over the pooled Te+Th+Tu clips, plus overall
accuracy. z = 1.96 (95%) and 2.81 (99.5%, Bonferroni over 11 types). Each model at its CPU best layer, and all at L3.

    python zfeval/experiments/ensemble_tet/tet_arm_contrast.py   -> ensemble_tet/tet_arm_contrast.json
"""
from __future__ import annotations
import json, time
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm
from common import task, probe, layer_of, N_JOBS, HERE

R19 = ["run19_avesteacher", "run19_seed2"]
E13 = ["run24_bioxnl12", "run24_bioxnl12_s2"]
R16 = ["run16_compute4x", "run16_seed2", "run16_seed3"]
VAN = ["run31_vanilla_lr1e4"]          # run31_vanilla_lr1e4_s2 has no CPU cache yet (E36 seed 2 not scored)
ARMS = {"aveslabel(run19+E13)": R19 + E13, "ownlabel(run16+vanilla)": R16 + VAN, "run19": R19, "E13": E13,
        "run16": R16, "vanilla_lr1e4": VAN}
CONTRASTS = [("aveslabel(run19+E13)", "ownlabel(run16+vanilla)"),   # the headline the roadmap lead names
             ("aveslabel(run19+E13)", "run16"), ("aveslabel(run19+E13)", "vanilla_lr1e4"),
             ("run19", "run16"), ("E13", "run16")]
Z95, Z995 = norm.ppf(0.975), norm.ppf(1 - 0.005 / 2)


def main():
    t0 = time.time()
    y, birds, classes, BOOT = task()
    tags = sorted({t for v in ARMS.values() for t in v})
    jobs = [(t, l) for t in tags for l in ("best", 3)]
    outs = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(probe)(t, l, y, birds) for t, l in jobs)
    H = {k: (p.argmax(1) == y).astype(float) for k, p in zip(jobs, outs)}
    names = classes + ["TTT", "overall"]
    masks = [y == k for k in range(len(classes))] + [np.isin(y, [classes.index(c) for c in ("Te", "Th", "Tu")]),
                                                     np.ones(len(y), bool)]
    counts = {n: int(m.sum()) for n, m in zip(names, masks)}

    def metric(h, idx=None):          # per-type recall vector for one correctness vector (optionally resampled)
        if idx is None:
            return np.array([h[m].mean() for m in masks])
        return np.array([h[idx][m[idx]].mean() if m[idx].any() else np.nan for m in masks])

    res = dict(status="EXPLORATORY, not pre-registered (the per-type heatmap was read before this analysis)",
               protocol="CPU caches; cv_acc leave-birds-out SGKF(5) seed 0; arm metric = mean of seeds' recall; bird "
                        "bootstrap 2000 seed 0 + pooled seed term (seed_mean_pairs_cpu.py); observed differences; "
                        "z 1.96 (95%) and 2.81 (99.5%, Bonferroni over 11 types)",
               counts=counts, arms=ARMS, per_model={}, contrasts={})
    for layer in ("best", 3):
        lk = "best" if layer == "best" else "L3"
        for t in tags:
            res["per_model"].setdefault(t, {})[lk] = dict(layer=layer_of(t, layer),
                                                         **{n: float(v) for n, v in zip(names, metric(H[(t, layer)]))})
        for a, b in CONTRASTS:
            ha = np.mean([H[(t, layer)] for t in ARMS[a]], 0); hb = np.mean([H[(t, layer)] for t in ARMS[b]], 0)
            obs = metric(ha) - metric(hb)
            bs = np.array([metric(ha, s) - metric(hb, s) for s in BOOT])
            vb = np.nanvar(bs, 0, ddof=1)
            sa = np.array([metric(H[(t, layer)]) for t in ARMS[a]]); sb = np.array([metric(H[(t, layer)]) for t in ARMS[b]])
            df = (len(sa) - 1) + (len(sb) - 1)
            ss = sum(np.var(x, 0, ddof=1) * (len(x) - 1) for x in (sa, sb) if len(x) > 1)
            s2 = ss / df
            se = np.sqrt(vb + s2 * (1 / len(sa) + 1 / len(sb)))
            blo, bhi = np.nanpercentile(bs, [2.5, 97.5], 0)
            per = {}
            for k, n in enumerate(names):
                lo, hi = obs[k] - Z95 * se[k], obs[k] + Z95 * se[k]
                lo9, hi9 = obs[k] - Z995 * se[k], obs[k] + Z995 * se[k]
                per[n] = dict(mean_a=float(sa[:, k].mean()), mean_b=float(sb[:, k].mean()), diff=float(obs[k]),
                              lo_birds=float(blo[k]), hi_birds=float(bhi[k]), seed_sd_pooled=float(np.sqrt(s2[k])),
                              lo=float(lo), hi=float(hi), lo995=float(lo9), hi995=float(hi9),
                              resolved=bool(lo > 0 or hi < 0), resolved995=bool(lo9 > 0 or hi9 < 0),
                              seeds_a=[round(float(v), 4) for v in sa[:, k]], seeds_b=[round(float(v), 4) for v in sb[:, k]])
            key = f"{a} | {b} | {lk}"
            res["contrasts"][key] = dict(n_a=len(sa), n_b=len(sb), seed_df=df, per_type=per)
            print(f"\n{key}  (n {len(sa)} vs {len(sb)}, df {df})")
            for n in names:
                r = per[n]
                print(f"  {n:7s} n={counts[n]:4d} {r['mean_a']:.3f} vs {r['mean_b']:.3f} {r['diff']:+.3f}  birds "
                      f"[{r['lo_birds']:+.3f},{r['hi_birds']:+.3f}]  +seeds (sd {r['seed_sd_pooled']:.3f}) "
                      f"[{r['lo']:+.3f},{r['hi']:+.3f}]{' *' if r['resolved'] else ''}{'*' if r['resolved995'] else ''}",
                      flush=True)
    out = HERE / "tet_arm_contrast.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"-> {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

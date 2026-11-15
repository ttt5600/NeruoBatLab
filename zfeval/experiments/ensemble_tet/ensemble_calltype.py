#!/usr/bin/env python
"""Part 1: decision ensemble on 11-class call type, the pre-committed list in README.md (E1-E5, C1-C2).

Ensemble = unweighted mean of members' out-of-fold probabilities (same folds, seed 0), no fitted weights.
Comparator = the member with the highest acc11_cpu. Paired bird bootstrap (2000, seed 0), observed difference,
95% and Bonferroni (m=7) intervals. Secondary: fixed layer 3; fold splits 0-9 (no interval); per-type recall E1.

    python zfeval/experiments/ensemble_tet/ensemble_calltype.py   -> ensemble_tet/ensemble_calltype.json
"""
from __future__ import annotations
import json, time
import numpy as np
from joblib import Parallel, delayed
from common import task, probe, pct, layer_of, CPU, N_JOBS, HERE, _cv, feats

WB, WS, BXL, R19, E13 = "hf-whisper-base-enc", "hf-whisper-small-enc", "birdaves-bioxn-large", "run19_seed2", "run24_bioxnl12_s2"
ENS = {"E1": (WB, WS, BXL), "E2": (WB, BXL), "E3": (WB, R19), "E4": (WB, WS, BXL, R19), "E5": (WS, BXL, E13)}
CAT = {"C1": ENS["E1"], "C2": ENS["E2"]}
M = len(ENS) + len(CAT)
ALPHA_B = 0.05 / M
MEMBERS = sorted({t for v in ENS.values() for t in v})


def main():
    t0 = time.time()
    y, birds, classes, BOOT = task()
    # sanity: the split-k re-implementation equals cv_acc at split 0
    P0 = _cv(feats(WB, "best"), y, birds, 0)
    assert np.allclose(P0, probe(WB, "best", y, birds)), "split re-implementation differs from cv_acc"

    jobs = [(t, "best", 0) for t in MEMBERS] + [(t, 3, 0) for t in MEMBERS] + [(CAT[c], "best", 0) for c in CAT] \
        + [(t, "best", s) for t in MEMBERS for s in range(1, 10)]
    outs = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(probe)(t, l, y, birds, s) for t, l, s in jobs)
    P = {(t if isinstance(t, str) else "+".join(t), l, s): p for (t, l, s), p in zip(jobs, outs)}
    print(f"probes done {time.time() - t0:.0f}s", flush=True)

    hit = lambda p: (p.argmax(1) == y).astype(float)
    acc = lambda p: float(hit(p).mean())

    def contrast(pa, pb):
        ha, hb = hit(pa), hit(pb)
        bs = np.array([ha[s].mean() - hb[s].mean() for s in BOOT])
        lo, hi = pct(bs, 0.05); blo, bhi = pct(bs, ALPHA_B)
        return dict(acc_a=float(ha.mean()), acc_b=float(hb.mean()), diff=float(ha.mean() - hb.mean()),
                    lo95=lo, hi95=hi, lo_bonf=blo, hi_bonf=bhi, resolved95=bool(lo > 0 or hi < 0),
                    resolved_bonf=bool(blo > 0 or bhi < 0),
                    wording=("ensemble better" if blo > 0 else "ensemble worse" if bhi < 0 else "not distinguishable")
                    + " (Bonferroni m=%d)" % M)

    def recall(p):
        pr = p.argmax(1)
        return {c: float((pr[y == k] == k).mean()) for k, c in enumerate(classes)}

    res = dict(protocol="CPU caches; cv_acc leave-birds-out SGKF(5); members at CPU best layer (and fixed L3 secondary); "
                        "mean of out-of-fold probabilities, no fitted weights; paired bird bootstrap 2000 seed 0; "
                        f"observed differences; Bonferroni m={M} -> two-sided alpha {ALPHA_B:.5f}",
               members={t: dict(layer_best=layer_of(t, "best"), acc_best=acc(P[(t, "best", 0)]),
                                acc_best_rescore_json=CPU[t]["acc11_cpu"], acc_L3=acc(P[(t, 3, 0)])) for t in MEMBERS},
               primary={}, secondary_L3={}, splits={}, per_type_E1={}, error_overlap={})
    for name, mem in list(ENS.items()) + list(CAT.items()):
        best = max(mem, key=lambda t: CPU[t]["acc11_cpu"])
        pe = P[("+".join(mem), "best", 0)] if name in CAT else np.mean([P[(t, "best", 0)] for t in mem], 0)
        r = dict(members=list(mem), kind="concat" if name in CAT else "mean", best_member=best,
                 **contrast(pe, P[(best, "best", 0)]))
        res["primary"][name] = r
        print(f"{name} {r['kind']:6s} {'+'.join(mem)}  {r['acc_a']:.4f} vs best {best} {r['acc_b']:.4f}  "
              f"{r['diff']:+.4f} 95% [{r['lo95']:+.4f}, {r['hi95']:+.4f}] Bonf [{r['lo_bonf']:+.4f}, {r['hi_bonf']:+.4f}]"
              f"  {r['wording']}", flush=True)
        if name in ENS:
            bl3 = max(mem, key=lambda t: acc(P[(t, 3, 0)]))
            pe3 = np.mean([P[(t, 3, 0)] for t in mem], 0)
            res["secondary_L3"][name] = dict(best_member_L3=bl3, **contrast(pe3, P[(bl3, 3, 0)]))
            ds = []
            for s in range(10):
                pes = np.mean([P[(t, "best", s)] for t in mem], 0)
                ds.append(acc(pes) - acc(P[(best, "best", s)]))
            res["splits"][name] = dict(diffs=[round(d, 4) for d in ds], mean=float(np.mean(ds)), min=float(np.min(ds)),
                                       n_positive=int(np.sum(np.array(ds) > 0)))
            print(f"    L3: {res['secondary_L3'][name]['diff']:+.4f} [{res['secondary_L3'][name]['lo95']:+.4f}, "
                  f"{res['secondary_L3'][name]['hi95']:+.4f}]  splits mean {np.mean(ds):+.4f} pos {res['splits'][name]['n_positive']}/10",
                  flush=True)
    # error overlap between members (031 reported error correlation): fraction of clips both get wrong, and phi
    for i, a in enumerate(MEMBERS):
        for b in MEMBERS[i + 1:]:
            ea, eb = 1 - hit(P[(a, "best", 0)]), 1 - hit(P[(b, "best", 0)])
            res["error_overlap"][f"{a} | {b}"] = dict(both_wrong=float((ea * eb).mean()), phi=float(np.corrcoef(ea, eb)[0, 1]))
    pE1 = np.mean([P[(t, "best", 0)] for t in ENS["E1"]], 0)
    rE, rB = recall(pE1), recall(P[(WB, "best", 0)])
    res["per_type_E1"] = {c: dict(recall_E1=rE[c], recall_whisper_base=rB[c], diff=rE[c] - rB[c]) for c in classes}
    out = HERE / "ensemble_calltype.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"-> {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

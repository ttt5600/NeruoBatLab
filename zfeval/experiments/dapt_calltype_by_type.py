#!/usr/bin/env python
"""Per-call-type recall of every DAPT checkpoint against its parent, with the controls the 075 critic asked for.

Reads the CPU call-type embeddings (features/ct11_<tag>_cpuemb.npy from calltype_cpu_rescore.py; the MPS caches are
wrong for 2-8 frame clips) and each model's CPU best 11-class layer from analysis/calltype_cpu_rescore.json. Board
probe throughout (cv_acc: leave-birds-out StratifiedGroupKFold(5) seed 0, StandardScaler + LogisticRegression), and
every model's pooled accuracy is asserted to reproduce calltype_cpu_rescore.json. All differences are OBSERVED, with
a 95% bird bootstrap (2,000 resamples of the 48 birds, seed 0, the same resamples for every row).

  dapt.best[tag]      recall minus parent per type + overall, each model at its own CPU best layer
  dapt.L3[tag]        the same with both models at layer 3 (no per-model layer choice)
  downloads.best/L3   the six other AVES/BirdAVES downloads minus aves-base-bio, same two readings: what ANY other
                      encoder does against aves-base-bio's particular layer, the control a "DAPT does X" claim needs
  pattern             per type: DAPT endpoints above/below parent (point, resolved), and the downloads' range
  seed_pairs          per-type differences for the four same-recipe seed pairs
  round1              the 12 zebra-finch-only checkpoints minus aves-base-bio, best layer and L3
  bird_generalisation per-type recall under a random clip split (StratifiedKFold(5) seed 0; a diagnostic, never a
                      score) vs leave-birds-out; share of each group's errors the leaky split removes; and E23's extra
                      gain on the leaky split with a bird-bootstrap interval
  error_share         share of each model's errors whose true type is Te/Ne/Th/Tu

    python zfeval/experiments/dapt_calltype_by_type.py   -> analysis/dapt_calltype_by_type.json
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
from calltype11 import collect, KEEP11                                     # noqa: E402
from aves_calltype import cv_acc                                           # noqa: E402
from sklearn.model_selection import StratifiedKFold                        # noqa: E402
from sklearn.pipeline import make_pipeline                                 # noqa: E402
from sklearn.preprocessing import StandardScaler                           # noqa: E402
from sklearn.linear_model import LogisticRegression                        # noqa: E402

FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
OUT = ANA / "dapt_calltype_by_type.json"
CPU = json.loads((ANA / "calltype_cpu_rescore.json").read_text())["models"]
REF = "aves-base-bio"
PARENT = {t: REF for t in [
    "dapt5e5_step15000", "dapt1e4_step15000", "daptshort_5e5_step2500", "daptlr_1e5_step15000",
    "daptfreeze_5e5_step15000", "daptreplay_5e5_step15000", "daptreplay3_5e5_step15000",
    "daptv3aves_5e5_step15000", "daptv3aves_2e4_step15000", "daptv3aves_2e4s2_step15000",
    "daptv3aves_2e4m8_step15000", "daptv3aves_2e4s2m8_step15000", "daptv4two_2e4_step15000",
    "daptv4two_2e4s2_step15000", "daptv4aveso_2e4_step15000", "daptv4aveso_2e4s2_step15000",
    "daptv4two45k_2e4_step45000"]}
PARENT["daptv4bird_2e4_step15000"] = "birdaves-biox-base"
DOWNLOADS = ["aves-base-core", "aves-base-all", "aves-base-nonbio", "birdaves-biox-base", "birdaves-biox-large",
             "birdaves-bioxn-large"]
ROUND1 = [f"dapt{lr}_step{s}" for lr in ("5e5", "1e4") for s in (2500, 5000, 7500, 10000, 12500, 15000)]
PAIRS = [("daptv4two_2e4_step15000", "daptv4two_2e4s2_step15000"),
         ("daptv4aveso_2e4_step15000", "daptv4aveso_2e4s2_step15000"),
         ("daptv3aves_2e4_step15000", "daptv3aves_2e4s2_step15000"),
         ("daptv3aves_2e4m8_step15000", "daptv3aves_2e4s2m8_step15000")]
SOFT = ["Te", "Ne", "Th", "Tu"]
E23 = "daptv4two_2e4s2_step15000"


def main():
    rows = [r for r in collect() if r[3] in KEEP11]
    birds = np.array([r[1].lower() for r in rows])
    tt = np.array([r[3] for r in rows]); classes = sorted(set(tt))
    y = np.array([classes.index(t) for t in tt]); K = len(classes)
    rng = np.random.default_rng(0); ub = np.unique(birds); bidx = {b: np.where(birds == b)[0] for b in ub}
    BOOT = [np.concatenate([bidx[b] for b in rng.choice(ub, len(ub))]) for _ in range(2000)]
    emb = {}

    def X_of(tag, layer):
        if tag not in emb:
            emb[tag] = np.load(FEAT / f"ct11_{tag}_cpuemb.npy", mmap_mode="r")
        return np.ascontiguousarray(emb[tag][:, layer])

    hits = {}

    def hit(tag, layer=None):
        """Out-of-fold correctness at the CPU best layer (asserted) or at a fixed layer."""
        key = (tag, layer)
        if key not in hits:
            best = int(CPU[tag]["best_layer_cpu"])
            a, _, P = cv_acc(X_of(tag, best if layer is None else layer), y, birds, return_proba=True)
            if layer is None:
                assert abs(a - CPU[tag]["acc11_cpu"]) < 1e-9, (tag, a, CPU[tag]["acc11_cpu"])
            hits[key] = ((P.argmax(1) == y).astype(float), P.argmax(1), best if layer is None else layer, a)
        return hits[key]

    def rec(h, idx=None):
        idx = np.arange(len(y)) if idx is None else idx
        with np.errstate(invalid="ignore"):
            return np.bincount(y[idx], weights=h[idx], minlength=K) / np.bincount(y[idx], minlength=K)

    def diff(ha, hb):
        obs = rec(ha) - rec(hb)
        bs = np.array([rec(ha, s) - rec(hb, s) for s in BOOT])
        lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
        ba = np.array([ha[s].mean() - hb[s].mean() for s in BOOT])
        out = {c: dict(diff=float(obs[k]), lo=float(lo[k]), hi=float(hi[k]), resolved=bool(lo[k] > 0 or hi[k] < 0))
               for k, c in enumerate(classes)}
        la, ha_ = np.percentile(ba, [2.5, 97.5])
        out["ALL"] = dict(diff=float(ha.mean() - hb.mean()), lo=float(la), hi=float(ha_), resolved=bool(la > 0 or ha_ < 0))
        return out

    def row(tag, parent, L3):
        a = hit(tag, 3 if L3 else None); b = hit(parent, 3 if L3 else None)
        return dict(parent=parent, layer=a[2], parent_layer=b[2], acc11=a[3], vs_parent=diff(a[0], b[0]))

    twelve = lambda t: len(CPU[t]["per_layer_acc11_cpu"]) == 12
    res = dict(classes=classes, n_clips=int(len(y)), n_birds=int(len(ub)),
               counts={c: int((y == k).sum()) for k, c in enumerate(classes)},
               embeddings="CPU (features/ct11_<tag>_cpuemb.npy); MPS caches are wrong for 2-8 frame clips",
               protocol="board probe (cv_acc); bird bootstrap 2000 seed 0; observed differences",
               dapt=dict(best={}, L3={}), downloads=dict(best={}, L3={}), round1=dict(best={}, L3={}))
    for t, p in PARENT.items():
        res["dapt"]["best"][t] = row(t, p, False)
        res["dapt"]["L3"][t] = row(t, p, True)
        print(f"{t:34s} L{res['dapt']['best'][t]['layer']} {res['dapt']['best'][t]['acc11']:.4f} vs {p}: "
              f"{res['dapt']['best'][t]['vs_parent']['ALL']['diff']:+.4f} | L3 {res['dapt']['L3'][t]['vs_parent']['ALL']['diff']:+.4f}",
              flush=True)
    for t in DOWNLOADS:
        res["downloads"]["best"][t] = row(t, REF, False)
        if twelve(t):
            res["downloads"]["L3"][t] = row(t, REF, True)
    for t in ROUND1:
        res["round1"]["best"][t] = row(t, REF, False)["vs_parent"]["ALL"]
        res["round1"]["L3"][t] = row(t, REF, True)["vs_parent"]["ALL"]

    def counts(block, keys):
        out = {}
        for c in classes + ["ALL"]:
            v = [block[t]["vs_parent"][c] for t in keys]
            out[c] = dict(n=len(v), above_point=sum(x["diff"] > 0 for x in v), below_point=sum(x["diff"] < 0 for x in v),
                          above_resolved=sum(x["resolved"] and x["diff"] > 0 for x in v),
                          below_resolved=sum(x["resolved"] and x["diff"] < 0 for x in v),
                          median=float(np.median([x["diff"] for x in v])),
                          min=float(min(x["diff"] for x in v)), max=float(max(x["diff"] for x in v)))
        return out
    aves_parent = [t for t in PARENT if PARENT[t] == REF]
    res["pattern"] = dict(dapt_best=counts(res["dapt"]["best"], aves_parent), dapt_L3=counts(res["dapt"]["L3"], aves_parent),
                          downloads_best=counts(res["downloads"]["best"], DOWNLOADS),
                          downloads_L3=counts(res["downloads"]["L3"], list(res["downloads"]["L3"])),
                          note="dapt_* counts exclude E14 (parent birdaves-biox-base); downloads are vs aves-base-bio")
    res["seed_pairs"] = {f"{a} | {b}": {c: [res["dapt"]["best"][a]["vs_parent"][c]["diff"],
                                             res["dapt"]["best"][b]["vs_parent"][c]["diff"]] for c in classes + ["ALL"]}
                         for a, b in PAIRS}

    # bird-generalisation diagnostic (random clip split vs leave-birds-out), with the critic's error-removal view
    si = np.isin(y, [classes.index(c) for c in SOFT])
    bg, rs_hit = {}, {}
    for t in (REF, E23):
        X = X_of(t, int(CPU[t]["best_layer_cpu"])); pr = np.zeros(len(y), int)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
            pr[te] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000)).fit(X[tr], y[tr]).predict(X[te])
        h_lbo = hit(t)[0]; h_rs = (pr == y).astype(float); rs_hit[t] = h_rs
        e_lbo, e_rs = h_lbo == 0, h_rs == 0
        bg[t] = dict(random_split={c: float(v) for c, v in zip(classes, rec(h_rs))},
                     leave_birds_out={c: float(v) for c, v in zip(classes, rec(h_lbo))},
                     acc_random_split=float(h_rs.mean()), acc_leave_birds_out=float(h_lbo.mean()),
                     soft_errors_lbo=int((e_lbo & si).sum()), soft_errors_random=int((e_rs & si).sum()),
                     other_errors_lbo=int((e_lbo & ~si).sum()), other_errors_random=int((e_rs & ~si).sum()))
        bg[t]["share_soft_errors_removed"] = 1 - bg[t]["soft_errors_random"] / max(1, bg[t]["soft_errors_lbo"])
        bg[t]["share_other_errors_removed"] = 1 - bg[t]["other_errors_random"] / max(1, bg[t]["other_errors_lbo"])
    g_rs = rs_hit[E23] - rs_hit[REF]; g_lbo = hit(E23)[0] - hit(REF)[0]
    obs = float(g_rs.mean() - g_lbo.mean())
    bs = np.array([g_rs[s].mean() - g_lbo[s].mean() for s in BOOT])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    bg["E23_extra_gain_on_random_split"] = dict(diff=obs, lo=float(lo), hi=float(hi), resolved=bool(lo > 0 or hi < 0),
                                                gain_random=float(g_rs.mean()), gain_lbo=float(g_lbo.mean()))
    res["bird_generalisation"] = bg
    es = {}
    for t in (REF, E23):
        w = hit(t)[1] != y
        es[t] = dict(errors=int(w.sum()), share_soft=float(si[w].mean()))
    res["error_share"] = dict(types=SOFT, share_of_clips=float(si.mean()), models=es)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

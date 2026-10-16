#!/usr/bin/env python
"""Per-call-type recall of every DAPT checkpoint against its parent, and where the call-type errors come from.

Writes the numbers notebooks/08_calltype_errors.ipynb shows (sections 8, 10, 11) to a JSON so a finding can cite
them. Board protocol throughout: cached clip embeddings at each model's reported best 11-class layer; cv_acc
(leave-birds-out StratifiedGroupKFold(5), seed 0, StandardScaler + LogisticRegression); every model's pooled
accuracy is asserted to reproduce its <tag>_calltype.json / aves_variants_calltype.json value.

  per_checkpoint[tag]: recall minus parent's per call type and overall, OBSERVED difference with a 95% bird
      bootstrap (2,000 resamples of the 48 birds, seed 0; same resamples for every row)
  pattern[type]: how many checkpoints are above / below their parent (point estimate, and resolved)
  bird_generalisation: per-type recall under a random clip split (StratifiedKFold(5), seed 0; test birds' other
      clips in training -- a diagnostic, never a score) against leave-birds-out, for aves-base-bio and E23
  error_share: share of each model's errors whose true type is Te/Ne/Th/Tu

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
AVJ = json.loads((ANA / "aves_variants_calltype.json").read_text())["models"]
CACHE = {"aves-base-bio": "ct11_aves_emb.npy"}
# every DAPT checkpoint that is a 15k-update (or longer) endpoint, with its parent; the round-1 intermediate
# checkpoints are the trajectory, read from <tag>_bootstrap.json instead
PARENT = {t: "aves-base-bio" for t in [
    "dapt5e5_step15000", "dapt1e4_step15000", "daptshort_5e5_step2500", "daptlr_1e5_step15000",
    "daptfreeze_5e5_step15000", "daptreplay_5e5_step15000", "daptreplay3_5e5_step15000",
    "daptv3aves_5e5_step15000", "daptv3aves_2e4_step15000", "daptv3aves_2e4s2_step15000",
    "daptv3aves_2e4m8_step15000", "daptv3aves_2e4s2m8_step15000", "daptv4two_2e4_step15000",
    "daptv4two_2e4s2_step15000", "daptv4aveso_2e4_step15000", "daptv4aveso_2e4s2_step15000",
    "daptv4two45k_2e4_step45000"]}
PARENT["daptv4bird_2e4_step15000"] = "birdaves-biox-base"
SOFT = ["Te", "Ne", "Th", "Tu"]


def reported(tag):
    if tag in AVJ:
        return AVJ[tag]["best_11"]["layer"], AVJ[tag]["best_11"]["acc"]
    d = json.loads((ANA / f"{tag}_calltype.json").read_text())
    return d["best_layer11"], d["acc11"]


def main():
    rows = [r for r in collect() if r[3] in KEEP11]
    birds = np.array([r[1].lower() for r in rows])
    tt = np.array([r[3] for r in rows]); classes = sorted(set(tt))
    y = np.array([classes.index(t) for t in tt]); K = len(classes)
    rng = np.random.default_rng(0); ub = np.unique(birds); bidx = {b: np.where(birds == b)[0] for b in ub}
    BOOT = [np.concatenate([bidx[b] for b in rng.choice(ub, len(ub))]) for _ in range(2000)]

    def X_of(tag):
        layer, _ = reported(tag)
        return np.ascontiguousarray(np.load(FEAT / CACHE.get(tag, f"ct11_{tag}_emb.npy"), mmap_mode="r")[:, layer])

    def hit_of(tag):
        layer, acc_rep = reported(tag)
        a, _, P = cv_acc(X_of(tag), y, birds, return_proba=True)
        assert abs(a - acc_rep) < 1e-9, (tag, a, acc_rep)
        return (P.argmax(1) == y).astype(float), P.argmax(1), layer, a

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
        la, hia = np.percentile(ba, [2.5, 97.5])
        out["ALL"] = dict(diff=float(ha.mean() - hb.mean()), lo=float(la), hi=float(hia), resolved=bool(la > 0 or hia < 0))
        return out

    parents = {p: hit_of(p) for p in set(PARENT.values())}
    res = dict(classes=classes, n_clips=int(len(y)), n_birds=int(len(ub)),
               counts={c: int((y == k).sum()) for k, c in enumerate(classes)},
               birds_per_type={c: int(len(set(birds[y == k]))) for k, c in enumerate(classes)},
               protocol="board probe (cv_acc), own reported best layer; bird bootstrap 2000, seed 0; observed differences",
               parents={p: dict(layer=int(v[2]), acc11=v[3]) for p, v in parents.items()},
               per_checkpoint={})
    hits = {}
    for t, p in PARENT.items():
        h, pr, layer, a = hit_of(t)
        hits[t] = (h, pr)
        res["per_checkpoint"][t] = dict(parent=p, layer=int(layer), acc11=a, vs_parent=diff(h, parents[p][0]))
        print(f"{t:34s} L{layer} {a:.4f}  vs {p}: {res['per_checkpoint'][t]['vs_parent']['ALL']['diff']:+.4f}", flush=True)
    pat = {}
    for c in classes + ["ALL"]:
        v = [res["per_checkpoint"][t]["vs_parent"][c] for t in PARENT]
        pat[c] = dict(n=len(v), above_point=sum(x["diff"] > 0 for x in v), below_point=sum(x["diff"] < 0 for x in v),
                      above_resolved=sum(x["resolved"] and x["diff"] > 0 for x in v),
                      below_resolved=sum(x["resolved"] and x["diff"] < 0 for x in v),
                      median_diff=float(np.median([x["diff"] for x in v])))
    res["pattern"] = pat
    # seed pairs: does the per-type direction replicate within a recipe?
    pairs = [("daptv4two_2e4_step15000", "daptv4two_2e4s2_step15000"),
             ("daptv4aveso_2e4_step15000", "daptv4aveso_2e4s2_step15000"),
             ("daptv3aves_2e4_step15000", "daptv3aves_2e4s2_step15000"),
             ("daptv3aves_2e4m8_step15000", "daptv3aves_2e4s2m8_step15000")]
    res["seed_pairs"] = {f"{a} | {b}": {c: [res["per_checkpoint"][a]["vs_parent"][c]["diff"],
                                             res["per_checkpoint"][b]["vs_parent"][c]["diff"]] for c in classes + ["ALL"]}
                         for a, b in pairs}
    # bird generalisation diagnostic
    bg = {}
    for t in ("aves-base-bio", "daptv4two_2e4s2_step15000"):
        X = X_of(t); pr = np.zeros(len(y), int)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
            pr[te] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000)).fit(X[tr], y[tr]).predict(X[te])
        h_lbo = parents[t][0] if t in parents else hits[t][0]
        bg[t] = dict(random_split={c: float(v) for c, v in zip(classes, rec((pr == y).astype(float)))},
                     leave_birds_out={c: float(v) for c, v in zip(classes, rec(h_lbo))},
                     acc_random_split=float((pr == y).mean()), acc_leave_birds_out=float(h_lbo.mean()))
    res["bird_generalisation"] = bg
    # where the errors are
    si = [classes.index(c) for c in SOFT]
    es = {}
    for t in ("aves-base-bio", "daptv4two_2e4s2_step15000"):
        pr = parents[t][1] if t in parents else hits[t][1]
        w = pr != y
        es[t] = dict(errors=int(w.sum()), share_soft=float(np.isin(y[w], si).mean()))
    res["error_share"] = dict(types=SOFT, share_of_clips=float(np.isin(y, si).mean()), models=es)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

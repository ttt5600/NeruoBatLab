#!/usr/bin/env python
"""E16 label-free half: does time expansion give bat embeddings more structure that recurs across individuals?

    python3 e16_labelfree.py

E11's held-out stability (calltype_labelfree.ward / transfer_ari, unchanged) with BEANS bats individuals as the
held-out groups instead of zebra finch birds. Protocol pre-registered in harness/experiments.yaml E16
(labelfree_protocol) before any score was computed:
  clips   BEANS bats valid + test (4,000 clips, 10 individuals), the features cached by beans_probe.py
  groups  B = 50 random halvings of the 10 individuals (5 vs 5), seed 0, identical for every arm
  layer   3; per-dimension standardisation; k = 30 primary, 11 secondary
  arms    run16_compute4x and aves-base-bio at plain, @te2, @te4, @lpte4
  readout score(arm) - score(plain) per model, paired over halvings, 2.5-97.5 percentile over halvings
  check   a column-shuffled null for one arm must sit near 0
Labels only define the halvings; the score never sees them. Writes analysis/e16_labelfree.json.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calltype_labelfree import ward, transfer_ari, summary          # noqa: E402,F401  (E11's functions)
from beans_probe import BEANS, FEAT, ANA                             # noqa: E402

LAYER, KS, B, SEED = 3, (30, 11), 50, 0
MODELS = ["run16_compute4x", "aves-base-bio"]
SUFFIXES = ["", "@te2", "@te4", "@lpte4"]
OUT = ANA / "e16_labelfree.json"


def individuals():
    cfg = {d["name"]: d for d in yaml.safe_load(open(BEANS / "datasets.yml"))}["bats"]
    return np.concatenate([pd.read_csv(BEANS / cfg[f"{s}_data"])["label"].astype(str).to_numpy()
                           for s in ("valid", "test")])


def features(tag):
    X = np.concatenate([np.load(FEAT / f"bats_{s}_{tag}.npy", mmap_mode="r")[:, LAYER].astype(np.float64)
                        for s in ("valid", "test")])
    return StandardScaler().fit_transform(X)


def halvings(groups):
    ug = np.unique(groups); rng = np.random.default_rng(SEED)
    return [np.isin(groups, ug[rng.permutation(len(ug))[: len(ug) // 2]]) for _ in range(B)]


def score(X, sp):
    return {k: np.array([transfer_ari(X[m], X[~m], k) for m in sp]) for k in KS}


def main():
    g = individuals()
    sp = halvings(g)
    print(f"[e16 label-free] {len(g)} clips, {len(np.unique(g))} individuals, {B} halvings, layer {LAYER}", flush=True)
    raw, out = {}, dict(protocol=__doc__.split("Protocol")[1].split("Labels")[0].strip(), n_clips=int(len(g)),
                        individuals=int(len(np.unique(g))), layer=LAYER, halvings=B, seed=SEED, models={})
    for m in MODELS:
        for suf in SUFFIXES:
            tag = m + suf
            X = features(tag)
            assert len(X) == len(g), f"{tag}: {len(X)} rows vs {len(g)} labels"
            raw[tag] = score(X, sp)
            out["models"][tag] = {f"k{k}": summary(raw[tag][k]) for k in KS}
            print(f"  {tag:24s} " + "  ".join(f"k{k} {raw[tag][k].mean():.4f}" for k in KS), flush=True)
        for suf in SUFFIXES[1:]:
            tag = m + suf
            out["models"][tag]["vs_plain"] = {f"k{k}": summary(raw[tag][k] - raw[m][k]) for k in KS}
            v = raw[tag][KS[0]] - raw[m][KS[0]]
            lo, hi = np.percentile(v, [2.5, 97.5])
            print(f"  {tag:24s} - plain  k{KS[0]} {v.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]"
                  f"{'  *' if lo > 0 or hi < 0 else ''}", flush=True)
    rng = np.random.default_rng(1)
    Xn = features(MODELS[0]).copy()
    for j in range(Xn.shape[1]):
        Xn[:, j] = Xn[rng.permutation(len(Xn)), j]
    null = score(Xn, sp)
    out["null_column_shuffled"] = {"tag": MODELS[0], **{f"k{k}": summary(null[k]) for k in KS}}
    print(f"  null (column-shuffled {MODELS[0]}) " + "  ".join(f"k{k} {null[k].mean():.4f}" for k in KS), flush=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

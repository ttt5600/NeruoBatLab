#!/usr/bin/env python
"""Label-free score for every model: do its clusters reappear in birds it has not seen? (E11)

    python3 calltype_labelfree.py              # every cached ct11 embedding
    python3 calltype_labelfree.py --only TAG   # add or refresh one model (merges into the JSON)

Finding 050's bird-held-out stability, the label-free criterion that ranked encoders the right way
(compactness criteria ranked them backwards). For each of B random halvings of the 48 birds: Ward-
cluster each half into k groups, carry half A's centroids onto half B's clips, and score that against
B's own clustering with ARI (chance-corrected, so a larger k earns nothing for free). Every model sees
the SAME halvings, so model differences are paired over splits.

Choices fixed before any score was read (E11, 2026-10-12):
  layer   3 for 12-layer models, 6 for 24-layer models (no layer is picked on labels or on the score)
  k       30 primary (inside 050's k ~13-140 band where this criterion tracked accuracy); 11 secondary
  B       50 halvings, seed 0; per-dimension standardisation, as in 050
  interval  2.5-97.5 percentile over halvings (the halvings overlap, so this is a resampling spread,
          not an independent-sample CI); paired difference vs aves-base-bio the same way
  check   Spearman rho, across models, between the score and 11-class call type at the same layer
          and at each model's best layer; a column-shuffled null for one model must sit near 0
Writes analysis/labelfree_stability.json.
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import calltype11 as C11                                                  # noqa: E402

A = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
OUT = A / "labelfree_stability.json"
KS, B, SEED, REF = (30, 11), 50, 0, "aves-base-bio"
EMB_ALIAS = {"aves-base-bio": "aves"}          # tag -> ct11 file stem where they differ


def tags():
    stems = sorted(p.name[5:-8] for p in C11.FEAT.glob("ct11_*_emb.npy"))
    inv = {v: k for k, v in EMB_ALIAS.items()}
    return [inv.get(s, s) for s in stems]


def emb(tag):
    return np.load(C11.FEAT / f"ct11_{EMB_ALIAS.get(tag, tag)}_emb.npy", mmap_mode="r")


def calltype_by_layer(tag):
    """Per-layer 11-class accuracy from the stored call-type JSONs (never recomputed here)."""
    f = A / f"{tag}_calltype.json"
    if f.exists():
        return json.load(open(f))["per_layer_acc11"]
    av = json.load(open(A / "aves_variants_calltype.json"))
    rec = av.get(tag) or av.get("models", {}).get(tag)
    return rec["per_layer_acc_11"] if rec else None


def ward(X, k):
    return fcluster(linkage(X, method="ward"), t=k, criterion="maxclust") - 1


def transfer_ari(Xa, Xb, k):
    la, lb = ward(Xa, k), ward(Xb, k)
    C = np.vstack([Xa[la == c].mean(0) for c in range(k)])
    pred = ((Xb ** 2).sum(1)[:, None] - 2 * Xb @ C.T + (C ** 2).sum(1)[None]).argmin(1)
    return ARI(lb, pred)


def splits(birds):
    ub = np.unique(birds); rng = np.random.default_rng(SEED)
    out = []
    for _ in range(B):
        a = rng.permutation(len(ub))[: len(ub) // 2]
        out.append(np.isin(birds, ub[a]))
    return out


def score(X, sp):
    return {k: np.array([transfer_ari(X[m], X[~m], k) for m in sp]) for k in KS}


def summary(v):
    lo, hi = np.percentile(v, [2.5, 97.5])
    return dict(mean=float(v.mean()), lo=float(lo), hi=float(hi))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only"); a = ap.parse_args()
    rows = [r for r in C11.collect() if r[3] in C11.KEEP11]
    birds = np.array([r[1].lower() for r in rows])
    sp = splits(birds)
    out = json.load(open(OUT)) if OUT.exists() else {}
    out.update(settings=dict(ks=list(KS), B=B, seed=SEED, ref=REF, n_clips=len(rows), n_birds=int(len(np.unique(birds))),
                             layer_rule="3 for 12-layer models, 6 for 24-layer models", primary_k=KS[0]))
    models = out.setdefault("models", {})
    per_split = {}
    todo = [a.only] if a.only else tags()
    for tag in sorted(set(todo) | {REF}):
        E = emb(tag); layer = 3 if E.shape[1] <= 13 else 6
        X = StandardScaler().fit_transform(np.asarray(E[:, layer], dtype=np.float64))
        s = score(X, sp); per_split[tag] = s
        ct = calltype_by_layer(tag)
        rec = dict(layer=layer, **{f"k{k}": summary(s[k]) for k in KS},
                   calltype_same_layer=ct[layer] if ct else None, calltype_best=max(ct) if ct else None)
        if tag in models and "k30" in models[tag]:
            assert abs(models[tag]["k30"]["mean"] - rec["k30"]["mean"]) < 1e-9, f"{tag}: k30 not reproduced"
        models[tag] = rec
        print(f"{tag:32s} L{layer}  k30 {rec['k30']['mean']:.4f} [{rec['k30']['lo']:.4f}, {rec['k30']['hi']:.4f}]"
              f"  k11 {rec['k11']['mean']:.4f}", flush=True)
    for tag in per_split:
        if tag == REF:
            continue
        for k in KS:
            d = per_split[tag][k] - per_split[REF][k]
            lo, hi = np.percentile(d, [2.5, 97.5])
            models[tag][f"k{k}_minus_ref"] = dict(diff=float(d.mean()), lo=float(lo), hi=float(hi),
                                                 resolved=bool(lo > 0 or hi < 0))
    if not a.only:                               # null + rank check need the full set
        E = emb("run16_compute4x"); X = StandardScaler().fit_transform(np.asarray(E[:, 3], dtype=np.float64))
        rng = np.random.default_rng(SEED)
        Xn = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
        out["null_run16_shuffled"] = {f"k{k}": summary(v) for k, v in score(Xn, sp[:10]).items()}
        ok = [t for t in models if models[t]["calltype_same_layer"] is not None]
        rank = {}
        for k in KS:
            sc = [models[t][f"k{k}"]["mean"] for t in ok]
            for col in ("calltype_same_layer", "calltype_best"):
                r, p = spearmanr(sc, [models[t][col] for t in ok])
                rank[f"k{k}_vs_{col}"] = dict(rho=float(r), p=float(p), n=len(ok))
        out["rank_check"] = rank
        print("null (run16 L3, columns shuffled, 10 splits):", out["null_run16_shuffled"])
        print(json.dumps(rank, indent=1))
    OUT.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

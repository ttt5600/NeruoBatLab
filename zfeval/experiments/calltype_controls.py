#!/usr/bin/env python
"""Is the call-type comparison able to see a difference at all, and are the models just the same model?

calltype_abx.py and calltype_cluster_by_type.py put run16 and AVES within seed noise of each other.
Before reading that as "tied", four checks:

  1. the ruler: ABX error with SHUFFLED labels (must be ~0.5) and for a plain log-mel spectrogram,
     mean-pooled like the encoders (the floor every encoder should clear by a lot).
  2. are the spaces the same? linear CKA (on standardised features) and 10-nearest-neighbour overlap
     between every pair of models, with run16 vs its own seed 2 as the "same recipe" reference.
  3. do they fail on the SAME clips? out-of-fold probe predictions (the board's protocol, asserted to
     reproduce its accuracy); P(model B wrong | model A wrong) against B's base error rate.
  4. DAPT (AVES continued on zebra finch, replay x3, the one DAPT checkpoint with call-type
     embeddings) through the same ABX as everything else.

  python calltype_controls.py   -> analysis/calltype_controls.json
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import calltype11 as C11                                                  # noqa: E402
from aves_calltype import load_audio, cv_acc                              # noqa: E402
from calltype_abx import abx_bird_arrays, overall, pair_matrix            # noqa: E402

ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
OUT = ANA / "calltype_controls.json"
ABX = json.load(open(ANA / "calltype_abx.json"))
PC = json.load(open(ANA / "calltype_per_class.json"))
FILES = {"run11": "ct11_run11_emb.npy", "run16": "ct11_run16_compute4x_emb.npy",
         "run16 seed 2": "ct11_run16_seed2_emb.npy", "run19": "ct11_run19_avesteacher_emb.npy",
         "AVES bio": "ct11_aves_emb.npy", "AVES core": "ct11_aves-base-core_emb.npy",
         "AVES all": "ct11_aves-base-all_emb.npy", "DAPT replay x3": "ct11_daptreplay3_5e5_step15000_emb.npy"}


def std(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-8)


def cka(X, Y):
    X, Y = std(X), std(Y)
    return float(np.linalg.norm(Y.T @ X) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def knn(X, k=10):
    Z = std(X); Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    S = Z @ Z.T; np.fill_diagonal(S, -np.inf)
    return np.argpartition(-S, k, axis=1)[:, :k]


def logmel(paths):
    import torch, torchaudio
    mel = torchaudio.transforms.MelSpectrogram(16000, n_fft=400, hop_length=160, n_mels=64)
    out = np.zeros((len(paths), 64), dtype=np.float32)
    with torch.no_grad():
        for i, p in enumerate(paths):
            w = load_audio(p)
            if w.shape[-1] < 400:
                w = torch.nn.functional.pad(w, (0, 400 - w.shape[-1]))
            out[i] = torch.log(mel(w)[0] + 1e-6).mean(1).numpy()
    return out


def main():
    t0 = time.time()
    rows = [r for r in C11.collect() if r[3] in C11.KEEP11]
    paths = [r[0] for r in rows]
    birds = np.array([r[1].lower() for r in rows])
    tt = np.array([r[3] for r in rows])
    classes = sorted(set(tt))
    y = np.array([classes.index(t) for t in tt])
    K, ub = len(classes), np.unique(birds)
    assert len(y) == 3412 and K == 11
    out = {}

    # layer per model: the one calltype_abx.py chose out of fold (both halves agreed for every model)
    L = {m: ABX["models"][m]["layers_oof"]["half_a"] for m in FILES if m in ABX["models"]}
    for m in L:
        assert L[m] == ABX["models"][m]["layers_oof"]["half_b"], m
    E = {m: np.asarray(np.load(C11.FEAT / f, mmap_mode="r")[:, L.get(m, 0)], dtype=np.float32)
         for m, f in FILES.items() if m in L}

    # ---- 4. DAPT: pick its layer out of fold exactly as calltype_abx.py does
    half = np.random.default_rng(0).permutation(len(ub)) < len(ub) // 2
    D = np.load(C11.FEAT / FILES["DAPT replay x3"], mmap_mode="r")
    per_layer = [abx_bird_arrays(np.asarray(D[:, l], dtype=np.float32), y, birds, ub, K) for l in range(D.shape[1])]
    Lf = {h: int(np.argmin([overall(c[half != h]) for c, _ in per_layer])) for h in (True, False)}
    cross = np.where(half[:, None, None], per_layer[Lf[True]][0], per_layer[Lf[False]][0])
    Mc = pair_matrix(cross)
    out["dapt_abx"] = dict(layers_oof=Lf, cross_error=overall(cross),
                           per_type={c: float(np.nanmean(np.delete(Mc[i], i))) for i, c in enumerate(classes)})
    L["DAPT replay x3"] = Lf[True]
    E["DAPT replay x3"] = np.asarray(D[:, Lf[True]], dtype=np.float32)
    print(f"[DAPT] layers {Lf}  cross-bird ABX error {out['dapt_abx']['cross_error']:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- 1. the ruler
    rng = np.random.default_rng(0)
    shuf = abx_bird_arrays(E["run16"], rng.permutation(y), birds, ub, K)[0]
    spec = logmel(paths)
    sc, _ = abx_bird_arrays(spec, y, birds, ub, K)
    Ms = pair_matrix(sc)
    out["ruler"] = dict(run16_shuffled_labels=overall(shuf), logmel64=overall(sc),
                        logmel64_per_type={c: float(np.nanmean(np.delete(Ms[i], i))) for i, c in enumerate(classes)})
    print(f"[ruler] shuffled labels {overall(shuf):.4f}   log-mel {overall(sc):.4f}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- 2. are the spaces the same?
    names = list(E)
    nn = {m: knn(E[m]) for m in names}
    sim = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ov = float(np.mean([len(set(nn[a][j]) & set(nn[b][j])) / 10 for j in range(len(y))]))
            sim[f"{a} | {b}"] = dict(cka=cka(E[a], E[b]), knn10_overlap=ov)
    out["similarity"] = dict(layers=L, pairs=sim)
    print(f"[similarity] done ({time.time()-t0:.0f}s)", flush=True)

    # ---- 3. do they fail on the same clips? (board protocol: each model's own board layer)
    board = {"run16": ("ct11_run16_compute4x_emb.npy", "run16"), "run16 seed 2": ("ct11_run16_seed2_emb.npy", "run16 seed 2"),
             "run19": ("ct11_run19_avesteacher_emb.npy", "run19"), "AVES bio": ("ct11_aves_emb.npy", "AVES bio"),
             "AVES core": ("ct11_aves-base-core_emb.npy", "AVES core"), "DAPT replay x3": ("ct11_daptreplay3_5e5_step15000_emb.npy", "DAPT replay x3")}
    wrong, prob = {}, {}
    for m, (f, lab) in board.items():
        r = PC["models"][lab]
        acc, _, P = cv_acc(np.ascontiguousarray(np.load(C11.FEAT / f, mmap_mode="r")[:, r["layer"]]), y, birds, return_proba=True)
        assert abs(acc - r["acc11"]) < 1e-9, f"{m}: {acc} != board {r['acc11']}"
        wrong[m], prob[m] = P.argmax(1) != y, P

    # ---- 5. complementary? mean of two models' probabilities vs the better member, and vs a
    # same-recipe pair (run16 + seed 2) so a gain from averaging alone is visible
    idx = {g: np.where(birds == g)[0] for g in ub}
    sels = [np.concatenate([idx[g] for g in rng.choice(ub, len(ub))]) for _ in range(2000)]
    ens = {}
    for a, b in [("run16", "run16 seed 2"), ("AVES bio", "AVES core"), ("run16", "AVES bio"),
                 ("run16", "AVES core"), ("run19", "AVES core"),
                 ("DAPT replay x3", "AVES bio"), ("DAPT replay x3", "AVES core"), ("DAPT replay x3", "run16")]:
        hit_e = (prob[a] + prob[b]).argmax(1) == y
        best = max((a, b), key=lambda m: (~wrong[m]).mean())
        hit_b = ~wrong[best]
        d = np.array([hit_e[s].mean() - hit_b[s].mean() for s in sels])
        lo, hi = np.percentile(d, [2.5, 97.5])
        ens[f"{a} + {b}"] = dict(acc=float(hit_e.mean()), better_member=best, member_acc=float(hit_b.mean()),
                                 diff=float(hit_e.mean() - hit_b.mean()), lo=float(lo), hi=float(hi),
                                 resolved=bool(lo > 0 or hi < 0))
    out["ensemble"] = ens
    ov = {}
    for a in wrong:
        for b in wrong:
            if a < b:
                ov[f"{a} | {b}"] = dict(err_a=float(wrong[a].mean()), err_b=float(wrong[b].mean()),
                                        p_b_wrong_given_a_wrong=float(wrong[b][wrong[a]].mean()),
                                        jaccard=float((wrong[a] & wrong[b]).sum() / (wrong[a] | wrong[b]).sum()))
    allw = np.all(np.stack(list(wrong.values())), axis=0)
    anyw = np.any(np.stack(list(wrong.values())), axis=0)
    out["error_overlap"] = dict(pairs=ov, n_wrong_by_all=int(allw.sum()), n_wrong_by_any=int(anyw.sum()),
                                wrong_by_all_by_type={c: int(allw[y == i].sum()) for i, c in enumerate(classes)},
                                wrong_by_all_birds=int(len(set(birds[allw]))))
    OUT.write_text(json.dumps(out, indent=2))

    print("\nrepresentation similarity (CKA, 10-NN overlap)")
    for k, v in sim.items():
        if "run16 |" in k or "| run16" in k or k.startswith("AVES bio | AVES") or "DAPT" in k:
            print(f"  {k:<32} CKA {v['cka']:.3f}   knn {v['knn10_overlap']:.3f}")
    print("\nerror overlap (probe, board protocol)")
    for k, v in ov.items():
        print(f"  {k:<32} err {v['err_a']:.3f}/{v['err_b']:.3f}  P(B wrong | A wrong) {v['p_b_wrong_given_a_wrong']:.3f}  jaccard {v['jaccard']:.3f}")
    print(f"  wrong by ALL {len(wrong)} models: {allw.sum()} clips ({out['error_overlap']['wrong_by_all_birds']} birds); "
          f"by any: {anyw.sum()};  by type {out['error_overlap']['wrong_by_all_by_type']}")
    print("\nensemble (mean of probabilities) vs its better member, bird bootstrap")
    for k, v in ens.items():
        print(f"  {k:<26} {v['acc']:.4f} vs {v['better_member']} {v['member_acc']:.4f}: {v['diff']:+.4f} "
              f"[{v['lo']:+.4f}, {v['hi']:+.4f}]{'  *' if v['resolved'] else ''}")
    print(f"\nDAPT per-type ABX: " + " ".join(f"{c}:{v:.3f}" for c, v in out['dapt_abx']['per_type'].items()))
    print(f"-> {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

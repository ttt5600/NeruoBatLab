#!/usr/bin/env python
"""Is the call-type information MISSING from run11, or just in the wrong basis?

Frozen linear probe on 11-class call type, leave-birds-out: run11 L3 0.8118, AVES L3 0.8453
(analysis/calltype11.json). Two readings of that gap, and they have opposite consequences for the
lab:

  (a) run11's representation does not CONTAIN what separates these call types -- colony pretraining
      spent its capacity elsewhere, and no readout will recover it.
  (b) run11's representation contains it but not along directions a linear probe on that basis can
      reach, and AVES's basis happens to expose it.

A learned map run11 L3 -> AVES L3, fitted on TRAINING birds only and then probed for call type on
held-out birds, separates them. If the mapped features probe near 0.845 the information was there
(b). If they stay near 0.812 it was not (a).

Two things must be said before any number is read, because they are both structural:

  1. A LINEAR map cannot help, by construction. Multinomial logistic regression is invariant under
     an invertible linear reparameterisation of its inputs, so probing X @ W.T asks for a linear
     separator inside the row space of W. If W has full rank 768 the hypothesis class is unchanged
     and the accuracy can only move by however much the L2 penalty and the standardiser see a
     different geometry. The linear arm is therefore a NULL ARM -- it is run here precisely so the
     reader can see it come out at run11's own number, which is the evidence that the pipeline is
     not accidentally leaking AVES.
  2. No map can ADD information. A map is a function of run11 L3 alone; every arm here is a probe on
     a deterministic transform of run11 L3. So reading (b) is not "distillation recovered what was
     lost", it is "the information was already decodable, non-linearly, and the AVES targets were a
     useful thing to point a nonlinear map at". That makes one control mandatory:

       mlp_direct   an MLP of the SAME shape (768 -> 512 -> 11) trained directly on call-type labels
                    on raw run11 L3. If it also reaches ~0.845, the MLP-map gain is nonlinearity and
                    has nothing to do with AVES. If it does not, the AVES targets are supplying the
                    training signal that the 11 labels alone do not.

Arms:
  raw_run11 / raw_aves            frozen probes -- reproduction guard against 0.8118 / 0.8453
  map_linear                      ridge run11 L3 -> AVES L3, alpha picked on a bird-disjoint inner
                                  split by mean out-of-sample R^2; probe the mapped features
  map_mlp                         768 -> 512 (GELU) -> 768, MSE on standardised AVES L3, epoch
                                  picked on inner-split R^2; probe the mapped features
  mlp_direct                      the capacity control above
  map_linear_rev / map_mlp_rev    the same maps in the other direction (AVES L3 -> run11 L3), probed
                                  for call type. Asymmetry is informative: if AVES predicts run11
                                  better than the reverse, run11's representation is closer to a
                                  coarsening of AVES's than the other way round.
  concat_mapped                   [run11 L3, mapped] 1536-d, in case the map is complementary rather
                                  than a replacement.

Protocol: cohort and split are bit-identical to calltype11 / aves_calltype -- 3412 clips, 11 classes,
48 birds, majority 0.17966, StratifiedGroupKFold(5, shuffle=True, random_state=0) grouped by bird.
The map is refitted inside every outer fold on that fold's training birds only; test-fold clips are
transformed by it but never contribute to fitting it, to the alpha choice, or to the epoch choice.
Probe is StandardScaler + LogisticRegression(max_iter=4000), the same estimator cv_acc uses.
R^2 per AVES dimension is reported out-of-sample (pooled over the five test folds) and in-sample, so
the reader can see how good the map actually is rather than inferring it from the probe.
Every accuracy comparison is bootstrapped over the 48 BIRDS with aves_calltype.bird_bootstrap.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from calltype11 import collect, KEEP11                                        # noqa: E402
from aves_calltype import bird_bootstrap                                      # noqa: E402

FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
OUT = ANA / "distillation.json"
LAYER = 3                             # each encoder's best frozen layer on this task
NFOLD, SEED = 5, 0
N_CLIPS, N_CLASSES, N_BIRDS, MAJORITY = 3412, 11, 48, 0.17966002344665885
PROBE_REF = {"run11": 0.8118405627198124, "aves": 0.8452520515826495}
ALPHAS = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5]


def save(key, payload):
    """Write into OUT (overridable with --out so this can run alongside another experiment that is
    already appending to analysis/distillation.json; the pieces are merged afterwards)."""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d = json.loads(OUT.read_text()) if OUT.exists() else {}
    d.setdefault(key, {}).update(payload)
    OUT.write_text(json.dumps(d, indent=2))


def load_cohort():
    rows = [r for r in collect() if r[3] in KEEP11]
    birds = np.array([r[1].lower() for r in rows])
    tt = np.array([r[3] for r in rows])
    classes = sorted(set(tt))
    y = np.array([classes.index(t) for t in tt])
    maj = float(np.bincount(y).max() / len(y))
    assert len(y) == N_CLIPS and len(classes) == N_CLASSES and len(set(birds)) == N_BIRDS
    assert abs(maj - MAJORITY) < 1e-9
    return y, birds, classes, maj


def folds_of(y, birds):
    from sklearn.model_selection import StratifiedGroupKFold
    cv = StratifiedGroupKFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
    return list(cv.split(np.zeros((len(y), 1)), y, birds))


def inner_split(idx, y, birds):
    from sklearn.model_selection import StratifiedGroupKFold
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    a, b = next(iter(cv.split(np.zeros((len(idx), 1)), y[idx], birds[idx])))
    return idx[a], idx[b]


def probe_oof(X, y, birds, folds):
    """StandardScaler + LogisticRegression per fold -- the estimator cv_acc uses, unchanged."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    P = np.zeros((len(y), len(np.unique(y))))
    for tr, te in folds:
        est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000))
        est.fit(X[tr], y[tr])
        P[te] = est.predict_proba(X[te])
    return float((P.argmax(1) == y).mean()), P


def probe_oof_prefit(Xs, y, folds):
    """Same, but each fold has its OWN feature matrix (the map differs per fold)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    P = np.zeros((len(y), len(np.unique(y))))
    for f, (tr, te) in enumerate(folds):
        X = Xs[f]
        est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000))
        est.fit(X[tr], y[tr])
        P[te] = est.predict_proba(X[te])
    return float((P.argmax(1) == y).mean()), P


# ------------------------------------------------------------------ maps
def zstats(X):
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    return mu, sd


def fit_ridge_map(Xtr, Ttr, Xva, Tva, log):
    """Ridge with alpha chosen by mean out-of-sample R^2 on a bird-disjoint inner split."""
    from sklearn.linear_model import Ridge
    best = (None, -np.inf, None)
    for al in ALPHAS:
        r = Ridge(alpha=al).fit(Xtr, Ttr)
        Pv = r.predict(Xva)
        r2 = 1.0 - ((Tva - Pv) ** 2).sum(0) / ((Tva - Tva.mean(0)) ** 2).sum(0).clip(1e-12)
        m = float(np.mean(r2))
        log(f"      alpha {al:>9.3g}  inner-val mean R^2 {m:+.4f}")
        if m > best[1]:
            best = (al, m, None)
    al = best[0]
    return Ridge(alpha=al).fit(np.vstack([Xtr, Xva]), np.vstack([Ttr, Tva])), al, best[1]


class MapMLP(nn.Module):
    def __init__(self, d_in=768, h=512, d_out=768):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.GELU(), nn.Linear(h, d_out))

    def forward(self, x):
        return self.net(x)


def fit_mlp_map(Xtr, Ttr, Xva, Tva, device, hp, log):
    """MSE regression to standardised targets; epoch count picked on inner-val mean R^2."""
    def run(Xa, Ta, n_epochs, Xb=None, Tb=None):
        torch.manual_seed(hp["seed"])
        m = MapMLP(Xa.shape[1], hp["hidden"], Ta.shape[1]).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["epochs"])
        xa = torch.from_numpy(Xa).to(device)
        ta = torch.from_numpy(Ta).to(device)
        xb = None if Xb is None else torch.from_numpy(Xb).to(device)
        rng = np.random.default_rng(hp["seed"])
        curve = []
        for ep in range(n_epochs):
            m.train()
            perm = torch.from_numpy(rng.permutation(len(Xa))).to(device)
            for s in range(0, len(Xa), hp["batch"]):
                b = perm[s:s + hp["batch"]]
                loss = F.mse_loss(m(xa[b]), ta[b])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sch.step()
            if xb is not None:
                m.eval()
                with torch.no_grad():
                    Pv = m(xb).cpu().numpy()
                r2 = 1.0 - ((Tb - Pv) ** 2).sum(0) / ((Tb - Tb.mean(0)) ** 2).sum(0).clip(1e-12)
                curve.append(float(np.mean(r2)))
        return m, curve

    _, curve = run(Xtr, Ttr, hp["epochs"], Xva, Tva)
    e = int(np.argmax(curve))
    log(f"      MLP map: best epoch {e} inner-val mean R^2 {curve[e]:+.4f} "
        f"(final {curve[-1]:+.4f})")
    m, _ = run(np.vstack([Xtr, Xva]), np.vstack([Ttr, Tva]), e + 1)
    m.eval()
    return m, e, curve[e]


def mlp_direct(Xtr, ytr, Xva, yva, Xall, device, hp, log):
    """Capacity control: MLP 768 -> hidden -> 11 on raw features, trained on the labels."""
    def run(Xa, ya, n_epochs, Xb=None, yb=None):
        torch.manual_seed(hp["seed"])
        m = MapMLP(Xa.shape[1], hp["hidden"], N_CLASSES).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["epochs"])
        xa = torch.from_numpy(Xa).to(device)
        ya_t = torch.from_numpy(ya).to(device)
        xb = None if Xb is None else torch.from_numpy(Xb).to(device)
        rng = np.random.default_rng(hp["seed"])
        curve = []
        for ep in range(n_epochs):
            m.train()
            perm = torch.from_numpy(rng.permutation(len(Xa))).to(device)
            for s in range(0, len(Xa), hp["batch"]):
                b = perm[s:s + hp["batch"]]
                loss = F.cross_entropy(m(xa[b]), ya_t[b])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sch.step()
            if xb is not None:
                m.eval()
                with torch.no_grad():
                    curve.append(float((m(xb).argmax(1).cpu().numpy() == yb).mean()))
        return m, curve

    _, curve = run(Xtr, ytr, hp["epochs"], Xva, yva)
    e = int(np.argmax(curve))
    log(f"      mlp_direct: best epoch {e} inner-val acc {curve[e]:.4f} (final {curve[-1]:.4f})")
    m, _ = run(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]), e + 1)
    m.eval()
    with torch.no_grad():
        P = torch.softmax(m(torch.from_numpy(Xall).to(device)), -1).cpu().numpy()
    del m
    gc.collect()
    return P, e, curve[e]


def r2_per_dim(T, P):
    return 1.0 - ((T - P) ** 2).sum(0) / ((T - T.mean(0)) ** 2).sum(0).clip(1e-12)


def r2_summary(r2):
    q = np.percentile(r2, [5, 25, 50, 75, 95])
    return dict(mean=float(np.mean(r2)), median=float(q[2]), p5=float(q[0]), p25=float(q[1]),
                p75=float(q[3]), p95=float(q[4]), min=float(r2.min()), max=float(r2.max()),
                n_dims_above_0=int((r2 > 0).sum()), n_dims_above_0p5=int((r2 > 0.5).sum()),
                per_dim=[float(x) for x in r2])



# ------------------------------------------------------------------ interpretation
def interpretation(res, boot):
    """State reading (a) vs reading (b) strictly.

    The trap this function exists to avoid: a bird-level bootstrap over 48 birds on a 3412-clip
    cohort has intervals about +-0.025 wide, so "not distinguishable from AVES" happens to arms that
    are 0.020 BELOW AVES. That is a statement about power, not about equivalence, and it must not be
    read as "the gap is closed". An arm only supports reading (b) if it is distinguishably ABOVE
    run11 *and* not distinguishable from AVES; it supports reading (a) if it is not distinguishable
    from run11. Anything else is inconclusive and is labelled as such.

    The headline arm is map_mlp -- the brief asks about probing the MAPPED run11 features.
    concat_map_mlp is reported beside it but is a different object: [run11 L3 | mapped], which still
    needs run11's own 768 dims at inference and is not "run11 in the AVES basis".
    """
    run11, aves = res["raw_run11"]["acc"], res["raw_aves"]["acc"]
    gap = aves - run11
    out = dict(run11_acc=run11, aves_acc=aves, gap=gap, arms={},
               interval_width_note=(
                   "the bird bootstrap interval on this cohort is roughly +-0.025 wide, so "
                   "'not_distinguishable from AVES' is compatible with being 0.02 below it; it is a "
                   "power statement, not an equivalence claim"))
    for k in [k for k in res if k.startswith(("map_", "concat_map_", "mlp_direct"))]:
        if not isinstance(res[k], dict) or "acc" not in res[k]:
            continue
        vr = boot.get(f"{k}__vs__raw_run11", {})
        va = boot.get(f"{k}__vs__raw_aves", {})
        if k.endswith("_rev"):
            # AVES -> run11: the (a)/(b) labels are about run11's content and do not apply. What
            # this arm measures is how much accuracy a map of this quality DESTROYS: it starts from
            # AVES's own features, so anything below raw_aves is pure map loss and is the ceiling on
            # what the forward arm could ever have carried.
            out["arms"][k] = dict(acc=res[k]["acc"], vs_run11=vr, vs_aves=va,
                                  accuracy_lost_by_the_map=float(aves - res[k]["acc"]),
                                  fraction_of_gap_closed=None,
                                  verdict="map_lossiness_diagnostic")
            continue
        above_run11 = vr.get("verdict") == "a_better"
        below_aves = va.get("verdict") == "b_better"
        if above_run11 and not below_aves:
            verdict = "supports_b_information_present_but_not_linearly_accessible"
        elif not above_run11 and below_aves:
            verdict = "supports_a_information_absent_from_run11_L3"
        elif above_run11 and below_aves:
            verdict = "partial_gain_still_below_aves"
        else:
            verdict = "inconclusive_underpowered"
        out["arms"][k] = dict(acc=res[k]["acc"],
                              fraction_of_gap_closed=float((res[k]["acc"] - run11) / gap)
                              if gap else None,
                              vs_run11=vr, vs_aves=va, verdict=verdict)
    head = out["arms"].get("map_mlp")
    lin = out["arms"].get("map_linear")
    cc = out["arms"].get("concat_map_mlp")
    ctrl = out["arms"].get("mlp_direct")
    out["headline_arm"] = "map_mlp"
    out["reading"] = (
        f"map_mlp {head['acc']:.4f} vs run11 {run11:.4f} vs AVES {aves:.4f}: "
        f"{head['verdict']}. " if head else "") + (
        f"The linear map ({lin['acc']:.4f}) is the null arm and behaves like one. " if lin else "") + (
        f"The capacity control mlp_direct ({ctrl['acc']:.4f}, {ctrl['verdict']}) says whether any "
        f"nonlinear gain needed the AVES targets at all. " if ctrl else "") + (
        f"concat [run11 | mapped] reaches {cc['acc']:.4f} ({cc['verdict']}) but is not a remap of "
        f"run11 -- it keeps run11's own dimensions." if cc else "")
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="map_linear,map_mlp,mlp_direct,map_linear_rev,map_mlp_rev")
    ap.add_argument("--out", default=None, help="write results here instead of distillation.json")
    ap.add_argument("--interpret-only", action="store_true",
                    help="recompute the interpretation from an existing results file and exit")
    a = ap.parse_args()
    global OUT
    if a.out:
        OUT = Path(a.out)
    if a.interpret_only:
        d = json.loads(OUT.read_text())["exp2_cross_encoder_feature_distillation"]
        res, boot = d["arms"], d["bootstrap"]
        res["interpretation"] = interpretation(res, boot)
        save("exp2_cross_encoder_feature_distillation", {"arms": res})
        print(json.dumps(res["interpretation"]["arms"], indent=2))
        print("\nREADING:", res["interpretation"]["reading"])
        return
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    hp = dict(epochs=a.epochs, hidden=a.hidden, batch=a.batch, lr=a.lr, wd=a.wd, seed=a.seed)
    print(f"[device] {device}", flush=True)

    y, birds, classes, maj = load_cohort()
    R = np.load(FEAT / "ct11_run11_emb.npy")[:, LAYER].astype(np.float32)
    A = np.load(FEAT / "ct11_aves_emb.npy")[:, LAYER].astype(np.float32)
    assert R.shape == A.shape == (len(y), 768)
    folds = folds_of(y, birds)
    for f, (tr, te) in enumerate(folds):
        assert not set(birds[tr]) & set(birds[te]), f"fold {f} leaks birds"
    print(f"[cohort] {len(y)} clips, {len(classes)} classes, {len(set(birds))} birds, "
          f"majority {maj:.5f}; layer L{LAYER}")
    print(f"[split] StratifiedGroupKFold({NFOLD}, shuffle, seed {SEED}) by bird, "
          f"test sizes {[len(te) for _, te in folds]}", flush=True)

    meta = dict(n_clips=int(len(y)), n_classes=len(classes), n_birds=int(len(set(birds))),
                majority=maj, layer=LAYER, classes=classes,
                split=f"leave-birds-out StratifiedGroupKFold({NFOLD}, shuffle=True, "
                      f"random_state={SEED})",
                probe="StandardScaler + LogisticRegression(max_iter=4000)",
                frozen_probe_reference=PROBE_REF, hp=hp, ridge_alphas=ALPHAS, device=device,
                bootstrap_unit="birds (aves_calltype.bird_bootstrap, 2000 resamples)")
    save("exp2_cross_encoder_feature_distillation", {"meta": meta})

    # ------------------------- reproduction guard
    print("\n=== reproduction guard: raw frozen probes ===", flush=True)
    res, P = {}, {}
    for nm, X in (("raw_run11", R), ("raw_aves", A)):
        acc, p = probe_oof(X, y, birds, folds)
        ref = PROBE_REF["run11" if nm.endswith("run11") else "aves"]
        res[nm] = dict(acc=acc, published=ref, diff=acc - ref)
        P[nm] = p
        print(f"  {nm:10s} {acc:.4f}   published {ref:.4f}   diff {acc-ref:+.6f}", flush=True)
    worst = max(abs(res[k]["diff"]) for k in ("raw_run11", "raw_aves"))
    res["reproduction_max_abs_diff"] = worst
    res["reproduction_passed"] = bool(worst < 1e-6)
    print(f"  max |diff| {worst:.2e} -> {'PASS' if worst < 1e-6 else 'FAIL'}", flush=True)
    save("exp2_cross_encoder_feature_distillation", {"arms": res})

    arms = [x for x in a.arms.split(",") if x.strip()]
    for direction in ("fwd", "rev"):
        Xsrc, Xtgt = (R, A) if direction == "fwd" else (A, R)
        sname, tname = ("run11", "aves") if direction == "fwd" else ("aves", "run11")
        for kind in ("linear", "mlp"):
            arm = f"map_{kind}" + ("" if direction == "fwd" else "_rev")
            if arm not in arms:
                continue
            print(f"\n=== {arm}: {sname} L{LAYER} -> {tname} L{LAYER} ===", flush=True)
            mapped = []                       # one (N, 768) matrix per fold
            oos_pred = np.zeros_like(Xtgt)
            oos_tgt = np.zeros_like(Xtgt)
            ins_r2, chosen = [], []
            t0 = time.time()
            for f, (tr_all, te) in enumerate(folds):
                tr, va = inner_split(tr_all, y, birds)
                mu_s, sd_s = zstats(Xsrc[tr_all])
                mu_t, sd_t = zstats(Xtgt[tr_all])
                Zs = (Xsrc - mu_s) / sd_s                      # (N, 768) standardised on train
                Zt = (Xtgt - mu_t) / sd_t
                print(f"    fold {f}: map train {len(tr)} / inner-val {len(va)} "
                      f"(birds {len(set(birds[tr]))}/{len(set(birds[va]))}), test {len(te)}",
                      flush=True)
                if kind == "linear":
                    mdl, al, r2v = fit_ridge_map(Zs[tr], Zt[tr], Zs[va], Zt[va],
                                                 lambda s: print(s, flush=True))
                    pred_all = mdl.predict(Zs).astype(np.float32)
                    chosen.append(dict(fold=f, alpha=al, inner_val_mean_r2=r2v))
                    print(f"      chose alpha {al:g} (inner-val mean R^2 {r2v:+.4f})", flush=True)
                else:
                    mdl, e, r2v = fit_mlp_map(Zs[tr], Zt[tr], Zs[va], Zt[va], device, hp,
                                              lambda s: print(s, flush=True))
                    with torch.no_grad():
                        pred_all = mdl(torch.from_numpy(Zs).to(device)).cpu().numpy()
                    chosen.append(dict(fold=f, epoch=e, inner_val_mean_r2=r2v))
                    del mdl
                    gc.collect()
                mapped.append(pred_all)
                oos_pred[te] = pred_all[te]
                oos_tgt[te] = Zt[te]
                ins_r2.append(r2_per_dim(Zt[tr_all], pred_all[tr_all]))
            r2_oos = r2_per_dim(oos_tgt, oos_pred)
            acc, p = probe_oof_prefit(mapped, y, folds)
            P[arm] = p
            res[arm] = dict(acc=acc, direction=f"{sname}->{tname}", selection=chosen,
                            r2_out_of_sample=r2_summary(r2_oos),
                            r2_in_sample=r2_summary(np.mean(ins_r2, 0)),
                            sec=round(time.time() - t0, 1))
            print(f"  {arm}: probe acc {acc:.4f}   out-of-sample R^2 mean "
                  f"{np.mean(r2_oos):+.4f} median {np.median(r2_oos):+.4f} "
                  f"({(r2_oos>0.5).sum()}/768 dims above 0.5)", flush=True)
            # concat only for the forward direction -- that is the one being asked about
            if direction == "fwd":
                cm = [np.hstack([Xsrc, m]) for m in mapped]
                acc_c, pc = probe_oof_prefit(cm, y, folds)
                res[f"concat_{arm}"] = dict(acc=acc_c,
                                            note=f"[run11 L{LAYER} | {arm} output] 1536-d")
                P[f"concat_{arm}"] = pc
                print(f"  concat_{arm}: {acc_c:.4f}", flush=True)
                del cm
            del mapped, oos_pred, oos_tgt
            gc.collect()
            save("exp2_cross_encoder_feature_distillation", {"arms": res})

    # ------------------------- capacity control
    if "mlp_direct" in arms:
        print(f"\n=== mlp_direct: MLP(768 -> {a.hidden} -> 11) on raw run11 L{LAYER} labels ===",
              flush=True)
        Pd = np.zeros((len(y), N_CLASSES))
        info = []
        for f, (tr_all, te) in enumerate(folds):
            tr, va = inner_split(tr_all, y, birds)
            mu, sd = zstats(R[tr_all])
            Z = (R - mu) / sd
            print(f"    fold {f}:", flush=True)
            Pall, e, vacc = mlp_direct(Z[tr], y[tr], Z[va], y[va], Z, device, hp,
                                       lambda s: print(s, flush=True))
            Pd[te] = Pall[te]
            info.append(dict(fold=f, epoch=e, inner_val_acc=vacc))
        acc = float((Pd.argmax(1) == y).mean())
        res["mlp_direct"] = dict(acc=acc, selection=info)
        P["mlp_direct"] = Pd
        print(f"  mlp_direct: {acc:.4f}", flush=True)
        save("exp2_cross_encoder_feature_distillation", {"arms": res})

    # ------------------------- bootstraps over birds
    print("\n=== cluster bootstrap over the 48 birds (2000 resamples) ===", flush=True)
    boot = {}
    for aa in [k for k in P if k not in ("raw_run11", "raw_aves")]:
        for bb in ("raw_run11", "raw_aves"):
            r = bird_bootstrap(y, P[aa], P[bb], birds)
            boot[f"{aa}__vs__{bb}"] = r
            print(f"  {aa:22s} - {bb:10s}: {r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]  "
                  f"{r['verdict']}", flush=True)
    r = bird_bootstrap(y, P["raw_aves"], P["raw_run11"], birds)
    boot["raw_aves__vs__raw_run11"] = r
    print(f"  {'raw_aves':22s} - {'raw_run11':10s}: {r['delta']:+.4f} "
          f"[{r['lo']:+.4f}, {r['hi']:+.4f}]  {r['verdict']}", flush=True)
    save("exp2_cross_encoder_feature_distillation", {"bootstrap": boot})

    res["interpretation"] = interpretation(res, boot)
    save("exp2_cross_encoder_feature_distillation", {"arms": res})
    interp = res["interpretation"]

    print("\n" + "=" * 72)
    print(f"{'arm':26s} {'acc':>8s} {'vs run11':>10s} {'vs AVES':>10s}   R^2(oos) mean")
    print("-" * 72)
    for k in ["raw_run11", "raw_aves"] + [k for k in res if k.startswith(("map_", "concat_", "mlp_d"))]:
        v = res.get(k)
        if not isinstance(v, dict) or "acc" not in v:
            continue
        dr = v["acc"] - res["raw_run11"]["acc"]
        da = v["acc"] - res["raw_aves"]["acc"]
        r2 = v.get("r2_out_of_sample", {}).get("mean")
        print(f"{k:26s} {v['acc']:8.4f} {dr:+10.4f} {da:+10.4f}   "
              f"{'' if r2 is None else f'{r2:+.4f}'}")
    print("=" * 72)
    print(f"gap AVES-run11 {interp['gap']:+.4f}")
    for k, v in interp["arms"].items():
        frac = ("   n/a " if v.get("fraction_of_gap_closed") is None
                else f"{v['fraction_of_gap_closed']*100:6.1f}%")
        print(f"  {k:20s} closes {frac} of the gap -> {v['verdict']}")
    print(f"READING: {interp['reading']}")
    print(f"NOTE: {interp['interval_width_note']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

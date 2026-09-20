#!/usr/bin/env python
"""Can ONE model do what the run11+AVES probability average does, on detection?

The best vocalization detector this repo has is not a model, it is an average: mean the
probabilities of a linear probe on run11 L0 and a linear probe on AVES L6 and in-distribution AUC
goes 0.9687 / 0.9674 -> 0.9732, +0.0047 [+0.0034, +0.0061] over run11 by a paired block bootstrap
(analysis/ensemble_detect.json). That is a real gain and it is also a deployment tax: two 94.4M
parameter encoders have to see every second of audio to collect it.

So: train a single student to reproduce the ensemble's OUTPUT while reading only ONE encoder, and
ask whether the ensemble's advantage survives the compression. This is the classic
"ensemble -> single model" distillation question (Hinton 2015), and the interesting part here is
that the two teachers are architecturally identical and differ only in pretraining corpus, with
error correlation 0.857 -- high enough that most of what they know is shared, low enough that the
average helps.

The student reads FROZEN frame features from one encoder (features/frames_30min.npz F0 for run11,
features/aves_frames_matched.npz F6 for AVES) -- the same features the teacher probes read, so the
only difference between student and teacher is the readout and its training signal. That is also
exactly the deployment quantity of interest: a head on frozen features costs one encoder pass.

Arms, and why each one is needed to interpret the others:

  teacher_run11 / teacher_aves   sklearn LogisticRegression probes, the established baselines.
                                 Reproduction guard: these must recover 0.9687 / 0.9674 / 0.9732
                                 or nothing below is comparable to the published ensemble.
  linear  + ce                   student = Linear(768,1), hard labels. The control that says what
                                 the torch training loop alone does to the sklearn number.
  linear  + kl                   student = Linear(768,1), soft ensemble targets only. A LINEAR
                                 student cannot represent an average of two linear probes on
                                 DIFFERENT features, so this arm is expected to fail; it is here to
                                 establish that the ceiling is a function-class ceiling.
  linear  + kl_ce                both terms, w=0.5.
  mlp     + ce                   Linear(768,256)-GELU-Linear(256,1), hard labels. The CAPACITY
                                 control. If this already matches the ensemble, the ensemble's gain
                                 was nonlinearity in one encoder's features, not two encoders.
  mlp     + kl / kl_ce           the actual distillation arm. A gain over mlp+ce is attributable to
                                 the teacher's soft targets; a gain over linear+ce that mlp+ce also
                                 shows is not.

Every arm is run on BOTH feature sources (run11 and AVES), because "does the init matter" is half
the question and the two encoders are not interchangeable: run11's best detection layer is L0 and
AVES's is L6.

Protocol, held identical to ensemble_detect.py so the numbers are comparable:
  split    zfeval.splits.choose_cv(zeros, centers, 5, seed 0) -> contiguous 60 s blocks, 31 blocks,
           StratifiedGroupKFold(5) over blocks. Adjacent 20 ms frames are near-duplicates, so a
           random frame split would leak.
  teacher  fitted per fold on that fold's TRAINING frames only. Soft targets for training frames are
           the teacher's in-sample predictions (standard KD); the ensemble's score on test frames is
           its out-of-fold prediction, which is what cross_val_predict produced for the published
           0.9732. No teacher ever sees a test frame of its own fold.
  student  two-stage, so the student is matched to the teacher on DATA and still gets honest model
           selection. Stage A trains on a block-disjoint 80% of the training fold for a fixed 400
           epochs (AdamW lr 1e-3, wd 0, batch 1024, cosine decay to 0) and records validation AUC
           against HARD labels every epoch; e* = argmax. Stage B retrains on the FULL training fold
           for e*+1 epochs on the same schedule shape and predicts the test fold. The test fold is
           never touched by either stage.
           Calibration of this loop matters, and it is checked: a linear student with hard labels
           must land on the sklearn probe it is imitating. It does -- fold 0 train BCE 0.1143 vs
           sklearn's 0.1139 at 400 epochs with wd=0. An earlier version of this script used 120
           epochs with wd=0.01 and early stopping on val AUC with patience 20; it scored the linear
           CE student at 0.9365 against the same probe's 0.9688, because block-structured validation
           AUC is not monotone and patience 20 kept selecting epoch 0. A distillation result read off
           that loop would have been an optimizer artefact.
  scoring  pooled out-of-fold AUC/AP over all 90,061 frames; paired moving-block bootstrap with
           block=1500 frames (30 s), the unit ensemble_detect.py used.

Temperature: T=1 by default, so KL to the teacher's probabilities is soft-target cross-entropy minus
the teacher's (constant) entropy -- identical gradients, and the measured KL is reported as well.
--temperature runs the Hinton T>1 variant with the T^2 loss scale on the best arm.
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
from zfeval import metrics as mx, splits as sp                               # noqa: E402

FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
OUT = ANA / "distillation.json"
PRED = FEAT / "distill_detect_preds.npz"
L_RUN11, L_AVES = 0, 6              # each encoder's pre-committed best detection layer
NFOLD, SEED = 5, 0
# published references this script must reproduce before anything else is believable
REF = {"run11": 0.9687310882257707, "aves": 0.9673653088292173, "mean": 0.9731871699004724}


# ------------------------------------------------------------------ json checkpointing
def save(key, payload):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d = json.loads(OUT.read_text()) if OUT.exists() else {}
    d.setdefault(key, {}).update(payload)
    OUT.write_text(json.dumps(d, indent=2))


def standardize(Xtr_f16, Xte_f16):
    """fp16 store -> fp32 standardized, train statistics only, done in place to keep RAM down.

    Equivalent to make_pipeline(StandardScaler(), ...): StandardScaler uses ddof=0.
    """
    Xtr = Xtr_f16.astype(np.float32)
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd[sd == 0] = 1.0
    Xtr -= mu
    Xtr /= sd
    Xte = Xte_f16.astype(np.float32)
    Xte -= mu
    Xte /= sd
    return Xtr, Xte


# ------------------------------------------------------------------ teacher
def fit_teachers(Xr, Xa, y, folds, cache=True):
    """Per-fold logistic probes on each encoder. Returns in-sample train probs and OOF test probs."""
    from sklearn.linear_model import LogisticRegression
    cp = FEAT / "distill_detect_teacher.npz"
    if cache and cp.exists():
        d = np.load(cp)
        print(f"[teacher] cached {cp.name}")
        return {"run11": d["tr_run11"], "aves": d["tr_aves"]}, \
               {"run11": d["oof_run11"], "aves": d["oof_aves"]}
    ins = {"run11": np.zeros((NFOLD, len(y)), np.float32),
           "aves": np.zeros((NFOLD, len(y)), np.float32)}
    oof = {"run11": np.zeros(len(y), np.float32), "aves": np.zeros(len(y), np.float32)}
    for nm, X in (("run11", Xr), ("aves", Xa)):
        for f, (tr, te) in enumerate(folds):
            t0 = time.time()
            Xtr, Xte = standardize(X[tr], X[te])
            est = LogisticRegression(max_iter=4000, C=1.0).fit(Xtr, y[tr])
            ins[nm][f, tr] = est.predict_proba(Xtr)[:, 1]
            oof[nm][te] = est.predict_proba(Xte)[:, 1]
            print(f"  [teacher] {nm} fold {f}: {len(tr)} train frames, "
                  f"{time.time()-t0:.0f}s, n_iter {est.n_iter_[0]}", flush=True)
            del Xtr, Xte, est
            gc.collect()
    if cache:
        np.savez(cp, tr_run11=ins["run11"], tr_aves=ins["aves"],
                 oof_run11=oof["run11"], oof_aves=oof["aves"])
    return ins, oof


# ------------------------------------------------------------------ student
class MLP(nn.Module):
    def __init__(self, d=768, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_student(kind, device, seed):
    torch.manual_seed(seed)
    m = nn.Linear(768, 1) if kind == "linear" else MLP()
    return m.to(device)


def student_logits(m, x):
    out = m(x)
    return out.squeeze(-1) if out.dim() > 1 else out


def kd_loss(z, y_hard, l_soft, w, T):
    """(1-w) * BCE(hard) + w * T^2 * KL(teacher_T || student_T), binary form."""
    ce = F.binary_cross_entropy_with_logits(z, y_hard)
    if w == 0.0:
        return ce, ce, torch.zeros((), device=z.device)
    pt = torch.sigmoid(l_soft / T)
    ls = z / T
    # KL(pt || ps) = -H(pt) + CE(pt, ps)
    ce_soft = F.binary_cross_entropy_with_logits(ls, pt)
    ent = -(pt * torch.log(pt.clamp_min(1e-9)) + (1 - pt) * torch.log((1 - pt).clamp_min(1e-9)))
    kl = ce_soft - ent.mean()
    return (1 - w) * ce + w * (T ** 2) * kl, ce, kl


def inner_split(tr, y, blocks):
    """Block-disjoint validation slice of a training fold, for early stopping only."""
    from sklearn.model_selection import StratifiedGroupKFold
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    a, b = next(iter(cv.split(np.zeros((len(tr), 1)), y[tr], blocks[tr])))
    return tr[a], tr[b]


def _run(kind, Xtr, ytr, ls_tr, w, T, device, hp, n_epochs, Xeval, yeval, log, tag):
    """Train for exactly n_epochs on (Xtr, ytr) with cosine decay; return per-epoch eval AUC and
    the final predictions on Xeval."""
    from sklearn.metrics import roc_auc_score
    m = make_student(kind, device, hp["seed"])
    opt = torch.optim.AdamW(m.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["max_epochs"])
    Xt = torch.from_numpy(Xtr).to(device)
    Yt = torch.from_numpy(ytr.astype(np.float32)).to(device)
    Lt = torch.from_numpy(ls_tr.astype(np.float32)).to(device)
    Xe = torch.from_numpy(Xeval).to(device)
    rng = np.random.default_rng(hp["seed"])
    n = len(ytr)
    curve, last = [], None
    for ep in range(n_epochs):
        m.train()
        perm = torch.from_numpy(rng.permutation(n)).to(device)
        tot = totkl = 0.0
        for s in range(0, n, hp["batch"]):
            b = perm[s:s + hp["batch"]]
            loss, ce, kl = kd_loss(student_logits(m, Xt[b]), Yt[b], Lt[b], w, T)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
            totkl += float(kl.detach()) * len(b)
        sch.step()
        m.eval()
        with torch.no_grad():
            last = torch.sigmoid(student_logits(m, Xe)).cpu().numpy()
        auc = float(roc_auc_score(yeval, last)) if yeval is not None else float("nan")
        curve.append(auc)
        if ep % 100 == 0 or ep == n_epochs - 1:
            log(f"      {tag} ep{ep:3d} loss {tot/n:.4f} kl {totkl/n:.4f} auc {auc:.4f}")
    with torch.no_grad():
        train_bce = float(F.binary_cross_entropy_with_logits(
            student_logits(m, Xt), Yt))
    del m, opt, sch, Xt, Yt, Lt, Xe
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return curve, last, train_bce


def train_student(kind, Xfull, yfull, ls_full, itr, iva, Xte, w, T, device, hp, log):
    """Stage A on the inner 80% to pick the epoch, stage B on the full training fold to predict."""
    curve, _, _ = _run(kind, Xfull[itr], yfull[itr], ls_full[itr], w, T, device, hp,
                       hp["max_epochs"], Xfull[iva], yfull[iva], log, "A")
    e = int(np.argmax(curve))
    log(f"      selected epoch {e} (inner-val AUC {curve[e]:.4f}, "
        f"final-epoch {curve[-1]:.4f})")
    _, p_te, bce = _run(kind, Xfull, yfull, ls_full, w, T, device, hp, e + 1, Xte, None, log, "B")
    return dict(epoch=e, val_auc=curve[e], val_auc_final_epoch=curve[-1],
                val_curve_max=float(np.max(curve)), train_bce=bce, test=p_te)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="linear:ce,linear:kl,linear:kl_ce,mlp:ce,mlp:kl,mlp:kl_ce")
    ap.add_argument("--sources", default="run11,aves")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--w", type=float, default=0.5, help="mixing weight for the kl_ce arms")
    ap.add_argument("--bootstrap-only", action="store_true",
                    help="skip training; bootstrap and tabulate whatever is already in the "
                         "results file and the cached prediction npz")
    a = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[device] {device}  torch {torch.__version__}", flush=True)

    Z = np.load(FEAT / "frames_30min.npz", allow_pickle=True)
    AV = np.load(FEAT / "aves_frames_matched.npz", allow_pickle=True)
    y, centers = Z["y"].astype(int), Z["centers"]
    Xr, Xa = Z[f"F{L_RUN11}"], AV[f"F{L_AVES}"]          # left as float16 until a fold needs them
    assert Xr.shape == Xa.shape == (len(y), 768)
    cv, g, desc = sp.choose_cv(np.zeros(len(y), dtype="<U3"), centers, n_splits=NFOLD, seed=SEED)
    folds = list(cv.split(np.zeros((len(y), 1)), y, g))
    print(f"[data] {len(y)} frames, prevalence {y.mean():.4f}, run11 L{L_RUN11} + AVES L{L_AVES}")
    print(f"[split] {desc}; test sizes {[len(te) for _, te in folds]}", flush=True)

    meta = dict(n_frames=int(len(y)), prevalence=float(y.mean()), split=desc,
                layers={"run11": L_RUN11, "aves": L_AVES}, bootstrap_block=1500,
                published_reference=REF,
                hp=dict(epochs=a.epochs, batch=a.batch, lr=a.lr, wd=a.wd,
                        seed=a.seed, temperature=a.temperature, w_kl_ce=a.w,
                        optimizer="AdamW + cosine decay",
                        selection="stage A on inner 80% picks the epoch by val AUC vs hard labels; "
                                  "stage B retrains on the full training fold for that many epochs",
                        teacher="per-fold LogisticRegression(max_iter=4000, C=1.0) on each encoder, "
                                "probabilities averaged; soft targets are in-sample train probs"))
    save("exp1_ensemble_distillation_detection", {"meta": meta})

    # ---------------- teachers + reproduction guard
    print("\n=== teachers (per-fold logistic probes) ===", flush=True)
    ins, oof = fit_teachers(Xr, Xa, y, folds)
    P = {"run11": oof["run11"], "aves": oof["aves"]}
    P["mean"] = (P["run11"] + P["aves"]) / 2
    guard = {}
    print("\n=== reproduction guard vs analysis/ensemble_detect.json ===")
    for nm in ("run11", "aves", "mean"):
        s = mx.score(y, P[nm], desc)
        guard[nm] = dict(auc=s.auc, ap=s.ap, published_auc=REF[nm], diff=s.auc - REF[nm])
        print(f"  {nm:6s} AUC {s.auc:.4f}  published {REF[nm]:.4f}  diff {s.auc-REF[nm]:+.5f}")
    worst = max(abs(v["diff"]) for v in guard.values())
    guard["max_abs_diff"] = worst
    guard["passed"] = bool(worst < 1e-3)
    print(f"  max |diff| {worst:.5f}  ->  {'PASS' if worst < 1e-3 else 'FAIL'}", flush=True)
    save("exp1_ensemble_distillation_detection", {"reproduction_guard": guard})
    if not guard["passed"]:
        print("  WARNING: teachers do not reproduce the published ensemble; every number below "
              "is internally consistent but NOT comparable to 0.9732.")

    # soft targets: logit of the averaged probability, per fold, in-sample on the training frames
    def soft_logit(f):
        p = (ins["run11"][f] + ins["aves"][f]) / 2
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).astype(np.float32)

    # resume: arms already in the results file are kept and skipped, so the sweep can be run in
    # several sessions (and so a long arm list can be trimmed mid-sweep without losing what is done)
    results = {}
    if OUT.exists():
        prev = json.loads(OUT.read_text()).get("exp1_ensemble_distillation_detection", {})
        results = dict(prev.get("students", {}))
        if results:
            print(f"[resume] {len(results)} student arms already in {OUT.name}: "
                  f"{sorted(results)}", flush=True)
    arms = [x for x in a.arms.split(",") if x.strip()]
    sources = [x for x in a.sources.split(",") if x.strip()]
    hp = dict(max_epochs=a.epochs, batch=a.batch, lr=a.lr, wd=a.wd, seed=a.seed)
    W = {"ce": 0.0, "kl": 1.0, "kl_ce": a.w}
    preds = {f"teacher_{k}": v for k, v in P.items()}
    if PRED.exists():
        d0 = np.load(PRED)
        for k in d0.files:
            if k != "y" and k not in preds:
                preds[k] = d0[k]

    if a.bootstrap_only:
        arms = []
        sources = []
        print(f"[bootstrap-only] using {len(results)} stored arms", flush=True)
    for src in sources:
        X = Xr if src == "run11" else Xa
        for arm in arms:
            kind, mode = arm.split(":")
            key = f"{src}|{kind}|{mode}"
            if key in results and key in preds:
                print(f"[skip] {key} already done: AUC {results[key]['auc']:.4f}", flush=True)
                continue
            print(f"\n=== student {key}  (w={W[mode]}, T={a.temperature}) ===", flush=True)
            p_oof = np.zeros(len(y), np.float32)
            fold_info = []
            for f, (tr_all, te) in enumerate(folds):
                tr, va = inner_split(tr_all, y, g)
                Xtr, Xte = standardize(X[tr_all], X[te])
                # inner val rows are a subset of tr_all; index into the standardized block
                pos = {int(i): k for k, i in enumerate(tr_all)}
                itr = np.array([pos[int(i)] for i in tr], dtype=np.int64)
                iva = np.array([pos[int(i)] for i in va], dtype=np.int64)
                ls = soft_logit(f)[tr_all]
                best = train_student(kind, Xtr, y[tr_all], ls, itr, iva, Xte,
                                     W[mode], a.temperature, device, hp,
                                     lambda s: print(s, flush=True))
                p_oof[te] = best["test"]
                fold_info.append(dict(fold=f, n_train_stageA=len(tr), n_val=len(va),
                                      n_train_stageB=len(tr_all), n_test=len(te),
                                      selected_epoch=best["epoch"], val_auc=best["val_auc"],
                                      val_auc_final_epoch=best["val_auc_final_epoch"],
                                      train_bce_stageB=best["train_bce"]))
                print(f"    fold {f}: epoch {best['epoch']} inner-val AUC {best['val_auc']:.4f}, "
                      f"stage-B train BCE {best['train_bce']:.4f}", flush=True)
                del Xtr, Xte
                gc.collect()
            s = mx.score(y, p_oof, desc)
            results[key] = dict(auc=s.auc, ap=s.ap, folds=fold_info)
            preds[key] = p_oof
            print(f"  {key}: pooled OOF AUC {s.auc:.4f}  AP {s.ap:.4f}", flush=True)
            del p_oof
            save("exp1_ensemble_distillation_detection", {"students": results})
            np.savez(PRED, y=y, **preds)

    # ---------------- bootstraps
    from sklearn.metrics import roc_auc_score, average_precision_score
    print("\n=== paired moving-block bootstrap, block=1500 frames (30 s) ===", flush=True)
    boot = {}
    pairs = []
    for k in sorted(results):
        src = k.split("|")[0]
        pairs += [(k, "teacher_mean"), (k, f"teacher_{src}")]
        if k.endswith("|kl") and f"{src}|{k.split('|')[1]}|ce" in results:
            pairs.append((k, f"{src}|{k.split('|')[1]}|ce"))     # soft targets vs hard labels
    pairs += [("teacher_mean", "teacher_run11"), ("teacher_mean", "teacher_aves")]
    for mname, mfun in (("auc", roc_auc_score), ("ap", average_precision_score)):
        for aa, bb in pairs:
            if aa not in preds or bb not in preds:
                continue
            r = mx.paired_bootstrap(y, preds[aa], preds[bb], block=1500, n=2000, seed=0, metric=mfun)
            boot[f"{aa}__vs__{bb}__{mname}"] = r
            print(f"  {mname.upper():3s} {aa:22s} - {bb:14s}: {r['delta']:+.4f} "
                  f"[{r['lo']:+.4f}, {r['hi']:+.4f}]  {r['verdict']}", flush=True)
    save("exp1_ensemble_distillation_detection", {"bootstrap": boot})

    # ---------------- table
    print("\n" + "=" * 78)
    print(f"{'arm':28s} {'AUC':>8s} {'AP':>8s}   {'vs ensemble AUC':>28s}")
    print("-" * 78)
    for nm in ("teacher_run11", "teacher_aves", "teacher_mean"):
        s = mx.score(y, preds[nm], desc)
        print(f"{nm:28s} {s.auc:8.4f} {s.ap:8.4f}")
    for k, v in results.items():
        r = boot.get(f"{k}__vs__teacher_mean__auc", {})
        d = (f"{r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] {r['verdict']}"
             if r else "")
        print(f"{k:28s} {v['auc']:8.4f} {v['ap']:8.4f}   {d}")
    print("=" * 78)
    print(f"wrote {OUT}  and  {PRED}")


if __name__ == "__main__":
    main()

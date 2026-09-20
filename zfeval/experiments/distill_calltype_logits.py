#!/usr/bin/env python
"""Can a generic-audio teacher hand its call-type advantage to the colony-pretrained student?

Where this starts. On 11-class call type, leave-birds-out over 48 birds, 3412 clips
(majority 0.1797):

    frozen linear probe   run11 L3 0.8118    AVES L3 0.8453
    full fine-tune        run11    0.8306    AVES    0.8607        (mean over 5 folds, 2 seeds)

run11 loses under every protocol tried. Fine-tuning lifts both by about the same amount and does not
reorder them (analysis/finetune_calltype.json). The remaining question is not which encoder is
better -- that is settled -- but whether the lab's own weights can be TAUGHT: run11 fine-tuned with
the true labels AND the teacher's full output distribution, which is the standard reason knowledge
distillation works at all (Hinton 2015: the teacher's wrong-class probabilities carry which call
types resemble which, a similarity structure the one-hot labels destroy).

Two thresholds, and they are very different in strength:
  beat plain fine-tuned run11 (0.8306)  -- plausible; soft labels regularise, and this cohort has
                                           11 classes with counts from tens to hundreds.
  beat fine-tuned AVES (0.8607)         -- the interesting one. It would mean colony pretraining is
                                           worth keeping as long as it is taught, which is the only
                                           story under which the released weights still matter.

TEACHER: the FROZEN-PROBE AVES L3 model, not fine-tuned AVES. Stated plainly because it caps the
result -- this teacher scores 0.8453 leave-birds-out, which is below fine-tuned AVES's 0.8607, so
asking the student to pass 0.8607 is asking it to beat its teacher by 0.015. The choice is a compute
one (a fine-tuned-AVES teacher costs another five fine-tuning runs to produce fold-matched soft
labels, ~40 min, and is the obvious follow-up); it is not a claim that the probe teacher is the
better one. A teacher at 0.8453 is still well above the plain fine-tuned student at 0.8306, so there
is real headroom for the distillation term to exploit.

Soft labels are CROSS-FITTED, and that is not a detail. The textbook recipe -- fit the teacher on the
training clips, read its probabilities back off the same clips -- produces in-sample accuracy
0.9997-1.0000 on this cohort, because 3412 clips in 768 dimensions are nearly linearly separable.
Those targets are one-hot to 1e-4, the KL term becomes the CE term, and the w sweep would measure
nothing at all. Instead an inner bird-grouped StratifiedGroupKFold(5) inside each outer training fold
produces out-of-fold teacher probabilities for every training clip: real accuracy, real confusions,
and still no leakage, since every inner fold is a subset of the outer training birds. Both schemes'
soft-label accuracy and entropy are recorded in the results.
Reusing one global set of out-of-fold teacher probabilities would have been cheaper and would have
leaked: fold k's training clips would then carry soft labels from probes fitted on folds containing
fold k's TEST birds.

Loss, per clip:  (1 - w) * CE(y_true)  +  w * T^2 * KL(teacher_T || student_T)
  w = 0    plain fine-tuning; must reproduce run11|full_ft|seed0 = 0.8306162 from
           analysis/finetune_calltype.json, since the code path is then arithmetically identical.
           That is the correctness guard for everything else in this file.
  w = 1    pure distillation, the labels never seen directly.
  sweep    {0, 0.3, 0.7, 1.0} as specified, T = 1 by default.

Everything else is held bit-identical to finetune_calltype.py's full_ft arm -- same cohort assertion,
same StratifiedGroupKFold(5, shuffle=True, random_state=0) by bird, same bird-disjoint inner
validation for early stopping, same padding-invariant masked feature extractor (torchaudio's batched
extractor is NOT padding-invariant for these models and silently corrupts short clips), same AdamW,
enc_lr 1e-5, head_lr 1e-3, wd 0.01, clip 1.0, 10% warmup, batch 8, <=8 epochs, patience 3,
encoder_layer_drop 0. Nothing is tuned per arm.

Results append to analysis/distillation.json after every fold; comparisons are bootstrapped over the
48 birds, which is the independent unit here.
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
import finetune_calltype as FT                                                # noqa: E402
from aves_calltype import bird_bootstrap                                      # noqa: E402

FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
OUT = ANA / "distillation.json"
KEY = "exp3_logit_distillation_calltype"
LAYER = 3
# the anchors this script is measured against, from analysis/finetune_calltype.json
ANCHOR = {"run11_full_ft_seed0": 0.8306162223157509,
          "run11_full_ft_mean_2seeds": 0.8304913906667888,
          "aves_full_ft_seed0": 0.8607225536634434,
          "aves_full_ft_mean_2seeds": 0.8600485860713637,
          "run11_frozen_probe_L3": 0.8118405627198124,
          "aves_frozen_probe_L3": 0.8452520515826495}


def save(payload):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d = json.loads(OUT.read_text()) if OUT.exists() else {}
    d.setdefault(KEY, {}).update(payload)
    OUT.write_text(json.dumps(d, indent=2))


# ------------------------------------------------------------------ teacher
def teacher_soft_labels(y, birds, folds, mode, log):
    """Per-fold AVES L3 logistic probe -> soft labels for that fold's TRAINING clips.

    How the soft labels are produced matters more than anything else in this file, and the obvious
    choice is the wrong one. Fitting the probe on the training clips and reading its probabilities
    back off those same clips -- textbook KD -- gives in-sample accuracy 0.9997-1.0000 here: 3412
    clips, 768 features and 11 classes are very nearly linearly separable, so the teacher's training
    predictions are one-hot to within 1e-4. Distilling from those targets is distilling from the
    labels, the KL term collapses onto the CE term, and the whole w sweep would be measuring nothing.
    Measured entropies for both schemes are reported in the results so this is visible rather than
    asserted.

    So the default is CROSS-FITTED targets: inside each outer training fold, a second bird-grouped
    StratifiedGroupKFold(5) produces out-of-fold probabilities for every training clip. Those have
    the teacher's real accuracy (~0.845) and its real confusions, and they still cannot leak -- every
    inner fold is a subset of the outer training birds, and the outer test birds appear in no fit at
    any level.

    mode="insample" reproduces the naive variant for comparison.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedGroupKFold
    A = np.load(FEAT / "ct11_aves_emb.npy")[:, LAYER].astype(np.float32)
    assert A.shape[0] == len(y)

    def pipe():
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000))

    S = np.zeros((len(folds), len(y), FT.N_CLASSES), dtype=np.float32)
    O = np.zeros((len(y), FT.N_CLASSES), dtype=np.float32)
    tr_acc, tr_ent = [], []
    for f, (tr, te) in enumerate(folds):
        est = pipe().fit(A[tr], y[tr])
        O[te] = est.predict_proba(A[te])                       # out-of-fold, for the teacher's score
        if mode == "insample":
            S[f, tr] = est.predict_proba(A[tr])
        else:
            inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
            for itr, ite in inner.split(np.zeros((len(tr), 1)), y[tr], birds[tr]):
                e = pipe().fit(A[tr][itr], y[tr][itr])
                S[f, tr[ite]] = e.predict_proba(A[tr][ite])
        a = float((S[f, tr].argmax(1) == y[tr]).mean())
        ent = float(-(S[f, tr] * np.log(np.clip(S[f, tr], 1e-9, None))).sum(1).mean())
        tr_acc.append(a)
        tr_ent.append(ent)
        log(f"  [teacher] fold {f}: {len(tr)} train clips ({len(set(birds[tr]))} birds), "
            f"soft-label acc on them {a:.4f}, mean entropy {ent:.4f} nats")
    acc = float((O.argmax(1) == y).mean())
    log(f"  [teacher] mode={mode}; pooled out-of-fold acc {acc:.4f} "
        f"(reference {ANCHOR['aves_frozen_probe_L3']:.4f}, "
        f"diff {acc-ANCHOR['aves_frozen_probe_L3']:+.6f}); "
        f"soft labels: acc {np.mean(tr_acc):.4f}, entropy {np.mean(tr_ent):.4f} nats "
        f"(ln 11 = {np.log(11):.4f} would be uniform)")
    del A
    gc.collect()
    return S, O, acc, dict(mode=mode, soft_label_acc_per_fold=tr_acc,
                           soft_label_acc=float(np.mean(tr_acc)),
                           soft_label_entropy_nats=float(np.mean(tr_ent)),
                           uniform_entropy_nats=float(np.log(11)))


# ------------------------------------------------------------------ KD loss
def kd_terms(z, y_hard, p_teacher, w, T):
    """Per-clip (1-w)*CE + w*T^2*KL(teacher_T || student_T), summed -- caller divides by batch size.

    Summing rather than averaging is required: finetune_calltype accumulates gradients over
    micro-batches of one optimiser batch, so each micro-batch must contribute its own sum.
    """
    ce = F.cross_entropy(z, y_hard, reduction="none")
    if w == 0.0:
        return ce.sum(), ce.sum().detach(), torch.zeros((), device=z.device)
    logp_t = torch.log(p_teacher.clamp_min(1e-9))
    pt = torch.softmax(logp_t / T, dim=-1)
    logp_s = F.log_softmax(z / T, dim=-1)
    kl = (pt * (torch.log(pt.clamp_min(1e-9)) - logp_s)).sum(-1)
    return ((1 - w) * ce + w * (T ** 2) * kl).sum(), ce.sum().detach(), kl.sum().detach()


def train_fold_kd(tr, va, te, flat, off, y, birds, lens_arr, soft, w, T, device, hp, log):
    """finetune_calltype.train_fold's full_ft arm, with the KD term added to the loss."""
    torch.manual_seed(hp["seed"])
    np.random.seed(hp["seed"])
    enc = FT.build_model("run11", device, layer_drop=hp["layer_drop"])
    model = FT.Clf(enc).to(device)
    torch.manual_seed(hp["seed"])
    model.head.reset_parameters()

    enc_params = list(enc.parameters())
    groups = [{"params": enc_params, "lr": hp["enc_lr"]},
              {"params": model.head.parameters(), "lr": hp["head_lr"]}]
    opt = torch.optim.AdamW(groups, weight_decay=hp["wd"])
    rng = np.random.default_rng(hp["seed"])
    nb = len(FT.make_batches(tr, lens_arr, hp["batch"]))
    total = nb * hp["epochs"]
    warm = max(1, int(hp["warmup_frac"] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 1.0)
    St = torch.from_numpy(soft).to(device)

    hist = []
    best = dict(val=-1.0, epoch=-1, test=float("nan"))
    best_proba = None
    for ep in range(hp["epochs"]):
        model.train()
        tot = totce = totkl = 0.0
        nseen = 0
        t0 = time.time()
        params = [p for g in groups for p in g["params"]]
        for gi, b in enumerate(FT.make_batches(tr, lens_arr, hp["batch"], rng=rng)):
            opt.zero_grad(set_to_none=True)
            for mb in FT.split_micro(b, lens_arr):
                x, L = FT.pack(flat, off, mb, device)
                z = model(x, L)
                yt = torch.from_numpy(y[mb]).to(device)
                s, ce, kl = kd_terms(z, yt, St[mb], w, T)
                (s / len(b)).backward()
                tot += float(s.detach())
                totce += float(ce)
                totkl += float(kl)
                nseen += len(mb)
                del x, L, z, yt, s, ce, kl
            if hp["clip"]:
                torch.nn.utils.clip_grad_norm_(params, hp["clip"])
            opt.step()
            sched.step()
            if gi and gi % 100 == 0:
                log(f"        .. {gi} batches, {time.time()-t0:.0f}s, "
                    f"loss {tot/max(nseen,1):.4f} ce {totce/max(nseen,1):.4f} "
                    f"kl {totkl/max(nseen,1):.4f}")
        trl, trce, trkl = tot / nseen, totce / nseen, totkl / nseen
        vacc = FT.evaluate(model, flat, off, va, y, lens_arr, device, hp["batch"])
        tacc = float("nan")
        if vacc > best["val"]:
            tacc = FT.evaluate(model, flat, off, te, y, lens_arr, device, hp["batch"])
            best = dict(val=vacc, epoch=ep, test=tacc)
            best_proba = predict_proba(model, flat, off, te, lens_arr, device, hp["batch"])
        hist.append(dict(epoch=ep, train_loss=trl, train_ce=trce, train_kl=trkl, val_acc=vacc,
                         test_acc=None if np.isnan(tacc) else tacc, sec=round(time.time() - t0, 1)))
        log(f"      ep{ep} loss {trl:.4f} ce {trce:.4f} kl {trkl:.4f}  val {vacc:.4f}  "
            f"test {'--    ' if np.isnan(tacc) else f'{tacc:.4f}'}  ({time.time()-t0:.0f}s)")
        if not np.isfinite(trl):
            log(f"      ep{ep} train loss not finite -- DIVERGED, aborting fold")
            break
        if hp.get("patience") and ep - best["epoch"] >= hp["patience"]:
            log(f"      early stop: no val improvement for {hp['patience']} epochs "
                f"(best ep{best['epoch']} val {best['val']:.4f})")
            break

    del model, enc, opt, sched, St
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    best["epochs_run"] = len(hist)
    return best, hist, best_proba


@torch.inference_mode()
def predict_proba(model, flat, off, ids, lens_arr, device, batch):
    """Softmax probabilities for `ids`, in the order of `ids`, for the bird bootstrap."""
    model.eval()
    P = np.zeros((len(ids), FT.N_CLASSES), dtype=np.float32)
    pos = {int(v): k for k, v in enumerate(ids)}
    for b in FT.make_batches(ids, lens_arr, batch):
        for mb in FT.split_micro(b, lens_arr):
            x, L = FT.pack(flat, off, mb, device)
            p = torch.softmax(model(x, L), -1).cpu().numpy()
            for k, i in enumerate(mb):
                P[pos[int(i)]] = p[k]
            del x, L
    return P


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="0,0.3,0.7,1.0", help="KD mixing weights to sweep")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--enc-lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--layer-drop", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", default=None, help="comma-separated fold subset (default: all 5)")
    ap.add_argument("--teacher-mode", default="crossfit", choices=["crossfit", "insample"],
                    help="how the teacher's soft labels on training clips are produced; see "
                         "teacher_soft_labels()")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    global OUT
    if a.out:
        OUT = Path(a.out)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[device] {device}  torch {torch.__version__}", flush=True)

    paths, y, birds, classes, src, maj = FT.load_cohort()
    print(f"[cohort] {len(y)} clips, {len(classes)} classes, {len(set(birds))} birds, "
          f"majority {maj:.5f}", flush=True)
    flat, off = FT.load_wavs(paths)
    lens_arr = np.diff(off)
    folds = FT.outer_folds(y, birds)
    for f, (tr, te) in enumerate(folds):
        assert not set(birds[tr]) & set(birds[te]), f"fold {f} leaks birds"
    sizes = [len(te) for _, te in folds]
    print(f"[split] StratifiedGroupKFold(5, shuffle, seed 0) by bird; test sizes {sizes}",
          flush=True)

    print(f"\n=== teacher: frozen AVES L3 logistic probe, refitted per fold "
          f"({a.teacher_mode} soft labels) ===", flush=True)
    soft, teach_oof, teach_acc, tinfo = teacher_soft_labels(
        y, birds, folds, a.teacher_mode, lambda s: print(s, flush=True))
    # the naive alternative, measured rather than dismissed
    if a.teacher_mode == "crossfit":
        _, _, _, tinfo_ins = teacher_soft_labels(y, birds, folds, "insample", lambda s: None)
        tinfo["insample_comparison"] = tinfo_ins
        print(f"  [teacher] naive in-sample alternative would give soft-label acc "
              f"{tinfo_ins['soft_label_acc']:.4f}, entropy "
              f"{tinfo_ins['soft_label_entropy_nats']:.4f} nats -- effectively one-hot, which is "
              f"why cross-fitted targets are the default", flush=True)

    hp = dict(seed=a.seed, epochs=a.epochs, batch=a.batch, enc_lr=a.enc_lr, head_lr=a.head_lr,
              wd=a.wd, clip=a.clip, warmup_frac=a.warmup_frac, layer_drop=a.layer_drop,
              patience=a.patience)
    d0 = json.loads(OUT.read_text()).get(KEY, {}) if OUT.exists() else {}
    runs = d0.get("runs", {})
    save(dict(meta=dict(
        n_clips=int(len(y)), n_classes=len(classes), n_birds=int(len(set(birds))), majority=maj,
        classes=classes, layer_teacher=LAYER, device=device, torch=torch.__version__,
        split="leave-birds-out StratifiedGroupKFold(5, shuffle=True, random_state=0)",
        student="run11, full fine-tune (all parameters), head Linear(768,11) on length-aware mean",
        teacher="frozen AVES L3 StandardScaler+LogisticRegression(max_iter=4000), refitted per fold "
                "on that fold's training birds; soft labels for training clips are cross-fitted "
                "(inner bird-grouped StratifiedGroupKFold(5) inside the training fold)",
        teacher_soft_labels=tinfo,
        teacher_oof_acc=teach_acc, temperature=a.temperature, hp=hp, anchors=ANCHOR,
        loss="(1-w)*CE(true) + w*T^2*KL(teacher_T || student_T)",
        bootstrap_unit="birds (2000 resamples)"),
        teacher=dict(oof_acc=teach_acc, reference=ANCHOR["aves_frozen_probe_L3"],
                     diff=teach_acc - ANCHOR["aves_frozen_probe_L3"], soft_labels=tinfo)))

    P = {"teacher_aves_probe": teach_oof}
    for wtxt in [x for x in a.weights.split(",") if x.strip()]:
        w = float(wtxt)
        key = f"w{wtxt}"
        prev = runs.get(key)
        if prev and prev.get("complete"):
            print(f"[skip] {key} already complete")
            if prev.get("proba_file") and Path(prev["proba_file"]).exists():
                P[key] = np.load(prev["proba_file"])
            continue
        accs = list(prev["fold_acc"]) if prev else []
        hists = list(prev.get("history", [])) if prev else []
        print(f"\n=== run11 full_ft + KD, w={w}, T={a.temperature} ===", flush=True)
        Pk = np.zeros((len(y), FT.N_CLASSES), dtype=np.float32)
        if prev and prev.get("proba_file") and Path(prev["proba_file"]).exists():
            Pk = np.load(prev["proba_file"])
        t_arm = time.time()
        for f, (tr_all, te) in enumerate(folds):
            if f < len(accs):
                continue
            tr, va = FT.inner_val(tr_all, y, birds)
            print(f"    fold {f}: train {len(tr)} val {len(va)} test {len(te)} "
                  f"(birds {len(set(birds[tr]))}/{len(set(birds[va]))}/{len(set(birds[te]))})",
                  flush=True)
            best, hist, proba = train_fold_kd(tr, va, te, flat, off, y, birds, lens_arr,
                                              soft[f], w, a.temperature, device, hp,
                                              lambda s: print(s, flush=True))
            accs.append(best["test"])
            hists.append(hist)
            if proba is not None:
                Pk[te] = proba
            pf = str(FEAT / f"distill_ct_proba_{key}.npy")
            np.save(pf, Pk)
            runs[key] = dict(w=w, T=a.temperature, fold_acc=accs, fold_n=sizes[:len(accs)],
                             mean=float(np.mean(accs)),
                             pooled=float(np.average(accs, weights=sizes[:len(accs)])),
                             std=float(np.std(accs)), history=hists, proba_file=pf,
                             complete=False)
            save(dict(runs=runs))
            print(f"    fold {f} -> test {best['test']:.4f} at best-val epoch {best['epoch']} "
                  f"(val {best['val']:.4f})", flush=True)
        if key not in runs:                      # every fold was already on disk
            runs[key] = dict(prev)
        runs[key]["complete"] = True
        runs[key]["sec"] = round(time.time() - t_arm, 1)
        P[key] = Pk
        save(dict(runs=runs))
        print(f"  {key}: mean {np.mean(accs):.4f} pooled "
              f"{np.average(accs, weights=sizes):.4f}  folds {[round(x,4) for x in accs]}  "
              f"({(time.time()-t_arm)/60:.1f} min)", flush=True)

    # ------------------------- guard: w=0 must reproduce plain fine-tuning
    # Not bit-exactly, and the reason is worth stating. Epoch 0 of fold 0 reproduces
    # analysis/finetune_calltype.json exactly (train loss 1.1073, val 0.8397, test 0.8003), so the
    # data, the split, the batching and the seeding all match. From epoch 1 the trajectories separate
    # by one or two CLIPS (val 0.8664 vs 0.8702 = 1/262; test 0.8515 vs 0.8540 = 2/781): MPS
    # reductions are not bit-reproducible across processes, and the KD loss forms the same quantity
    # in a different order ((sum/len).backward() instead of (sum/len) built inside cross_entropy).
    # Once the weights differ in the last bits the early-stopping epoch can move, so the tolerance
    # here is a few clips, not 1e-9. The anchor's own seed-to-seed spread is 0.0002 (0.8306 vs
    # 0.8304), which is the scale that matters. Every KD arm is compared against THIS run's w=0 arm,
    # not against the stored anchor, so the comparison is internal and this guard is only a sanity
    # check that the reimplementation is the same experiment.
    guard = None
    if "w0" in runs and runs["w0"].get("complete"):
        got = runs["w0"]["mean"]
        d = got - ANCHOR["run11_full_ft_seed0"]
        guard = dict(w0_mean=got, reference_run11_full_ft_seed0=ANCHOR["run11_full_ft_seed0"],
                     reference_run11_full_ft_mean_2seeds=ANCHOR["run11_full_ft_mean_2seeds"],
                     diff=d, anchor_seed_spread=abs(ANCHOR["run11_full_ft_seed0"]
                                                    - ANCHOR["run11_full_ft_mean_2seeds"]) * 2,
                     tolerance=0.01, passed=bool(abs(d) < 0.01),
                     note="bit-exactness is not achievable on MPS across processes; epoch 0 of fold "
                          "0 matched the anchor exactly and the trajectories then separate by 1-2 "
                          "clips. KD arms are compared against this run's own w=0 arm.")
        print(f"\n[guard] w=0 mean {got:.4f} vs stored plain full_ft "
              f"{ANCHOR['run11_full_ft_seed0']:.4f}  diff {d:+.4f} "
              f"-> {'within tolerance' if guard['passed'] else 'OUTSIDE 0.01 TOLERANCE'}", flush=True)
        save(dict(w0_reproduction_guard=guard))

    # ------------------------- bootstraps over birds
    print("\n=== cluster bootstrap over the 48 birds ===", flush=True)
    boot = {}
    keys = [k for k in P if k.startswith("w")]
    base = "w0" if "w0" in P else None
    for k in keys:
        for other in ([base] if base and k != base else []) + ["teacher_aves_probe"]:
            if other is None or other == k or other not in P:
                continue
            r = bird_bootstrap(y, P[k], P[other], birds)
            boot[f"{k}__vs__{other}"] = r
            print(f"  {k:8s} - {other:18s}: {r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]  "
                  f"{r['verdict']}", flush=True)
    save(dict(bootstrap=boot))

    # ------------------------- table
    print("\n" + "=" * 96)
    print(f"{'arm':22s} {'per-fold':44s} {'mean':>8s} {'pooled':>8s}  vs plain FT  vs FT-AVES")
    print("-" * 96)
    for k in sorted(runs):
        v = runs[k]
        fa = " ".join(f"{x:.4f}" for x in v["fold_acc"])
        print(f"run11 KD {k:13s} {fa:44s} {v['mean']:8.4f} {v['pooled']:8.4f}  "
              f"{v['mean']-ANCHOR['run11_full_ft_seed0']:+11.4f} "
              f"{v['mean']-ANCHOR['aves_full_ft_seed0']:+11.4f}"
              f"{'' if v.get('complete') else '  [INCOMPLETE]'}")
    print("-" * 96)
    for nm, val in (("run11 plain full_ft (seed0)", ANCHOR["run11_full_ft_seed0"]),
                    ("AVES  plain full_ft (seed0)", ANCHOR["aves_full_ft_seed0"]),
                    ("teacher: AVES L3 probe", teach_acc),
                    ("run11 frozen probe L3", ANCHOR["run11_frozen_probe_L3"])):
        print(f"{nm:22s} {'':44s} {val:8.4f}")
    print("=" * 96)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

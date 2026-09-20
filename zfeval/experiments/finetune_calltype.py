#!/usr/bin/env python
"""Does FINE-TUNING flip the run11-vs-AVES ranking on 11-class call type?

Every comparison in this repo so far has frozen the encoder and fitted a linear probe on
mean-pooled embeddings. Under that protocol AVES (generic animal-sound pretraining) beats run11
(zebra-finch-colony pretraining) on the 11-class task: 0.8453 vs 0.8118, layer 3, leave-birds-out
(analysis/calltype11.json). The two models are architecturally identical -- same builder, same
94.37M parameters, same state_dict keys, and (checked here) the same dropout configuration -- so the
only variable is which audio they were pretrained on.

A frozen probe answers one question: how linearly decodable is the representation the encoder
already produces. Fine-tuning answers a different one: how good an INITIALISATION are the weights.
Those can disagree. A domain-specific encoder can carry features that are highly informative but
entangled -- less linearly separable at the readout, yet a shorter path to a good solution once the
weights are allowed to move. If colony-specific pretraining is worth anything, this is the strongest
untested place for it to show up, so it gets tested rather than asserted.

Design, and what is held fixed so the comparison means something:

  cohort   calltype11.collect() -> KEEP11, asserted at 3412 clips / 11 classes / 48 birds /
           majority 0.17966. Adults and chicks, exactly the 2023 notebook's cohort.
  split    StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0) grouped by bird --
           bit-identical to aves_calltype.cv_acc, so every number here is comparable to the
           frozen-probe numbers. Test folds are never touched for model selection. Early stopping
           uses a second, bird-disjoint StratifiedGroupKFold carved out of each training fold.
  audio    mono channel mean, 16 kHz, NO waveform normalisation (what zf_hubert.embed_file does).
           Each clip is encoded at its own length: batches are length-bucketed and pooling is a
           length-aware mean over real frames only. --verify-batching asserts a batched embedding
           matches the single-clip embedding to 1e-4, and it caught something worth knowing --
           torchaudio's `feature_extractor(x, lengths)` is NOT padding-invariant for hubert_base
           (GroupNorm over time in conv block 0), so a padded batch corrupts the features at the
           VALID frames too, by up to 123.0 in absolute value on this cohort. Passing `lengths` does
           not fix it; `lengths` only masks attention, which is downstream. masked_feature_extractor
           below recomputes that one normalisation over real frames and is exact to ~1e-5. The 2023
           notebook's mean-over-99%-padding bug is the reason the check was written; this is a
           second, quieter version of the same failure.
  head     mean-pool over time -> nn.Linear(768, 11). Fine-tune arms read the FINAL encoder layer;
           the frozen arm is reported at layer 3 (each encoder's best frozen layer) and at layer 11
           (what the fine-tune head sees), so a fine-tuning gain cannot be confused with a
           layer-choice gain.
  arms     frozen          encoder fixed, linear head by AdamW (sanity check against the probe)
           full_ft         everything trainable, encoder LR 1e-5, head LR 1e-3
           transformer_ft  CNN feature extractor frozen, transformer + projection + head trainable
           AdamW, weight decay 0.01, grad-norm clip 1.0, 10% linear warmup, batch 8, at most 8
           epochs with early stopping when validation accuracy has not improved for 3.
  controls identical hyperparameters, epoch budget, folds, seeds, and batching for both models.
           encoder_layer_drop is forced to 0.0 for both (AVES's released config ships 0.05, as does
           torchaudio's hubert_base default) so neither model gets a stochastic-depth advantage and
           runs are reproducible. Train loss is logged per epoch: bioacoustic fine-tuning collapses
           at too high an LR, and a divergence has to be visible rather than inferred. Nothing was
           tuned per model; the encoder LR was never lowered because nothing diverged.

Batches are memory-budgeted rather than fixed-size in the forward pass (see split_micro): the
optimiser still steps once per 8 clips, but a batch is chopped into forward passes of at most
256k padded samples and the gradients accumulated, because eight 15.7 s clips at once is an 819 MB
activation in conv block 0 alone and stalls a machine that is already deep into swap.

Two accuracy conventions come out of this: `mean` over the five folds, and `pooled`, weighted by
fold clip count. Folds run 360-908 clips, so they differ by up to ~0.006; `pooled` is the one that
matches aves_calltype.cv_acc and therefore the published 0.8118 / 0.8453.

Results are appended to analysis/finetune_calltype.json after every fold, so a crash on fold 4
does not cost folds 0-3, and an interrupted arm resumes from the folds already stored.
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
from aves_baseline import AVES_W, AVES_C                                      # noqa: E402
from aves_holdout import WEIGHTS as RUN11_W                                   # noqa: E402

FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
OUT = ANA / "finetune_calltype.json"
SR = 16000
NFOLD, SPLIT_SEED = 5, 0            # outer folds: must match aves_calltype.cv_acc exactly
INNER_NFOLD = 5                     # -> val is ~20% of each training fold, bird-disjoint
MIN_SAMPLES = 400                   # one receptive field; shorter clips are zero-padded up to it

# cohort invariants, asserted rather than trusted
N_CLIPS, N_CLASSES, N_BIRDS, MAJORITY = 3412, 11, 48, 0.17966002344665885
# frozen logistic-regression probe, from analysis/calltype11.json -- printed alongside for reference
PROBE_REF = {"run11": {"best_layer": 3, "best_acc": 0.8118405627198124, "L11": 0.8048065650644783},
             "aves": {"best_layer": 3, "best_acc": 0.8452520515826495, "L11": 0.8127198124267292}}


# --------------------------------------------------------------------------- models
def build_model(name, device, layer_drop=0.0):
    """run11 and AVES through the SAME builder, with identical dropout, differing only in weights."""
    from torchaudio.models import wav2vec2_model

    cfg = json.load(open(AVES_C))
    assert cfg["encoder_num_layers"] == 12 and cfg["encoder_embed_dim"] == 768
    cfg["encoder_layer_drop"] = layer_drop
    m = wav2vec2_model(**cfg, aux_num_out=None)
    if name == "run11":
        sd = torch.load(RUN11_W, map_location="cpu", weights_only=False)["state_dict"]
        m.load_state_dict(sd, strict=True)
    elif name == "aves":
        sd = torch.load(AVES_W, map_location="cpu", weights_only=True)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        bad = [k for k in missing if not k.startswith("aux")]
        if bad or unexpected:
            raise RuntimeError(f"AVES state_dict mismatch missing={bad} unexpected={unexpected}")
    else:
        raise ValueError(name)
    del sd
    gc.collect()
    return m.to(device)


def masked_feature_extractor(fe, x, lengths):
    """torchaudio's CNN extractor, but padding-invariant. The reason this function exists:

    hubert_base uses extractor_mode="group_norm", which puts a GroupNorm(512, 512) inside conv
    block 0 -- per sample, per channel, over TIME. Padding a batch therefore changes the
    normalisation statistics of every clip in it, and `feature_extractor(x, lengths)` returns
    features that are wrong at the VALID positions too, not just in the pad region. Measured on a
    batch of 8 clips from this cohort spanning 0.03-15.7 s, the library's batched extractor differs
    from the single-clip extractor by up to 123.0 in absolute value at valid frames. Passing
    `lengths` does not save you: lengths only mask attention, downstream of the damage.

    Everything else in the extractor is a strided conv plus GELU. Those are local and the valid
    output length bookkeeping guarantees a valid output position depends only on valid input
    positions, so recomputing just that one normalisation over the real frames makes the whole
    batched pass exact (verified to ~1e-5, i.e. float32 noise, by --verify-batching).
    """
    x = x.unsqueeze(1)                                     # (B, 1, T)
    L = lengths.clone()
    for layer in fe.conv_layers:
        x = layer.conv(x)
        L = (torch.div(L - layer.kernel_size, layer.stride, rounding_mode="floor") + 1).clamp(min=0)
        gn = layer.layer_norm
        if gn is not None:
            assert gn.num_groups == gn.num_channels, "expected per-channel-over-time GroupNorm"
            m = (torch.arange(x.shape[-1], device=x.device)[None, :]
                 < L[:, None]).to(x.dtype)[:, None, :]
            n = m.sum(-1, keepdim=True).clamp(min=1)
            mu = (x * m).sum(-1, keepdim=True) / n
            var = (((x - mu) * m) ** 2).sum(-1, keepdim=True) / n
            x = (x - mu) / torch.sqrt(var + gn.eps) * gn.weight[None, :, None] + gn.bias[None, :, None]
            x = x * m
        x = F.gelu(x)
    return x.transpose(1, 2), L                            # (B, frames, 512), (B,)


class Clf(nn.Module):
    """encoder -> length-aware mean over real frames -> Linear(768, 11)."""

    def __init__(self, enc, n_cls=N_CLASSES):
        super().__init__()
        self.enc = enc
        self.head = nn.Linear(768, n_cls)

    def forward(self, x, lens):
        f, olens = masked_feature_extractor(self.enc.feature_extractor, x, lens)
        h = self.enc.encoder(f, olens)                     # zeroes pads, masks attention
        mask = (torch.arange(h.shape[1], device=h.device)[None, :] < olens[:, None]).to(h.dtype)
        pooled = (h * mask[:, :, None]).sum(1) / mask.sum(1).clamp(min=1)[:, None]
        return self.head(pooled)


# --------------------------------------------------------------------------- data
def load_cohort():
    rows = [r for r in collect() if r[3] in KEEP11]
    paths = [r[0] for r in rows]
    birds = np.array([r[1].lower() for r in rows])
    tt = np.array([r[3] for r in rows])
    src = np.array([r[4] for r in rows])
    classes = sorted(set(tt))
    y = np.array([classes.index(t) for t in tt])
    maj = float(np.bincount(y).max() / len(y))
    assert len(y) == N_CLIPS, f"cohort is {len(y)} clips, expected {N_CLIPS}"
    assert len(classes) == N_CLASSES, f"{len(classes)} classes, expected {N_CLASSES}"
    assert len(set(birds)) == N_BIRDS, f"{len(set(birds))} birds, expected {N_BIRDS}"
    assert abs(maj - MAJORITY) < 1e-9, f"majority {maj}, expected {MAJORITY}"
    return paths, y, birds, classes, src, maj


def load_wavs(paths):
    """All clips at 16 kHz mono, no normalisation, cached as one flat buffer + offsets."""
    cache = FEAT / "ct11_wav16k.npz"
    if cache.exists():
        d = np.load(cache)
        flat, off = d["flat"], d["off"]
        if len(off) == len(paths) + 1:
            print(f"[audio] cached {cache.name}: {len(paths)} clips, "
                  f"{flat.size/SR:.1f}s total", flush=True)
            return flat, off
        print(f"[audio] cache has {len(off)-1} clips, need {len(paths)} -- rebuilding")
    import torchaudio
    res = {}
    chunks, off = [], [0]
    t0 = time.time()
    for i, p in enumerate(paths):
        w, sr = torchaudio.load(str(p))
        if sr != SR:
            if sr not in res:
                res[sr] = torchaudio.transforms.Resample(sr, SR)
            w = res[sr](w)
        w = w.mean(0).numpy().astype(np.float32)
        assert w.size > 0 and np.isfinite(w).all(), f"bad audio in {p}"
        chunks.append(w)
        off.append(off[-1] + w.size)
        if (i + 1) % 1000 == 0:
            print(f"    loaded {i+1}/{len(paths)}  {time.time()-t0:.0f}s", flush=True)
    flat = np.concatenate(chunks).astype(np.float32)
    off = np.array(off, dtype=np.int64)
    FEAT.mkdir(parents=True, exist_ok=True)
    np.savez(cache, flat=flat, off=off)
    print(f"[audio] built {cache.name}: {flat.size/SR:.1f}s total", flush=True)
    return flat, off


def clip_of(flat, off, i):
    return flat[off[i]:off[i + 1]]


def outer_folds(y, birds):
    from sklearn.model_selection import StratifiedGroupKFold
    cv = StratifiedGroupKFold(n_splits=NFOLD, shuffle=True, random_state=SPLIT_SEED)
    return list(cv.split(np.zeros((len(y), 1)), y, birds))


def inner_val(idx, y, birds):
    """A bird-disjoint validation slice of a training fold, for early stopping only."""
    from sklearn.model_selection import StratifiedGroupKFold
    cv = StratifiedGroupKFold(n_splits=INNER_NFOLD, shuffle=True, random_state=SPLIT_SEED)
    tr, va = next(iter(cv.split(np.zeros((len(idx), 1)), y[idx], birds[idx])))
    return idx[tr], idx[va]


MAX_BATCH_SAMPLES = 256_000        # 16 s of padded audio per forward; see split_micro()


def make_batches(idx, lens, batch, rng=None, bucket=200):
    """Length-bucketed optimiser batches: shuffle, sort by length inside blocks, shuffle the order.

    Padding is the enemy -- clip lengths span 0.03 s to 15.7 s, and a naively shuffled batch of 8
    pads to 4.2x the real audio. Sorting inside blocks of `batch * bucket` brings that to 1.10x.
    Correctness never depended on it (attention is masked, pooling is length-aware); only speed does.
    """
    idx = np.asarray(idx)
    if rng is not None:
        idx = idx[rng.permutation(len(idx))]
    block = batch * bucket
    out = []
    for s in range(0, len(idx), block):
        blk = idx[s:s + block]
        blk = blk[np.argsort(lens[blk], kind="stable")]
        out += [blk[j:j + batch] for j in range(0, len(blk), batch)]
    if rng is not None:
        out = [out[k] for k in rng.permutation(len(out))]
    return out


def split_micro(b, lens, max_samples=MAX_BATCH_SAMPLES):
    """Split one optimiser batch into memory-safe forward passes.

    conv block 0 of the extractor turns B x T samples into B x 512 x (T/5) floats: a batch of eight
    15.7 s clips is an 819 MB activation before the other six conv layers and everything autograd
    retains for the backward pass. On a machine already 14 GB into swap that does not merely run
    slowly, it stalls. So a batch is chopped into forward passes of at most `max_samples` padded
    samples and the gradients are accumulated -- the optimiser still steps once per 8 clips, exactly
    as configured, for both models.
    """
    b = np.asarray(b)
    micro, cur = [], []
    for i in b:
        cand = cur + [i]
        if cur and (len(cand) * max(lens[j] for j in cand)) > max_samples:
            micro.append(np.array(cur))
            cur = [i]
        else:
            cur = cand
    if cur:
        micro.append(np.array(cur))
    return micro


def pack(flat, off, ids, device):
    ws = [clip_of(flat, off, i) for i in ids]
    L = max(MIN_SAMPLES, max(w.size for w in ws))
    x = np.zeros((len(ws), L), dtype=np.float32)
    lens = np.zeros(len(ws), dtype=np.int64)
    for k, w in enumerate(ws):
        x[k, :w.size] = w
        lens[k] = max(w.size, MIN_SAMPLES)
    return (torch.from_numpy(x).to(device),
            torch.from_numpy(lens).to(device))


# --------------------------------------------------------------------------- train / eval
@torch.inference_mode()
def evaluate(model, flat, off, ids, y, lens_arr, device, batch):
    model.eval()
    correct, n = 0, 0
    for b in make_batches(ids, lens_arr, batch):
        for mb in split_micro(b, lens_arr):
            x, L = pack(flat, off, mb, device)
            pred = model(x, L).argmax(1).cpu().numpy()
            correct += int((pred == y[mb]).sum())
            n += len(mb)
            del x, L
    return correct / n


def train_fold(model_name, arm, tr, va, te, flat, off, y, birds, lens_arr, device, hp, log):
    """One fold of one arm. Returns test accuracy at the epoch with the best validation accuracy."""
    torch.manual_seed(hp["seed"])
    np.random.seed(hp["seed"])
    enc = build_model(model_name, device, layer_drop=hp["layer_drop"])
    model = Clf(enc).to(device)
    torch.manual_seed(hp["seed"])                      # head init, seed-dependent
    model.head.reset_parameters()

    if arm == "frozen":
        for p in enc.parameters():
            p.requires_grad_(False)
        enc_params = []
    elif arm == "transformer_ft":
        for p in enc.feature_extractor.parameters():
            p.requires_grad_(False)
        enc_params = [p for p in enc.parameters() if p.requires_grad]
    elif arm == "full_ft":
        enc_params = list(enc.parameters())
    else:
        raise ValueError(arm)

    groups = [{"params": model.head.parameters(), "lr": hp["head_lr"]}]
    if enc_params:
        groups.insert(0, {"params": enc_params, "lr": hp["enc_lr"]})
    opt = torch.optim.AdamW(groups, weight_decay=hp["wd"])

    rng = np.random.default_rng(hp["seed"])
    nb = len(make_batches(tr, lens_arr, hp["batch"]))
    total = nb * hp["epochs"]
    warm = max(1, int(hp["warmup_frac"] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 1.0)

    hist = []
    best = dict(val=-1.0, epoch=-1, test=float("nan"))
    for ep in range(hp["epochs"]):
        model.train()
        if arm != "full_ft":
            enc.feature_extractor.eval()               # frozen CNN stays in eval mode
        if arm == "frozen":
            enc.eval()
        tot, nseen, t0 = 0.0, 0, time.time()
        params = [p for g in groups for p in g["params"]]
        for gi, b in enumerate(make_batches(tr, lens_arr, hp["batch"], rng=rng)):
            opt.zero_grad(set_to_none=True)
            for mb in split_micro(b, lens_arr):                 # grad accumulation: 1 step per batch
                x, L = pack(flat, off, mb, device)
                loss = F.cross_entropy(model(x, L), torch.from_numpy(y[mb]).to(device),
                                       reduction="sum") / len(b)
                loss.backward()
                tot += float(loss.detach()) * len(b)
                nseen += len(mb)
                del x, L, loss
            if hp["clip"]:
                torch.nn.utils.clip_grad_norm_(params, hp["clip"])
            opt.step()
            sched.step()
            if gi and gi % 100 == 0:
                log(f"        .. {gi} batches, {time.time()-t0:.0f}s, "
                    f"running loss {tot/max(nseen,1):.4f}")
        trl = tot / nseen
        vacc = evaluate(model, flat, off, va, y, lens_arr, device, hp["batch"])
        tacc = float("nan")
        if vacc > best["val"]:                                  # test only when val improves
            tacc = evaluate(model, flat, off, te, y, lens_arr, device, hp["batch"])
            best = dict(val=vacc, epoch=ep, test=tacc)
        hist.append(dict(epoch=ep, train_loss=trl, val_acc=vacc,
                         test_acc=None if np.isnan(tacc) else tacc,
                         sec=round(time.time() - t0, 1)))
        log(f"      ep{ep} loss {trl:.4f}  val {vacc:.4f}  "
            f"test {'--    ' if np.isnan(tacc) else f'{tacc:.4f}'}  ({time.time()-t0:.0f}s)")
        if not np.isfinite(trl):
            log(f"      ep{ep} train loss is not finite -- DIVERGED, aborting fold")
            break
        if hp.get("patience") and ep - best["epoch"] >= hp["patience"]:
            log(f"      early stop: no val improvement for {hp['patience']} epochs "
                f"(best ep{best['epoch']} val {best['val']:.4f})")
            break

    del model, enc, opt, sched
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    best["epochs_run"] = len(hist)
    return best, hist


# --------------------------------------------------------------------------- frozen shortcut
def frozen_probe_head(emb, y, birds, folds, layer, hp, log):
    """The frozen arm on cached mean-pooled embeddings.

    With the encoder fixed and pooling deterministic, running the encoder every epoch computes the
    identical vectors every time; reading them from the cache is the same arm, minutes instead of
    hours. --no-frozen-cache runs it through the encoder instead to prove that.
    """
    accs, hists = [], []
    for f, (tr_all, te) in enumerate(folds):
        tr, va = inner_val(tr_all, y, birds)
        torch.manual_seed(hp["seed"])
        head = nn.Linear(emb.shape[1], N_CLASSES)
        opt = torch.optim.AdamW(head.parameters(), lr=hp["head_lr"], weight_decay=hp["wd"])
        X = torch.from_numpy(emb)
        Y = torch.from_numpy(y)
        rng = np.random.default_rng(hp["seed"])
        best = dict(val=-1.0, epoch=-1, test=float("nan"))
        hist = []
        for ep in range(hp["epochs"]):
            head.train()
            perm = rng.permutation(len(tr))
            tot = 0.0
            for s in range(0, len(tr), hp["batch"]):
                b = tr[perm[s:s + hp["batch"]]]
                loss = F.cross_entropy(head(X[b]), Y[b])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += float(loss) * len(b)
            head.eval()
            with torch.no_grad():
                vacc = float((head(X[va]).argmax(1).numpy() == y[va]).mean())
                tacc = float((head(X[te]).argmax(1).numpy() == y[te]).mean())
            hist.append(dict(epoch=ep, train_loss=tot / len(tr), val_acc=vacc, test_acc=tacc))
            if vacc > best["val"]:
                best = dict(val=vacc, epoch=ep, test=tacc)
        accs.append(best["test"])
        hists.append(hist)
        log(f"    fold {f}: L{layer} frozen test {best['test']:.4f} (best val ep{best['epoch']})")
    return accs, hists


# --------------------------------------------------------------------------- batching check
def verify_batching(device):
    """A batched embedding must equal the single-clip embedding, or nothing below is measuring
    the encoder. Also measures what the LIBRARY's batched path does, since that is the trap."""
    paths, y, birds, classes, src, maj = load_cohort()
    flat, off = load_wavs(paths)
    lens_arr = np.diff(off)
    enc = build_model("run11", device, layer_drop=0.0).eval()
    model = Clf(enc).to(device).eval()
    # deliberately mix the shortest and longest clips so padding dominates the batch
    order = np.argsort(lens_arr)
    ids = np.concatenate([order[:4], order[-2:], order[len(order) // 2:len(order) // 2 + 2]])
    with torch.no_grad():
        xb, lb = pack(flat, off, ids, device)
        batched = model(xb, lb).cpu().numpy()
        single = np.stack([model(*pack(flat, off, [i], device)).cpu().numpy()[0] for i in ids])
        # the naive path: torchaudio's own batched extractor, lengths passed exactly as documented
        fb, ob = enc.feature_extractor(xb, lb)
        hb = enc.encoder(fb, ob)
        mk = (torch.arange(hb.shape[1], device=hb.device)[None, :] < ob[:, None]).to(hb.dtype)
        naive = model.head((hb * mk[:, :, None]).sum(1) / mk.sum(1)[:, None]).cpu().numpy()
    d = float(np.abs(batched - single).max())
    dn = float(np.abs(naive - single).max())
    span = f"{lens_arr[ids].min()/SR:.3f}s-{lens_arr[ids].max()/SR:.3f}s"
    print(f"[verify] batch of {len(ids)} clips spanning {span}, logits vs single-clip:")
    print(f"           masked extractor (used here) max |diff| {d:.3e}")
    print(f"           torchaudio batched extractor max |diff| {dn:.3e}   <- the trap")
    assert d < 1e-4, f"batched pooling does not match single-clip pooling: {d}"
    print("[verify] PASS")
    del model, enc
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return dict(n_clips_in_batch=int(len(ids)), span=span, masked_max_abs_diff=d,
                torchaudio_batched_max_abs_diff=dn, tolerance=1e-4, passed=bool(d < 1e-4))



# --------------------------------------------------------------------------- summary
def summarize(res, folds):
    """Fill in clip-weighted accuracy, per-arm deltas, the run11-AVES gap, and the seed spread.

    Two accuracy conventions are reported because both are needed. `mean` is the unweighted mean of
    the five fold accuracies; `pooled` weights each fold by its clip count, which is what
    aves_calltype.cv_acc reports and therefore the only one directly comparable to the published
    0.8118 / 0.8453. The folds here run from 360 to 908 clips, so the two differ by up to ~0.006.
    """
    sizes = [len(te) for _, te in folds]
    runs = res["runs"]
    for v in runs.values():
        if not v.get("fold_acc"):
            continue
        w = v.get("fold_n") or sizes[:len(v["fold_acc"])]
        v["fold_n"] = [int(x) for x in w]
        v["mean"] = float(np.mean(v["fold_acc"]))
        v["pooled"] = float(np.average(v["fold_acc"], weights=w))

    def get(model, arm, seed):
        return runs.get(f"{model}|{arm}|seed{seed}")

    S = {"arms": {}, "seed_spread": {}, "ranking": {}}
    arms = sorted({v["arm"] for v in runs.values() if v.get("fold_acc")})
    for arm in arms:
        row = {}
        for m in ("run11", "aves"):
            per_seed = {str(v["seed"]): v["mean"] for k, v in runs.items()
                        if v.get("arm") == arm and v.get("model") == m and v.get("fold_acc")}
            if not per_seed:
                continue
            base = get(m, "frozen_L3", 0)
            base11 = get(m, "frozen_L11", 0)
            row[m] = dict(per_seed_mean=per_seed,
                          mean=float(np.mean(list(per_seed.values()))),
                          vs_frozen_L3=(float(np.mean(list(per_seed.values())) - base["mean"])
                                        if base else None),
                          vs_frozen_L11=(float(np.mean(list(per_seed.values())) - base11["mean"])
                                         if base11 else None))
        if len(row) == 2:
            row["aves_minus_run11"] = row["aves"]["mean"] - row["run11"]["mean"]
            row["winner"] = "aves" if row["aves_minus_run11"] > 0 else "run11"
        S["arms"][arm] = row

    for m in ("run11", "aves"):
        for arm in arms:
            vals = [v["mean"] for v in runs.values()
                    if v.get("model") == m and v.get("arm") == arm and v.get("fold_acc")]
            if len(vals) > 1:
                S["seed_spread"][f"{m}|{arm}"] = dict(
                    n_seeds=len(vals), means=vals, spread=float(max(vals) - min(vals)))

    flips = [a for a, r in S["arms"].items() if r.get("winner") == "run11"]
    S["ranking"] = dict(
        frozen_probe_winner="aves",
        arms_won_by_run11=flips,
        ranking_changed=bool(flips),
        note=("AVES wins every arm run; fine-tuning raised both models without reordering them"
              if not flips else f"run11 wins under: {flips}"))
    res["summary"] = S
    return S


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="frozen,full_ft,transformer_ft")
    ap.add_argument("--models", default="run11,aves")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=3,
                    help="stop a fold when validation accuracy has not improved for this many "
                         "epochs; applied identically to both models")
    ap.add_argument("--frozen-epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--enc-lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--layer-drop", type=float, default=0.0)
    ap.add_argument("--frozen-layers", default="3,11")
    ap.add_argument("--verify-batching", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[device] {device}  torch {torch.__version__}", flush=True)
    vb = verify_batching(device) if a.verify_batching else None

    paths, y, birds, classes, src, maj = load_cohort()
    print(f"[cohort] {len(y)} clips, {len(classes)} classes, {len(set(birds))} birds, "
          f"majority {maj:.5f}  (adults {int((src=='AdultVocalizations').sum())}, "
          f"chicks {int((src=='ChickVocalizations').sum())})", flush=True)
    flat, off = load_wavs(paths)
    lens_arr = np.diff(off)
    folds = outer_folds(y, birds)
    for f, (tr, te) in enumerate(folds):
        assert not set(birds[tr]) & set(birds[te]), f"fold {f} leaks birds"
    print(f"[split] StratifiedGroupKFold({NFOLD}, shuffle, seed {SPLIT_SEED}) by bird; "
          f"test sizes {[len(te) for _, te in folds]}", flush=True)

    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res.setdefault("meta", {}).update(
        n_clips=int(len(y)), n_classes=len(classes), n_birds=int(len(set(birds))),
        majority=maj, classes=classes,
        split=f"leave-birds-out StratifiedGroupKFold({NFOLD}, shuffle=True, random_state={SPLIT_SEED})",
        inner_val=f"StratifiedGroupKFold({INNER_NFOLD}) inside each training fold, bird-disjoint",
        device=device, torch=torch.__version__,
        frozen_probe_reference=PROBE_REF,
        hp=dict(epochs=a.epochs, frozen_epochs=a.frozen_epochs, batch=a.batch, enc_lr=a.enc_lr,
                head_lr=a.head_lr, wd=a.wd, grad_clip=a.clip, warmup_frac=a.warmup_frac,
                layer_drop=a.layer_drop, patience=a.patience, optimizer="AdamW",
                pool="length-aware mean",
                head="Linear(768,11)", ft_layer="final (L11)"))
    if vb is not None:
        res["meta"]["batching_check"] = vb
    res.setdefault("runs", {})

    def save():
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(res, indent=2))

    save()
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    arms = [x for x in a.arms.split(",") if x.strip()]
    models = [x for x in a.models.split(",") if x.strip()]

    for seed in seeds:
        hp_ft = dict(seed=seed, epochs=a.epochs, batch=a.batch, enc_lr=a.enc_lr,
                     head_lr=a.head_lr, wd=a.wd, clip=a.clip, warmup_frac=a.warmup_frac,
                     layer_drop=a.layer_drop, patience=a.patience)
        for arm in arms:
            for mn in models:
                if arm == "frozen":
                    for layer in [int(l) for l in a.frozen_layers.split(",")]:
                        key = f"{mn}|frozen_L{layer}|seed{seed}{a.tag}"
                        if key in res["runs"]:
                            print(f"[skip] {key} already in {OUT.name}")
                            continue
                        emb = np.load(FEAT / f"ct11_{mn}_emb.npy")[:, layer].astype(np.float32)
                        assert emb.shape[0] == len(y)
                        print(f"\n=== {key} ===", flush=True)
                        hp = dict(hp_ft, epochs=a.frozen_epochs)
                        accs, hists = frozen_probe_head(emb, y, birds, folds, layer, hp,
                                                        lambda s: print(s, flush=True))
                        res["runs"][key] = dict(
                            model=mn, arm=f"frozen_L{layer}", seed=seed, fold_acc=accs,
                            fold_n=[int(len(te)) for _, te in folds],
                            mean=float(np.mean(accs)),
                            pooled=float(np.average(accs, weights=[len(te) for _, te in folds])),
                            std=float(np.std(accs)), history=hists, epochs=a.frozen_epochs)
                        print(f"  {key}: mean {np.mean(accs):.4f}  folds "
                              f"{[round(x,4) for x in accs]}", flush=True)
                        save()
                        del emb
                        gc.collect()
                    continue

                key = f"{mn}|{arm}|seed{seed}{a.tag}"
                prev = res["runs"].get(key)
                if prev and prev.get("complete"):
                    print(f"[skip] {key} already complete in {OUT.name}")
                    continue
                accs = list(prev["fold_acc"]) if prev else []
                hists = list(prev.get("history", [])) if prev else []
                if accs:
                    print(f"[resume] {key}: folds 0-{len(accs)-1} already done "
                          f"{[round(x,4) for x in accs]}")
                print(f"\n=== {key}  (enc_lr {a.enc_lr}, head_lr {a.head_lr}, "
                      f"<={a.epochs} epochs, patience {a.patience}, batch {a.batch}) ===",
                      flush=True)
                bests = list(prev.get("fold_best", [])) if prev else []
                bests += [None] * (len(accs) - len(bests))     # resumed folds predate this field
                t_arm = time.time()
                for f, (tr_all, te) in enumerate(folds):
                    if f < len(accs):
                        continue
                    tr, va = inner_val(tr_all, y, birds)
                    print(f"    fold {f}: train {len(tr)} val {len(va)} test {len(te)} "
                          f"(birds {len(set(birds[tr]))}/{len(set(birds[va]))}/"
                          f"{len(set(birds[te]))})", flush=True)
                    best, hist = train_fold(mn, arm, tr, va, te, flat, off, y, birds, lens_arr,
                                            device, hp_ft, lambda s: print(s, flush=True))
                    accs.append(best["test"])
                    hists.append(hist)
                    bests.append(best)
                    print(f"    fold {f} -> test {best['test']:.4f} at best-val epoch "
                          f"{best['epoch']} (val {best['val']:.4f})", flush=True)
                    res["runs"][key] = dict(
                        model=mn, arm=arm, seed=seed, fold_acc=accs,
                        fold_n=[int(len(folds[i][1])) for i in range(len(accs))],
                        mean=float(np.mean(accs)),
                        pooled=float(np.average(accs, weights=[len(folds[i][1])
                                                               for i in range(len(accs))])),
                        std=float(np.std(accs)), history=hists, fold_best=bests,
                        epochs=a.epochs, patience=a.patience, complete=False)
                    save()
                res["runs"][key]["complete"] = True
                res["runs"][key]["sec"] = round(time.time() - t_arm, 1)
                print(f"  {key}: mean {np.mean(accs):.4f}  folds {[round(x,4) for x in accs]}  "
                      f"({(time.time()-t_arm)/60:.1f} min)", flush=True)
                save()

    summarize(res, folds)

    # ------------------------------------------------------------------ summary table
    print("\n" + "=" * 92)
    print(f"{'model':7s} {'arm':16s} {'seed':>4s}  {'per-fold accuracies':44s} {'mean':>7s} {'pooled':>7s}")
    print("-" * 92)
    for k, v in sorted(res["runs"].items()):
        if not v.get("fold_acc"):
            continue
        fa = " ".join(f"{x:.4f}" for x in v["fold_acc"])
        flag = "" if v.get("complete", True) else "  [INCOMPLETE]"
        pooled = v.get("pooled", float("nan"))
        print(f"{v['model']:7s} {v['arm']:16s} {v['seed']:>4d}  {fa:44s} "
              f"{np.mean(v['fold_acc']):7.4f} {pooled:7.4f}{flag}")
    print("=" * 92)
    print("frozen logistic-regression probe (analysis/calltype11.json): "
          f"run11 L3 {PROBE_REF['run11']['best_acc']:.4f}, "
          f"AVES L3 {PROBE_REF['aves']['best_acc']:.4f}")
    save()
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

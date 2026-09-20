#!/usr/bin/env python
"""Is the READOUT the bottleneck, not the encoder?

Every encoder comparison in this repo reads a model out the same way: mean-pool its frames over
time, then fit a linear probe. On the 11-class call-type task that gives

    run11 (zebra-finch colony HuBERT)  L3  0.8118
    aves-base-bio (generic animal SSL) L3  0.8453        (analysis/calltype11.json)

and the whole "generic pretraining wins" conclusion rests on it. But mean pooling is not a neutral
measurement device. It is a *destructive* one: it averages a call's frames into one vector and throws
away every bit of temporal structure -- onset, offset, the order of a frequency sweep, where in the
call the distinguishing energy sits. A recent bioacoustics review reports BEATs going from 94.10 to
97.98 BEANS AUROC purely by replacing linear probing with ATTENTIVE probing: same weights, same
frozen encoder, different readout. If that happens here, the run11-vs-AVES ranking is a fact about
mean pooling, not about the encoders.

The specific hypothesis being tested, and why it is not symmetric: if run11's zebra-finch pretraining
encodes call identity in a temporally LOCALISED way (a few frames at the onset, say) while AVES
encodes it diffusely, then mean pooling dilutes run11 more than it dilutes AVES, and a readout that
can *select* frames should recover more for run11 than for AVES. That predicts a specific pattern --
attentive pooling closes or reverses the gap -- which is distinguishable from the null (both models
rise by the same amount and the ranking survives).

Readouts, applied identically to both encoders, cheapest first:

  mean            mean over real frames -> StandardScaler -> LogisticRegression   (the baseline)
  meanstd         [mean ; std] concatenated (1536-d) -> same linear probe
  max             max over real frames -> same linear probe                       (diagnostic)
  first/mid/last  a single frame -> same linear probe                             (diagnostic:
                  where in a call does the information sit?)
  meanmax         [mean ; max] (1536-d) -> same linear probe
  multi           [mean ; std ; max ; first ; mid ; last] (4608-d) -> same linear probe. This is the
                  confound-free version of "give the readout temporal structure": it adds temporal
                  information while keeping the baseline's OWN optimiser and L2 strength, so it
                  cannot be beaten by a regularisation artefact the way a fresh AdamW head can.
  mean_adamw      mean-pool -> nn.Linear, trained by AdamW                        (CONTROL: isolates
                  "the head was trained by AdamW instead of LBFGS" from "the pooling changed")
  attn            learned scoring vector over frames -> softmax -> weighted mean -> nn.Linear
  attn_cat        [attentive-pooled ; masked mean] (1536-d) -> nn.Linear. Attention can only ADD.
  attn4           4-head attentive pooling, heads concatenated (3072-d) -> nn.Linear
  trf             project 768->256, 2 transformer blocks over frames, attentive pool -> nn.Linear

Two implementation traps, both of which silently produce plausible wrong numbers:

  (1) The cached embeddings ct11_{run11,aves}_emb.npy are ALREADY mean-pooled (3412, 12, 768). They
      can only serve arm 1. Everything else needs per-clip FRAME SEQUENCES, so this script
      re-extracts them (layers 1 and 3) as a ragged float16 store + offsets index.

  (2) torchaudio's feature extractor is not padding-invariant for these models: extractor_mode
      "group_norm" normalises over TIME inside conv block 0, so a zero-padded batch corrupts short
      clips at their VALID frames, and passing `lengths` does not help (lengths only mask attention,
      downstream of the damage). Frame extraction therefore runs ONE CLIP AT A TIME, and
      `--stage verify` measures the size of that failure, shows finetune_calltype's
      masked_feature_extractor repairs it, and checks that the batched *pooling heads* here -- whose
      softmax/mean/std must mask the padded positions -- agree with the single-clip path to ~1e-5.

Protocol, held fixed across every arm and both encoders: calltype11.collect() -> KEEP11, asserted at
3412 clips / 11 classes / 48 birds / majority 0.17966; outer split
StratifiedGroupKFold(5, shuffle=True, random_state=0) grouped by BIRD, bit-identical to
aves_calltype.cv_acc; learned heads trained by AdamW with early stopping on a bird-disjoint
validation slice carved out of the TRAINING birds only (test birds are never touched for model
selection); identical hyperparameters, epoch budget and seeds for both encoders; >=2 seeds on the
learned arms, and a difference smaller than the seed spread is reported as not a difference. Every
headline comparison gets a cluster bootstrap over the 48 birds via aves_calltype.bird_bootstrap.

Results are checkpointed into analysis/attentive_probe.json after every arm.
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
import torch.nn.functional as Fn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from calltype11 import collect, KEEP11                                       # noqa: E402
from aves_calltype import cv_acc, bird_bootstrap, NFOLD, SEED                # noqa: E402
from finetune_calltype import (build_model, masked_feature_extractor,        # noqa: E402
                              load_wavs, clip_of, inner_val, MIN_SAMPLES)

FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
OUT = ANA / "attentive_probe.json"
SR = 16000
LAYERS = (1, 3)                      # 3 = each encoder's best frozen layer; 1 = a shallow control
ENCODERS = ("run11", "aves")

N_CLIPS, N_CLASSES, N_BIRDS, MAJORITY = 3412, 11, 48, 0.17966002344665885
BASELINE_REF = {"run11": 0.8118405627198124, "aves": 0.8452520515826495}      # calltype11.json, L3

# learned-head hyperparameters -- identical for both encoders and every learned arm
HP = dict(lr=1e-3, batch=64, max_frames=16384,
          max_epochs=80, patience=12, attn_hidden=128, trf_dim=256, trf_layers=2, trf_heads=4)

# Regularisation is grid-searched PER FOLD on the bird-disjoint validation slice, never on test.
# It has to be: a first pass with a single setting (wd 0.01, no dropout) drove the training loss to
# 0.003 and put the AdamW mean-pool control BELOW the sklearn mean-pool baseline it is supposed to
# reproduce (aves 0.8356/0.8414 vs 0.8455) -- i.e. the learned arms were losing to the baseline on
# regularisation strength, not on pooling. Comparing attentive pooling against mean pooling under a
# setting that handicaps every learned arm would have measured the handicap. Each arm and each
# encoder gets the same grid and the same selection rule.
GRID = [(wd, dr) for wd in (0.01, 0.1, 1.0) for dr in (0.0, 0.5)]


# --------------------------------------------------------------------------- cohort
def load_cohort():
    rows = [r for r in collect() if r[3] in KEEP11]
    paths = [r[0] for r in rows]
    birds = np.array([r[1].lower() for r in rows])
    tt = np.array([r[3] for r in rows])
    classes = sorted(set(tt))
    y = np.array([classes.index(t) for t in tt])
    maj = float(np.bincount(y).max() / len(y))
    assert len(y) == N_CLIPS, f"cohort is {len(y)} clips, expected {N_CLIPS}"
    assert len(classes) == N_CLASSES, f"{len(classes)} classes, expected {N_CLASSES}"
    assert len(set(birds)) == N_BIRDS, f"{len(set(birds))} birds, expected {N_BIRDS}"
    assert abs(maj - MAJORITY) < 1e-9, f"majority {maj}, expected {MAJORITY}"
    print(f"[cohort] {len(y)} clips, {len(classes)} classes, {len(set(birds))} birds, "
          f"majority {maj:.5f}", flush=True)
    return paths, y, birds, classes


def outer_folds(y, birds):
    """Bit-identical to aves_calltype.cv_acc's split. Asserted, not assumed: StratifiedGroupKFold
    must not look at the VALUES of X, or the folds here and there would differ."""
    from sklearn.model_selection import StratifiedGroupKFold
    cv = StratifiedGroupKFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
    a = list(cv.split(np.zeros((len(y), 1)), y, birds))
    b = list(cv.split(np.random.default_rng(0).normal(size=(len(y), 768)), y, birds))
    for (t1, e1), (t2, e2) in zip(a, b):
        assert np.array_equal(t1, t2) and np.array_equal(e1, e2), \
            "StratifiedGroupKFold depends on X's values -- splits are not comparable"
    return a


# --------------------------------------------------------------------------- frame store
def store_paths(enc):
    return {l: FEAT / f"ct11_frames_{enc}_L{l}.npy" for l in LAYERS}, FEAT / "ct11_frames_off.npy"


@torch.no_grad()
def extract_frames(enc, flat, off, device):
    """Per-clip frame sequences for LAYERS, ONE CLIP AT A TIME (group_norm is not pad-invariant).

    Stored as a concatenated (total_frames, 768) float16 array per layer plus a shared int64 offsets
    index, so a clip is store[off[i]:off[i+1]]. 3412 clips -> 67,858 frames -> ~104 MB per layer.
    """
    sp, op = store_paths(enc)
    if all(p.exists() for p in sp.values()) and op.exists():
        print(f"[frames] cached {enc}: " + ", ".join(p.name for p in sp.values()), flush=True)
        return

    n = len(off) - 1
    nfr = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        T = max(off[i + 1] - off[i], MIN_SAMPLES)
        nfr[i + 1] = nfr[i] + max((T - 400) // 320 + 1, 1)
    total = int(nfr[-1])
    print(f"[frames] {enc}: {n} clips -> {total} frames, "
          f"{total*768*2/1e6:.0f} MB per layer", flush=True)

    store = {l: np.lib.format.open_memmap(sp[l].with_suffix(".tmp.npy"), mode="w+",
                                          dtype=np.float16, shape=(total, 768)) for l in LAYERS}
    m = build_model(enc, device).eval()
    need = max(LAYERS) + 1
    t0 = time.time()
    for i in range(n):
        w = clip_of(flat, off, i)
        x = torch.from_numpy(np.asarray(w, dtype=np.float32)).unsqueeze(0)
        if x.shape[-1] < MIN_SAMPLES:                    # calltype11.embed_all does exactly this
            x = Fn.pad(x, (0, MIN_SAMPLES - x.shape[-1]))
        feats, _ = m.extract_features(x.to(device), None, num_layers=need)
        got = feats[0].shape[1]
        exp = nfr[i + 1] - nfr[i]
        assert got == exp, f"clip {i}: encoder gave {got} frames, index expects {exp}"
        for l in LAYERS:
            store[l][nfr[i]:nfr[i + 1]] = feats[l][0].cpu().numpy().astype(np.float16)
        del feats
        if (i + 1) % 500 == 0:
            print(f"    {enc} {i+1}/{n}  {time.time()-t0:.0f}s", flush=True)
    for l in LAYERS:
        store[l].flush()
    del store, m
    gc.collect()
    for l in LAYERS:
        sp[l].with_suffix(".tmp.npy").rename(sp[l])
    np.save(op, nfr)
    print(f"[frames] {enc} written, {time.time()-t0:.0f}s", flush=True)


def load_store(enc, layer):
    sp, op = store_paths(enc)
    return np.load(sp[layer], mmap_mode="r"), np.load(op)


# --------------------------------------------------------------------------- fixed poolings
def fixed_pool(F, nfr, kind):
    """(N, D) or (N, 2D) pooled vectors from the ragged frame store. No learning, no masking bugs:
    each clip is sliced at its own length so padding never enters."""
    n = len(nfr) - 1
    D = F.shape[1]
    W = {"meanstd": 2 * D, "meanmax": 2 * D, "multi": 6 * D}.get(kind, D)
    out = np.zeros((n, W), dtype=np.float32)
    for i in range(n):
        h = np.asarray(F[nfr[i]:nfr[i + 1]], dtype=np.float32)
        if kind == "mean":
            out[i] = h.mean(0)
        elif kind == "meanstd":
            out[i] = np.concatenate([h.mean(0), h.std(0)])
        elif kind == "max":
            out[i] = h.max(0)
        elif kind == "first":
            out[i] = h[0]
        elif kind == "mid":
            out[i] = h[h.shape[0] // 2]
        elif kind == "last":
            out[i] = h[-1]
        elif kind == "meanmax":
            out[i] = np.concatenate([h.mean(0), h.max(0)])
        elif kind == "multi":
            out[i] = np.concatenate([h.mean(0), h.std(0), h.max(0),
                                     h[0], h[h.shape[0] // 2], h[-1]])
        else:
            raise ValueError(kind)
    return out


# --------------------------------------------------------------------------- learned heads
class MeanHead(nn.Module):
    """Control arm: the baseline pooling, but the head trained by AdamW like the attentive ones."""

    def __init__(self, D, n_cls, drop=0.0, **_):
        super().__init__()
        self.drop = nn.Dropout(drop)
        self.out = nn.Linear(D, n_cls)

    def forward(self, h, mask):
        m = mask.unsqueeze(-1)
        return self.out(self.drop((h * m).sum(1) / m.sum(1).clamp(min=1)))


class AttnHead(nn.Module):
    """Attentive pooling: score every frame, softmax over TIME, take the weighted mean.

    `mask` is 1 at real frames. Padded positions are set to -inf BEFORE the softmax -- not zeroed
    after -- because a zeroed-after softmax still lets padding steal probability mass from the real
    frames, which is exactly the bug class this whole script exists to avoid.
    """

    def __init__(self, D, n_cls, heads=1, hidden=128, drop=0.0, **_):
        super().__init__()
        self.heads = heads
        self.score = nn.Sequential(nn.Linear(D, hidden), nn.Tanh(), nn.Linear(hidden, heads))
        self.drop = nn.Dropout(drop)
        self.out = nn.Linear(D * heads, n_cls)

    def pool(self, h, mask):
        s = self.score(h)                                        # (B, T, H)
        s = s.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        a = torch.softmax(s, dim=1)                              # over TIME
        p = torch.einsum("bth,btd->bhd", a, h)                   # (B, H, D)
        return p.reshape(p.shape[0], -1)

    def forward(self, h, mask):
        return self.out(self.drop(self.pool(h, mask)))


class AttnCatHead(AttnHead):
    """Attentive pooling CONCATENATED with the plain masked mean.

    Why this arm exists: a softmax pooling head CAN reproduce the mean (constant scores give uniform
    weights), and near initialisation it roughly does -- so `attn` starts at mean-pool and any loss
    relative to mean-pool is the optimiser walking away from it and overfitting the scorer. Giving
    the classifier both vectors makes selection strictly additive: it can only help if the selected
    frames carry something the average does not.
    """

    def __init__(self, D, n_cls, heads=1, hidden=128, drop=0.0, **_):
        super().__init__(D, n_cls, heads=heads, hidden=hidden, drop=drop)
        self.out = nn.Linear(D * (heads + 1), n_cls)

    def forward(self, h, mask):
        m = mask.unsqueeze(-1)
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.out(self.drop(torch.cat([self.pool(h, mask), mean], -1)))


class TrfHead(nn.Module):
    """A small transformer over the frame sequence, then attentive pooling."""

    def __init__(self, D, n_cls, dim=256, layers=2, heads=4, hidden=128, drop=0.0, **_):
        super().__init__()
        self.proj = nn.Linear(D, dim)
        lay = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=2 * dim, dropout=0.1,
                                         batch_first=True, norm_first=True)
        self.trf = nn.TransformerEncoder(lay, layers)
        self.attn = AttnHead(dim, n_cls, heads=1, hidden=hidden, drop=drop)

    def forward(self, h, mask):
        z = self.trf(self.proj(h), src_key_padding_mask=(mask == 0))
        z = torch.nan_to_num(z)            # a fully-padded row cannot occur, but be explicit
        return self.attn(z, mask)


HEADS = {"mean_adamw": (MeanHead, {}),
         "attn": (AttnHead, dict(heads=1)),
         "attn4": (AttnHead, dict(heads=4)),
         "attn_cat": (AttnCatHead, dict(heads=1)),
         "trf": (TrfHead, dict(dim=HP["trf_dim"], layers=HP["trf_layers"], heads=HP["trf_heads"]))}


def build_head(arm, D, n_cls, drop=0.0):
    """One construction path, so verify() and train_head() cannot instantiate different heads."""
    cls, kw = HEADS[arm]
    return cls(D, n_cls, hidden=HP["attn_hidden"], drop=drop, **kw)


# --------------------------------------------------------------------------- batching
def make_batches(idx, lens, batch, max_frames, rng=None, bucket=40):
    """Length-bucketed batches. Padding is capped two ways: sort by length inside blocks, and cap
    B * Tmax frames so a batch of 782-frame clips cannot blow the activation budget."""
    idx = np.asarray(idx)
    if rng is not None:
        idx = idx[rng.permutation(len(idx))]
    out = []
    block = batch * bucket
    for s in range(0, len(idx), block):
        blk = idx[s:s + block]
        blk = blk[np.argsort(lens[blk], kind="stable")]
        cur = []
        for i in blk:
            cand = cur + [i]
            if cur and (len(cand) > batch or
                        len(cand) * max(lens[j] for j in cand) > max_frames):
                out.append(np.array(cur)); cur = [i]
            else:
                cur = cand
        if cur:
            out.append(np.array(cur))
    if rng is not None:
        out = [out[k] for k in rng.permutation(len(out))]
    return out


def pack(F, nfr, ids, mu, sd, device):
    """(B, Tmax, D) standardised frames + (B, Tmax) 1/0 mask, padding zeroed."""
    lens = [int(nfr[i + 1] - nfr[i]) for i in ids]
    T, D = max(lens), F.shape[1]
    x = np.zeros((len(ids), T, D), dtype=np.float32)
    m = np.zeros((len(ids), T), dtype=np.float32)
    for k, i in enumerate(ids):
        h = np.asarray(F[nfr[i]:nfr[i + 1]], dtype=np.float32)
        x[k, :h.shape[0]] = (h - mu) / sd
        m[k, :h.shape[0]] = 1.0
    return torch.from_numpy(x).to(device), torch.from_numpy(m).to(device)


def frame_stats(F, nfr, ids):
    """Per-dimension mean/std over the TRAINING frames only."""
    D = F.shape[1]
    s = np.zeros(D, dtype=np.float64); ss = np.zeros(D, dtype=np.float64); n = 0
    for i in ids:
        h = np.asarray(F[nfr[i]:nfr[i + 1]], dtype=np.float64)
        s += h.sum(0); ss += (h ** 2).sum(0); n += h.shape[0]
    mu = s / n
    sd = np.sqrt(np.maximum(ss / n - mu ** 2, 0)) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


# --------------------------------------------------------------------------- train / eval
@torch.no_grad()
def predict(model, F, nfr, ids, lens, mu, sd, device, n_cls):
    model.eval()
    P = np.zeros((len(ids), n_cls), dtype=np.float32)
    pos = {i: k for k, i in enumerate(ids)}
    for b in make_batches(ids, lens, HP["batch"], HP["max_frames"]):
        x, m = pack(F, nfr, b, mu, sd, device)
        p = torch.softmax(model(x, m).float(), 1).cpu().numpy()
        for k, i in enumerate(b):
            P[pos[i]] = p[k]
        del x, m
    return P


def train_head(arm, F, nfr, tr, va, y, lens, mu, sd, device, n_cls, seed, wd, drop):
    torch.manual_seed(seed)
    model = build_head(arm, F.shape[1], n_cls, drop=drop).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=wd)
    rng = np.random.default_rng(seed)
    best, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(HP["max_epochs"]):
        model.train()
        tot, nb = 0.0, 0
        for b in make_batches(tr, lens, HP["batch"], HP["max_frames"], rng=rng):
            x, m = pack(F, nfr, b, mu, sd, device)
            loss = Fn.cross_entropy(model(x, m), torch.from_numpy(y[b]).to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
            del x, m, loss
        Pv = predict(model, F, nfr, va, lens, mu, sd, device, n_cls)
        acc = float((Pv.argmax(1) == y[va]).mean())
        hist.append(dict(epoch=ep, loss=tot / max(nb, 1), val_acc=acc))
        if acc > best + 1e-6:
            best, bad = acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= HP["patience"]:
                break
    model.load_state_dict(best_state)
    return model, best, hist


def run_learned(arm, F, nfr, y, birds, folds, device, seed):
    n_cls = int(y.max() + 1)
    lens = np.diff(nfr)
    P = np.zeros((len(y), n_cls), dtype=np.float32)
    per_fold, logs = [], []
    for k, (tr_all, te) in enumerate(folds):
        tr, va = inner_val(tr_all, y, birds)
        assert not (set(birds[tr]) & set(birds[va])), "val birds leak into train"
        assert not (set(birds[tr_all]) & set(birds[te])), "test birds leak into train"
        mu, sd = frame_stats(F, nfr, tr_all)
        t0 = time.time()
        best = None
        grid = []
        for wd, drop in GRID:
            model, vb, hist = train_head(arm, F, nfr, tr, va, y, lens, mu, sd, device,
                                         n_cls, seed, wd, drop)
            grid.append(dict(wd=wd, drop=drop, val=vb, epochs=len(hist),
                             first_loss=hist[0]["loss"], last_loss=hist[-1]["loss"]))
            if best is None or vb > best[0] + 1e-9:
                best = (vb, wd, drop, model, hist)
            else:
                del model
            gc.collect()
        vb, wd, drop, model, hist = best
        P[te] = predict(model, F, nfr, te, lens, mu, sd, device, n_cls)
        a = float((P[te].argmax(1) == y[te]).mean())
        per_fold.append(a)
        logs.append(dict(fold=k, n_train=len(tr), n_val=len(va), n_test=len(te),
                         val_best=vb, chosen_wd=wd, chosen_drop=drop, epochs=len(hist), acc=a,
                         first_loss=hist[0]["loss"], last_loss=hist[-1]["loss"], grid=grid))
        print(f"    fold {k}: val {vb:.4f}  test {a:.4f}  (wd {wd} drop {drop}, "
              f"{len(hist)} ep, loss {hist[0]['loss']:.3f}->{hist[-1]['loss']:.3f}, "
              f"{time.time()-t0:.0f}s)", flush=True)
        del model
        gc.collect()
    pooled = float((P.argmax(1) == y).mean())
    return pooled, per_fold, P, logs


# --------------------------------------------------------------------------- verification
@torch.no_grad()
def verify(flat, off, device, out):
    """Three checks, all reported as numbers rather than asserted-and-forgotten."""
    res = {}
    lens = np.diff(off)
    # a batch spanning the cohort's length range, which is where padding does its damage
    order = np.argsort(lens)
    ids = [int(order[0]), int(order[5]), int(order[len(order) // 2]),
           int(order[int(0.9 * len(order))]), int(order[-1])]
    m = build_model("run11", device).eval()
    fe = m.feature_extractor

    ws = [np.asarray(clip_of(flat, off, i), dtype=np.float32) for i in ids]
    ws = [np.pad(w, (0, max(0, MIN_SAMPLES - w.size))) for w in ws]
    L = max(w.size for w in ws)
    X = torch.zeros(len(ws), L)
    LN = torch.zeros(len(ws), dtype=torch.long)
    for k, w in enumerate(ws):
        X[k, :w.size] = torch.from_numpy(w); LN[k] = w.size
    X, LN = X.to(device), LN.to(device)

    single = [fe(torch.from_numpy(w).unsqueeze(0).to(device), None)[0][0] for w in ws]

    naive, nl = fe(X, LN)
    e_naive = max(float((naive[k, :single[k].shape[0]] - single[k]).abs().max()) for k in range(len(ws)))
    fixed, fl = masked_feature_extractor(fe, X, LN)
    e_fixed = max(float((fixed[k, :single[k].shape[0]] - single[k]).abs().max()) for k in range(len(ws)))
    res["clip_seconds"] = [round(w.size / SR, 3) for w in ws]
    res["torchaudio_batched_extractor_max_abs_err_at_valid_frames"] = e_naive
    res["masked_feature_extractor_max_abs_err_at_valid_frames"] = e_fixed
    print(f"[verify] clips {res['clip_seconds']} s")
    print(f"[verify] torchaudio feature_extractor(x, lengths), batched vs single-clip, "
          f"max |err| at VALID frames: {e_naive:.3f}   <- NOT padding-invariant")
    print(f"[verify] masked_feature_extractor, same comparison: {e_fixed:.2e}", flush=True)
    assert e_fixed < 1e-4, f"masked extractor is not exact ({e_fixed})"
    del m, fe, single, naive, fixed
    gc.collect()

    # (3) the batched POOLING heads must equal the single-clip path
    torch.manual_seed(0)
    D, n_cls = 768, N_CLASSES
    rng = np.random.default_rng(0)
    seqs = [rng.normal(size=(t, D)).astype(np.float32) for t in (1, 3, 7, 61, 300)]
    T = max(s.shape[0] for s in seqs)
    Xb = np.zeros((len(seqs), T, D), dtype=np.float32)
    Mb = np.zeros((len(seqs), T), dtype=np.float32)
    for k, s in enumerate(seqs):
        Xb[k, :s.shape[0]] = s; Mb[k, :s.shape[0]] = 1
    Xb_t, Mb_t = torch.from_numpy(Xb), torch.from_numpy(Mb)
    res["head_batch_vs_single_max_abs_err"] = {}
    for arm in ("mean_adamw", "attn", "attn_cat", "attn4", "trf"):
        torch.manual_seed(0)
        h = build_head(arm, D, n_cls).eval()
        with torch.no_grad():
            ob = h(Xb_t, Mb_t)
            e = 0.0
            for k, s in enumerate(seqs):
                os_ = h(torch.from_numpy(s).unsqueeze(0), torch.ones(1, s.shape[0]))
                e = max(e, float((ob[k] - os_[0]).abs().max()))
        res["head_batch_vs_single_max_abs_err"][arm] = e
        print(f"[verify] head {arm:11s} batched vs single-clip max |err|: {e:.2e}", flush=True)
        assert e < 1e-4, f"{arm} pooling is not padding-masked correctly ({e})"
        del h
    # a paranoia check that masking actually matters: the same batch with padding UNmasked
    torch.manual_seed(0)
    h = build_head("attn", D, n_cls).eval()
    with torch.no_grad():
        good = h(Xb_t, Mb_t)
        bad = h(Xb_t, torch.ones_like(Mb_t))
    res["attn_unmasked_vs_masked_max_abs_err"] = float((good - bad).abs().max())
    print(f"[verify] attn with padding UNmasked differs by {res['attn_unmasked_vs_masked_max_abs_err']:.3f} "
          f"-- the mask is load-bearing", flush=True)
    out["verify"] = res
    save(out)


def save(out):
    ANA.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=False))



# --------------------------------------------------------------------------- report
def load_P(e, l, arm, seed=None):
    f = FEAT / (f"ap_P_{e}_L{l}_{arm}.npy" if seed is None
                else f"ap_P_{e}_L{l}_{arm}_s{seed}.npy")
    return np.load(f) if f.exists() else None


def dd_bootstrap(y, Pa1, Pa0, Pb1, Pb0, groups, n=2000, seed=0):
    """Difference of differences over BIRDS: (a1-a0) - (b1-b0).

    The asymmetry question -- "does the new readout help run11 MORE than it helps AVES" -- is a
    difference of two gains, so its uncertainty has to be resampled as one quantity. Doing it by
    eye from two separate CIs overstates the evidence, because the same birds are hard for both.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    d = []
    for _ in range(n):
        sel = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        ya = y[sel]
        ga = (Pa1[sel].argmax(1) == ya).mean() - (Pa0[sel].argmax(1) == ya).mean()
        gb = (Pb1[sel].argmax(1) == ya).mean() - (Pb0[sel].argmax(1) == ya).mean()
        d.append(ga - gb)
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(delta=float(d.mean()), lo=float(lo), hi=float(hi),
                verdict=("a_gains_more" if lo > 0 else "b_gains_more" if hi < 0
                         else "not_distinguishable"))


ALL_ARMS = ["mean", "meanstd", "max", "first", "mid", "last", "meanmax", "multi",
            "mean_adamw", "attn", "attn_cat", "attn4", "trf"]


def report(y, birds, out, layers, seeds):
    res = out.get("results", {})
    rep = out.setdefault("report", {})

    def entries(e, l, arm):
        """(label, acc, per_fold, P) for every stored run of this arm, one per seed if learned."""
        if arm in FIXED_ARMS:
            k = f"{e}|L{l}|{arm}"
            if k in res:
                return [(k, res[k]["acc"], res[k]["per_fold"], load_P(e, l, arm))]
            return []
        o = []
        for s in seeds:
            k = f"{e}|L{l}|{arm}|s{s}"
            if k in res:
                o.append((k, res[k]["acc"], res[k]["per_fold"], load_P(e, l, arm, s)))
        return o

    for l in layers:
        print(f"\n{'='*104}\nLAYER {l}: readout comparison, 11-class call type, "
              f"leave-birds-out (majority {out['meta']['majority']:.4f})\n{'='*104}")
        print(f"{'encoder':7s} {'readout':11s} {'seed':4s} {'per-fold accuracies':44s} "
              f"{'mean':>7s} {'vs mean-pool':>13s}")
        base = {e: (res.get(f"{e}|L{l}|mean", {}).get("acc"), load_P(e, l, "mean"))
                for e in ENCODERS}
        tbl = []
        for e in ENCODERS:
            b_acc, b_P = base[e]
            for arm in ALL_ARMS:
                for k, acc, ff, P in entries(e, l, arm):
                    sd = k.split("|")[3][1:] if arm in LEARNED_ARMS else "-"
                    d = f"{acc-b_acc:+.4f}" if b_acc is not None else "n/a"
                    print(f"{e:7s} {arm:11s} {sd:4s} "
                          f"{' '.join(f'{v:.4f}' for v in ff):44s} {acc:7.4f} {d:>13s}")
                    tbl.append(dict(encoder=e, layer=l, readout=arm, seed=sd, acc=acc,
                                    per_fold=ff, delta_vs_meanpool=acc - b_acc
                                    if b_acc is not None else None))
        rep[f"L{l}_table"] = tbl

        # seed spread on the learned arms -- the yardstick every difference is measured against
        spread = {}
        for e in ENCODERS:
            for arm in LEARNED_ARMS:
                a = [v[1] for v in entries(e, l, arm)]
                if len(a) > 1:
                    spread[f"{e}|{arm}"] = dict(accs=a, spread=float(max(a) - min(a)),
                                                mean=float(np.mean(a)))
        rep[f"L{l}_seed_spread"] = spread
        if spread:
            print("\n  seed spread (max-min over seeds) on the learned arms:")
            for k, v in spread.items():
                print(f"    {k:18s} {['%.4f' % x for x in v['accs']]}  spread {v['spread']:.4f}")
            mx = max(v["spread"] for v in spread.values())
            print(f"    -> largest seed spread {mx:.4f}; a gap smaller than this is not a gap")

        # bootstrap: each readout vs that encoder's own mean-pool baseline
        bs = {}
        print("\n  cluster bootstrap over the 48 birds, readout vs the SAME encoder's mean-pool:")
        for e in ENCODERS:
            b_acc, b_P = base[e]
            if b_P is None:
                continue
            for arm in ALL_ARMS:
                if arm == "mean":
                    continue
                for k, acc, ff, P in entries(e, l, arm):
                    if P is None:
                        continue
                    r = bird_bootstrap(y, P, b_P, birds)
                    bs[k] = r
                    print(f"    {k:26s} {r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]  "
                          f"{'BETTER' if r['verdict']=='a_better' else 'worse' if r['verdict']=='b_better' else 'ns'}")
        rep[f"L{l}_bootstrap_vs_meanpool"] = bs

        # ...and vs the AdamW mean-pool CONTROL, seed-matched. The sklearn baseline and the learned
        # arms are trained by different optimisers with different L2 strengths, so "attn vs mean"
        # mixes a pooling change with a trainer change. Against mean_adamw at the same seed, the
        # pooling is the only thing that differs.
        bc = {}
        print("\n  cluster bootstrap, learned readout vs the AdamW mean-pool control (same seed):")
        for e in ENCODERS:
            ctrl = {k.split("|")[3]: P for k, _, _, P in entries(e, l, "mean_adamw")}
            for arm in LEARNED_ARMS:
                if arm == "mean_adamw":
                    continue
                for k, acc, ff, P in entries(e, l, arm):
                    tag = k.split("|")[3]
                    if P is None or ctrl.get(tag) is None:
                        continue
                    r = bird_bootstrap(y, P, ctrl[tag], birds)
                    bc[k] = r
                    print(f"    {k:26s} {r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]  "
                          f"{'BETTER' if r['verdict']=='a_better' else 'worse' if r['verdict']=='b_better' else 'ns'}")
        rep[f"L{l}_bootstrap_vs_mean_adamw"] = bc

        # bootstrap: run11 vs AVES under each readout -- does the ranking survive?
        rk = {}
        print("\n  cluster bootstrap over the 48 birds, run11 vs AVES under each readout:")
        for arm in ALL_ARMS:
            ra, aa = entries("run11", l, arm), entries("aves", l, arm)
            for (kr, ar, _, Pr), (ka, aav, _, Pa) in zip(ra, aa):
                if Pr is None or Pa is None:
                    continue
                r = bird_bootstrap(y, Pr, Pa, birds)
                tag = kr.split("|")[-1] if arm in LEARNED_ARMS else ""
                rk[f"{arm}|{tag}"] = dict(run11=ar, aves=aav, **r)
                print(f"    {arm:11s} {tag:3s} run11 {ar:.4f}  aves {aav:.4f}  "
                      f"diff {r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]  "
                      f"{'run11 wins' if r['verdict']=='a_better' else 'AVES wins' if r['verdict']=='b_better' else 'tie'}")
        rep[f"L{l}_bootstrap_run11_vs_aves"] = rk

        # the asymmetry question: does a readout help run11 MORE than it helps AVES?
        dd = {}
        print("\n  difference of differences (gain for run11) - (gain for AVES), bootstrapped:")
        for arm in ALL_ARMS:
            if arm == "mean":
                continue
            ra, aa = entries("run11", l, arm), entries("aves", l, arm)
            for (kr, _, _, Pr), (ka, _, _, Pa) in zip(ra, aa):
                if Pr is None or Pa is None or base["run11"][1] is None or base["aves"][1] is None:
                    continue
                r = dd_bootstrap(y, Pr, base["run11"][1], Pa, base["aves"][1], birds)
                tag = kr.split("|")[-1] if arm in LEARNED_ARMS else ""
                dd[f"{arm}|{tag}"] = r
                print(f"    {arm:11s} {tag:3s} {r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]  "
                      f"{r['verdict']}")
        rep[f"L{l}_dd_bootstrap"] = dd

        # Where an attentive readout could possibly help: clips long enough to have structure.
        # 57.5% of this cohort is <= 8 frames (<= ~0.18 s) and 21% is <= 3 frames, so for most clips
        # "attend over time" has almost nothing to attend over. Broken out so the headline is not
        # read as "attentive pooling does not work" when it may be "these clips are too short".
        nfr_all = np.load(FEAT / "ct11_frames_off.npy")
        FR = np.diff(nfr_all)
        edges = [1, 4, 8, 14, int(FR.max()) + 1]
        bins = [(edges[i], edges[i + 1] - 1) for i in range(len(edges) - 1)]
        lb = {}
        print("\n  accuracy by clip length in FRAMES (a frame is 20 ms of hop):")
        hdr = "  ".join(f"{a}-{b}f n={int(((FR>=a)&(FR<=b)).sum())}" for a, b in bins)
        print(f"    {'encoder/readout':28s} {hdr}")
        for e in ENCODERS:
            for arm in ALL_ARMS:
                ent = entries(e, l, arm)
                if not ent:
                    continue
                k, acc, ff, P = ent[0]
                if P is None:
                    continue
                row = []
                for a, b in bins:
                    m = (FR >= a) & (FR <= b)
                    row.append(float((P[m].argmax(1) == y[m]).mean()))
                lb[f"{e}|{arm}"] = dict(bins=[list(bb) for bb in bins],
                                        n=[int(((FR >= a) & (FR <= b)).sum()) for a, b in bins],
                                        acc=row)
                print(f"    {e+'/'+arm:28s} " + "  ".join(f"{v:.4f}      " for v in row))
        rep[f"L{l}_by_clip_length"] = lb
        rep["frame_count_distribution"] = dict(
            n=int(len(FR)), min=int(FR.min()), p25=float(np.percentile(FR, 25)),
            median=float(np.median(FR)), p75=float(np.percentile(FR, 75)),
            p90=float(np.percentile(FR, 90)), max=int(FR.max()), mean=float(FR.mean()),
            frac_le_8=float((FR <= 8).mean()), frac_le_3=float((FR <= 3).mean()))
    save(out)


# --------------------------------------------------------------------------- main
FIXED_ARMS = ["mean", "meanstd", "max", "first", "mid", "last", "meanmax", "multi"]
LEARNED_ARMS = ["mean_adamw", "attn", "attn_cat", "attn4", "trf"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["verify", "extract", "fixed", "learned", "report", "all"])
    ap.add_argument("--arms", default=None, help="comma list; default = all for the stage")
    ap.add_argument("--layers", default="3,1")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--encoders", default="run11,aves")
    ap.add_argument("--json", default=None,
                    help="write to a different results file (used to run a second stage in "
                         "parallel without two processes clobbering one JSON; merge afterwards)")
    ap.add_argument("--device", default=None,
                    help="default: mps for frame extraction, cpu for the pooling heads. The heads "
                         "are tiny (a 768->128->1 scorer); measured on this machine an epoch takes "
                         "0.17 s on cpu and 3.26 s on mps, because per-kernel dispatch overhead "
                         "dominates at this size. Extraction, which is a 94M-parameter forward, is "
                         "the opposite way round.")
    a = ap.parse_args()
    if a.json:
        global OUT
        OUT = ANA / a.json

    device = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    head_device = a.device or "cpu"
    print(f"[device] extraction {device}, pooling heads {head_device}", flush=True)
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    paths, y, birds, classes = load_cohort()
    out.setdefault("meta", {}).update(
        n_clips=int(len(y)), n_classes=len(classes), n_birds=int(len(set(birds))),
        classes=classes, majority=float(np.bincount(y).max() / len(y)),
        split=f"leave-birds-out StratifiedGroupKFold({NFOLD}), seed {SEED}",
        baseline_ref=BASELINE_REF, hp=HP, layers=list(LAYERS))
    folds = outer_folds(y, birds)
    flat, off = load_wavs(paths)
    encs = a.encoders.split(",")
    layers = [int(v) for v in a.layers.split(",")]
    seeds = [int(v) for v in a.seeds.split(",")]

    if a.stage in ("verify", "all"):
        verify(flat, off, device, out)

    if a.stage in ("extract", "all"):
        for e in encs:
            extract_frames(e, flat, off, device)

    # frame store must reproduce the cached mean-pooled embeddings it is meant to replace
    if a.stage in ("extract", "fixed", "all"):
        chk = out.setdefault("frame_store_vs_cached_meanpool", {})
        for e in encs:
            ref = np.load(FEAT / f"ct11_{e}_emb.npy", mmap_mode="r")
            for l in layers:
                F, nfr = load_store(e, l)
                M = fixed_pool(F, nfr, "mean")
                d = float(np.abs(M - np.asarray(ref[:, l], dtype=np.float32)).max())
                chk[f"{e}_L{l}"] = d
                print(f"[check] {e} L{l}: mean of float16 frames vs cached float32 mean-pool, "
                      f"max |diff| {d:.2e}", flush=True)
                assert d < 0.05, f"frame store for {e} L{l} does not match the cached embeddings"
                del F, M
                gc.collect()
            del ref
        save(out)

    res = out.setdefault("results", {})

    if a.stage in ("fixed", "all"):
        arms = (a.arms.split(",") if a.arms else FIXED_ARMS)
        for e in encs:
            for l in layers:
                F, nfr = load_store(e, l)
                for arm in arms:
                    if arm not in FIXED_ARMS:
                        continue
                    key = f"{e}|L{l}|{arm}"
                    if key in res:
                        print(f"[skip] {key} = {res[key]['acc']:.4f}"); continue
                    t0 = time.time()
                    X = fixed_pool(F, nfr, arm)
                    acc, ff, P = cv_acc(X, y, birds, return_proba=True)
                    res[key] = dict(encoder=e, layer=l, readout=arm, learned=False,
                                    acc=acc, per_fold=ff, dim=int(X.shape[1]),
                                    seconds=round(time.time() - t0, 1))
                    np.save(FEAT / f"ap_P_{e}_L{l}_{arm}.npy", P)
                    print(f"[fixed] {key:26s} dim {X.shape[1]:5d}  acc {acc:.4f}  "
                          f"folds {[round(v,4) for v in ff]}  {time.time()-t0:.0f}s", flush=True)
                    save(out)
                    del X, P
                    gc.collect()
                del F
                gc.collect()

    if a.stage in ("learned", "all"):
        arms = (a.arms.split(",") if a.arms else LEARNED_ARMS)
        for l in layers:
            for e in encs:
                F, nfr = load_store(e, l)
                for arm in arms:
                    if arm not in LEARNED_ARMS:
                        continue
                    for s in seeds:
                        key = f"{e}|L{l}|{arm}|s{s}"
                        if key in res:
                            print(f"[skip] {key} = {res[key]['acc']:.4f}"); continue
                        print(f"[learned] {key}", flush=True)
                        t0 = time.time()
                        acc, ff, P, logs = run_learned(arm, F, nfr, y, birds, folds,
                                                       head_device, s)
                        res[key] = dict(encoder=e, layer=l, readout=arm, learned=True, seed=s,
                                        acc=acc, per_fold=ff, folds=logs,
                                        seconds=round(time.time() - t0, 1))
                        np.save(FEAT / f"ap_P_{e}_L{l}_{arm}_s{s}.npy", P)
                        print(f"[learned] {key:30s} acc {acc:.4f}  "
                              f"folds {[round(v,4) for v in ff]}  {time.time()-t0:.0f}s", flush=True)
                        save(out)
                        del P
                        gc.collect()
                del F
                gc.collect()

    if a.stage in ("report", "all"):
        report(y, birds, out, layers, seeds)

    save(out)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

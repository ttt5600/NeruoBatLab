#!/usr/bin/env python
"""Every AVES / BirdAVES checkpoint against run11 on call-type classification.

calltype11.py answered the one-versus-one question and the answer went the wrong way for the
release model: on the 11-class, 3412-clip, leave-birds-out task, aves-base-bio L3 = 0.8453 beat
run11 L3 = 0.8118. That single comparison leaves two escape hatches open, and this script closes
both.

  1. "aves-base-bio was a lucky checkpoint." AVES ships three pretraining subsets (core, bio, all)
     and BirdAVES ships three more that add bird-heavy data (biox-base, biox-large, bioxn-large).
     If colony-specific pretraining is worth anything, run11 should at least beat the middle of
     that family, not lose to one member of it. Six checkpoints is a distribution, not an anecdote.
  2. "run11 lost on capacity, not on corpus." aves-base-core/bio/all and birdaves-biox-base are
     architecturally IDENTICAL to run11 -- same 7-layer conv extractor, 768-d, 12 blocks,
     320-sample hop, same state_dict keys -- so for those four the ONLY variable is which audio the
     encoder saw during pretraining. That is why those four carry the argument. The 1024-d /
     24-block larges are included so the capacity axis is visible separately and cannot be confused
     with the corpus axis; their embed_dim and layer count are read from each checkpoint's own
     model_config.json and asserted against what the model actually returns, never assumed.

Protocol is copied from calltype11.py without change, because the whole point is comparability to
the 0.8118 / 0.8453 pair already on the record:

  * cohort: calltype11.collect() + KEEP11 -- 3412 clips, 11 classes, 48 birds, majority 0.1797,
    adults AND chicks. Asserted, not hoped for.
  * audio: mono channel mean, resampled to 16 kHz with a CACHED resampler (the source is 44.1 kHz
    and rebuilding the filter kernel per clip is most of the wall time), and NO waveform
    normalisation -- AVES was pretrained on un-normalised audio through a group_norm extractor and
    zf_hubert.embed_file does not normalise either, so raw is native for both sides.
  * each clip encoded AT ITS OWN LENGTH. The April-2023 notebook zero-padded every clip to 250,606
    samples and mean-pooled over a frame axis that was ~99% padding; that single bug is what made
    AVES look bad, and it is not repeated here.
  * mean-pool over real frames -> one vector per clip per layer, ALL layers swept.
  * probe: StandardScaler + LogisticRegression(max_iter=4000), StratifiedGroupKFold(5, seed 0)
    grouped by bird, i.e. leave-birds-out. A random split inflates this task by about +0.11, so the
    grouping is load-bearing and is not a detail.
  * the adults-only 8-class arm runs too, so every checkpoint also ties back to the released
    8-class number.

Significance is a cluster bootstrap over the 48 BIRDS (aves_calltype.bird_bootstrap), not over
clips: clips from one bird are not independent draws, and a clip-level interval would be narrower
than the data can support. Only the best checkpoint of each family is bootstrapped against run11 --
doing all six would be six looks at one comparison.

Resource shape: one model in memory at a time, embeddings cached to features/ct11_<tag>_emb.npy and
best-layer probabilities to features/ct11_<tag>_P{11,8}.npy, so a crash costs minutes and not hours.
"""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from calltype11 import collect, KEEP11, KEEP8                              # noqa: E402
from aves_calltype import load_audio, cv_acc, bird_bootstrap               # noqa: E402
from aves_holdout import load_run11                                       # noqa: E402

EXT = Path.home() / "zf_labelset/external/aves"
FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
OUT = ANA / "aves_variants_calltype.json"

# tag -> (family, embedding cache). aves-base-bio reuses the cache calltype11.py already wrote, so
# its numbers here come out bit-identical to the ones already on the record.
CKPTS = [
    ("aves-base-core",       "aves",     "ct11_aves-base-core_emb.npy"),
    ("aves-base-bio",        "aves",     "ct11_aves_emb.npy"),
    ("aves-base-all",        "aves",     "ct11_aves-base-all_emb.npy"),
    ("birdaves-biox-base",   "birdaves", "ct11_birdaves-biox-base_emb.npy"),
    ("birdaves-biox-large",  "birdaves", "ct11_birdaves-biox-large_emb.npy"),
    ("birdaves-bioxn-large", "birdaves", "ct11_birdaves-bioxn-large_emb.npy"),
]
EXPECT = dict(n_clips=3412, n_classes=11, n_birds=48, majority=0.1797)


def build_variant(tag):
    """Build from the checkpoint's OWN config and load its weights. Returns (model, info)."""
    from torchaudio.models import wav2vec2_model
    cfg_p = EXT / f"{tag}.torchaudio.model_config.json"
    w_p = EXT / f"{tag}.torchaudio.pt"
    for p in (cfg_p, w_p):
        if not p.exists():
            raise FileNotFoundError(p)
    cfg = json.load(open(cfg_p))
    cfg["encoder_layer_drop"] = 0.0          # layerdrop is a training-only trick; off for eval
    m = wav2vec2_model(**cfg, aux_num_out=None)
    sd = torch.load(w_p, map_location="cpu", weights_only=True)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not k.startswith("aux")]   # masked-prediction head; never used
    if bad or unexpected:
        raise RuntimeError(f"{tag} state_dict mismatch\n  missing={bad}\n  unexpected={unexpected}")
    n_par = sum(p.numel() for p in m.parameters())
    cnn = sum(p.numel() for p in m.feature_extractor.parameters())
    info = dict(params=int(n_par), params_cnn=int(cnn), params_transformer=int(n_par - cnn),
                embed_dim=int(cfg["encoder_embed_dim"]), n_layers=int(cfg["encoder_num_layers"]),
                n_heads=int(cfg["encoder_num_heads"]),
                ff=int(cfg["encoder_ff_interm_features"]),
                extractor_mode=cfg["extractor_mode"],
                ckpt_bytes=int(w_p.stat().st_size),
                missing_keys=list(missing), unexpected_keys=list(unexpected))
    return m.eval(), info


@torch.no_grad()
def embed_all(model, paths, device, n_layers, dim, tag=""):
    """(N, n_layers, dim), each clip at its own length, mean-pooled over real frames only."""
    out = np.zeros((len(paths), n_layers, dim), dtype=np.float32)
    t0 = time.time()
    for i, p in enumerate(paths):
        w = load_audio(p)
        if w.shape[-1] < 400:                       # shorter than one receptive field
            w = torch.nn.functional.pad(w, (0, 400 - w.shape[-1]))
        feats, _ = model.extract_features(w.to(device), None)
        if len(feats) != n_layers:
            raise RuntimeError(f"{tag}: config says {n_layers} layers, model gave {len(feats)}")
        for l in range(n_layers):
            f = feats[l].squeeze(0)
            if f.shape[-1] != dim:
                raise RuntimeError(f"{tag}: config dim {dim}, model gave {f.shape[-1]}")
            out[i, l] = f.mean(0).cpu().numpy()
        if (i + 1) % 500 == 0:
            print(f"    {tag} {i+1}/{len(paths)}  {time.time()-t0:.0f}s", flush=True)
    print(f"    {tag} embedded {len(paths)} clips in {time.time()-t0:.0f}s", flush=True)
    return out


def sweep(E, y, groups, tag=""):
    """Per-layer leave-birds-out accuracy; returns (accs, best_layer, P_at_best_layer)."""
    accs, Ps = [], []
    t0 = time.time()
    for l in range(E.shape[1]):
        a, _, P = cv_acc(np.ascontiguousarray(E[:, l]), y, groups, return_proba=True)
        accs.append(a); Ps.append(P)
    b = int(np.argmax(accs))
    print(f"    {tag} probe sweep {E.shape[1]} layers in {time.time()-t0:.0f}s", flush=True)
    return accs, b, Ps[b]


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[device] {device}", flush=True)

    # ---------------- cohort, exactly calltype11's
    rows = [r for r in collect() if r[3] in KEEP11]
    paths = [r[0] for r in rows]
    birds = np.array([r[1].lower() for r in rows])
    tt = np.array([r[3] for r in rows])
    src = np.array([r[4] for r in rows])
    classes = sorted(set(tt))
    y = np.array([classes.index(t) for t in tt])
    maj = float(np.bincount(y).max() / len(y))
    print(f"[cohort] {len(y)} clips, {len(classes)} classes, {len(set(birds))} birds, "
          f"majority {maj:.4f}  (adults {int((src=='AdultVocalizations').sum())}, "
          f"chicks {int((src=='ChickVocalizations').sum())})", flush=True)
    assert len(y) == EXPECT["n_clips"], f"cohort is {len(y)} clips, expected {EXPECT['n_clips']}"
    assert len(classes) == EXPECT["n_classes"], f"{len(classes)} classes, expected 11"
    assert len(set(birds)) == EXPECT["n_birds"], f"{len(set(birds))} birds, expected 48"
    assert abs(maj - EXPECT["majority"]) < 1e-3, f"majority {maj}, expected ~0.1797"

    m8 = np.array([t in KEEP8 for t in tt]) & (src == "AdultVocalizations")
    c8 = sorted(set(tt[m8])); y8 = np.array([c8.index(t) for t in tt[m8]])
    b8 = birds[m8]
    maj8 = float(np.bincount(y8).max() / len(y8))
    print(f"[cohort-8] {int(m8.sum())} adult clips, {len(c8)} classes, {len(set(b8))} birds, "
          f"majority {maj8:.4f}", flush=True)

    out = {"cohort11": dict(n_clips=int(len(y)), n_classes=len(classes), classes=classes,
                            n_birds=int(len(set(birds))), majority=maj,
                            n_adult=int((src == "AdultVocalizations").sum()),
                            n_chick=int((src == "ChickVocalizations").sum())),
           "cohort8": dict(n_clips=int(m8.sum()), n_classes=len(c8), classes=c8,
                           n_birds=int(len(set(b8))), majority=maj8),
           "split": "leave-birds-out StratifiedGroupKFold(5), seed 0",
           "probe": "StandardScaler + LogisticRegression(max_iter=4000)",
           "audio": "mono channel mean, 16 kHz, no waveform normalisation, per-clip length",
           "models": {}}

    # resume: a model already in the JSON whose best-layer probabilities are also on disk is not
    # re-probed. Delete the JSON (not just the .npy caches) if the protocol above ever changes.
    if OUT.exists():
        try:
            out["models"].update(json.loads(OUT.read_text()).get("models", {}))
            print(f"[resume] {len(out['models'])} model(s) already scored in {OUT.name}", flush=True)
        except Exception as e:
            print(f"[resume] ignoring unreadable {OUT.name}: {e}", flush=True)

    # ---------------- phase 1: verify every checkpoint loads, and count its parameters
    print("\n=== checkpoint inventory (each built from its own config, weights loaded) ===",
          flush=True)
    INFO = {}
    for tag, fam, _ in CKPTS:
        m, info = build_variant(tag)
        info["family"] = fam
        INFO[tag] = info
        print(f"  {tag:22s} {info['params']/1e6:7.2f}M ({info['params_cnn']/1e6:.2f}M cnn + "
              f"{info['params_transformer']/1e6:.2f}M tf)  dim {info['embed_dim']}  "
              f"L{info['n_layers']}  heads {info['n_heads']}  ff {info['ff']}  "
              f"missing={info['missing_keys']} unexpected={info['unexpected_keys']}", flush=True)
        del m; gc.collect()

    Pbest = {}

    def score(tag, E, info):
        pa = FEAT / f"ct11_{tag}_P11.npy"
        pb = FEAT / f"ct11_{tag}_P8.npy"
        rec = dict(info)
        if pa.exists() and pb.exists() and tag in out["models"]:
            P11, P8 = np.load(pa), np.load(pb)
        else:
            a11, i11, P11 = sweep(E, y, birds, tag=f"{tag}/11")
            a8, i8, P8 = sweep(E[m8], y8, b8, tag=f"{tag}/8")
            rec.update(per_layer_acc_11=a11, best_11=dict(layer=i11, acc=a11[i11]),
                       per_layer_acc_8=a8, best_8=dict(layer=i8, acc=a8[i8]))
            out["models"][tag] = rec
            np.save(pa, P11); np.save(pb, P8)
        Pbest[tag] = (P11, P8)
        r = out["models"][tag]
        print(f"  [{tag}] 11-class best L{r['best_11']['layer']} {r['best_11']['acc']:.4f} | "
              f"8-class best L{r['best_8']['layer']} {r['best_8']['acc']:.4f}", flush=True)
        OUT.write_text(json.dumps(out, indent=2))       # persist after every model

    # ---------------- phase 2: run11, from the cache calltype11.py wrote
    print("\n=== run11 (zebra-finch-pretrained reference) ===", flush=True)
    r11p = FEAT / "ct11_run11_emb.npy"
    m = load_run11("cpu")
    n_par = sum(p.numel() for p in m.parameters())
    cnn = sum(p.numel() for p in m.feature_extractor.parameters())
    r11info = dict(params=int(n_par), params_cnn=int(cnn), params_transformer=int(n_par - cnn),
                   embed_dim=768, n_layers=12, n_heads=12, ff=3072, family="run11",
                   pretraining="HuBERT-base SSL on 120 zebra finch colony recordings")
    print(f"  run11 {n_par/1e6:.2f}M params ({cnn/1e6:.2f}M cnn + {(n_par-cnn)/1e6:.2f}M tf)",
          flush=True)
    if r11p.exists():
        E = np.load(r11p)
        print(f"  [cache] {r11p.name} {E.shape}", flush=True)
        del m; gc.collect()
    else:
        m = m.to(device)
        E = embed_all(m, paths, device, 12, 768, tag="run11")
        np.save(r11p, E); del m; gc.collect()
    if E.shape != (len(y), 12, 768):
        raise RuntimeError(f"run11 embeddings {E.shape} != {(len(y), 12, 768)}")
    score("run11", E, r11info)
    del E; gc.collect()

    # ---------------- phase 3: every variant, one model in memory at a time
    for tag, fam, cache in CKPTS:
        print(f"\n=== {tag} ===", flush=True)
        info = INFO[tag]
        nl, dim = info["n_layers"], info["embed_dim"]
        cp = FEAT / cache
        if cp.exists():
            E = np.load(cp)
            print(f"  [cache] {cache} {E.shape}", flush=True)
            if E.shape != (len(y), nl, dim):
                raise RuntimeError(f"{cache} shape {E.shape} != {(len(y), nl, dim)}; delete it")
        else:
            m, _ = build_variant(tag)
            m = m.to(device)
            print(f"  embedding {len(paths)} clips, {nl} layers x {dim}-d", flush=True)
            E = embed_all(m, paths, device, nl, dim, tag=tag)
            np.save(cp, E)
            del m; gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
        score(tag, E, info)
        del E; gc.collect()

    # ---------------- family winners vs run11, cluster bootstrap over the 48 birds
    print("\n=== cluster bootstrap over birds: family best vs run11 ===", flush=True)
    boots = {}
    for fam in ("aves", "birdaves"):
        tags = [t for t, f, _ in CKPTS if f == fam]
        w11 = max(tags, key=lambda t: out["models"][t]["best_11"]["acc"])
        w8 = max(tags, key=lambda t: out["models"][t]["best_8"]["acc"])
        r = bird_bootstrap(y, Pbest["run11"][0], Pbest[w11][0], birds)
        r8 = bird_bootstrap(y8, Pbest["run11"][1], Pbest[w8][1], b8)
        boots[fam] = dict(winner_11=w11, winner_8=w8,
                          run11_minus_best_11=r, run11_minus_best_8=r8)
        print(f"  {fam:9s} 11-class: {w11} {out['models'][w11]['best_11']['acc']:.4f} vs run11 "
              f"{out['models']['run11']['best_11']['acc']:.4f} -> "
              f"{r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]  {r['verdict']}")
        print(f"  {fam:9s}  8-class: {w8} {out['models'][w8]['best_8']['acc']:.4f} vs run11 "
              f"{out['models']['run11']['best_8']['acc']:.4f} -> "
              f"{r8['delta']:+.4f} [{r8['lo']:+.4f}, {r8['hi']:+.4f}]  {r8['verdict']}", flush=True)
    out["bootstrap"] = dict(
        note="delta = run11 minus the named checkpoint; a_better means run11 wins, "
             "b_better means the AVES checkpoint wins",
        **boots)

    # ---------------- the table
    order = ["run11"] + [t for t, _, _ in CKPTS]
    hdr = (f"{'checkpoint':22s} {'params':>9s} {'dim':>5s} {'nL':>3s} "
           f"{'bL11':>5s} {'acc11':>7s} {'bL8':>4s} {'acc8':>7s}")
    print("\n" + hdr)
    lines = [hdr]
    for t in order:
        m_ = out["models"][t]
        lines.append(f"{t:22s} {m_['params']/1e6:8.2f}M {m_['embed_dim']:5d} {m_['n_layers']:3d} "
                     f"{m_['best_11']['layer']:5d} {m_['best_11']['acc']:7.4f} "
                     f"{m_['best_8']['layer']:4d} {m_['best_8']['acc']:7.4f}")
        print(lines[-1])
    out["table"] = lines

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

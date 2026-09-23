#!/usr/bin/env python
"""The whole AVES/BirdAVES family against run11 on vocalisation detection, in one table.

aves_baseline.py and aves_holdout.py each compared run11 to a SINGLE off-the-shelf encoder
(aves-base-bio) and found a tie in-distribution (0.9687 vs 0.9674 AUC) and a loss on the BirdPark
holdout (0.865 vs 0.901). One checkpoint is not a family. Hagiwara's release ships three AVES
variants that differ only in which slice of AudioSet/VGGSound they were trained on (core / bio /
all) plus three BirdAVES variants that add a large bird-heavy corpus (biox-base, biox-large,
bioxn-large), and a "generic encoders match a colony-specific one" claim that rests on the single
checkpoint that happened to be downloaded first is one lucky draw away from being wrong in either
direction. So: every released variant, both evaluations, one canonical table.

Four things this script is careful about, because each one is an easy way to manufacture a result:

1. Normalisation is a confound, not a detail. run11 consumes (x - mu) / sqrt(var + 1e-5) over the
   20 s chunk; every AVES checkpoint was pretrained on raw waveforms (the fairseq config records
   normalize: False, and the extractor is group_norm). Running the AVES family only our way hobbles
   it and running it only its way changes two variables at once, so every checkpoint is run BOTH
   ways and both numbers are printed. run11 is also run on raw audio, which is not its native input
   and is included as a control rather than as a fair reading of run11.

2. Layer choice is pre-committed on ZF and frozen. Each (checkpoint, normalisation) arm picks its
   layer by out-of-fold AUC on the 30 min of ZF colony audio, and BirdPark is then scored at that
   layer only. Every layer is still written to the JSON so the choice is auditable, but a layer
   chosen on the BirdPark test set is never headlined -- with six layers times two normalisations
   times seven checkpoints there are 84 chances to find a winner on a 5925-frame test set, and a
   max over 84 is not a measurement.

3. The layer grid is read from each config, never assumed. The two large variants are 24 layers of
   1024 dims, not 12 of 768, so the grid is taken at matched RELATIVE depth
   (base 0/1/3/6/9/11, large 0/2/6/12/18/23) rather than at matched absolute index, which would
   compare run11's middle to a large model's first quarter.

4. BirdPark cannot support a tight interval and the script says so out loud. 5925 frames at a 320
   sample hop is 118.5 s, which is FOUR independent 30 s blocks. The block=1500 bootstrap is
   reported because it is the honest unit, the block=500 bootstrap is reported because it shows how
   much of the interval width is the block size, and a per-block win count is reported because with
   four blocks that is closer to what the data can actually say than any confidence interval is.

Features are written to features/detvar/ as float16 memmaps, one (checkpoint, normalisation,
dataset) per file, and skipped if present -- a crash mid-family costs one frame pass, not eleven.
One model is resident at a time, which is why the family is driven one --models at a time on a
machine under memory pressure.

Usage
    detection_variants.py                        # every checkpoint, encode + score + bootstrap
    detection_variants.py --models run11,aves-base-bio
    detection_variants.py --encode-only          # frame passes only
    detection_variants.py --score-only           # reuse cached features / predictions
    detection_variants.py --report               # reprint the canonical table from the JSON

Validation that this is the same pipeline as the two scripts it replaces: run11 matched L0
reproduces 0.9687 / 0.8681 in-distribution and 0.8648 / 0.8121 on BirdPark to four decimals, and
the recomputed aves-base-bio features are bit-identical to the published aves_frames_matched.npz
(max abs diff 0.0, recorded in the JSON under "consistency").
"""
from __future__ import annotations
import argparse, gc, json, os, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from zfeval import metrics as mx, splits as sp                              # noqa: E402
from aves_holdout import load_run11                                         # noqa: E402

SR, HOP, RF = 16000, 320, 400
SPAN = int(round(79434253 * SR / 44100))
AUDIO = Path.home() / "zf_labelset/audio/111021-000.wav"
AVES_DIR = Path.home() / "zf_labelset/external/aves"
BP = Path.home() / "zf_labelset/external/birdpark"
FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
DETDIR = FEAT / "detvar"
# Domain-adaptive-pretraining exports (export_weights.py output: encoder_config +
# state_dict + meta). Named dapt*, one file per step-spaced checkpoint, so a whole
# training trajectory can be run through this same table rather than a parallel script
# that would drift from its normalisation and layer-selection doctrine.
DAPT_DIR = Path.home() / "zf_labelset/external/dapt"
OUTJSON = ANA / "detection_variants.json"

BLOCK, BLOCK_SMALL = 1500, 500          # 30 s and 10 s of frames
TAGS = ("matched", "native")            # run11-style chunk norm; raw AVES-native waveform
# large variants last on purpose: they are 4x the frame-pass cost and 1.26 GB each on disk
CKPTS = ["run11", "aves-base-core", "aves-base-bio", "aves-base-all",
         "birdaves-biox-base", "birdaves-biox-large", "birdaves-bioxn-large"]


def layer_grid(n_layers: int) -> list[int]:
    """Six layers at matched relative depth, so 12- and 24-layer models are compared like for like.

    n=12 -> [0, 1, 3, 6, 9, 11];  n=24 -> [0, 2, 6, 12, 18, 23].
    Includes both ends because the known ZF trend is a plateau over the shallow half followed by a
    decline, and a grid that skipped layer 0 or the last layer would hide both ends of it.
    """
    return sorted({0, max(1, n_layers // 12), n_layers // 4,
                   n_layers // 2, (3 * n_layers) // 4, n_layers - 1})


def spec(name: str) -> dict:
    if name == "run11":
        return dict(kind="run11", n_layers=12, dim=768, weights=None, cfg=None)
    # Any self-describing export goes through this branch, not just DAPT checkpoints: run15
    # (from scratch on the combined corpus) is written by the same export_weights.py and carries
    # its own encoder_config, so it needs no config file and cannot drift from what it trained as.
    if name.startswith("dapt") or (DAPT_DIR / f"{name}.pt").exists():
        # Self-describing: export_weights.py embeds the constructor kwargs beside the weights,
        # so a DAPT checkpoint needs no config file of its own and cannot drift from the
        # architecture it was actually trained with.
        w = DAPT_DIR / f"{name}.pt"
        if not w.exists():
            raise FileNotFoundError(f"{w} -- export it first with export_weights.py")
        cfg = dict(torch.load(w, map_location="cpu", weights_only=False)["encoder_config"])
        return dict(kind="dapt", cfg=cfg, n_layers=int(cfg["encoder_num_layers"]),
                    dim=int(cfg["encoder_embed_dim"]), weights=w)
    cfg = json.load(open(AVES_DIR / f"{name}.torchaudio.model_config.json"))
    return dict(kind="aves", cfg=cfg, n_layers=int(cfg["encoder_num_layers"]),
                dim=int(cfg["encoder_embed_dim"]),
                weights=AVES_DIR / f"{name}.torchaudio.pt")


def load_model(name: str, s: dict, device: str):
    """load_aves generalised to any released config; strict about what may be missing."""
    if s["kind"] == "run11":
        m = load_run11(device)
    else:
        from torchaudio.models import wav2vec2_model
        cfg = dict(s["cfg"])
        cfg["encoder_layer_drop"] = 0.0          # eval-time determinism; layerdrop is training-only
        m = wav2vec2_model(**cfg, aux_num_out=None)
        if s["kind"] == "dapt":
            sd = torch.load(s["weights"], map_location="cpu", weights_only=False)["state_dict"]
        else:
            sd = torch.load(s["weights"], map_location="cpu", weights_only=True)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        # the released checkpoints carry no masked-prediction head, which we never use
        bad = [k for k in missing if not k.startswith("aux")]
        if bad or unexpected:
            raise RuntimeError(f"{name} state_dict mismatch\n  missing={bad}\n  unexpected={unexpected}")
        m = m.eval().to(device)
    n_param = sum(p.numel() for p in m.parameters())
    print(f"[load] {name}: {n_param/1e6:.2f}M params, {s['n_layers']} layers, dim {s['dim']}",
          flush=True)
    return m, int(n_param)


def frame_pass(model, rec, span, device, normalize, layers, dim, out: Path, chunk_sec=20.0):
    """Identical frame grid to aves_baseline.frame_pass -- frame i covers [i*HOP, i*HOP+RF).

    Writes (len(layers), nF, dim) float16 straight into a memmap so peak RAM is one chunk, and only
    renames the file into place once every frame is filled: a half-written file that got 'skipped'
    on the next run would silently corrupt the table.
    """
    nF = (span - RF) // HOP + 1
    tmp = out.with_name(out.name + ".partial.npy")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(len(layers), nF, dim))
    C = int(chunk_sec * SR)
    filled, t0 = 0, time.time()
    for c in range((span + C - 1) // C):
        lo = c * C
        x = rec[lo:min(span, lo + C + RF - HOP)]
        if len(x) < RF:
            break
        if normalize:
            mu, var = float(x.mean()), float(x.var())
            xin = ((x - mu) / np.sqrt(var + 1e-5)).astype(np.float32)
        else:
            xin = x.astype(np.float32)
        with torch.no_grad():
            feats, _ = model.extract_features(torch.from_numpy(xin).unsqueeze(0).to(device), None)
        if len(feats) <= layers[-1]:
            raise RuntimeError(f"model returned {len(feats)} layers, need index {layers[-1]}")
        i0 = lo // HOP
        take = min(feats[0].shape[1], C // HOP, nF - i0)
        for k, l in enumerate(layers):
            arr[k, i0:i0 + take] = feats[l][0, :take].cpu().numpy().astype(np.float16)
        del feats
        filled += take
        if c % 25 == 0:
            print(f"    chunk {c}  {filled}/{nF}  {time.time()-t0:.0f}s", flush=True)
    arr.flush()
    del arr
    gc.collect()
    if filled < nF - 2:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"only filled {filled}/{nF} frames -- grid misaligned")
    os.replace(tmp, out)
    print(f"  wrote {out.name}  ({nF} frames, {time.time()-t0:.0f}s)", flush=True)
    return nF


def pipe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(CKPTS))
    ap.add_argument("--encode-only", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--report", action="store_true",
                    help="print the canonical table from the existing JSON and exit")
    args = ap.parse_args()
    names = [n for n in args.models.split(",") if n]
    if args.report:
        report()
        return

    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score, average_precision_score

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[device] {device}", flush=True)
    DETDIR.mkdir(parents=True, exist_ok=True)

    # ---------------- reference labels / frame grid (ZF in-distribution)
    FR = np.load(FEAT / "frames_30min.npz", allow_pickle=True)
    y_zf, centers, energy_zf = FR["y"], FR["centers"], FR["energy"]
    nF_ref = len(y_zf)

    # ---------------- BirdPark audio, labels, frame grid
    bp, sr = sf.read(BP / "birdpark_16k.wav", dtype="float32")
    assert sr == SR and bp.ndim == 1, f"BirdPark must be 16 kHz mono, got sr={sr} ndim={bp.ndim}"
    iv_bp = np.load(BP / "birdpark_labels.npz", allow_pickle=True)["merged"]     # seconds
    ann_end = float(iv_bp.max())
    nF_bp = (len(bp) - RF) // HOP + 1
    t_bp = (np.arange(nF_bp) * HOP + RF / 2) / SR
    y_bp_full = np.zeros(nF_bp, dtype=int)
    for a, b in iv_bp:
        y_bp_full[(t_bp >= a) & (t_bp <= b)] = 1
    keep = t_bp <= ann_end                          # never score past the annotated span
    y_bp = y_bp_full[keep]
    print(f"[birdpark] {len(bp)/SR:.1f}s, {len(iv_bp)} events, annotated to {ann_end:.1f}s; "
          f"{keep.sum()} scorable frames, prevalence {y_bp.mean():.4f}", flush=True)
    print(f"[zf] {nF_ref} frames, prevalence {y_zf.mean():.4f}", flush=True)

    # ---------------- encode
    params = {}
    if not args.score_only:
        need = min(SPAN + 20 * SR, sf.info(AUDIO).frames)
        rec, _ = sf.read(AUDIO, dtype="float32", frames=need)
        for name in names:
            s = spec(name)
            layers = layer_grid(s["n_layers"])
            todo = [(tag, ds) for tag in TAGS for ds in ("zf", "bp")
                    if not (DETDIR / f"{ds}_{name}_{tag}.npy").exists()]
            if not todo:
                print(f"[skip] {name}: all 4 feature files present")
                continue
            model, n_param = load_model(name, s, device)
            params[name] = n_param
            try:
                for tag in TAGS:
                    norm = (tag == "matched")
                    p_zf = DETDIR / f"zf_{name}_{tag}.npy"
                    if not p_zf.exists():
                        print(f"\n=== {name} / {tag} / ZF 30 min, layers {layers} ===", flush=True)
                        nF = frame_pass(model, rec, SPAN, device, norm, layers, s["dim"], p_zf)
                        if nF != nF_ref:
                            raise RuntimeError(f"frame count {nF} != reference {nF_ref}")
                    p_bp = DETDIR / f"bp_{name}_{tag}.npy"
                    if not p_bp.exists():
                        print(f"=== {name} / {tag} / BirdPark ===", flush=True)
                        nb = frame_pass(model, bp, len(bp), device, norm, layers, s["dim"], p_bp)
                        if nb != nF_bp:
                            raise RuntimeError(f"BP frame count {nb} != {nF_bp}")
            finally:
                del model
                gc.collect()
                if device == "mps":
                    torch.mps.empty_cache()
        del rec
        gc.collect()
    if args.encode_only:
        print("[encode-only] done")
        return

    # ---------------- CV on ZF: contiguous 60 s time blocks
    grp = np.zeros(nF_ref, dtype="<U3")
    cv, g, desc = sp.choose_cv(grp, centers, n_splits=5, seed=0)
    print(f"[cv] {desc}", flush=True)

    out = {"split": desc, "zf": dict(n_frames=int(nF_ref), prevalence=float(y_zf.mean())),
           "birdpark": dict(n_frames_scored=int(keep.sum()), prevalence=float(y_bp.mean()),
                            n_events=int(len(iv_bp)), annotated_sec=ann_end,
                            independent_30s_blocks=float(keep.sum() / BLOCK)),
           "layer_grids": {}, "params": {}, "tags": list(TAGS),
           "in_distribution": {}, "zf_to_bp": {}, "precommit": {}, "consistency": {},
           "bootstrap": {},
           "block_frames": {"main": BLOCK, "small": BLOCK_SMALL},
           "caveats": [
               "BirdPark is 118.5 s = 5925 frames ~ 4 independent 30 s blocks. Every BirdPark "
               "interval here is underpowered; block=500 and the per-block win counts are given "
               "because a 4-block bootstrap CI is barely a measurement.",
               "The ZF in-distribution split is contiguous 60 s time blocks WITHIN one recording, "
               "so it holds out time, not birds and not recordings.",
               "Layers are pre-committed on ZF and frozen for BirdPark; the per-layer BirdPark "
               "columns in zf_to_bp are for auditing only and must not be maxed over.",
               "'native' = raw waveform, which is AVES's pretraining condition; for run11 'native' "
               "is OFF-condition input and its native arm is 'matched'."]}
    if OUTJSON.exists():                            # resume: keep arms already scored
        try:
            prev = json.loads(OUTJSON.read_text())
            for k in ("in_distribution", "zf_to_bp", "precommit", "layer_grids", "params",
                      "consistency", "bootstrap"):
                out[k].update(prev.get(k, {}))
        except Exception as e:
            print(f"[warn] could not reuse {OUTJSON.name}: {e}")

    # ---------------- baselines that are not encoders
    print("\n=== non-encoder baselines ===", flush=True)
    pr = cross_val_predict(pipe(), energy_zf.reshape(-1, 1), y_zf, groups=g, cv=cv,
                           method="predict_proba")[:, 1]
    s_ = mx.score(y_zf, pr, desc)
    out["in_distribution"]["logenergy"] = {"L0": dict(auc=s_.auc, ap=s_.ap)}
    print(f"  ZF  logenergy        AUC {s_.auc:.4f}  AP {s_.ap:.4f}", flush=True)
    en_bp = np.array([20 * np.log10(np.sqrt((bp[i * HOP:i * HOP + RF] ** 2).mean()) + 1e-12)
                      for i in range(nF_bp)])[keep]
    s_ = mx.score(y_bp, en_bp, "ZF->BP")
    out["zf_to_bp"]["logenergy"] = {"L0": dict(auc=s_.auc, ap=s_.ap)}
    print(f"  BP  logenergy        AUC {s_.auc:.4f}  AP {s_.ap:.4f}", flush=True)

    # ---------------- score every arm, one checkpoint at a time
    for name in names:
        s = spec(name)
        layers = layer_grid(s["n_layers"])
        out["layer_grids"][name] = layers
        if name in params:
            out["params"][name] = params[name]
        pf = DETDIR / f"preds_{name}.npz"
        if pf.exists():
            print(f"\n=== {name}: predictions cached, re-reading ===", flush=True)
            _z = np.load(pf)
            P = {k: _z[k] for k in _z.files}
        else:
            P = {}
            print(f"\n=== {name}: in-distribution ZF, then ZF -> BirdPark ===", flush=True)
            for tag in TAGS:
                Z = np.load(DETDIR / f"zf_{name}_{tag}.npy", mmap_mode="r")
                B = np.load(DETDIR / f"bp_{name}_{tag}.npy", mmap_mode="r")
                assert Z.shape[1] == nF_ref and B.shape[1] == nF_bp, \
                    f"{name}/{tag} shapes {Z.shape} {B.shape}"
                for k, l in enumerate(layers):
                    X = np.asarray(Z[k], dtype=np.float32)
                    p_in = cross_val_predict(pipe(), X, y_zf, groups=g, cv=cv,
                                             method="predict_proba")[:, 1]
                    est = pipe().fit(X, y_zf)
                    p_bp = est.predict_proba(np.asarray(B[k], dtype=np.float32))[:, 1][keep]
                    P[f"in_{tag}_L{l}"] = p_in.astype(np.float32)
                    P[f"bp_{tag}_L{l}"] = p_bp.astype(np.float32)
                    si = mx.score(y_zf, p_in, desc)
                    sb = mx.score(y_bp, p_bp, "ZF->BP")
                    print(f"  {tag:7s} L{l:<2d}  ZF AUC {si.auc:.4f} AP {si.ap:.4f} | "
                          f"BP AUC {sb.auc:.4f} AP {sb.ap:.4f}", flush=True)
                    del X, est
                    gc.collect()
                del Z, B
                gc.collect()
            np.savez(pf, **P)
        # record
        out["in_distribution"].setdefault(name, {})
        out["zf_to_bp"].setdefault(name, {})
        for tag in TAGS:
            for l in layers:
                si = mx.score(y_zf, P[f"in_{tag}_L{l}"], desc)
                sb = mx.score(y_bp, P[f"bp_{tag}_L{l}"], "ZF->BP")
                out["in_distribution"][name][f"{tag}_L{l}"] = dict(auc=si.auc, ap=si.ap)
                out["zf_to_bp"][name][f"{tag}_L{l}"] = dict(auc=sb.auc, ap=sb.ap)
        # pre-commit the layer on ZF, per normalisation and jointly, and freeze it for BirdPark
        pc = {}
        for tag in TAGS:
            best = max(layers, key=lambda l: out["in_distribution"][name][f"{tag}_L{l}"]["auc"])
            pc[tag] = dict(layer=int(best),
                           zf=out["in_distribution"][name][f"{tag}_L{best}"],
                           bp=out["zf_to_bp"][name][f"{tag}_L{best}"])
        jt = max(TAGS, key=lambda t: pc[t]["zf"]["auc"])
        pc["joint"] = dict(tag=jt, **pc[jt])
        out["precommit"][name] = pc
        print(f"  pre-committed on ZF: {jt} L{pc[jt]['layer']}  "
              f"ZF AUC {pc[jt]['zf']['auc']:.4f} -> BP AUC {pc[jt]['bp']['auc']:.4f}", flush=True)
        OUTJSON.write_text(json.dumps(out, indent=2))

    # ---------------- internal consistency vs the already-published feature files
    if "run11" in names and (DETDIR / "zf_run11_matched.npy").exists():
        Z = np.load(DETDIR / "zf_run11_matched.npy", mmap_mode="r")
        lg = layer_grid(12)
        d = {}
        for l in (0, 6):
            a = np.asarray(Z[lg.index(l)], dtype=np.float32)
            b = FR[f"F{l}"].astype(np.float32)
            d[f"run11_L{l}_max_abs_diff_vs_frames_30min"] = float(np.abs(a - b).max())
            del a, b
            gc.collect()
        out["consistency"].update(d)
        print(f"\n[consistency] {d}", flush=True)
        del Z
    if "aves-base-bio" in names and (FEAT / "aves_frames_matched.npz").exists():
        Z = np.load(DETDIR / "zf_aves-base-bio_matched.npy", mmap_mode="r")
        A = np.load(FEAT / "aves_frames_matched.npz", allow_pickle=True)
        lg = layer_grid(12)
        d = {}
        for l in (0, 6):
            d[f"aves_bio_L{l}_max_abs_diff_vs_aves_frames"] = float(
                np.abs(np.asarray(Z[lg.index(l)], dtype=np.float32) - A[f"F{l}"].astype(np.float32)).max())
            gc.collect()
        out["consistency"].update(d)
        print(f"[consistency] {d}", flush=True)
        del Z, A
    gc.collect()

    # ---------------- bootstraps against run11's pre-committed arm
    if "run11" in out["precommit"]:
        r11 = out["precommit"]["run11"]["joint"]
        rtag, rlay = r11["tag"], r11["layer"]
        Pr = np.load(DETDIR / "preds_run11.npz")
        ref_in, ref_bp = Pr[f"in_{rtag}_L{rlay}"], Pr[f"bp_{rtag}_L{rlay}"]
        out["reference_arm"] = dict(name="run11", tag=rtag, layer=int(rlay))
        print(f"\n=== bootstraps vs run11 {rtag} L{rlay} ===", flush=True)
        out.setdefault("bootstrap", {})
        for name in [n for n in names if n != "run11"]:
            pc = out["precommit"][name]["joint"]
            P = np.load(DETDIR / f"preds_{name}.npz")
            a_in, a_bp = P[f"in_{pc['tag']}_L{pc['layer']}"], P[f"bp_{pc['tag']}_L{pc['layer']}"]
            rec_ = {"arm": f"{pc['tag']}_L{pc['layer']}"}
            for nm, f in (("auc", roc_auc_score), ("ap", average_precision_score)):
                r = mx.paired_bootstrap(y_zf, a_in, ref_in, block=BLOCK, n=2000, seed=0, metric=f)
                rec_[f"indist_{nm}_block{BLOCK}"] = r
                print(f"  {name:22s} IN  {nm.upper():3s} minus_run11 {r['delta']:+.4f} "
                      f"[{r['lo']:+.4f}, {r['hi']:+.4f}] {r['verdict']}", flush=True)
            for blk in (BLOCK, BLOCK_SMALL):
                for nm, f in (("auc", roc_auc_score), ("ap", average_precision_score)):
                    r = mx.paired_bootstrap(y_bp, a_bp, ref_bp, block=blk, n=2000, seed=0, metric=f)
                    rec_[f"bp_{nm}_block{blk}"] = r
                    print(f"  {name:22s} BP  {nm.upper():3s} b{blk:<4d} minus_run11 "
                          f"{r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] {r['verdict']}",
                          flush=True)
            for blk in (BLOCK, BLOCK_SMALL):
                rec_[f"bp_blockwise_block{blk}"] = blockwise(y_bp, a_bp, ref_bp, blk, roc_auc_score)
            rec_[f"bp_blockwise_vs_logenergy_block{BLOCK}"] = blockwise(
                y_bp, a_bp, en_bp, BLOCK, roc_auc_score)
            out["bootstrap"][name] = rec_
            OUTJSON.write_text(json.dumps(out, indent=2))
        # run11 vs energy, and the family best vs energy, for scale
        for nm, f in (("auc", roc_auc_score), ("ap", average_precision_score)):
            r = mx.paired_bootstrap(y_bp, ref_bp, en_bp, block=BLOCK, n=2000, seed=0, metric=f)
            out["bootstrap"].setdefault("run11_vs_logenergy", {})[f"bp_{nm}_block{BLOCK}"] = r
            print(f"  run11 vs logenergy BP {nm.upper():3s} {r['delta']:+.4f} "
                  f"[{r['lo']:+.4f}, {r['hi']:+.4f}] {r['verdict']}", flush=True)

    OUTJSON.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUTJSON}")


def report():
    """The canonical table, rebuilt from the JSON so it can never drift from what was computed.

    Also writes back a `summary` block (rankings of the pre-committed arms, plus the auditable
    maxima over every arm) so the JSON is self-contained -- and labels those maxima as AUDIT ONLY,
    because a max over 84 arms is not an estimate of anything.
    """
    d = json.loads(OUTJSON.read_text())
    pc = d.get("precommit", {})
    if pc:
        rank = {}
        for ds, met in (("zf", "auc"), ("zf", "ap"), ("bp", "auc"), ("bp", "ap")):
            rank[f"{ds}_{met}"] = [[m, pc[m]["joint"][ds][met]] for m in
                                   sorted(pc, key=lambda m: -pc[m]["joint"][ds][met])]
        audit = {}
        for ds, key in (("zf", "in_distribution"), ("bp", "zf_to_bp")):
            cand = [(v["auc"], m, k) for m in d[key] if m != "logenergy"
                    for k, v in d[key][m].items()]
            a, m, k = max(cand)
            audit[f"best_{ds}_auc_any_arm_AUDIT_ONLY"] = dict(model=m, arm=k, auc=a)
        # what the layer discipline COST each model on BirdPark: if pre-committing on ZF hurt run11
        # less than it hurt the AVES family, then the AVES wins are understated, not manufactured
        cost = {}
        for m in pc:
            bb = max(d["zf_to_bp"][m].items(), key=lambda kv: kv[1]["auc"])
            cost[m] = dict(precommitted_arm=f"{pc[m]['joint']['tag']}_L{pc[m]['joint']['layer']}",
                           precommitted_bp_auc=pc[m]["joint"]["bp"]["auc"],
                           best_bp_arm=bb[0], best_bp_auc=bb[1]["auc"],
                           cost_of_precommitting=pc[m]["joint"]["bp"]["auc"] - bb[1]["auc"])
        d["summary"] = dict(precommit_ranking=rank, precommit_cost_on_birdpark=cost, **audit,
                            precommit_criterion="max out-of-fold AUC on the ZF in-distribution set",
                            n_arms_scored=sum(len(d["in_distribution"][m])
                                              for m in d["in_distribution"] if m != "logenergy"))
        OUTJSON.write_text(json.dumps(d, indent=2))
    print(f"split: {d['split']}")
    print(f"ZF in-distribution: {d['zf']['n_frames']} frames, prevalence {d['zf']['prevalence']:.4f}")
    b = d["birdpark"]
    print(f"BirdPark holdout:   {b['n_frames_scored']} frames, prevalence {b['prevalence']:.4f}, "
          f"{b['n_events']} events, {b['annotated_sec']:.1f}s annotated "
          f"= {b['independent_30s_blocks']:.1f} independent 30 s blocks")
    ref = d.get("reference_arm", {})
    hdr = (f"{'checkpoint':22s} {'params':>8s} {'norm':8s} {'L*':>3s} "
           f"{'ZF AUC':>7s} {'ZF AP':>7s} {'BP AUC':>7s} {'BP AP':>7s}")
    print("\n" + hdr); print("-" * len(hdr))
    # Iterate the union, not CKPTS: names passed via --models that are not in the built-in list
    # (e.g. dapt* checkpoints) are scored and written to JSON, and would otherwise be computed and
    # then silently dropped from the printed table by the `continue` below.
    scored = list(d.get("precommit", {}))
    for name in CKPTS + [n for n in scored if n not in CKPTS]:
        if name not in d.get("precommit", {}):
            continue
        pc = d["precommit"][name]
        pr = d.get("params", {}).get(name)
        ps = f"{pr/1e6:.2f}M" if pr else "?"
        for tag in d["tags"]:
            mark = " <-" if pc["joint"]["tag"] == tag else ""
            print(f"{name:22s} {ps:>8s} {tag:8s} {pc[tag]['layer']:3d} "
                  f"{pc[tag]['zf']['auc']:7.4f} {pc[tag]['zf']['ap']:7.4f} "
                  f"{pc[tag]['bp']['auc']:7.4f} {pc[tag]['bp']['ap']:7.4f}{mark}")
    for nm, lab in (("logenergy", "log-energy"),):
        if nm in d["in_distribution"]:
            i = d["in_distribution"][nm]["L0"]; o = d["zf_to_bp"][nm]["L0"]
            print(f"{lab:22s} {'-':>8s} {'-':8s} {'-':>3s} "
                  f"{i['auc']:7.4f} {i['ap']:7.4f} {o['auc']:7.4f} {o['ap']:7.4f}")
    print("\n'L*' = layer pre-committed by max out-of-fold AUC on the ZF in-distribution set; "
          "'<-' = the normalisation that arm was chosen under.\n"
          "log-energy is a 1-D probe on frame log-RMS, not an encoder.\n"
          "For run11, 'native' means raw audio, which is NOT its pretraining condition -- it is a "
          "control, not a fair reading of run11.")
    if "summary" in d:
        print(f"\npre-committed rankings ({d['summary']['n_arms_scored']} arms scored in total):")
        for k, v in d["summary"]["precommit_ranking"].items():
            print(f"  {k:7s} " + "  ".join(f"{m}={a:.4f}" for m, a in v))
        for k in ("best_zf_auc_any_arm_AUDIT_ONLY", "best_bp_auc_any_arm_AUDIT_ONLY"):
            a = d["summary"][k]
            print(f"  {k}: {a['model']} {a['arm']} {a['auc']:.4f}")
        print("  what pre-committing the layer on ZF cost each model on BirdPark AUC:")
        for m, c in d["summary"]["precommit_cost_on_birdpark"].items():
            print(f"    {m:22s} {c['precommitted_arm']:12s} {c['precommitted_bp_auc']:.4f}  vs "
                  f"best arm {c['best_bp_arm']:12s} {c['best_bp_auc']:.4f}  "
                  f"({c['cost_of_precommitting']:+.4f})")
    for c in d.get("caveats", []):
        print(f"  ! {c}")
    if ref:
        print(f"\nbootstraps vs {ref['name']} {ref['tag']} L{ref['layer']} "
              f"(positive = the AVES variant is ahead)")
        for name, r in d.get("bootstrap", {}).items():
            if name == "run11_vs_logenergy":
                continue
            def f(k):
                v = r.get(k)
                return (f"{v['delta']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}] {v['verdict']}"
                        if v else "n/a")
            print(f"  {name:22s} {r.get('arm','')}")
            print(f"    in-dist  AUC {f('indist_auc_block1500')}")
            print(f"    in-dist  AP  {f('indist_ap_block1500')}")
            print(f"    BirdPark AUC {f('bp_auc_block1500')}   (b500 {f('bp_auc_block500')})")
            print(f"    BirdPark AP  {f('bp_ap_block1500')}   (b500 {f('bp_ap_block500')})")
            for k in ("bp_blockwise_block1500", "bp_blockwise_block500"):
                bw = r.get(k)
                if bw:
                    print(f"    per-block AUC wins vs run11 (block {bw['block']}): "
                          f"{bw['a_wins']}/{bw['n_blocks_used']}")


def blockwise(y, p_a, p_b, block, metric):
    """Per-block win count: with four independent blocks this is what BirdPark can actually say."""
    y = np.asarray(y).astype(int)
    rows, wins, used = [], 0, 0
    for s0 in range(0, len(y), block):
        sl = slice(s0, min(len(y), s0 + block))
        yy = y[sl]
        if len(np.unique(yy)) < 2 or (sl.stop - sl.start) < block // 2:
            rows.append(dict(start=int(s0), n=int(sl.stop - sl.start), a=None, b=None,
                             note="skipped (single class or short)"))
            continue
        a, b = float(metric(yy, p_a[sl])), float(metric(yy, p_b[sl]))
        rows.append(dict(start=int(s0), n=int(sl.stop - sl.start), a=a, b=b))
        used += 1
        wins += int(a > b)
    return dict(block=int(block), n_blocks_used=used, a_wins=int(wins), blocks=rows)


if __name__ == "__main__":
    main()

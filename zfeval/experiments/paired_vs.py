#!/usr/bin/env python3
"""Paired detection bootstrap between ANY two scored models, each at its own pre-committed arm.

detection_variants.py bootstraps every model against run11 only. Some questions need a different
reference -- E5 asks whether the x3 replay arm beats the x1 arm. This reuses detection_variants'
saved predictions (detvar/preds_<name>.npz), its labels, block sizes and blockwise(), and the same
zfeval.metrics.paired_bootstrap call (n=2000, seed 0): nothing about the protocol changes, only the
reference. `--check` reproduces an existing vs-run11 record from detection_variants.json exactly.

    paired_vs.py --a daptreplay3_5e5_step15000 --b daptreplay_5e5_step15000 [--out paired_vs.json]
    paired_vs.py --check daptreplay_5e5_step15000
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detection_variants as dv                                  # noqa: E402  (main is guarded)
from zfeval import metrics as mx                                 # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score  # noqa: E402


def labels():
    y_zf = np.load(dv.FEAT / "frames_30min.npz", allow_pickle=True)["y"]
    bp, sr = sf.read(dv.BP / "birdpark_16k.wav", dtype="float32")
    assert sr == dv.SR and bp.ndim == 1
    iv = np.load(dv.BP / "birdpark_labels.npz", allow_pickle=True)["merged"]
    nF = (len(bp) - dv.RF) // dv.HOP + 1
    t = (np.arange(nF) * dv.HOP + dv.RF / 2) / dv.SR
    y = np.zeros(nF, dtype=int)
    for a, b in iv:
        y[(t >= a) & (t <= b)] = 1
    return y_zf, y[t <= float(iv.max())]


def arm_preds(name, D):
    pc = D["precommit"][name]["joint"]
    P = np.load(dv.DETDIR / f"preds_{name}.npz")
    return f"{pc['tag']}_L{pc['layer']}", P[f"in_{pc['tag']}_L{pc['layer']}"], P[f"bp_{pc['tag']}_L{pc['layer']}"]


def compare(a, b, D, y_zf, y_bp):
    arm_a, a_in, a_bp = arm_preds(a, D)
    arm_b, b_in, b_bp = arm_preds(b, D)
    rec = {"a": a, "b": b, "arm_a": arm_a, "arm_b": arm_b,
           "note": "delta is the RESAMPLE MEAN (zfeval.metrics); observed = metric(a) - metric(b) below"}
    for nm, f in (("auc", roc_auc_score), ("ap", average_precision_score)):
        rec[f"indist_{nm}_block{dv.BLOCK}"] = mx.paired_bootstrap(y_zf, a_in, b_in, block=dv.BLOCK, n=2000, seed=0, metric=f)
        rec[f"indist_{nm}_observed"] = float(f(y_zf, a_in) - f(y_zf, b_in))
    for blk in (dv.BLOCK, dv.BLOCK_SMALL):
        for nm, f in (("auc", roc_auc_score), ("ap", average_precision_score)):
            rec[f"bp_{nm}_block{blk}"] = mx.paired_bootstrap(y_bp, a_bp, b_bp, block=blk, n=2000, seed=0, metric=f)
    for nm, f in (("auc", roc_auc_score), ("ap", average_precision_score)):
        rec[f"bp_{nm}_observed"] = float(f(y_bp, a_bp) - f(y_bp, b_bp))
    for blk in (dv.BLOCK, dv.BLOCK_SMALL):
        rec[f"bp_blockwise_auc_block{blk}"] = dv.blockwise(y_bp, a_bp, b_bp, blk, roc_auc_score)
        rec[f"bp_blockwise_ap_block{blk}"] = dv.blockwise(y_bp, a_bp, b_bp, blk, average_precision_score)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a"); ap.add_argument("--b")
    ap.add_argument("--check", help="reproduce detection_variants' vs-run11 record for this model")
    ap.add_argument("--out", default="paired_vs.json")
    args = ap.parse_args()
    D = json.loads(dv.OUTJSON.read_text())
    y_zf, y_bp = labels()
    if args.check:
        rec = compare(args.check, "run11", D, y_zf, y_bp)
        old = D["bootstrap"][args.check]
        keys = [k for k in old if isinstance(old[k], dict) and "lo" in old[k]]
        bad = [k for k in keys if any(old[k][f] != rec[k][f] for f in ("delta", "lo", "hi"))]
        bw = old[f"bp_blockwise_block{dv.BLOCK}"]["a_wins"] == rec[f"bp_blockwise_auc_block{dv.BLOCK}"]["a_wins"]
        print(f"[check] {args.check}: {len(keys)} intervals, mismatches {bad}, AUC block wins match {bw}")
        sys.exit(0 if not bad and bw else 1)
    rec = compare(args.a, args.b, D, y_zf, y_bp)
    out = dv.ANA / args.out
    allrec = json.loads(out.read_text()) if out.exists() else {}
    allrec[f"{args.a}__vs__{args.b}"] = rec
    out.write_text(json.dumps(allrec, indent=2))
    for k, v in rec.items():
        if isinstance(v, dict) and "lo" in v:
            print(f"  {k:24s} [{v['lo']:+.4f}, {v['hi']:+.4f}] {v['verdict']}")
        elif k.endswith("_observed"):
            print(f"  {k:24s} {v:+.4f}")
        elif k.startswith("bp_blockwise"):
            print(f"  {k:24s} a wins {v['a_wins']} of {v['n_blocks_used']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

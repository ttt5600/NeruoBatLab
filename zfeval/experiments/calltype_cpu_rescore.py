#!/usr/bin/env python
"""Re-encode the 11-class call-type clips on CPU for every model and re-score call type beside the MPS numbers.

Why: on this Mac (torch 2.8.0) MPS scaled_dot_product_attention returns wrong outputs for sequences of 2-8 frames
(T=1 and T>=9 agree with CPU to ~1e-6; CPU fp32 agrees with fp64 to ~1e-5). The median call-type clip is ~7 frames,
so 1,940 of 3,412 clips were mis-encoded in every MPS cache (features/ct11_<tag>_emb.npy). Nothing is overwritten:
CPU embeddings go to features/ct11_<tag>_cpuemb.npy and scores to analysis/calltype_cpu_rescore.json, beside the
original JSONs. Same protocol as the board (embed_all, mean-pooled per clip at its own length; cv_acc leave-birds-out
StratifiedGroupKFold(5) seed 0; StandardScaler + LogisticRegression; 11-class best layer).

Resumable: a model whose _cpuemb.npy exists is not re-encoded.

    python zfeval/experiments/calltype_cpu_rescore.py [--only tag1,tag2]
"""
from __future__ import annotations
import argparse, json, sys, time, traceback, warnings
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
from calltype11 import collect, KEEP11                                     # noqa: E402
from aves_calltype import cv_acc                                           # noqa: E402
from aves_variants_calltype import embed_all                               # noqa: E402
from detection_variants import spec, load_model                            # noqa: E402

FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
OUT = ANA / "calltype_cpu_rescore.json"
AVJ = json.loads((ANA / "aves_variants_calltype.json").read_text())["models"]
MPS_CACHE = {"aves-base-bio": "ct11_aves_emb.npy", "run11": "ct11_run11_emb.npy"}
FIRST = ["aves-base-bio", "daptv4two_2e4s2_step15000", "run16_compute4x", "run11", "birdaves-biox-base",
         "hf-whisper-small-enc", "run16_seed2", "run16_seed3", "run19_avesteacher"]


def reported(tag):
    if tag in AVJ:
        v = AVJ[tag]
        return list(v["per_layer_acc_11"]), v["best_11"]["layer"], v["best_11"]["acc"]
    d = json.loads((ANA / ("run15_calltype.json" if tag == "run15_combined" else f"{tag}_calltype.json")).read_text())
    return list(d["per_layer_acc11"]), d["best_layer11"], d["acc11"]


def all_tags():
    tags = list(AVJ)
    for p in sorted(ANA.glob("*_calltype.json")):
        t = p.name[: -len("_calltype.json")]
        if t in ("aves", "aves_variants", "finetune", "calltype11"):
            continue
        t = "run15_combined" if t == "run15" else t
        if t not in tags and "per_layer_acc11" in json.loads(p.read_text()):
            tags.append(t)
    return FIRST + [t for t in tags if t not in FIRST]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default="")
    only = [t for t in ap.parse_args().only.split(",") if t]
    torch.set_num_threads(8)
    rows = [r for r in collect() if r[3] in KEEP11]
    paths = [r[0] for r in rows]
    birds = np.array([r[1].lower() for r in rows])
    tt = np.array([r[3] for r in rows]); classes = sorted(set(tt))
    y = np.array([classes.index(t) for t in tt])
    assert len(y) == 3412 and len(set(birds)) == 48
    res = json.loads(OUT.read_text()) if OUT.exists() else dict(
        why="torch 2.8 MPS SDPA wrong for 2-8 frame sequences; CPU re-encode of the 11-class call-type clips",
        protocol="embed_all on cpu; cv_acc leave-birds-out StratifiedGroupKFold(5) seed 0; 11-class", models={})
    for tag in (only or all_tags()):
        try:
            t0 = time.time()
            cpu = FEAT / f"ct11_{tag}_cpuemb.npy"
            if cpu.exists():
                E = np.load(cpu)
            else:
                s = spec(tag)
                model, _ = load_model(tag, s, "cpu")
                model.eval()
                E = embed_all(model, paths, "cpu", s["n_layers"], s["dim"], tag=tag)
                np.save(cpu, E)
                del model
            mps_layers, mps_best, mps_acc = reported(tag)
            accs = [cv_acc(np.ascontiguousarray(E[:, l]), y, birds)[0] for l in range(E.shape[1])]
            b = int(np.argmax(accs))
            M = np.load(FEAT / MPS_CACHE.get(tag, f"ct11_{tag}_emb.npy"), mmap_mode="r")
            d = np.abs(np.asarray(E) - np.asarray(M)).max(axis=2)
            res["models"][tag] = dict(per_layer_acc11_cpu=accs, best_layer_cpu=b, acc11_cpu=accs[b],
                                      per_layer_acc11_mps=mps_layers, best_layer_mps=mps_best, acc11_mps=mps_acc,
                                      acc11_L3_cpu=accs[3] if len(accs) > 3 else None,
                                      n_clips_changed=int((d.max(1) > 1e-2).sum()))
            OUT.write_text(json.dumps(res, indent=1))
            print(f"[{tag}] cpu best L{b} {accs[b]:.4f} | mps best L{mps_best} {mps_acc:.4f} | "
                  f"diff {accs[b] - mps_acc:+.4f} | clips changed {res['models'][tag]['n_clips_changed']} | "
                  f"{time.time() - t0:.0f}s", flush=True)
        except Exception as e:                                      # keep going; report at the end
            print(f"[{tag}] FAILED: {e!r}", flush=True); traceback.print_exc()
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

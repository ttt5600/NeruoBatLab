#!/usr/bin/env python
"""One table of every model we trained, with its single change, verified data, and CPU call type.

Reads only score files (never re-computes): calltype_cpu_rescore.json (best layer + layer 3), seed_pairs_cpu.json and
seed_pairs_cpu_L3.json (vs run16, 95% bird bootstrap), calltype_split_sensitivity_cpu.json (10-split mean). Hours are the
Savio training TSVs summed on 2026-10-21 (ZF 120 files 116.03 h, FSD50K 643 files 108.39 h, both spot-checked against
the audio headers). Missing entries print as '-' rather than being filled in.

    python zfeval/experiments/our_models_table.py            -> markdown table on stdout + analysis/our_models_table.json
"""
from __future__ import annotations
import json
from pathlib import Path

ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
ZF, FSD, BOTH = "ZF 116.0 h", "FSD50K 108.4 h", "ZF + FSD50K 224.4 h"
SPEC, ITER2, AVESL, TWO, MFCC = ("spectrogram k-means", "run16 layer-6 k-means", "AVES layer-6 k-means",
                                 "run16 + AVES layer-6 k-means", "MFCC k-means")
# tag: (name, what changed and against what, data, labels, family)
OURS = {
    "run11": ("run11", "first release: 1 GPU, k=100", ZF, SPEC + " (k=100)", "scratch"),
    "run15_combined": ("run15", "run11 on the combined corpus, k=200", BOTH, SPEC, "scratch"),
    "run16_compute4x": ("run16", "run15 on 4 GPUs (4x audio per update) -- REFERENCE", BOTH, SPEC, "scratch"),
    "run16_seed2": ("run16 seed 2", "run16, another seed", BOTH, SPEC, "scratch"),
    "run16_seed3": ("run16 seed 3", "run16, another seed", BOTH, SPEC, "scratch"),
    "run17_accum2": ("run17", "run16 + 2x audio per update", BOTH, SPEC, "scratch"),
    "run18_iter2": ("run18", "run16, labels from run16 itself (iteration 2)", BOTH, ITER2, "scratch"),
    "run19_avesteacher": ("run19", "run16, labels from AVES", BOTH, AVESL, "scratch"),
    "run19_seed2": ("run19 seed 2", "run19, another seed", BOTH, AVESL, "scratch"),
    "run20_fsdonly": ("run20", "run16 on general audio only", FSD, SPEC, "scratch"),
    "run20_seed2": ("run20 seed 2", "run20, another seed", FSD, SPEC, "scratch"),
    "run21_zfonly": ("run21", "run16 on zebra finch only", ZF, SPEC, "scratch"),
    "run22_avesfsd": ("run22", "run20 with AVES labels", FSD, AVESL, "scratch"),
    "run23_mfcc": ("run23", "run16 with stock MFCC labels (the control)", BOTH, MFCC, "scratch"),
    "run29_avesrecipe": ("E29", "run16 with AVES's masking / lr / warmup", BOTH, SPEC, "scratch"),
    "run29_avesrecipe_s2": ("E29 seed 2", "E29, another seed", BOTH, SPEC, "scratch"),
    "daptv4two_2e4_step15000": ("E19 DAPT", "AVES bio continued on ZF + FSD50K", "AVES 360 h, then ZF stream", TWO, "dapt"),
    "daptv4two_2e4s2_step15000": ("E23 DAPT", "E19, another seed", "AVES 360 h, then ZF stream", TWO, "dapt"),
    "daptv4bird_2e4_step15000": ("E14 DAPT", "E19 from BirdAVES biox-base", "BirdAVES 2,570 h, then ZF stream", TWO, "dapt"),
    "daptv4bxl_2e4_step15000": ("E27 DAPT", "E19 from BirdAVES bioxn-large (315M)", "BirdAVES 3,076 h, then ZF stream", TWO, "dapt"),
    "daptv4bxl_2e4s2_step15000": ("E27 DAPT seed 2", "E27, another seed", "BirdAVES 3,076 h, then ZF stream", TWO, "dapt"),
}
REFS = {"aves-base-bio": "AVES bio (download)", "birdaves-bioxn-large": "BirdAVES bioxn-large (download)",
        "hf-whisper-base-enc": "Whisper base (download)"}


def load(name):
    p = ANA / name
    return json.loads(p.read_text()) if p.exists() else {}


def main():
    cpu = load("calltype_cpu_rescore.json").get("models", {})
    pb, p3 = load("seed_pairs_cpu.json").get("pairs", {}), load("seed_pairs_cpu_L3.json").get("pairs", {})
    ss = load("calltype_split_sensitivity_cpu.json").get("models", {})
    pc = load("model_param_counts.json").get("models", {})

    def pair(pairs, tag):
        """A minus run16, whichever order the pair was stored in (reversed pairs are negated)."""
        d = pairs.get(f"{tag} | run16_compute4x")
        if d: return d
        r = pairs.get(f"run16_compute4x | {tag}")
        return dict(diff=-r["diff"], lo=-r["hi"], hi=-r["lo"]) if r else None

    def ci(d):
        if not d: return "-"
        r = d["lo"] > 0 or d["hi"] < 0
        return f"{d['diff']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]{' *' if r else ''}"

    rows = []
    for tag, meta in list(OURS.items()) + [(t, (n, "", "", "", "ref")) for t, n in REFS.items()]:
        c = cpu.get(tag)
        if not c:
            continue
        name, what, data, labels, fam = meta
        rows.append(dict(tag=tag, name=name, what=what, data=data, labels=labels, family=fam,
                         params_m=round(pc[tag]["params"] / 1e6, 1) if tag in pc else None,
                         best=round(c["acc11_cpu"], 4), best_layer=c["best_layer_cpu"],
                         l3=round(c["per_layer_acc11_cpu"][3], 4),
                         split10=round(ss[tag]["mean"], 4) if tag in ss else None,
                         vs16_best=None if tag == "run16_compute4x" else pair(pb, tag),
                         vs16_l3=None if tag == "run16_compute4x" else pair(p3, tag)))
    (ANA / "our_models_table.json").write_text(json.dumps(rows, indent=1))
    print("| Model | One change | Data | Labels | Call type (layer) | Layer 3 | 10-split | vs run16, best layer | vs run16, layer 3 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        s10 = "-" if r["split10"] is None else f"{r['split10']:.4f}"
        ref = r["tag"] == "run16_compute4x"
        print(f"| {r['name']} | {r['what']} | {r['data']} | {r['labels']} | {r['best']:.4f} (L{r['best_layer']}) | "
              f"{r['l3']:.4f} | {s10} | {'reference' if ref else ci(r['vs16_best'])} | "
              f"{'reference' if ref else ci(r['vs16_l3'])} |")


if __name__ == "__main__":
    main()

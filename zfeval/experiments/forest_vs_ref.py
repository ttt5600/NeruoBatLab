#!/usr/bin/env python
"""Every model on the CPU call-type board against ONE reference model: the rows behind the report doc's forest plots
("Every model against vanilla HuBERT", widget a557a9c6-097f; "Against the best download", widget 95e3ed0b-092d).

Single-model pairs from seed_pairs_cpu.py (bird bootstrap only -- these intervals ignore seed noise, sd ~0.011; use
seed_mean_pairs_cpu.py for arm claims), at each model's best layer and at layer 3. Intermediate DAPT trajectory checkpoints
are left out (picking the best one would inflate). Writes analysis/forest_vs_<REF>.json: rows grouped scratch / dapt / dl,
sorted by difference, each {slug, name, g, acc, d, lo, hi, l3, res} with res in above / below / nd, plus the shuffled-label
control (far off any sensible axis) as `controls`.

    python zfeval/experiments/forest_vs_ref.py run31_vanilla_lr1e4            # computes missing pairs, then writes rows
    python zfeval/experiments/forest_vs_ref.py run31_vanilla_lr1e4 --rows-only  # pairs already in seed_pairs_cpu[_L3].json
"""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"

SCRATCH = {"run19_seed2": "run19 seed 2", "run22_avesfsd": "run22", "run19_avesteacher": "run19", "run23_mfcc": "run23",
           "run18_iter2": "run18", "run16_seed3": "run16 seed 3", "run16_compute4x": "run16", "run20_seed2": "run20 seed 2",
           "run29_avesrecipe": "E29", "run17_accum2": "run17", "run20_fsdonly": "run20", "run29_avesrecipe_s2": "E29 seed 2",
           "run15_combined": "run15", "run16_seed2": "run16 seed 2", "run11": "run11", "run21_zfonly": "run21",
           "run24_bioxnl12": "E13", "run24_bioxnl12_s2": "E13 seed 2", "run30_logmel": "E30", "run30_logmel_s2": "E30 seed 2",
           "run37_randproj": "E37 random projection", "run37_randproj_s2": "E37 random projection seed 2",
           "run31_vanilla": "E36 stock lr (step 23,405)", "run31_vanilla_s2": "E36 stock lr seed 2",
           "run31_vanilla_lr1e4": "E36 vanilla lr 1e-4", "run31_vanilla_lr1e4_s2": "E36 vanilla lr 1e-4 seed 2"}
DAPT = {"daptv3aves_2e4s2_step15000": "E20", "daptv3aves_2e4s2m8_step15000": "E28b", "daptshort_5e5_step2500": "R2 short",
        "daptv3aves_2e4_step15000": "E18", "daptv4two_2e4_step15000": "E19", "daptlr_1e5_step15000": "R2 lr 1e-5",
        "daptv4two45k_2e4_step45000": "E22", "daptv4two_2e4s2_step15000": "E23", "daptv4aveso_2e4_step15000": "E21",
        "daptv4bxl_2e4s2_step15000": "E27 seed 2", "daptreplay_5e5_step15000": "R2 replay", "daptfreeze_5e5_step15000": "R2 freeze",
        "daptreplay3_5e5_step15000": "E5", "dapt5e5_step15000": "R1 lr 5e-5", "daptv3aves_5e5_step15000": "E17",
        "daptv4aveso_2e4s2_step15000": "E24", "daptv3aves_2e4m8_step15000": "E28a", "daptv4bxl_2e4_step15000": "E27",
        "daptv4bird_2e4_step15000": "E14", "dapt1e4_step15000": "R1 lr 1e-4"}
DL = {"hf-whisper-base-enc": "Whisper base", "hf-whisper-small-enc": "Whisper small", "hf-whisper-tiny-enc": "Whisper tiny",
      "birdaves-bioxn-large": "BirdAVES bioxn-large", "birdaves-biox-base": "BirdAVES biox-base",
      "birdaves-biox-large": "BirdAVES biox-large", "aves-base-core": "AVES core", "aves-base-all": "AVES all",
      "aves-base-bio": "AVES bio", "aves-base-nonbio": "AVES nonbio", "hf-ast-audioset": "AST (AudioSet)",
      "hf-wavlm-base-plus": "WavLM base+", "hf-hubert-base-ls960": "HuBERT base (speech)", "hf-wav2vec2-base": "wav2vec2 base",
      "hf-data2vec-audio-base": "data2vec audio base"}
CONTROLS = {"run37_shuffle": "E37 shuffled labels"}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        raise SystemExit(__doc__)
    ref = args[0]
    cpu = json.loads((ANA / "calltype_cpu_rescore.json").read_text())["models"]
    groups = [("scratch", SCRATCH), ("dapt", DAPT), ("dl", DL), ("control", CONTROLS)]
    tags = [t for _, m in groups for t in m if t != ref and t in cpu and (FEAT / f"ct11_{t}_cpuemb.npy").exists()]
    if "--rows-only" not in sys.argv:
        pairs = [f"{t}:{ref}" for t in tags]
        for extra in ([], ["--layer", "3"]):
            subprocess.run([sys.executable, "seed_pairs_cpu.py", *extra, *pairs], cwd=HERE, check=True)
    p = json.loads((ANA / "seed_pairs_cpu.json").read_text())["pairs"]
    q = json.loads((ANA / "seed_pairs_cpu_L3.json").read_text())["pairs"]
    out = dict(reference=ref, ref_acc=None, ref_l3=None, rows=[], controls=[], missing=[],
               note="bird bootstrap only (seed_pairs_cpu); intervals ignore seed noise")
    for g, m in groups:
        rows = []
        for t, name in m.items():
            if t not in tags:
                continue
            k = f"{t} | {ref}"
            if k not in p or k not in q:
                out["missing"].append(t)
                continue
            r, r3 = p[k], q[k]
            out["ref_acc"], out["ref_l3"] = round(r["acc_b"], 4), round(r3["acc_b"], 4)
            rows.append(dict(slug=t.replace("_", "-").replace(".", "-"), name=name, g=g, acc=round(r["acc_a"], 4),
                             d=round(r["diff"], 4), lo=round(r["lo"], 4), hi=round(r["hi"], 4), l3=round(r3["diff"], 4),
                             res="above" if r["lo"] > 0 else "below" if r["hi"] < 0 else "nd"))
        rows.sort(key=lambda r: -r["d"])
        out["controls" if g == "control" else "rows"] += rows
    dst = ANA / f"forest_vs_{ref}.json"
    dst.write_text(json.dumps(out, indent=1))
    n = len(out["rows"])
    print(f"{ref} ({out['ref_acc']}): {n} models, {sum(r['res'] == 'above' for r in out['rows'])} above, "
          f"{sum(r['res'] == 'below' for r in out['rows'])} below; controls {[(c['name'], c['d']) for c in out['controls']]}; "
          f"missing pairs {out['missing']} -> {dst}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Row files for the Call Type Breakdown dashboard, from the analysis JSONs.

  analysis/calltype_per_class_cpu.json          (zfeval/experiments/calltype_per_class.py --cpu)
  analysis/calltype_split_sensitivity_cpu.json  (zfeval/experiments/calltype_split_sensitivity.py --cpu)
  analysis/model_param_counts.json              (zfeval/experiments/model_param_counts.py)
  analysis/calltype_cpu_rescore.json            (the board's CPU call-type scores)
-> docs/calltype_dashboard/{call_types,recall,vs_run16,confusion,splits,model_specs,head_to_head}.json

Since 2026-10-19 every number is from CPU embeddings (finding 076). The Oct 9 version read the MPS files
calltype_per_class.json / calltype_split_sensitivity.json, whose short clips the Mac GPU mis-encoded.
Parameter counts are read from the weights, never typed; the recipe text below cites where each fact comes from.
"""
import json
from pathlib import Path

ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
OUT = Path(__file__).resolve().parents[1] / "docs/calltype_dashboard"
NAMES = dict(Ag="Wsst (aggressive)", Be="Begging", DC="Distance call", Di="Distress", LT="Long tonal",
             Ne="Nest", So="Song", Te="Tet", Th="Thuk", Tu="Tuck", Wh="Whine")
GROUP = dict(ours="Ours, from scratch", dapt="AVES continued on zebra finch", aves="AVES, downloaded",
             other="Off the shelf, no animal audio")

ZF, FSD = "zebra finch colony recordings", "FSD50K general audio"
SPEC = "k-means k=200 on a 4000-d spectrogram (iteration 1)"
# tag -> (pretraining audio, corpus hours, training targets). Sources: knowledge/findings 051-061 and the
# slurm/train_run*.sh scripts for ours; the 2022 checkpoint cfg + AVES paper + earthspecies/aves README for AVES
# (docs/aves_method_and_transfer_literature.md); HuBERT paper + fairseq hubert_base_librispeech.yaml; Whisper paper.
RECIPE = {
    "run11": (ZF, "116", "k-means k=100 on a 4000-d spectrogram (iteration 1)"),
    "run15_combined": (f"{ZF} + {FSD}", "224", SPEC),
    "run16_compute4x": (f"{ZF} + {FSD}", "224", SPEC),
    "run16_seed2": (f"{ZF} + {FSD}", "224", SPEC),
    "run16_seed3": (f"{ZF} + {FSD}", "224", SPEC),
    "run17_accum2": (f"{ZF} + {FSD}", "224", SPEC),
    "run18_iter2": (f"{ZF} + {FSD}", "224", "k-means k=200 on run16 layer 6 (iteration 2)"),
    "run19_avesteacher": (f"{ZF} + {FSD}", "224", "k-means k=200 on AVES bio layer 6"),
    "run19_seed2": (f"{ZF} + {FSD}", "224", "k-means k=200 on AVES bio layer 6"),
    "run20_fsdonly": (FSD, "108", "the combined corpus's spectrogram codebook, k=200"),
    "run21_zfonly": (ZF, "116", "the combined corpus's spectrogram codebook, k=200"),
    "daptreplay3_5e5_step15000": (f"AVES bio's 360 h, then {ZF} + 41.8% FSD50K replay", "360 + ZF stream",
                                  "run11's spectrogram codebook, k=100"),
    "aves-base-bio": ("general audio + animal sounds (FSD50K, AudioSet, VGGSound)", "360",
                      "iteration 2: k-means k=200 on a HuBERT's layer 6 (iteration 1: MFCC, k=200)"),
    "aves-base-core": ("general audio (FSD50K + AudioSet)", "153", "as AVES bio"),
    "aves-base-all": ("all AVES sources", "5,054", "as AVES bio"),
    "birdaves-biox-base": ("AVES bio + Xeno-canto birds", "2,570", "not published"),
    "birdaves-biox-large": ("AVES bio + Xeno-canto birds", "2,570", "not published"),
    "birdaves-bioxn-large": ("AVES bio + Xeno-canto + iNaturalist", "3,076", "not published"),
    "hf-hubert-base-ls960": ("LibriSpeech read English speech", "960",
                             "iteration 2: k-means k=500 on layer 6 (iteration 1: MFCC, k=100)"),
    "hf-whisper-base-enc": ("multilingual web speech", "680,000", "supervised: transcripts (no k-means)"),
    "hf-whisper-small-enc": ("multilingual web speech", "680,000", "supervised: transcripts (no k-means)"),
}
INPUT = {"hf-whisper-base-enc": "80-bin log-mel spectrogram", "hf-whisper-small-enc": "80-bin log-mel spectrogram"}
WAVE = "raw waveform, 7-layer CNN"

pc = json.load(open(ANA / "calltype_per_class_cpu.json"))
ss = json.load(open(ANA / "calltype_split_sensitivity_cpu.json"))
pcount = json.load(open(ANA / "model_param_counts.json"))
cpu = json.load(open(ANA / "calltype_cpu_rescore.json"))["models"]
same = set(pcount["same_tensors_as_run16"])
PC = pcount["models"]
cl = pc["classes"]
order = list(pc["models"])
tag2lab = {v["tag"]: lab for lab, v in pc["models"].items()}
assert len(tag2lab) == len(order)


def mparams(tag):
    return round(PC[tag]["params"] / 1e6, 1)


def specs_row(lab, v):
    t = v["tag"]; p = PC[t]; data, hours, targets = RECIPE[t]
    return dict(model=lab, group=GROUP[v["group"]], model_order=order.index(lab), params_m=mparams(t),
                params=p["params"], layers=p["layers"], width=p["width"], input=INPUT.get(t, WAVE),
                same_network_as_run16=(t in same), data=data, hours=hours, targets=targets,
                calltype=round(v["acc11"], 4), best_layer=v["layer"])


# Four models side by side: the speech original, the AVES everyone cites, ours, and the large BirdAVES the
# board ranks first among AVES. Rows are attributes so the table reads across.
H2H = [("hubert", "hf-hubert-base-ls960"), ("aves", "aves-base-bio"), ("ours", "run16_compute4x"),
       ("large", "birdaves-bioxn-large")]
COUNT = {"hubert": "hubert-base-torchaudio"}   # torchaudio weights: same tensors as ours (HF adds one 768-d mask vector)


def h2h_rows():
    def per(f):
        return {k: f(k, t) for k, t in H2H}
    rows = [
        ("Parameters (counted from the weights)", per(lambda k, t: f"{PC[COUNT.get(k, t)]['params']:,}")),
        ("Same tensors as run16", per(lambda k, t: "yes" if COUNT.get(k, t) in same else "no: HuBERT-large network")),
        ("Transformer layers x width", per(lambda k, t: f"{PC[COUNT.get(k, t)]['layers']} x {PC[COUNT.get(k, t)]['width']}")),
        ("Attention heads, FFN width", per(lambda k, t: f"{PC[t]['heads']}, {PC[t]['ffn']}")),
        ("Input", per(lambda k, t: "raw 16 kHz waveform, 7-layer CNN (512 channels), 20 ms frames")),
        ("Objective", per(lambda k, t: "HuBERT masked prediction of k-means labels")),
        ("Pretraining audio", per(lambda k, t: RECIPE[t][0])),
        ("Hours of audio in the corpus", {**per(lambda k, t: RECIPE[t][1]),
                                          "ours": "224 (116 zebra finch + 108 FSD50K)"}),
        ("Training targets", dict(hubert=RECIPE["hf-hubert-base-ls960"][2], aves=RECIPE["aves-base-bio"][2],
                                  ours="k-means k=200 on a 4000-d spectrogram (iteration 1 only)",
                                  large="not published")),
        ("Updates", dict(hubert="400,000 (iteration 2; iteration 1: 250,000)", aves="100,000",
                         ours=f"{PC['run16_compute4x']['global_step']:,} (from the checkpoint)",
                         large="not published")),
        ("GPUs", dict(hubert="32", aves="1 (8x gradient accumulation)", ours="4", large="not published")),
        ("Audio per update", dict(hubert="2,800 s", aves="700 s", ours="322 s", large="not published")),
        ("Total audio seen in training", dict(hubert="311,111 h (iteration 2)", aves="19,444 h", ours="8,392 h",
                                              large="not published")),
        ("Call type, 11 classes (CPU, best layer)", per(lambda k, t: f"{cpu[t]['acc11_cpu']:.4f} (layer {cpu[t]['best_layer_cpu']})")),
        ("Where these facts come from", dict(
            hubert="HuBERT paper (Hsu et al. 2021); fairseq hubert_base_librispeech.yaml",
            aves="the 2022 checkpoint's own config, matching the AVES paper (ICASSP 2023)",
            ours="slurm/train_run16_compute4x.sh; finding 052",
            large="earthspecies/aves README and BirdAVES blog (Jun 2024); weights for size")),
    ]
    return [dict(attribute=a, order=i, **v) for i, (a, v) in enumerate(rows)]


files = {
    "call_types": [dict(call_type=c, name=NAMES[c], n_clips=pc["class_counts"][c], n_birds=pc["class_birds"][c],
                        chick_share=round(pc["class_chick_fraction"][c], 4)) for c in cl],
    "recall": [dict(model=m, group=GROUP[v["group"]], model_order=i, layer=v["layer"], call_type=c,
                    recall=round(v["recall"][c], 4), correct=v["confusion"][j][j], n=sum(v["confusion"][j]),
                    overall=round(v["acc11"], 4), params_m=mparams(v["tag"]))
               for i, (m, v) in enumerate(pc["models"].items()) for j, c in enumerate(cl)],
    "vs_run16": [dict(model=m, call_type=c, diff=round(x["diff"], 4), lo=round(x["lo"], 4), hi=round(x["hi"], 4),
                      resolved=x["resolved"])
                 for m, v in pc["models"].items() if "vs_ref" in v for c, x in v["vs_ref"].items()],
    "confusion": [dict(model=m, true_type=c, pred_type=cl[k], count=v["confusion"][j][k])
                  for m, v in pc["models"].items() for j, c in enumerate(cl) for k in range(len(cl))],
    "splits": [dict(model=tag2lab[t], group=GROUP[pc["models"][tag2lab[t]]["group"]], model_order=order.index(tag2lab[t]),
                    split=s, acc=round(a, 4), board_split=(s == 0), diff_vs_run16=round(v["diff_vs_run16"][i], 4))
               for t, v in ss["models"].items() if t in tag2lab for i, (s, a) in enumerate(zip(ss["seeds"], v["acc"]))],
    "model_specs": [specs_row(m, v) for m, v in pc["models"].items()],
    "head_to_head": h2h_rows(),
}
# every model on the page must have its split row set, and the board's split must be seed 0
assert {r["model"] for r in files["splits"]} == set(order), set(order) - {r["model"] for r in files["splits"]}
assert ss["seeds"][0] == 0
OUT.mkdir(parents=True, exist_ok=True)
for k, rows in files.items():
    (OUT / f"{k}.json").write_text(json.dumps(rows))
    print(k, len(rows), "rows", (OUT / f"{k}.json").stat().st_size, "bytes")

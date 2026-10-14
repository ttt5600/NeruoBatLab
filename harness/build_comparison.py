#!/usr/bin/env python
"""Rows for the encoder-comparison dashboard, every number read from analysis/ JSONs.

    python3 harness/build_comparison.py      -> docs/comparison/{models,beans,variables}.json

Descriptive columns (input, objective, pretraining data, hours) are facts about each checkpoint: ours from the
training scripts and TSVs, AVES from its checkpoint cfg / release notes, Hugging Face models from their model
cards. Scores come only from the JSONs named in each column's comment; nothing numeric is typed here except
those published corpus sizes.
"""
from __future__ import annotations
import json, statistics as st
from pathlib import Path

A = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
OUT = Path(__file__).resolve().parents[1] / "docs/comparison"
REF = "aves-base-bio"
CT_ALIAS = {"run15_combined": "run15"}

ZF, ZFFSD, MIX = "zebra finch colony recordings", "zebra finch + FSD50K general audio", "AVES-bio's 360 h, then ZF + FSD50K-eval stream"
# tag: (name, family, input, objective, pretraining data, hours, saw the test birds)
META = {
    "run11": ("run11", "ours", "waveform", "HuBERT masked prediction", ZF, 116, True),
    "run15_combined": ("run15", "ours", "waveform", "HuBERT masked prediction", ZFFSD, 224.42, True),
    "run16_compute4x": ("run16 (seed 1337)", "ours", "waveform", "HuBERT masked prediction", ZFFSD, 224.42, True),
    "run16_seed2": ("run16 seed 2", "ours", "waveform", "HuBERT masked prediction", ZFFSD, 224.42, True),
    "run16_seed3": ("run16 seed 3", "ours", "waveform", "HuBERT masked prediction", ZFFSD, 224.42, True),
    "run17_accum2": ("run17 (2x accumulation)", "ours", "waveform", "HuBERT masked prediction", ZFFSD, 224.42, True),
    "run18_iter2": ("run18 (iteration 2)", "ours", "waveform", "HuBERT masked prediction", ZFFSD, 224.42, True),
    "run19_avesteacher": ("run19 (AVES labels)", "ours", "waveform", "HuBERT masked prediction", ZFFSD, 224.42, True),
    "run20_fsdonly": ("run20 (FSD50K only)", "ours", "waveform", "HuBERT masked prediction", "FSD50K general audio", 108, False),
    "run21_zfonly": ("run21 (ZF only)", "ours", "waveform", "HuBERT masked prediction", ZF, 116, True),
    "run29_avesrecipe": ("E29 AVES recipe (seed 1337)", "ours", "waveform", "HuBERT masked prediction", ZFFSD, 224.42, True),
    "run29_avesrecipe_s2": ("E29 AVES recipe seed 2", "ours", "waveform", "HuBERT masked prediction", ZFFSD, 224.42, True),
    "daptreplay_5e5_step15000": ("DAPT replay x1", "DAPT", "waveform", "HuBERT masked prediction", MIX, None, True),
    "daptfreeze_5e5_step15000": ("DAPT frozen CNN", "DAPT", "waveform", "HuBERT masked prediction", MIX, None, True),
    "daptlr_1e5_step15000": ("DAPT lr 1e-5", "DAPT", "waveform", "HuBERT masked prediction", MIX, None, True),
    "daptshort_5e5_step2500": ("DAPT 2.5k steps", "DAPT", "waveform", "HuBERT masked prediction", MIX, None, True),
    "daptreplay3_5e5_step15000": ("DAPT replay x3", "DAPT", "waveform", "HuBERT masked prediction", MIX, None, True),
    "daptv3aves_5e5_step15000": ("E17 DAPT, AVES labels, lr 5e-5", "DAPT", "waveform", "HuBERT masked prediction", MIX, None, True),
    "daptv3aves_2e4_step15000": ("E18 DAPT, AVES labels, lr 2e-4", "DAPT", "waveform", "HuBERT masked prediction", MIX, None, True),
    "daptv3aves_2e4s2_step15000": ("E20 = E18 seed 2", "DAPT", "waveform", "HuBERT masked prediction", MIX, None, True),
    "daptv3aves_2e4m8_step15000": ("E28a = E18 + AVES masking", "DAPT", "waveform", "HuBERT masked prediction", MIX, None, True),
    "daptv3aves_2e4s2m8_step15000": ("E28b = E20 + AVES masking", "DAPT", "waveform", "HuBERT masked prediction", MIX, None, True),
    "daptv4two_2e4_step15000": ("E19 DAPT, two-teacher labels", "DAPT", "waveform", "HuBERT masked prediction", MIX, None, True),
    "daptv4two_2e4s2_step15000": ("E23 = E19 seed 2", "DAPT", "waveform", "HuBERT masked prediction", MIX, None, True),
    "daptv4aveso_2e4_step15000": ("E21 DAPT, AVES-only control", "DAPT", "waveform", "HuBERT masked prediction", MIX, None, True),
    "daptv4aveso_2e4s2_step15000": ("E24 = E21 seed 2", "DAPT", "waveform", "HuBERT masked prediction", MIX, None, True),
    "daptv4two45k_2e4_step45000": ("E22 = E19 x3 longer", "DAPT", "waveform", "HuBERT masked prediction", MIX, None, True),
    "daptv4bird_2e4_step15000": ("E14 DAPT from BirdAVES", "DAPT", "waveform", "HuBERT masked prediction", "BirdAVES base's data, then ZF + FSD50K-eval stream", None, True),
    "aves-base-core": ("AVES core", "AVES", "waveform", "HuBERT masked prediction", "general audio (FSD50K + AudioSet)", 153, False),
    "aves-base-bio": ("AVES bio", "AVES", "waveform", "HuBERT masked prediction", "general + animal audio", 360, False),
    "aves-base-nonbio": ("AVES nonbio", "AVES", "waveform", "HuBERT masked prediction", "non-animal audio, size-matched to bio", 360, False),
    "aves-base-all": ("AVES all", "AVES", "waveform", "HuBERT masked prediction", "all AVES sources", 5054, False),
    "birdaves-biox-base": ("BirdAVES biox base", "AVES", "waveform", "HuBERT masked prediction", "AVES bio + Xeno-canto birds", 2570, False),
    "birdaves-biox-large": ("BirdAVES biox large", "AVES", "waveform", "HuBERT masked prediction", "AVES bio + Xeno-canto birds", 2570, False),
    "birdaves-bioxn-large": ("BirdAVES bioxn large", "AVES", "waveform", "HuBERT masked prediction", "+ Xeno-canto + iNaturalist", 3076, False),
    "hf-hubert-base-ls960": ("HuBERT base (speech)", "other", "waveform", "HuBERT masked prediction", "LibriSpeech speech", 960, False),
    "hf-wav2vec2-base": ("wav2vec 2.0 base (speech)", "other", "waveform", "contrastive", "LibriSpeech speech", 960, False),
    "hf-wavlm-base-plus": ("WavLM base+ (speech)", "other", "waveform", "masked prediction + denoising", "speech (Libri-Light, GigaSpeech, VoxPopuli)", 94000, False),
    "hf-data2vec-audio-base": ("data2vec audio base (speech)", "other", "waveform", "regression on teacher layers", "LibriSpeech speech", 960, False),
    "hf-ast-audioset": ("AST (AudioSet)", "other", "log-mel patches", "supervised sound tagging", "AudioSet (ImageNet-initialised)", None, False),
    "hf-whisper-tiny-enc": ("Whisper tiny encoder", "other", "log-mel", "supervised speech recognition", "multilingual speech", 680000, False),
    "hf-whisper-base-enc": ("Whisper base encoder", "other", "log-mel", "supervised speech recognition", "multilingual speech", 680000, False),
    "hf-whisper-small-enc": ("Whisper small encoder", "other", "log-mel", "supervised speech recognition", "multilingual speech", 680000, False),
}


def load(name):
    p = A / name
    return json.load(open(p)) if p.exists() else {}


def calltype(tag, av):
    j = load(f"{CT_ALIAS.get(tag, tag)}_calltype.json")
    if j:
        pl = j["per_layer_acc11"]; return pl, j.get("params")
    m = av.get(tag)
    return (m["per_layer_acc_11"], m.get("params")) if m else (None, None)


def vs_ref(tag, pairs):
    """Observed difference vs aves-base-bio: calltype_pairs.json first, then <tag>_bootstrap.json (observed delta)."""
    if tag == REF:
        return None
    k = f"{tag} | {REF}"
    if k in pairs:
        d = pairs[k]["a_minus_b"]; return d["diff"], d["lo"], d["hi"]
    b = load(f"{tag}_bootstrap.json").get("vs", {}).get(tag, {}).get(REF)
    return (b["delta"], b["lo"], b["hi"]) if b else None


def main():
    av = load("aves_variants_calltype.json").get("models", {})
    pairs = load("calltype_pairs.json")
    det = load("detection_variants.json")
    pc, bp = det.get("precommit", {}), det.get("zf_to_bp", {})
    lf = load("labelfree_stability.json").get("models", {})
    rows = []
    for tag, (name, fam, inp, obj, data, hours, saw) in META.items():
        pl, params = calltype(tag, av)
        if pl is None:
            raise SystemExit(f"{tag}: no call-type numbers")
        best = max(range(len(pl)), key=lambda i: pl[i])
        if params is None:
            params = load(f"{tag}_calltype.json").get("params") or (av.get("run11", {}).get("params") if fam in ("ours", "DAPT") else None)
        v = vs_ref(tag, pairs)
        j = pc.get(tag, {}).get("joint")
        arms = bp.get(tag, {})
        rows.append(dict(
            tag=tag, name=name, family=fam, input=inp, objective=obj, data=data, hours=hours, saw_test_birds=saw,
            params_m=round(params / 1e6, 2) if params else None, layers=len(pl),
            calltype=round(pl[best], 4), best_layer=best, calltype_l3=round(pl[3], 4),
            diff_vs_aves=None if v is None else round(v[0], 4), lo=None if v is None else round(v[1], 4),
            hi=None if v is None else round(v[2], 4),
            resolved=None if v is None else bool(v[1] > 0 or v[2] < 0),
            zf_auc=round(j["zf"]["auc"], 4) if j else None,
            bp_ap_median=round(st.median(a["ap"] for a in arms.values()), 4) if arms else None,
            labelfree_k30=round(lf[tag]["k30"]["mean"], 4) if tag in lf else None))
    tasks = ["bats", "watkins", "dogs", "humbugdb", "esc50", "hiceas", "enabirds", "dcase"]
    beans = []
    for t in tasks:
        m = load(f"beans_{t}.json").get("models", {})
        for tag, rec in m.items():
            if tag in META and "test" in rec:
                beans.append(dict(task=t, tag=tag, name=META[tag][0], family=META[tag][1], test=round(rec["test"], 4)))
    variables = [  # what differs between our recipe and AVES-bio's, and where each stands
        dict(variable="Model size", ours="94.4M, 12 x 768", aves="94.4M, 12 x 768 (large: 315M, 24 x 1024)", status="closed",
             tests="AVES variants; E27 proposed for large", verdict="Identical for base models. Large BirdAVES is the best AVES on call type and BEANS."),
        dict(variable="Input front end", ours="raw waveform, 7-layer CNN", aves="raw waveform, same CNN", status="open",
             tests="E25 (Whisper, AST use log-mel)", verdict="Whisper encoders (log-mel) are the highest call-type scores so far; not yet tested inside our training."),
        dict(variable="Training objective", ours="HuBERT masked prediction", aves="HuBERT masked prediction", status="open",
             tests="E25", verdict="Supervised speech recognition (Whisper) scores higher than any masked-prediction model; contrastive and regression score lower."),
        dict(variable="Masking", ours="0.4 / 3 frames (34% of frames)", aves="0.8 / 10 frames (54%)", status="tested in DAPT",
             tests="E28 (DAPT, 2 seeds); E29 (from scratch, running)", verdict="In DAPT: no call-type change on either seed; detection gain on one seed only."),
        dict(variable="Learning rate / warmup", ours="1e-4, 3% warmup", aves="2e-4, 32% warmup", status="running",
             tests="E29", verdict="Bundled into E29 with masking."),
        dict(variable="Compute (audio processed)", ours="8,392 h (run16); 16,784 h (run17)", aves="19,444 h", status="closed",
             tests="run16, run17 (052, 054, 070)", verdict="run17 reached 86% of AVES with lower loss but no better call type."),
        dict(variable="Training labels", ours="iteration 1, spectrogram k-means", aves="iteration 2, HuBERT layer 6 k-means", status="closed",
             tests="run18, run19, E17-E24 (056, 059, 064, 065, 067)", verdict="Iteration 2 and AVES's own labels: no call-type gain; two-teacher labels' detection gain did not replicate."),
        dict(variable="Data domain", ours="half zebra finch, half general audio", aves="general + animal", status="closed",
             tests="run20, run21, AVES nonbio, speech models", verdict="AVES nonbio ties AVES bio; a speech-only HuBERT ties run16: domain barely matters here."),
        dict(variable="Data amount", ours="224 h", aves="360 h (all: 5,054 h)", status="partly",
             tests="AVES core/bio/all; E15 proposed", verdict="AVES 153 h vs 5,054 h tie on call type; larger corpora help BEANS more."),
        dict(variable="Continued training from AVES (DAPT)", ours="15k-45k updates on ZF + replay", aves="none", status="closed",
             tests="E17-E24, E28, E22 (064-068)", verdict="ZF detection AUC about +0.003 over AVES at every arm and seed; call type unchanged; BirdAVES start got worse."),
        dict(variable="Seeds", ours="2-3 per recipe", aves="one release each", status="measured",
             tests="057, 067, 070", verdict="Same-recipe seeds move call type by up to about 0.019, as much as most recipe changes."),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "variables.json").write_text(json.dumps(variables, indent=1))
    (OUT / "models.json").write_text(json.dumps(rows, indent=1))
    (OUT / "beans.json").write_text(json.dumps(beans, indent=1))
    print(f"models {len(rows)}, beans rows {len(beans)} -> {OUT}")


if __name__ == "__main__":
    main()

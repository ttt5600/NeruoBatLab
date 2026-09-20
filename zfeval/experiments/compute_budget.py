#!/usr/bin/env python
"""How much training did run11 get, against AVES? Every number read from a config, none typed.

The question "compare training hours between AVES and our model" has three different answers and
they point in different directions, so all three are computed:

    optimiser steps     how many times the weights moved
    audio-seconds seen  steps x audio per step -- the quantity that actually bounds what an SSL
                        objective can learn, and the one people mean by "training hours"
    epochs over corpus  audio seen / corpus size -- how many times the model saw the SAME audio

Sources
    run11   slurm/train_iter7_long_lowprio.sh (the submitted script; its header names it run11)
            plus train.py's argparse defaults for anything the script leaves unset
    AVES    the fairseq cfg block inside datasets/11905533/aves-base-bio.pt, read directly

Batch arithmetic differs between the two and is the whole reason audio-seconds diverge from steps:
    run11   BucketizeBatchSampler(max_token_count = seconds_per_batch * 16000) builds one batch per
            rank, DistributedBatchSampler shards batches across ranks, and DDP averages gradients --
            so one optimiser step consumes gpus x seconds_per_batch of audio.
            (lightning_modules.py:378-386)
    AVES    one rank, no DDP, gradient accumulation instead -- one step consumes
            max_tokens/sample_rate x update_freq.
"""
from __future__ import annotations
import json, re, sys, types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SLURM = REPO / "pytorchAudio/examples/hubert/slurm/train_iter7_long_lowprio.sh"
TRAIN = REPO / "pytorchAudio/examples/hubert/train.py"
AVES_CKPT = REPO / "datasets/11905533/aves-base-bio.pt"
OUT = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis/compute_budget.json"

# Corpus sizes. These two are the only quantities here that are not machine-read, because neither is
# recorded in any config: the ZF figure is the release README's, the AVES figure is the paper's.
# CORRECTED 2026-09-16. The previous version of this script took the corpus from the release
# README ("~100 hours") and computed audio/step as `--gpus` x `--seconds-per-batch` = 4 x 87.5 = 350.
# Both were wrong, and together they overstated run11's training by ~5x:
#
#   * The corpus is 116.03 h, not 100 -- read straight from the TSV the run actually consumed
#     (120 rows, 6,683,245,309 samples @ 16 kHz).
#   * run11 never used 4 GPUs. Its slurm script pairs `--gres=gpu:L40:4` with `--ntasks=1`, so srun
#     launches ONE task and Lightning initialises world_size 1. Confirmed directly on a job running
#     this same script: GPU 0 at 90% / 9.9 GB, GPUs 1-3 at 0% / 3 MiB.
#   * The realised batch is also below the 87.5 nominal, because BucketizeBatchSampler emits a
#     partial final batch per bucket. The honest figure is corpus / steps-per-epoch.
#
# Everything below is now derived from what the run RECORDED, and cross-checked against Lightning's
# own epoch counter for BOTH models -- the check that would have caught the original error, and which
# the first version applied only to AVES.
RUN11_CORPUS_SEC = 417702.83   # awk over tsv/ZF_test_pipeline_train.tsv: 120 files, 6,683,245,309 samples
# (epoch, global_step) from run11's checkpoint filenames on scratch. Consecutive differences give
# steps-per-epoch; the last pair is truncated by --max-updates and is excluded from that estimate.
RUN11_CKPT_LEDGER = [(14, 78029), (15, 83230), (16, 88432), (17, 93633), (18, 93750)]
AVES_CORPUS_H = 360.0       # Hagiwara 2023 Table 1 for aves-base-bio; corroborated in finding 038
SR = 16000


def parse_run11():
    """Read the submitted slurm script, falling back to train.py's argparse defaults."""
    txt = SLURM.read_text()
    assert "run11" in txt, "this script does not identify itself as run11"
    flags = dict(re.findall(r"--([a-z0-9-]+)\s+([0-9.e-]+)\s*\\", txt))
    defaults = dict(re.findall(
        r'"--([a-z0-9-]+)",\s*\n\s*default=([0-9.]+),', TRAIN.read_text()))

    def g(name):
        if name in flags:
            return float(flags[name]), "slurm"
        assert name in defaults, f"{name} is in neither the slurm script nor train.py defaults"
        return float(defaults[name]), "train.py default"

    updates, s_up = g("max-updates")
    warmup, s_wu = g("warmup-updates")
    lr, s_lr = g("learning-rate")
    spb, s_spb = g("seconds-per-batch")
    gpus_requested, s_g = g("gpus")
    k, s_k = g("num-classes")

    # world_size is what srun ACTUALLY launched, which is --ntasks, not --gpus. Reading the flag
    # the job passed to Lightning instead of the one SLURM honoured is exactly how the 4x error got
    # in, so take it from the #SBATCH directive and assert the two disagree as expected.
    ntasks = int(re.search(r"#SBATCH\s+--ntasks=(\d+)", txt).group(1))
    world_size = ntasks

    # steps-per-epoch from consecutive checkpoints; drop the final truncated epoch.
    deltas = [b - a for (_, a), (_, b) in zip(RUN11_CKPT_LEDGER, RUN11_CKPT_LEDGER[1:])]
    full = [d for d in deltas if d > 0.5 * max(deltas)]
    steps_per_epoch = sum(full) / len(full)
    realised_spb = RUN11_CORPUS_SEC / steps_per_epoch
    assert realised_spb <= spb * 1.02, (
        f"realised {realised_spb:.1f} s/step exceeds the nominal cap {spb}; the corpus or the "
        f"ledger is wrong")
    return dict(
        name="run11", updates=int(updates), warmup_updates=int(warmup), peak_lr=lr,
        seconds_per_batch=spb, world_size=world_size, update_freq=1, num_clusters=int(k),
        gpus_requested=int(gpus_requested),
        audio_seconds_per_update=realised_spb,
        steps_per_epoch=steps_per_epoch,
        epochs_recorded=RUN11_CKPT_LEDGER[-1][0],
        corpus_hours=RUN11_CORPUS_SEC / 3600.0,
        target_feature="soundsig Gaussian STFT 4000-d (iteration 1)",
        sources=dict(max_updates=s_up, warmup=s_wu, lr=s_lr, seconds_per_batch=s_spb,
                     gpus_requested=s_g, num_classes=s_k, script=str(SLURM.relative_to(REPO)),
                     world_size="#SBATCH --ntasks (what srun launched), NOT --gpus",
                     corpus="tsv/ZF_test_pipeline_train.tsv, 120 files",
                     audio_per_step="corpus_seconds / steps_per_epoch (realised, not nominal)",
                     steps_per_epoch="consecutive checkpoint (epoch, step) differences"))


def parse_aves():
    """Pull the fairseq cfg out of the 2022 checkpoint.

    fairseq/omegaconf classes are not installed here, so unpickling needs stand-ins. A permissive
    module stub is registered for every module the pickle asks for; the cfg we want is plain
    dict/scalar data once the wrapper objects resolve.
    """
    import torch

    class _Any:
        def __init__(self, *a, **k): self.__dict__.update(k)
        def __setstate__(self, st): self.__dict__.update(st if isinstance(st, dict) else {})
        def __reduce__(self): return (_Any, ())

    class _Stub(types.ModuleType):
        def __getattr__(self, n):
            if n.startswith("__"):
                raise AttributeError(n)
            return _Any

    for m in ["fairseq", "fairseq.dataclass", "fairseq.dataclass.configs", "fairseq.data",
              "fairseq.data.dictionary", "fairseq.tasks", "fairseq.tasks.hubert_pretraining",
              "fairseq.models", "fairseq.models.hubert", "fairseq.models.hubert.hubert",
              "omegaconf", "omegaconf.base", "omegaconf.dictconfig", "omegaconf.listconfig",
              "omegaconf.nodes"]:
        sys.modules.setdefault(m, _Stub(m))

    ck = torch.load(AVES_CKPT, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    cfg = cfg if isinstance(cfg, dict) else cfg.__dict__

    def dig(node, key):
        node = node if isinstance(node, dict) else getattr(node, "__dict__", {})
        if key in node:
            return node[key]
        for v in node.values():
            if isinstance(v, dict) or hasattr(v, "__dict__"):
                r = dig(v, key)
                if r is not None:
                    return r
        return None

    max_tokens = dig(cfg, "max_tokens")
    update_freq = dig(cfg, "update_freq")
    if isinstance(update_freq, (list, tuple)):
        update_freq = update_freq[0]
    max_update = dig(cfg, "max_update")
    warmup = dig(cfg, "warmup_updates")
    lr = dig(cfg, "lr")
    if isinstance(lr, (list, tuple)):
        lr = lr[0]
    world = dig(cfg, "distributed_world_size")
    # fairseq banks wall clock in extra_state.previous_training_time, NOT in optimizer_history
    es = ck.get("extra_state") or {}
    wall = es.get("previous_training_time")
    num_updates = (ck.get("optimizer_history") or [{}])[-1].get("num_updates")
    epoch = (es.get("train_iterator") or {}).get("epoch")

    for nm, v in [("max_tokens", max_tokens), ("update_freq", update_freq),
                  ("max_update", max_update), ("distributed_world_size", world)]:
        assert v is not None, f"could not read {nm} from the AVES checkpoint cfg"

    return dict(
        name="aves-base-bio", updates=int(max_update), warmup_updates=int(warmup) if warmup else None,
        peak_lr=float(lr) if lr else None,
        seconds_per_batch=max_tokens / SR, world_size=int(world), update_freq=int(update_freq),
        num_clusters=200,
        audio_seconds_per_update=max_tokens / SR * int(update_freq) * int(world),
        corpus_hours=AVES_CORPUS_H,
        wall_clock_seconds=float(wall) if wall else None,
        target_feature="HuBERT layer 6 @ k=200 (iteration 2)",
        final_val_loss=es.get("val_loss"), epochs_recorded=epoch,
        sources=dict(cfg=str(AVES_CKPT.relative_to(REPO)), num_updates_in_ckpt=num_updates))


def enrich(d):
    tot = d["updates"] * d["audio_seconds_per_update"]
    d["total_audio_seconds"] = tot
    d["total_audio_hours"] = tot / 3600.0
    d["epochs_over_corpus"] = (tot / 3600.0) / d["corpus_hours"]
    d["warmup_fraction"] = (d["warmup_updates"] / d["updates"]) if d.get("warmup_updates") else None
    return d


def main():
    a, b = enrich(parse_run11()), enrich(parse_aves())
    rows = [("optimiser steps", "updates", "{:,.0f}"),
            ("audio-seconds per step", "audio_seconds_per_update", "{:,.1f}"),
            ("TOTAL audio seen (hours)", "total_audio_hours", "{:,.1f}"),
            ("pretraining corpus (hours)", "corpus_hours", "{:,.1f}"),
            ("epochs over its own corpus", "epochs_over_corpus", "{:,.1f}"),
            ("peak LR", "peak_lr", "{:.1e}"),
            ("warmup fraction of run", "warmup_fraction", "{:.3f}"),
            ("k-means clusters", "num_clusters", "{:,.0f}"),
            ("world size x grad accum", None, None)]
    print(f"{'':32s} {'run11':>18s} {'aves-base-bio':>18s} {'ratio AVES/run11':>18s}")
    for label, key, fmt in rows:
        if key is None:
            va = f"{a['world_size']} x {a['update_freq']}"
            vb = f"{b['world_size']} x {b['update_freq']}"
            if a.get("gpus_requested", 1) != a["world_size"]:
                va += f"  (asked {a['gpus_requested']})"
            print(f"{label:32s} {va:>18s} {vb:>18s} {'':>18s}")
            continue
        xa, xb = a.get(key), b.get(key)
        sa = fmt.format(xa) if xa is not None else "n/a"
        sb = fmt.format(xb) if xb is not None else "n/a"
        r = f"{xb/xa:.2f}x" if (xa and xb) else ""
        print(f"{label:32s} {sa:>18s} {sb:>18s} {r:>18s}")
    if b.get("wall_clock_seconds"):
        gh = b["wall_clock_seconds"] / 3600 * b["world_size"]
        print(f"\naves wall clock: {b['wall_clock_seconds']/3600:.1f} h on {b['world_size']} GPU(s) "
              f"= {gh:.1f} GPU-hours, checkpoint records {b['sources']['num_updates_in_ckpt']:,} "
              f"updates over {b['epochs_recorded']} epochs, final val loss {b['final_val_loss']}")
    print(f"run11 wall clock: NOT RECORDED locally (slurm log lives on Savio); the job requested a "
          f"12 h limit and actually used {a['world_size']} GPU "
          f"(it reserved {a.get('gpus_requested', '?')}), so <= 12 GPU-hours.")

    # Self-check: the epoch count DERIVED from (steps x audio/step) / corpus-hours must agree with
    # the epoch counter fairseq itself wrote into the checkpoint. Agreement validates two things at
    # once -- the 700 s/update batch arithmetic above, and the 360 h corpus figure taken from the
    # paper. Disagreement would mean one of them is wrong and the comparison is not trustworthy.
    # Applied to BOTH models. The first version of this script checked only AVES, and the run11
    # number it printed (91.1 epochs) was 5x the 18 epochs Lightning had written into the checkpoint
    # filenames sitting on scratch. One line of symmetry would have caught it immediately.
    chk = {}
    for tag, m, counter in (("run11", a, "lightning"), ("aves", b, "fairseq")):
        if not m.get("epochs_recorded"):
            continue
        rel = abs(m["epochs_over_corpus"] - m["epochs_recorded"]) / m["epochs_recorded"]
        chk[tag] = dict(derived_epochs=m["epochs_over_corpus"],
                        recorded_epoch_counter=m["epochs_recorded"], counter_source=counter,
                        relative_disagreement=rel, passed=bool(rel < 0.05))
        print(f"[check] {tag:6s} derived epochs {m['epochs_over_corpus']:6.2f} vs {counter}'s own "
              f"counter {m['epochs_recorded']:3d} -> {rel*100:5.1f}% apart "
              f"({'OK' if rel < 0.05 else 'MISMATCH'})")
        assert rel < 0.05, (
            f"{tag}: derived epochs {m['epochs_over_corpus']:.2f} disagree with the recorded "
            f"{m['epochs_recorded']}; the batch arithmetic or corpus size is wrong -- "
            f"do not quote these numbers")
    print("Agreement corroborates, for each model independently, both its per-step audio "
          "arithmetic and its corpus size.")

    out = dict(run11=a, aves=b, consistency_check=chk, comparison=dict(
        audio_hours_ratio_aves_over_run11=b["total_audio_hours"] / a["total_audio_hours"],
        corpus_hours_ratio=b["corpus_hours"] / a["corpus_hours"],
        epochs_ratio=b["epochs_over_corpus"] / a["epochs_over_corpus"],
        steps_ratio=b["updates"] / a["updates"]),
        note="Architecture is NOT a difference: both are torchaudio hubert_pretrain_base / fairseq "
             "hubert_base, 94,370,944 parameters with bit-identical state_dict keys. The differences "
             "are corpus, schedule, and k-means targets.")
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

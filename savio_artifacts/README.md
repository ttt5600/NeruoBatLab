# Savio artifacts

Small, irreplaceable outputs pulled off Savio on 2026-09-20, when a Lustre OST
(`brc-OST006e`, index 110, pool `ddn_nvme3`) went offline and made roughly a
quarter of every directory on `/global/scratch` unreadable.

Nothing here can be regenerated without re-running the job that produced it, and
all of it is small. That is the entire selection rule. Checkpoints, corpora and
feature shards are deliberately **not** here — they are large, and either
re-downloadable from a public source or reproducible from a checkpoint.

## metrics/

Per-run Lightning training curves, copied from
`/global/scratch/users/jonathanswang/temp_train_<run>/lightning_logs/version_0/metrics.csv`.

| file | steps | what it is |
|---|---|---|
| `run7/8/9/10_metrics.csv` | — | the LR investigation; run10 is where lr 1e-4 fixed the 3.46/0.19 plateau |
| `run11_metrics.csv` | 15000 | **the release model.** ZF-specific pretraining, 116 h |
| `run12_iter2_metrics.csv` | 15000 | iteration 2, layer-6 teacher |
| `run13_layer3_metrics.csv` | 15000 | iteration 2, layer-3 teacher |
| `dapt_5e5_metrics.csv` | 15000 | round-1 DAPT from AVES weights, lr 5e-5 |
| `daptv2_replay_5e5_metrics.csv` | 15000 | round-2 replay arm — the one that worked |
| `daptv2_freeze_5e5_metrics.csv` | 15000 | round-2, frozen feature extractor |
| `daptv2_lr_1e5_metrics.csv` | 15000 | round-2, lr 1e-5 |
| `daptv2_short_5e5_metrics.csv` | 2500 | round-2, early stop |

Verified on copy: every file parses, and line counts track the step budgets
(1895 for the 15k-step runs, 52 for the 2500-step arm).

## analysis/

Evaluation outputs keyed by SLURM job id, so each traces back to a specific run:
cross-dataset transfer, frame-level detection, the k sweep at layer 3, the
new-dataset evaluations, and `run11_meta.json`. All parse as valid JSON.

## What is NOT backed up, and why

| | where it lives | if lost |
|---|---|---|
| model checkpoints | Savio + `release/zf_hubert_run11/weights` | retrain (~37 min/arm on an L40) |
| ZF corpus | Savio; public Elie & Theunissen release (CC BY 4.0, figshare) | re-download |
| FSD50K | Savio; public | re-download (~108 h of audio) |
| feature shards | Savio, and already deleted once | re-extract, ~9.5 CPU-hours |

The one genuine gap: run11's **preprocessed** ZF audio exists only on Savio. The
raw corpus is public, but the resampled/chunked version would have to be rebuilt
by rerunning `preprocess.py` from `--start-from tsv`.

## Update 2026-09-23 — run15 (the de-confounding run)

run15 is run11's exact recipe on the combined ZF + FSD50K corpus (224.42 h, k=200 labels). It
scored 0.8136 on 11-class call type against run11's 0.8118 — **not distinguishable** — and lost to
every AVES checkpoint by a resolved margin. See knowledge finding 051 and
`notebooks/06_where_we_are.ipynb`.

| file | what it is |
|---|---|
| `analysis/run15_calltype.json` | run15 per-layer and best-layer call-type accuracy, 11- and 8-class |
| `analysis/run15_bootstrap.json` | paired bird-bootstrap intervals, run15 vs run11 and three AVES checkpoints |
| `analysis/run15_compute.json` | run15 realised compute (measured) and run16 (projected, labelled as such) |
| `analysis/aves_variants_calltype.json` | run11 vs all six AVES checkpoints, same probe |
| `analysis/compute_budget.json` | run11 vs AVES training budgets, derived from recorded configs |

### scripts/ — diagnostics that ran on Savio and existed nowhere in git

| script | what it checks |
|---|---|
| `verify_corpus.py` | combined-corpus labels: frame count vs audio duration (50.00 fps), all 200 clusters used, perplexity |
| `full_tsv_check.py` | every one of the 763 TSV rows resolves and its frame count matches the file |
| `compare_labels.py` | codebook balance (entropy, perplexity, dead clusters) against run11 and the replay corpus |
| `attrib2.py` | were the 50 skipped FSD50K clips silent, or cancelled by the stereo downmix? (42 silent, 0 cancelled) |
| `attrib3.py` | replicates the skip guard exactly, post-resample, to account for the remaining 8 |

### Where the pipeline code lives

Everything that trains or preprocesses is in the **nested** repo `pytorchAudio/`, which this repo
does not track — see `https://github.com/theunissenlab/SpectrogramBasedBERT`, branches
`updated-hyperparams` and `jw/zf-hubert-dapt` (kept identical). The SLURM scripts are under
`examples/hubert/slurm/`; the ones behind this update are `build_combined_corpus.sh`,
`train_run15_combined.sh`, `smoke_ddp4.sh` and `train_run16_compute4x.sh` (both since run; see the 2026-09-25 update).

### Pulled 2026-09-23

| file | what it is |
|---|---|
| `metrics/run15/run15_v0_preempted_metrics.csv` | job 39132082, preempted at step 15649 (315 rows) |
| `metrics/run15/run15_v1_resumed_metrics.csv` | job 39164203, resumed from step 10029 to 93750 (1686 rows) |
| `scripts/ost_sweep.sh`, `logs/ost_sweep.log` | first-64-KB read of every corpus file after the OST outage: 51,234 files, 0 unreadable |
| `scripts/attrib_skips.py`, `logs/attrib_skips.log` | header scan of the 50 skipped FSD50K clips: 0 shorter than 100 ms |
| `logs/attrib2.log` | 42 of the 50 silent at native rate, 0 cancelled by the stereo downmix |
| `logs/attrib3.log` | the last 8: silent only AFTER resampling to 16 kHz -- see below |
| `logs/build_combined_39114877.log` | the combined-corpus build |

**The 50 skipped clips, fully accounted for.** 42 are genuinely silent (several are digital zero,
−240 dB). The other 8 pass the −70 dB floor at 44.1 kHz and fail it after downsampling to 16 kHz:
five sit within ~3 dB of the floor, but three are loud clips whose energy lies entirely above
8 kHz — `89555.wav` drops from −13.4 dB to −79.9 dB. Rejecting them is correct for a 16 kHz model,
which cannot represent that band at all. The same fact matters beyond FSD50K: **anything
ultrasonic — bat echolocation included — is discarded by this pipeline before the model sees it.**

The two resumed-run metrics files overlap in steps 10029–15649, because the preempted attempt ran
past its last checkpoint; concatenate by step and keep the resumed rows where they collide.

## Update 2026-09-25 — run16 (4x compute) and gradient accumulation

run16 = run15 with the four requested GPUs actually used (`--ntasks-per-node=4`), nothing else
changed: 322 s of audio per update instead of 80.5, 8,392 h seen in total. It is the first change
in this project that moved call-type accuracy: 0.8379 vs run15 0.8136, +0.0243 [+0.0099, +0.0419],
and none of the six AVES checkpoints is ahead of it by more than noise (they are still ahead on the
point estimate, by 0.007–0.017). Finding 052 in `knowledge/findings/`.

| file | what it is |
|---|---|
| `analysis/run16_compute4x_calltype.json` | run16 per-layer and best-layer call-type accuracy, 11- and 8-class |
| `analysis/run16_bootstrap.json` | paired bird-bootstrap intervals, run16 vs run11, run15 and all six AVES; also run16 at layer 3 |
| `analysis/run15_compute.json` | now also holds run16 MEASURED (from checkpoint names) next to its pre-run projection |
| `metrics/run16/run16_v{0,1,2}_metrics.csv` | job 39203444, one file per attempt: preempted 09:36, preempted 11:22, completed 15:18. Step ranges OVERLAP — each attempt re-ran from the last checkpoint; stitch with "later file wins" |
| `logs/train_run16_compute4x_39203444.log` | all three attempts (`--open-mode=append`); shows MEMBER 1/4..4/4 and both auto-requeues |
| `logs/smoke_accum_39267313.log` | accumulation smoke test #1: died at import (`NameError: Tuple`), fixed in `ac5fc50c` |
| `logs/smoke_accum_39267771.log` | accumulation smoke test #2: passed — the checkpoint records 300 batches against 150 optimiser steps. The log's own "150/150" progress-bar line is wrong; see below |

**Two traps found in these logs.** (1) `train_masked_accuracy` is a running mean since the process
last started (its counters are reset only by validation, which never runs), so it jumps at every
resume and cannot be compared across runs; compare `train_loss_step`. (2) The progress bar's total
is capped at `--max-updates`, so it cannot count batches under accumulation; read
`loops.fit_loop.epoch_loop.batch_progress` from the checkpoint.

**`--requeue` works on savio_lowprio.** run16 was preempted twice and SLURM put it back in the
queue on its own each time (2 and 6 minutes later); the resume block picked up `last.ckpt`. Use
`sacct --duplicates` to see the preempted attempts — plain `sacct -X` shows only the last one.

run17 = run16 + `--accumulate-grad-batches 2` (~644 s/update, ~92% of AVES per update) is Savio job
39268321, script `examples/hubert/slurm/train_run17_accum2.sh` in the pytorchAudio repo.

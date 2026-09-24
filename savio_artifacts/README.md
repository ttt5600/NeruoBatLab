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
`train_run15_combined.sh`, and — not yet run — `smoke_ddp4.sh` and `train_run16_compute4x.sh`.

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

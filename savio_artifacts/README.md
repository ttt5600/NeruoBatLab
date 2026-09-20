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

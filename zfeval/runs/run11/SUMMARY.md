# run11 (reference)

Generated 2026-09-12T09:50:43Z · git `e616e4b` (dirty tree)

Checkpoint `temp_train_run11/.../epoch=18-step=93750.ckpt` — epoch 18, step 93750, encoder fingerprint `c088f1bc47a356d0`

## Detection scores

| eval | layer | n | majority | AUC | AP | acc |
|---|---|---|---|---|---|---|
| neg_pool [leave-recordings-out (70 recordings, 5-fold)] | L0 | 3768 | 0.6258 | 0.9774 | 0.9887 | 0.9321 |
| neg_pool [leave-recordings-out (70 recordings, 5-fold)] | L1 | 3768 | 0.6258 | 0.9753 | 0.9878 | 0.9323 |
| neg_pool [leave-recordings-out (70 recordings, 5-fold)] | L2 | 3768 | 0.6258 | 0.9749 | 0.9878 | 0.9334 |
| neg_pool [leave-recordings-out (70 recordings, 5-fold)] | L3 | 3768 | 0.6258 | 0.9754 | 0.9881 | 0.9339 |
| neg_pool [leave-recordings-out (70 recordings, 5-fold)] | L4 | 3768 | 0.6258 | 0.9765 | 0.9887 | 0.9350 |
| neg_pool [leave-recordings-out (70 recordings, 5-fold)] | L5 | 3768 | 0.6258 | 0.9767 | 0.9887 | 0.9360 |
| neg_pool [leave-recordings-out (70 recordings, 5-fold)] | L6 | 3768 | 0.6258 | 0.9757 | 0.9882 | 0.9355 |
| neg_pool [leave-recordings-out (70 recordings, 5-fold)] | L7 | 3768 | 0.6258 | 0.9749 | 0.9877 | 0.9329 |
| neg_pool [leave-recordings-out (70 recordings, 5-fold)] | L8 | 3768 | 0.6258 | 0.9743 | 0.9873 | 0.9289 |
| neg_pool [leave-recordings-out (70 recordings, 5-fold)] | L9 | 3768 | 0.6258 | 0.9732 | 0.9868 | 0.9241 |
| neg_pool [leave-recordings-out (70 recordings, 5-fold)] | L10 | 3768 | 0.6258 | 0.9722 | 0.9863 | 0.9220 |
| neg_pool [leave-recordings-out (70 recordings, 5-fold)] | L11 | 3768 | 0.6258 | 0.9714 | 0.9861 | 0.9233 |
| soundsep_111021 [contiguous 60s time blocks within 1 recording(s) (31 blocks) -- NOT a recording holdout] | L0 | 1801 | 0.6757 | 0.9582 | 0.9820 | 0.9006 |
| soundsep_111021 [contiguous 60s time blocks within 1 recording(s) (31 blocks) -- NOT a recording holdout] | L1 | 1801 | 0.6757 | 0.9576 | 0.9817 | 0.9017 |
| soundsep_111021 [contiguous 60s time blocks within 1 recording(s) (31 blocks) -- NOT a recording holdout] | L2 | 1801 | 0.6757 | 0.9597 | 0.9821 | 0.9017 |
| soundsep_111021 [contiguous 60s time blocks within 1 recording(s) (31 blocks) -- NOT a recording holdout] | L3 | 1801 | 0.6757 | 0.9570 | 0.9811 | 0.9012 |
| soundsep_111021 [contiguous 60s time blocks within 1 recording(s) (31 blocks) -- NOT a recording holdout] | L4 | 1801 | 0.6757 | 0.9575 | 0.9812 | 0.8978 |
| soundsep_111021 [contiguous 60s time blocks within 1 recording(s) (31 blocks) -- NOT a recording holdout] | L5 | 1801 | 0.6757 | 0.9567 | 0.9804 | 0.8978 |
| soundsep_111021 [contiguous 60s time blocks within 1 recording(s) (31 blocks) -- NOT a recording holdout] | L6 | 1801 | 0.6757 | 0.9594 | 0.9818 | 0.9028 |
| soundsep_111021 [contiguous 60s time blocks within 1 recording(s) (31 blocks) -- NOT a recording holdout] | L7 | 1801 | 0.6757 | 0.9606 | 0.9820 | 0.9089 |
| soundsep_111021 [contiguous 60s time blocks within 1 recording(s) (31 blocks) -- NOT a recording holdout] | L8 | 1801 | 0.6757 | 0.9618 | 0.9827 | 0.9089 |
| soundsep_111021 [contiguous 60s time blocks within 1 recording(s) (31 blocks) -- NOT a recording holdout] | L9 | 1801 | 0.6757 | 0.9607 | 0.9824 | 0.9023 |
| soundsep_111021 [contiguous 60s time blocks within 1 recording(s) (31 blocks) -- NOT a recording holdout] | L10 | 1801 | 0.6757 | 0.9640 | 0.9837 | 0.9117 |
| soundsep_111021 [contiguous 60s time blocks within 1 recording(s) (31 blocks) -- NOT a recording holdout] | L11 | 1801 | 0.6757 | 0.9630 | 0.9830 | 0.9089 |
| neg_pool -> soundsep_111021 (held out) | L0 | 1801 | 0.6757 | 0.9557 | 0.9793 | 0.8812 |
| soundsep_111021 -> neg_pool (held out) | L10 | 3768 | 0.6258 | 0.9574 | 0.9788 | 0.8962 |

_Accuracy is meaningless without the majority rate beside it._

## Baselines

| eval | model | log-mel | log-energy | model − mel |
|---|---|---|---|---|
| neg_pool | 0.9774 | 0.9267 | 0.7911 | **+0.0506** |
| soundsep_111021 | 0.9640 | 0.8999 | 0.5097 | **+0.0640** |

_If the model does not beat log-mel, the pretraining bought nothing._

## What a false-alarm budget buys

**neg_pool** — 0.01 FA → 2061 calls (87.4%), 0.05 FA → 2184 calls (92.6%), 0.1 FA → 2230 calls (94.6%), 0.2 FA → 2268 calls (96.2%)
**soundsep_111021** — 0.01 FA → 847 calls (69.6%), 0.05 FA → 1034 calls (85.0%), 0.1 FA → 1116 calls (91.7%), 0.2 FA → 1161 calls (95.4%)
**neg_pool -> soundsep_111021** — 0.01 FA → 671 calls (55.1%), 0.05 FA → 1016 calls (83.5%), 0.1 FA → 1088 calls (89.4%), 0.2 FA → 1143 calls (93.9%)
**soundsep_111021 -> neg_pool** — 0.01 FA → 1868 calls (79.2%), 0.05 FA → 2023 calls (85.8%), 0.1 FA → 2117 calls (89.8%), 0.2 FA → 2208 calls (93.6%)

## Controls

- `neg_pool:shuffled_label`: PASS — shuffled AUC 0.4914 ± 0.0140 (null 0.5)
- `neg_pool:per_group`: median AUC 0.9839, IQR [0.9591, 0.9994], min 0.8571, 0/53 below 0.80
- `neg_pool:loudness_stratified`: PASS — within-band model 0.9629, energy 0.5641 (null 0.5)
- `neg_pool:loudness_matched_pairs`: PASS — model wins 0.9609 of 843 equal-loudness pairs, energy 0.5302 (null 0.5)
- `soundsep_111021:shuffled_label`: PASS — shuffled AUC 0.4964 ± 0.0222 (null 0.5)
- `soundsep_111021:per_group`: median AUC 0.9640, IQR [0.9640, 0.9640], min 0.9640, 0/1 below 0.80
- `soundsep_111021:loudness_stratified`: PASS — within-band model 0.9616, energy 0.5580 (null 0.5)
- `soundsep_111021:loudness_matched_pairs`: PASS — model wins 0.9451 of 528 equal-loudness pairs, energy 0.5114 (null 0.5)

## Embedding geometry

| layer | AUC | kNN-10 | silhouette | Fisher | PCA dim@90% |
|---|---|---|---|---|---|
| L0 | 0.9774 | 0.8830 | 0.195 | 0.5725 | 15 |
| L1 | 0.9753 | 0.8835 | 0.201 | 0.5697 | 21 |
| L2 | 0.9749 | 0.8936 | 0.197 | 0.5475 | 29 |
| L3 | 0.9754 | 0.8994 | 0.200 | 0.5354 | 36 |
| L4 | 0.9765 | 0.9039 | 0.210 | 0.5393 | 39 |
| L5 | 0.9767 | 0.9058 | 0.220 | 0.5434 | 42 |
| L6 | 0.9757 | 0.9047 | 0.216 | 0.5348 | 44 |
| L7 | 0.9749 | 0.9018 | 0.219 | 0.5327 | 46 |
| L8 | 0.9743 | 0.9037 | 0.215 | 0.5427 | 49 |
| L9 | 0.9732 | 0.8984 | 0.210 | 0.5451 | 50 |
| L10 | 0.9722 | 0.9026 | 0.197 | 0.5521 | 51 |
| L11 | 0.9714 | 0.9031 | 0.205 | 0.5632 | 51 |

## Files

- `SUMMARY.md`
- `checkpoint.json`
- `report.json`
- `windows_neg_pool.meta.json`
- `windows_neg_pool.npz`
- `windows_soundsep_111021.meta.json`
- `windows_soundsep_111021.npz`
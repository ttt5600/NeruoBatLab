# How sharp can the call-type test get? (2026-11-15, CPU only, no training)

Scripts: `cache_oof.py` (out-of-fold probe probabilities; board probe, each model at its CPU best layer; split 0
asserted to reproduce `calltype_cpu_rescore.json` acc11_cpu for all 70 models; splits 0-9 for 12 seed groups),
`power_analysis.py` (everything below, writes `power_analysis.json`), `show.py` (prints the headline numbers).
Cache (`cache/`, about 100 MB) is git-ignored. Rebuild it with `python cache_oof.py --split-models --all --leaky`
(about 4 min at n_jobs=4), then run `python power_analysis.py` (about 2 min).
Seed groups (12, df 13): run16 x3, run19, E13, run20, run29, run30, E19, bxl, aveso, v3aves, v3m8, vanilla_lr1e4 x2
(run31_vanilla_lr1e4_s2 was scored on this Mac while this ran; it is included).
These are exploratory numbers, not a finding. Nothing was critic-checked.

## 1. Where the interval width comes from (accuracy, JSON `typical_components_median_over_recipe_pairs.acc`)
Medians over the 66 two-seed recipe pairs:
| term | variance | share |
|---|---|---|
| birds (bird bootstrap, the tool's 2,000 draws) | 6.21e-5 | 50% |
| seeds, s^2(1/2+1/2), s pooled over 12 groups = 0.0060 | 3.59e-5 | 29% |
| fold split (var over SGKF random_state 0-9) | 2.51e-5 | 20% |
The resulting full half-width is 0.0216. The tool's own interval (birds + seeds) leaves out the split term: 0.0192.
- The birds term is mostly between-bird heterogeneity. Resampling clips inside each bird gives 1.96e-5 (32% of V_bird),
  and an iid clip bootstrap gives 2.02e-5. So the design effect of clustering is 3.4 (2-6.5 by statistic), and a per-clip
  paired test (McNemar-style) would give intervals about 1.85x too narrow. More clips per existing bird can remove at
  most the within part.
- About 40% of V_bird is seed-specific clip noise that averages out with more seeds (fit of V_bird(n) = a + c(1/n_a+1/n_b): a 3.9e-5, c 2.64e-5).
- Half of the split-0 "seed variance" is probe-split noise. The pooled seed sd is 0.0060 on split 0 and 0.0041 on 10-split
  means. run16's three seeds give 0.0111 on split 0 and 0.0047 on 10-split means (`seed_sd_by_group_acc`). run16 is the
  noisiest group. Most groups have split-0 sd of 0.001-0.006 (df 1 each).
- Named contrasts (`decomposition.*.acc`): run19|run16 +0.0182 at split 0 but +0.0115 as a 10-split mean, with shares
  birds 60%, seeds 19%, split 21%. run16|vanilla (2 seeds each) -0.0067 at split 0, +0.0005 as a 10-split mean.
  The board split is not neutral.
- Probe randomness: LogisticRegression (lbfgs) is deterministic, so the only probe randomness is the fold assignment.

## 2. Power (accuracy; `power_requirements_acc`, `power_acc_grid`)
The model is SE^2 = (a + 2c/n)*48/B + s^2*2/n + V_split/K, with B birds, n seeds per arm and K averaged splits.
- Today (2 v 2 seeds, 48 birds, 1 split): SE 0.0112. Power is 14% for a true 0.01 and 43% for a true 0.02.
- With 10 splits: power 58% for 0.02 (2 seeds) and 74% (5 seeds).
- True 0.02 at 80% power: 10 seeds/arm at 48 birds, or 60 birds at 5 seeds, or 104 birds at 2 seeds.
- True 0.01: at 48 birds the SE floor with unlimited seeds is 0.0064, so the half-width can never reach 0.01.
  It needs 108 birds at 10 seeds/arm (half-width <= 0.01) or 316 birds at 10 seeds (80% power).
- Bird-scaling check (`bird_scaling_check`, probe refit on 24/36-bird subsets): V_bird*B/48 = 1.08e-4 / 0.98e-4 / 1.59e-4
  at B = 24/36/48. That is no sign of worse-than-1/B scaling, but it is noisy and drawn from the same pool. Accuracy rises
  with B (run19_seed2: 0.814, 0.839, 0.855), so effect sizes could also change with more birds.

## 3. Other statistics (`statistics`)
Each statistic was scored on the same 66 recipe pairs at the full SE (birds + seeds + split). Null check: the 14
seed-vs-seed pairs inside one recipe.
| statistic | recipe pairs resolved | median abs z | null resolved |
|---|---|---|---|
| accuracy (board) | 4/66 | 1.01 | 0/14 |
| accuracy, 10-split mean | 3/66 | 1.06 | 0/14 |
| **mean p(true class)** | **12/66** | **1.24** | 0/14 |
| **mean p(true), 10-split** | **14/66** | 1.23 | 0/14 |
| mean log p(true) | 2/66 | 0.96 | 0/14 |
| balanced accuracy | 1/66 | 0.48 | 0/14 |
| bird-mean accuracy | 3/66 | 0.69 | 0/14 |
| Tet recall | 1/66 | 0.88 | 0/14 |
| Tet+Thuk+Tuck recall | 4/66 | 0.98 | 0/14 |
- Mean p(true) agrees with accuracy (sign 98%, Spearman 0.99 on split-0 differences). A z^2 ratio of 1.49 is worth about
  1.5x the birds, at no cost. In acc-equivalent units (slope 1.06), a true 0.02 reaches 80% power with 4 seeds/arm at
  48 birds and 10 splits (accuracy needs 10). A true 0.01 still needs 80+ birds (half-width) or 204 birds at 10 seeds (80% power).
- The balanced, per-bird and per-class statistics are noisier, because they average over small cells. Tet recall alone
  does not sharpen the test in general. It is high only on the AVES-label vs vanilla contrasts (z 1.73 for E13|vanilla),
  which is a hypothesis to pre-register, not a general sharpener.
- Caveat: p(true) also moves with probe calibration. Pre-register it before using it; 16 statistics were looked at here.

## 4. Label ceiling (`ceiling`)
- Across 69 non-shuffled models (split 0, best layers), 41 clips are wrong for every model (37 when the shuffle model is
  included), and 1,674 are right for every model. That is not the Mac-era 174.
- The 41 clips:
  - Types: Thuk 10 (3.4% of Thuk), Tuck 10 (4.2%), Tet 9 (1.5%). Every other type has 0-4.
  - Length: short, median 0.085 s against 0.154 s for all clips.
  - Birds: 16 birds, and 3 of them hold 59% (lblbla4548 9/189, yelgre5275 9/207, yelora2575 6/144). 37 adult, 4 chick.
  - The models agree on the wrong answer: 51% of these clips get the same wrong label from all 69 models (modal wrong
    labels Te 13, Ne 11, Th 9).
- Leaky random split (the bird's own other clips in training; Whisper base, bioxn-large, run19_seed2): 20 of the 41 stay
  wrong for all three, against a 2.5% all-three-wrong rate overall. About half are bird-generalisation failures. The
  other 20 or so (0.6% of clips) are label-error or ambiguous-soft-call candidates.
- No binding ceiling. Even if all 41 are unfixable, the cap is 0.988, against a best model of 0.8646. Clips that every
  model gets right or every model gets wrong (50%) carry no information about differences: a model pair differs on a
  median of 399 clips. Relabelling would barely sharpen arm contrasts.

## Recommendations
1. **Change the statistic and the readout (free, CPU minutes):**
   - Use mean p(true class), averaged over 10 fold splits.
   - Use one pre-committed layer rule, so the split-0 winner's curse is gone.
   - Use a pooled seed variance from all seed groups (df 13), not just the two arms' 2-3 seeds.
   This takes 2v2 power for a true 0.02 from 43% to 70%, and it resolves 14/66 recipe pairs against 4/66 for accuracy.
2. **Seeds, not birds, for 0.02 effects:** 4-5 seeds per arm with (1) gives about 80-84% power for a true 0.02. Each
   seed is one full Savio training run.
3. **Accept that 0.01 is out of reach on this dataset.** 48 birds is the whole Elie & Theunissen set, and the SE floor at
   unlimited seeds is about 0.006. Sub-0.02 recipe effects need roughly 100-200+ labelled birds (a new dataset), or a
   different task.

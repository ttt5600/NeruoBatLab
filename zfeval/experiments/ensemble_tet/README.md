# Call-type decision ensemble + Tet-focused arm contrast (CPU, no training)

Two roadmap leads (harness/ROADMAP.md "Leads"): decision ensemble on call type, and a Tet-focused arm test.
Everything here reads existing CPU caches `features/ct11_<tag>_cpuemb.npy` with the board probe (`aves_calltype.cv_acc`:
StandardScaler + LogisticRegression, leave-birds-out StratifiedGroupKFold(5) seed 0) and the bird bootstrap
(2,000 resamples of the 48 birds, seed 0, same resamples as seed_pairs_cpu.py). Differences are OBSERVED, not resample means.

## Part 1 pre-commitment (written and committed BEFORE any ensemble was computed)

Question: does averaging out-of-fold class probabilities of 2-4 frozen encoders beat the best single member on
11-class call type?

Rule (fixed now):
- Each member is read at its own CPU best layer (`calltype_cpu_rescore.json models.<tag>.best_layer_cpu`); its pooled
  accuracy is asserted to reproduce `acc11_cpu`. All members share the same folds (seed 0), so every clip's
  probability comes from a probe that never saw that clip's bird.
- Ensemble = unweighted arithmetic mean of the members' out-of-fold probability vectors, argmax. NO weights are fitted
  (no stacking), so there is nothing to leak from the test folds.
- Comparator = the member with the highest board accuracy (`acc11_cpu`), i.e. what we would ship without the ensemble.
- Family of tests (m = 7, Bonferroni): the five mean-ensembles E1-E5 and the two concatenation baselines C1-C2.
  An ensemble "beats its best member" only if the Bonferroni interval (two-sided 1 - 0.05/7, percentiles
  0.357 / 99.643) excludes zero. The 95% interval is also printed. If an interval crosses zero: "not distinguishable".
- Pre-committed list (nothing else will be reported as a candidate; any extra combo is labelled post hoc):
  - E1 = whisper-base + whisper-small + birdaves-bioxn-large  (top-3 downloads; PRIMARY)
  - E2 = whisper-base + birdaves-bioxn-large                   (best of two different families)
  - E3 = whisper-base + run19_seed2                            (best download + best of ours)
  - E4 = whisper-base + whisper-small + birdaves-bioxn-large + run19_seed2
  - E5 = whisper-small + birdaves-bioxn-large + run24_bioxnl12_s2  (E13 seed 2 replaces run19)
  - C1 = feature concatenation of E1's members (each at its best layer) into one probe, vs best member
  - C2 = feature concatenation of E2's members, vs best member
- Prior expectation (031 on detection; 043): mean helps, concatenation does not. Kill: if E1 is not distinguishable
  from whisper-base at the Bonferroni level, the decision ensemble is not a lead for call type at this test's power.
- Secondary readouts (not part of the decision): (a) every member at fixed layer 3 (no per-model layer selection;
  CRITIC.md check 4); (b) ensemble-minus-best-member over fold splits 0-9 (a consistency check, no interval: the
  splits reuse the same 48 birds); (c) per-type recall of E1 vs whisper-base.
- Known bias: best layers and the "top downloads" were chosen on this same board split, so member scores carry some
  winner's curse. It affects members and ensemble alike; the fixed-layer readout is the check.

## Part 2 (EXPLORATORY, not pre-registered)

Tet (+ Thuk/Tuck) recall, arm vs arm, seeds averaged, with seed_mean_pairs_cpu.py's seed term
(half-width = 1.96 sqrt(var_birds + s^2 (1/n_a + 1/n_b)), s^2 = pooled within-arm variance of the seeds' metric).
The per-type heatmap that motivated it was read before this analysis, so it is hypothesis-generating only.
A draft pre-registration for a confirmatory test is at the end of this file once results are in.

<!-- RESULTS BELOW ARE ADDED AFTER COMPUTATION -->

## Part 1 results (ensemble_calltype.json / .log; computed after the pre-commit, commit 2649db2)

All five members reproduce `acc11_cpu` exactly (asserted; `members.*.acc_best` = `acc_best_rescore_json`). Family m = 7,
Bonferroni two-sided alpha 0.00714. Comparator = best member by board accuracy.

| id | kind | members | ens acc | best member (acc) | diff | 95% | Bonferroni | verdict |
|---|---|---|---|---|---|---|---|---|
| E1 | mean | wb + ws + bxl | 0.8816 | wb 0.8646 | +0.0170 | [+0.0061, +0.0269] | [+0.0019, +0.0300] | ensemble better |
| E2 | mean | wb + bxl | 0.8810 | wb 0.8646 | +0.0164 | [+0.0076, +0.0254] | [+0.0046, +0.0283] | ensemble better |
| E3 | mean | wb + run19_seed2 | 0.8813 | wb 0.8646 | +0.0167 | [+0.0057, +0.0305] | [+0.0018, +0.0349] | ensemble better |
| E4 | mean | wb + ws + bxl + run19_seed2 | 0.8828 | wb 0.8646 | +0.0182 | [+0.0063, +0.0304] | [+0.0026, +0.0360] | ensemble better |
| E5 | mean | ws + bxl + run24_bioxnl12_s2 | 0.8810 | ws 0.8599 | +0.0211 | [+0.0098, +0.0310] | [+0.0065, +0.0339] | ensemble better |
| C1 | concat | wb + ws + bxl | 0.8713 | wb 0.8646 | +0.0067 | [-0.0041, +0.0181] | [-0.0076, +0.0221] | not distinguishable |
| C2 | concat | wb + bxl | 0.8693 | wb 0.8646 | +0.0047 | [-0.0080, +0.0164] | [-0.0134, +0.0210] | not distinguishable |

(wb = hf-whisper-base-enc L0, ws = hf-whisper-small-enc L3, bxl = birdaves-bioxn-large L1, run19_seed2 L1,
run24_bioxnl12_s2 L4.) The pre-committed kill did not fire: E1 beats whisper-base at the Bonferroni level.

Secondary readouts:
- Fixed layer 3 for every member (`secondary_L3`; comparator = best member at L3): E1 +0.0106, 95% [+0.0004, +0.0194],
  Bonferroni [-0.0032, +0.0226] (crosses zero). E2 +0.0234, E3 +0.0188, E4 +0.0226, E5 +0.0149, all with Bonferroni
  lower bounds > 0 (E5 only just, +0.0008). So with no layer selection the primary E1 holds at 95% only.
- Fold splits 0-9 (`splits`; members at their split-0 best layers): ensemble minus best member is positive on 10/10
  splits for every E; means E1 +0.0179, E2 +0.0158, E3 +0.0127, E4 +0.0205, E5 +0.0162 (no interval: same 48 birds).
- Where E1 gains over whisper-base (`per_type_E1`, recall, no intervals): Te 0.949 vs 0.917, Th 0.741 vs 0.669,
  Ne 0.876 vs 0.850; lower on Di 0.444 vs 0.556 (18 clips) and Tu 0.396 vs 0.417.
- Error overlap (`error_overlap`, phi of error indicators): 0.50-0.65; whisper-base vs whisper-small 0.652 is the most
  correlated pair, whisper vs bioxn-large ~0.50-0.51.

POST HOC controls (ensemble_controls.json / .log; chosen after seeing E1-E5; 95% only; not in the corrected family):
K1 run16 three seeds +0.0079 [-0.0030, +0.0187] vs best seed; K2 run19 two seeds +0.0015 [-0.0064, +0.0106];
K3 wb + run16_seed2 +0.0073 [-0.0044, +0.0202]; K4 wb + stock hf-hubert-base-ls960 +0.0059 [-0.0047, +0.0175];
K5 wb + ws (same family) +0.0026 [-0.0078, +0.0136]. None resolved. Averaging per se does not buy ~+0.017; it takes
two strong members from different families (Whisper + an AVES-family or our ZF model). Our run19_seed2 is as good a
partner for whisper-base as BirdAVES-large (E3 +0.0167 vs E2 +0.0164).

Why this is not a leak: no weights are fitted (plain mean), every member's probe is the board probe on the same
leave-birds-out folds, per-member accuracies reproduce the board exactly, and the paired bootstrap is over birds.

Caveats: (1) Best layers and the choice of top downloads came from this same split, so members carry winner's curse;
the fixed-L3 readout removes the layer part (E1 then 95% only). (2) The interval is over birds for these fixed
checkpoints; for E3/E5 it does not include our recipes' seed variance (K2 shows run19's two seeds barely complement
each other). (3) Costs 2-4 encoder passes; it is a ceiling for frozen readouts, not a better single HuBERT encoder,
so it does not directly advance the goal - but it says our ZF model carries call-type information Whisper lacks.
(4) Matches 031 (detection): mean helps, concatenation does not. (5) Not critic-checked; run /finding before quoting.

## Part 2 results (EXPLORATORY; tet_arm_contrast.json / .log)

Arms: aveslabel = run19_avesteacher, run19_seed2, run24_bioxnl12, run24_bioxnl12_s2 (4); ownlabel = run16_compute4x,
run16_seed2, run16_seed3, run31_vanilla_lr1e4 (4). run31_vanilla_lr1e4_s2 has no CPU cache yet, so vanilla is 1 seed.
Intervals = birds + seed term, 95% (99.5% also in the JSON). Cross-check: "run19 | run16 | best" overall +0.018
[-0.008, +0.044] reproduces seed_mean_pairs_cpu.json "run19 | run16" (+0.0182 [-0.0076, +0.0439]).

| contrast | readout | Te recall | Th | Tu | TTT (Te+Th+Tu, n 1143) | overall |
|---|---|---|---|---|---|---|
| aveslabel - ownlabel | best | 0.895 vs 0.835, +0.060 [+0.007, +0.113] | +0.072 [-0.034, +0.178] | -0.029 [-0.107, +0.049] | +0.044 [+0.002, +0.086] | +0.018 [+0.002, +0.035] |
| aveslabel - ownlabel | L3 | 0.884 vs 0.816, +0.068 [-0.001, +0.137] | +0.043 [-0.037, +0.123] | -0.006 [-0.077, +0.064] | +0.046 [-0.005, +0.097] | +0.019 [+0.002, +0.037] |
| aveslabel - run16 | best | +0.062 [-0.004, +0.127] | +0.093 [-0.026, +0.211] | -0.027 | +0.051 [+0.004, +0.098] | +0.019 [-0.001, +0.039] |
| aveslabel - vanilla | best | +0.055 [+0.0001, +0.110] | +0.011 | -0.034 | +0.025 [-0.025, +0.075] | +0.016 [-0.003, +0.034] |
| run19 - run16 | best | +0.056 [-0.019, +0.132] | +0.094 | -0.024 | +0.049 [-0.011, +0.109] | +0.018 [-0.008, +0.044] |
| E13 - run16 | best | +0.067 [+0.002, +0.132] | +0.092 | -0.031 | +0.053 [+0.004, +0.102] | +0.021 [-0.002, +0.043] |

Reading: the AVES-family-label arms are ahead on Tet by +0.055 to +0.083 in every contrast and readout (point
estimates; L3 rows in the JSON), resolved at 95% in 3 of 10 contrast-readouts (all at the best layer; the vanilla one
only by +0.0001), none at 99.5% (Bonferroni over 11 types). No other single type is resolved in any of the 10, a weak
built-in placebo in Tet's favour, but the 10 contrasts share models and are not independent. Thuk is +0.09 vs run16
but +0.011 vs vanilla and noisy (seed sd 0.02-0.035). Tuck: nothing (-0.034 to +0.003). Tet is not a much sharper
test than overall accuracy: in the headline contrast Tet carries 0.060 x 613 / 3412 = 0.011 of the +0.018 overall
difference, and the overall difference is resolved too (best and L3), at [+0.002, ...] lower bounds.
Caveats: the hypothesis came from the per-type heatmap on these same models (circular); aveslabel mixes two recipes
and ownlabel mixes run16 and vanilla (between-recipe spread inflates the pooled seed SD: conservative); vanilla has one
seed; "best" carries layer selection on the board split.

## Draft pre-registration for a confirmatory Tet test (not run; needs new runs = user approval)

- Question: does training a HuBERT on AVES-family teacher labels (run19 / E13 recipes) raise Tet recall over own-label
  training (run16 / vanilla lr 1e-4)?
- Independence: the confirmatory set must be models NOT used above, i.e. new seeds only (e.g. run19 seed 3, E13 seed 3
  vs run16 seed 4, vanilla lr1e4 seed 2 [40654081, in flight] and seed 3). Re-reading the 8 models above is not confirmatory.
- Primary metric: arm-mean Tet recall (613 Te clips), board probe (leave-birds-out SGKF(5) seed 0), every model at fixed
  layer 3 (no layer selection). Secondary, must agree in sign: each model at its CPU best layer; TTT recall.
- Test: arm difference with seed_mean_pairs_cpu.py's interval (bird bootstrap 2000 seed 0 + pooled seed term).
  Success: 95% lower bound > 0 at L3 AND best-layer point estimate > 0. Kill: L3 point estimate <= +0.02, or 95% upper
  bound < +0.03 -> "AVES labels do not buy Tet"; drop the lead.
- Power (rough, from tet_arm_contrast.json "aveslabel | ownlabel | best" Te: birds-only 95% width 0.094 -> bird SE
  ~0.024; pooled seed SD 0.015). With 3 new seeds per arm SE ~ sqrt(0.024^2 + 0.015^2 x 2/3) ~ 0.027: a true +0.06 gives
  power ~ Phi(0.06/0.027 - 1.96) ~ 0.6, a true +0.04 ~ 0.3. Bird variance dominates, so more seeds barely help, and the
  observed +0.06 is likely inflated by selection. A test worth paying for probably needs more birds (new labelled Tet
  clips from new birds), not more seeds.
- Cost: 4-6 training runs on Savio (paid A40) -> human gate. Nothing was launched here.

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

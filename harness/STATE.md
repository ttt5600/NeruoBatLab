# Lead state -- OVERWRITE every wake (<= 40 lines). Volatile facts only; history goes to pi_notes.md.
Updated: 2026-10-19 03:05 (lead: session_01EGDWEvYFd6zSmWQfUugJnH; user: "summarize please and keep going")

## USER PRIORITY: ZF specialization judged on CALL TYPE; skip BEANS. Deliver visually (notebooks/artifacts).
Memory: feedback_specialization_not_beans.md. Call type is CPU ONLY (076): never quote ct11_<tag>_emb.npy readers.

## Since 2026-10-18: call type, label-free, 10-split and chick all CPU (076); details in pi_notes.md
- score_run.sh + build_board.py read only CPU files (calltype_cpu_rescore, seed_pairs_cpu[_L3],
  calltype_split_sensitivity_cpu[_L3], labelfree_stability_cpu, chick_holdout_variants_cpu).
- 077, 078, 079 CONFIRMED after critic. 079: seed noise stays 0.015-0.02 (run16 seed pair resolved at fixed L3).

## 2026-10-19: chick holdout on CPU
- chick_holdout_variants.py --cpu: 34 models -> chick_holdout_variants_cpu.json (+ _cpu.npy caches). All 97 chick clips
  of 2-8 frames were mis-encoded on MPS. AUC moves <=0.005; AMI(k=4) -0.046..+0.150; 0 of 33 vs run11 resolved, no
  flips -> 048 stands. 080 CONFIRMED (critic changes applied; pointer in 048). score_run.sh + board use the CPU chick
  file; board v6 published -- no MPS-derived column left on it.

## In flight
- E12 = run22 (40090445): preempted 2x, requeued at step 41,018; Slurm est. restart 2026-10-19 14:39. Score with
  `bash harness/score_run.sh run22_avesfsd /global/scratch/users/jonathanswang/temp_train_run22_avesfsd 200 40090445`.

## Next (lead)
- Board v7: 8 off-the-shelf encoders added (CPU vs run16: Whisper base +0.0293 [+0.0076, +0.0508], small +0.0246
  [+0.0017, +0.0465] at L3, both ahead on 10/10 splits). 081 CONFIRMED (2 critic passes): Whisper-small's lead over
  run16 is 72-85% Tet (beyond run16's seed spread on Tet); vs AVES Tet+Nest; trails run16 on Begging; Tuck n.d.
  Digest 10-19 at the first wake after 09:00.
- E15 corpus research (post choice + size + licence in a digest, then wait a day) if the user wants E15.

## Waiting on the user
- E13 teacher (biox-base 0.8476 ~ AVES-bio 0.8438 on CPU; bioxn-large 0.8590 is large), E26 success + donors,
  E27 24-layer branch, E15 corpus, E33 (rec: no at +-2 st), 10-split headline, BeVo split, AVEX, WiSE-FT.
- Older: E30-32, label audit. Savio cert EXPIRED 2026-10-19 00:52 (notified): `sv login` before E12 starts ~14:39.

## Resume: /loop follow harness/PI.md  (first wake: read PI.md, AGENT.md, this file, knowledge/BRIEF.md, pi_status.sh)

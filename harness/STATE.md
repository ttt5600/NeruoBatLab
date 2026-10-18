# Lead state -- OVERWRITE every wake (<= 40 lines). Volatile facts only; history goes to pi_notes.md.
Updated: 2026-10-18 13:45 (lead: session_01EGDWEvYFd6zSmWQfUugJnH; user: "logged in keep going")

## USER PRIORITY: ZF specialization judged on CALL TYPE; skip BEANS. Deliver visually (notebooks/artifacts).
Memory: feedback_specialization_not_beans.md. Call type is CPU ONLY (076): never quote ct11_<tag>_emb.npy readers.

## This wake (2026-10-18): the pipeline moved to CPU call type everywhere
- score_run.sh: calltype_cpu_rescore.py --only TAG + seed_pairs_cpu.py TAG:aves-base-bio TAG:run16_compute4x +
  calltype_labelfree.py --cpu; MPS bootstrap dropped; DRY run vs run22 OK. Chick holdout still MPS (all models).
- Board (build_board.py) reads only CPU: calltype_cpu_rescore.json, seed_pairs_cpu.json (24 new X|run16 pairs),
  calltype_split_sensitivity_cpu.json (new --cpu mode), labelfree_stability_cpu.json (new --cpu mode).
- CPU vs run16: only bioxn-large +0.0237 [+0.0033, +0.0427] (best layer L1) and run21 -0.0226 resolved, of 29.
- 077 + 078 CONFIRMED after critic (changes applied; E9 result line corrected).
- 079 CONFIRMED (2 critic passes): 057's pair n.d. at best layers on CPU (+0.0155 [-0.0026, +0.0340]) but resolved at
  fixed L3 (+0.0185 [+0.0028, +0.0340]) on 10/10 splits; seed noise stays 0.015-0.02 (my ~0.01 draft was refuted).
  New: seed_pairs_cpu.py --layer, seed_pairs_cpu_L3.json, calltype_split_sensitivity_cpu_L3.json.
- Label-free on CPU (57 models): k30 vs same-layer call type rho 0.208 (p 0.12) -- same weak agreement as 069.
- Board republished v5 (https://claude.ai/artifact/3LSJjzp7pLS4JCWviA1wpb). Pushed: project ca69304.
- run15_calltype.py no longer stamps run15's corpus note on other tags. Digest 2026-10-18 drafted; task doc T3.

## In flight
- E12 = run22 (40090445): preempted 2x, requeued at step 41,018; Slurm est. restart 2026-10-19 14:39. Score with
  `bash harness/score_run.sh run22_avesfsd /global/scratch/users/jonathanswang/temp_train_run22_avesfsd 200 40090445`.

## Next (lead)
- Chick holdout on CPU for every model (048/076 scope). Board: HF encoders absent (Whisper-base 0.8646 leads CPU).
- E15 corpus research (post choice + size + licence in a digest, then wait a day) if the user wants E15.

## Waiting on the user
- E13 teacher (biox-base 0.8476 ~ AVES-bio 0.8438 on CPU; bioxn-large 0.8590 is large), E26 success + donors,
  E27 24-layer branch, E15 corpus, E33 (rec: no at +-2 st), 10-split headline, BeVo split, AVEX, WiSE-FT.
- Older: E30-32, label audit. Savio cert valid to 2026-10-19 00:52.

## Resume: /loop follow harness/PI.md  (first wake: read PI.md, AGENT.md, this file, knowledge/BRIEF.md, pi_status.sh)

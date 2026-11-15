# Lead state -- OVERWRITE every wake (<= 40 lines). Volatile facts only; history goes to pi_notes.md.
Updated: 2026-11-15 (lead session; E36/E37 judged, ensemble finding, power/data/design agents; branch zf-2026-11-15)
ON RESUME: read harness/ROADMAP.md (updated 2026-11-15) and harness/designs_2026-11-15.md; give where we are / leads / next.

## USER PRIORITY: ZF specialization judged on CALL TYPE; skip BEANS. Deliver visually. User wants A) more data B) other data
## types C) better embeddings / model class / fine-tuning. Call type is CPU ONLY (076). LOOPS OFF.

## Results 2026-11-15 (all in knowledge/findings, index rebuilt)
- 090 E36: vanilla lr1e-4 2 seeds 0.8388; run16/run19/E13/E30 all n.d. vs vanilla. Scoped to 4 arms (DAPT, run20, run29 open).
- 092 E37 randproj: NOT MET (L3 vs run23 resolved +0.0186, randproj above, df-1 seed term). Rule: pre-register one-seed sd.
- 091 ensemble (confirmed): whisper-base+small+bioxn-large prob-mean 0.8816 vs 0.8646, +0.017 Bonf [+0.002,+0.030]; nested
  +0.021 [+0.009,+0.035]. Code zfeval/experiments/ensemble_tet/. Concat n.d.
- Power (zfeval/experiments/power, NOT critic-checked): mean p(true) 10-split resolves 14/66 vs acc 4/66; 0.01 unreachable at 48 birds.
- Data scout: no other-lab ZF call-type labels exist; V33DA / D'Amelio identity sets; Rockefeller library (license unknown).

## In flight
- 40654080 E36 stock seed 2, paid A40 PENDING (Resources), est. start 2026-11-16 23:36, ~860-1,500 SU. Pointless for the
  E36 verdict -> ASKED USER: cancel / keep / move to lowprio. Do not act without the answer.

## Credits (check_usage.sh, 2026-11-15 00:24)
- fc_birdpow 7,688 / 300,000 SU used since 2026-06-17 (user is ~sole user). Since 10-28: 7,114 SU; A40 = 3.67 SU/CPU-h
  (derived); zf A40 ~6,840, compete A40 ~270. savio_lowprio = 0 SU. a40_gpu3_normal QoS GrpTRES cpu=128,gpu=16 (BRC-set).
- check_usage.sh lives at /global/home/groups/allhands/bin/check_usage.sh (not on PATH).

## Waiting on the user
- Goal scope: may the shipped model be non-HuBERT or an ensemble? Lock zf goal (sudo install.sh --lock).
- Approve next steps from ROADMAP (1 prereg sharper test, 2 X1 frozen screen, 3 X2 fine-tune -- all free CPU first).
- Savio cleanup yes/no (E13 feat ~165 GB, BEANS 14 GB). Merge branch zf-2026-11-15 into UMAP.

## Authorizations (unchanged)
- fc_birdpow FCA for E36/E37 only; anything else free savio_lowprio or Mac CPU unless the user says otherwise.

## Traps
- 2026-11-10 disk cleanup moved idle detvar files to Savio (same path under /global/scratch/users/jonathanswang);
  detection_variants.py needs features/detvar/preds_run11.npz (restored 2026-11-15). Restore others with rsync if a scorer dies.
- Worktrees lack untracked label files: run zfeval scripts from the MAIN checkout path (collect() returns 0 rows otherwise).
- calltype_split_sensitivity.py parses argv by hand: never pass --help (it starts a full run).
- score_run.sh prefers final.ckpt; seed_mean_pairs_cpu.py appends to shared JSONs. Teacher must be .eval().
- Single-seed comparators borrow the other arm's seed sd (df 1): pre-register the sd to use.

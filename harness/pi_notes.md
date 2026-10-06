# Lead agent working notes

Half-finished work, newest first. Each entry: date, what, where it stopped, next step.

## 2026-10-05 20:50 — E4 scripts built, Savio checks pending
- Savio cert expired 05:12 today; notified, no Savio work this wake. E1/E2 status unknown.
- Built `slurm/build_fsd_only_corpus.sh`, `scripts/subset_corpus.py`, `slurm/train_run20_fsdonly.sh`
  (pytorchAudio, added to scp FILES, not pushed to Savio). subset_corpus.py tested locally on a
  synthetic 4-row corpus: rows and labels stay aligned; no-match prefix aborts.
- Next (needs login): `sv push`; on login node `head -3` the combined TSV to confirm the FSD
  prefixes; `DRY=1 bash slurm/build_fsd_only_corpus.sh` — confirm hours and that kept vocabulary
  max id is 199; import check; then critic, then approve or needs-human. Then E5 (backlog 1b).

## 2026-10-05 21:05 — E4 approved, E6 twin added, both data sets built
- E4 critic: ACCEPT-WITH-CHANGES; applied (decision rule, twin E6, confounds). Both approved,
  data built on the login node (savio3 rejects savio_lowprio). They wait for E2/E3 to free slots.
- Next: when a slot frees, submit E4 then E6 (priority order). Backlog item 1b: E5 replay dose
  script (build_replay_corpus.sh 3 — check its qos: savio3 + savio_normal spends allocation).

## 2026-10-06 01:30 — COLLISION with the usage-watch auto-resume agent
- At 01:21 usage-watch auto-resumed `claude -p --continue` (PID 97510) on this project while this
  /loop lead was also awake. Both submitted E4: 39661997 (theirs) and 39661998 (mine), same script
  and exp_dir. Cancelled 39661998 while both were PENDING; 39661997 is THE E4 job.
- That agent rewrote experiments.yaml (yaml.dump reflow), marked E2 trained, and is scoring
  run16_seed2 (score_run.sh started 01:24). I stood down this wake to avoid a second scoring run
  and git races. Next wake: check E4's `job:` in experiments.yaml says 39661997, check
  datasets.tsv / jobs.json have no duplicate run20 rows, then write the E2 finding if it has not.
- Needs you: usage-watch auto-resume ignores the PI heartbeat. Either undesignate this project
  while the /loop lead runs, or have the auto-resume prompt check harness/.pi_heartbeat.
- 02:26 follow-up: the auto-resumed agent exited with its run16_seed2 scoring killed at the export
  step (no JSONs written). experiments.yaml correctly records E4 = 39661997. Removed the duplicate
  run20 row from datasets.tsv (pytorchAudio 275485e4). Re-running score_run.sh run16_seed2 myself.

## 2026-10-06 03:05 — E2 done (finding 057): seed gap 0.0155, resolved
- For the 09:00 digest "needs you": E3/E4/E6 are all one seed each, and 057 says a ~0.015 call-type
  gap can come from seed + run history. Proposal to put to the user: a second seed for whichever of
  E3/E4 lands within ~0.02 of run16, and a run16 seed 3 to get a third point. Costs ~6-7 h of
  lowprio each. Not self-approved: it changes the plan's shape, so it is the user's call.

## 2026-10-06 04:00 — E5 approved (repetition dose), critic changes applied
- Digest "needs you": (a) a --seed 2 run of the DAPT x1 replay arm (critic + 057); (b) the x1
  replay arm has no finding yet -- write one before E5 is scored (backlog, mine); (c) E5 can only
  say "not resolvable at ~0.05 AP" unless the effect is large -- run it anyway (1 GPU, lowprio)?
- Open technical item: score_run.sh was built for from-scratch runs; check it handles a DAPT
  checkpoint (100 classes, AVES init) before E5 finishes.

## 2026-10-06 05:20 — E5 scoring gap found (not urgent; E5 not yet submitted)
- score_run.sh exports DAPT ckpts the same way export_dapt_checkpoints.sh did (export_weights.py
  --num-classes), so step 1-2 are fine for E5 (NC=100).
- BUT detection_variants.py bootstraps only against run11 (hard-coded reference, line ~418). E5's
  condition (1) needs x3 minus x1. Plan: a separate zfeval/experiments/paired_vs.py that loads
  preds_<a>.npz and preds_<b>.npz and calls the SAME zfeval.metrics paired block bootstrap
  (no protocol change), cross-checked by reproducing one existing vs-run11 interval exactly.
  Build it before E5 finishes.
- 07:40 resolved: zfeval/experiments/paired_vs.py built; --check reproduces the stored vs-run11
  records bit-exactly (daptreplay_5e5_step15000, run19_avesteacher). E5 condition (1) uses it.

## 2026-10-06 12:05 — SECOND collision with usage-watch auto-resume (PID 12139, started 11:52)
- Both agents submitted E5 at 11:57:48: 39671591 (lead) and 39671592 (auto-resume). Cancelled
  39671591 while PENDING so the auto-resume agent's bookkeeping (it was mid-way through linking the
  _x3 exp dir) stays correct. 39671592 is THE E5 job.
- The lead stood down for this cycle; run20 (E4) is trained (COMPLETED 11:40, step 93750) and NOT
  yet scored by the lead -- check next wake whether the other agent scored it before starting.
- Needs you (again, now urgent): this is the 2nd duplicate submission. Undesignate this project
  from usage-watch auto-resume while the /loop lead runs.
- 12:44: auto-resume agent exited; it recorded E5 = 39671592 (correct) and E4 trained, but both of
  its run20 scoring attempts were killed mid-way (11:59 at export, 12:10 at encode 1500/3412). The
  lead re-runs score_run.sh run20_fsdonly. Pattern: the auto-resume agent's long background jobs
  die with it, so it should never start scoring.

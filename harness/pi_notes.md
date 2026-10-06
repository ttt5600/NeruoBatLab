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

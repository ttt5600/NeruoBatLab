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

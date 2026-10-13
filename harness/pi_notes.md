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

## 2026-10-06 18:40 — process correction (critic on 061)
- I had been writing each draft finding as `status: confirmed` and running build.py BEFORE the
  critic, so CONTEXT.md briefly listed unreviewed drafts as confirmed (056-061). PI.md says confirmed
  only after step 5. From now on: write drafts as `status: open`, flip to confirmed after applying
  the critic's verdict. 056-061 all ended ACCEPT-WITH-CHANGES with changes applied.

## 2026-10-06 21:10 — lead loop stopped (idle)
- E1-E6 done (findings 056-061); queue empty; E7 (run16 seed 3) and E8 (run19 seed 2) are
  needs-human pending the seed decision. Third idle wake in a row and the BRC cert expires 21:20,
  so the /loop lead stopped. Restart with `/loop follow harness/PI.md` after deciding on seeds.
- Open for the user: seeds (E7/E8); usage-watch auto-resume collisions (2 so far).

## 2026-10-11 13:05 — E10 (BEANS) started by hand; protocol problem before any download
- AVES's published BEANS row (arXiv 2210.14493 Table 2: wtkn 0.879, bat 0.748, cbi 0.598, hbdb 0.810,
  dogs 0.950, dcase 0.392, enab 0.555, hiceas 0.629, rfcx 0.130, gib 0.284, esc 0.773, sc 0.964) is
  FINE-TUNED: whole network except the CNN, Adam, batch 32, lr swept {1e-5, 5e-5, 1e-4} x 50 epochs,
  best on valid. E10 was approved as "no training", and its success criterion (reproduce AVES's
  published numbers) cannot be met by a frozen probe. Needs the user's call on protocol.
- Only aves-bio is published in Table 2; no per-task numbers for core/nonbio/all or BirdAVES there.
- Download sizes (HEAD Content-Length, nothing downloaded): bats 5.2 GB, watkins 0.9, dogs 0.9,
  humbugdb ~3.5 (3 parts found), dcase 1.9, enabirds 1.0, hiceas 1.4, hainan gibbons 11.9,
  speech commands 2.4, esc50 (git clone, small). cbi and rfcx are Kaggle competitions: need the
  user's Kaggle account + accepting each competition's rules. BEANS README says ~300 GB with the
  processed copies; Mac has 252 GB free.
- Licences: bats (Prat, Taub, Pratt, Yovel 2017, figshare 10.6084/m9.figshare.c.3666502) CC BY 4.0
  per the Scientific Data article; figshare record itself not checked. All others NOT YET CHECKED;
  BEANS's README lists none (BEANS code is MIT).

## 2026-10-11 20:45 — E10 progress; detection moved to Savio
- Done on the Mac (frozen probe, analysis/beans_<task>.json): bats, watkins, dogs, humbugdb, esc50.
  Every split CSV and audio file of all 10 non-Kaggle tasks matched BEANS's MD5s.
- Detection (hiceas, enabirds, dcase, hainan-gibbons) would hold the Mac GPU ~a day; moved to Savio
  as job 39836981 (beans_eval/zfeval/slurm/beans_detect.sh, 1 L40 lowprio, --requeue, cached per
  model x split). beans_detect.py checked against BEANS's own RecognitionDataset (windows, offsets,
  labels identical on valid+test of all 4) and metrics.AveragePrecision (identical to 6 dp).
- hiceas on the Mac got 8/11 models before it was stopped; keep that JSON to compare MPS vs CUDA.
- Savio login node 2026-10-11: its Lustre client lost OST006c ("Cannot send after transport endpoint
  shutdown"); hubert_env's python gave EIO there but reads fine from dtn00. Jobs use
  PYTHONNOUSERSITE=1. If E7/E8 fail on start, check this first.
- speech-commands (105k files) not run: too many files for Savio scratch; Mac run is ~11 h. Open.

## 2026-10-12 11:30 -- LOOP GOAL from the user (standing until met)
"Keep running things until we see improvement in call type and vocal detection; the eval is call type /
vocal detection; use the harness." Stop rule (mine, written before results): an experiment whose 11-class
call type beats the best AVES-base it is compared with (board protocol, bird-bootstrap interval excluding
zero) with ZF detection AUC not significantly worse, REPRODUCED on a second seed. Until then: score every
run, write the finding, submit the next approved entry, propose the next one from the evidence.
Evidence driving the queue (10-11, scripts zfeval/experiments/calltype_{cluster_by_type,abx,controls}.py,
JSONs in analysis/): run16 and AVES tie (ABX 0.1385 vs 0.139-0.146) but are complementary (prob-average
+0.016-0.019 over AVES, resolved; same-recipe pairs +0.002/+0.007 not); DAPT replay x3 drifted toward our
profile (trained on run11 labels). -> E17 (DAPT + AVES targets) and E18 (same, lr 2e-4), user-approved.
If E17/E18 do not beat AVES, next candidates: two-teacher targets (k-means on concatenated run16+AVES
layer features), distil the run16+AVES ensemble.
Savio cert expired 08:58; E17/E18 scripts built + on the push list, not yet submitted (need sv login).
E17/E18 go beyond max_concurrent (E7/E8 still pending since 10-08/09) on the user's direct approval.

## 2026-10-12 11:35 -- loop wake: E17/E18 submitted (39844771/39844772); BEANS detection done
- lab train's import check runs on the login node, whose python still gives EIO; ran the same check on
  dtn (imports OK, flags present) and sbatched from ~/jobs with a ledger line. It caught a real bug first:
  PYTHONNOUSERSITE=1 (copied from beans_detect.sh) would have hidden librosa -- removed.
- BEANS detection job 39836981 COMPLETED rc=0 (4 tasks). JSONs copied to analysis/ (Mac's 8-model hiceas
  kept as beans_hiceas_mac.json). AVES family ahead of run16 on hiceas/enabirds/dcase, several resolved;
  run19 again closes part (hiceas +0.138*, dcase +0.054*); gibbons mAPs tiny. NEXT: E10 finding (critic).

## 2026-10-12 13:35 -- E17/E18 done (finding 064 confirmed after critic); E19/E20/E21 submitted
- Critic overturned my framing: E18 "beats AVES on detection" is the replay-DAPT effect (058), not the targets;
  told the user. E19 amended before its log/scores were read (label NMI gate, L3, attribution via E21 control,
  mechanism = difference of ensemble gains). E21 (--only-b) submitted 39850689.
- TODO (resumable): republish docs/model_board.html (rebuilt locally) and the roadmap scoreboard artifacts; finding
  063 needs ours-vs-AVES BEANS intervals (scripts now merge reruns; classification on the Mac, detection on Savio).

## 2026-10-12 15:50 -- AGENT step 7 (task-list doc) checked
- Doc 1e00135c (Zebra Finch Encoder Tasks, rev 29): no row changed state (E10 and DAPT rounds 3-4 are not rows; "change nothing else"), so no edit.
- BEANS detection rerun vs AVES (job 39851650, 2 h limit) on its last task; E14 running; E22/E23/E24 queued.
- 2026-10-12 17:0x E14 scored (job 39851972): call type 0.8341 vs BirdAVES 0.8520, -0.0179 [-0.0343, -0.0023]
  (also at L3); detection not distinguishable from BirdAVES or E19; BirdPark pre-committed gain is an arm artefact
  (4-5/12 arms). Finding 066 confirmed after critic (ACCEPT-WITH-CHANGES, all applied). E14 done. Stop rule not met.
  Waiting on E23 (~17:05), E22 (~18:10), E24 (pending).
- 2026-10-12 17:25 E23 scored (job 39852046): call type 0.8564 L1, best DAPT so far but vs AVES +0.0111
  [-0.0060, +0.0274] n.d.; vs E19 (same recipe, seed 1) +0.0123 [+0.0016, +0.0241] RESOLVED -> seed-to-seed spread
  exceeds the bird-bootstrap interval (cf. 057). Stop rule not met. Detection at the AVES-init DAPT level (ZF AUC
  +0.0032 vs AVES, 12/12 arms). Its pre-committed arm (native_L1) is a weak BirdPark arm: pre-committed BP AUC -0.0672
  vs AVES but 12-arm AP median 0.8533 vs 0.8115 -- arm fragility again, do not quote either direction. E23 criterion
  (E23-E24) waits on E24 (pending). Finding to be written with E24.

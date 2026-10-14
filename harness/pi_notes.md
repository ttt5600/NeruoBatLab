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
- 2026-10-12 18:50 E24 scored (job 39852047). E23 criterion half-passed -> not met (pre-committed contrast +0.0002
  n.d.; 11/12 arms). Finding 067 confirmed after critic (ACCEPT-WITH-CHANGES, applied); 065 moved confirmed -> open
  (replication failed). Key lesson: arm contrast on call type flips sign between seeds; bird bootstrap holds the model
  fixed. E22 preempted ~18:00, resumed from last.ckpt at step ~25.7k, ETA ~19:15. Critic flagged 052 (left confirmed
  after 057) for the same review -- TODO next idle wake. Next idle: E11 (label-free score column).
- 2026-10-12 19:10 052 reviewed (critic flag from 067): run16 seed 2 n.d. from run15/run11 -> 052 moved to open with a
  replication caveat; E7 (run16 seed 3, job 39732808) STARTED running ~19:01 after 4 days pending; it is the tiebreaker.
- 2026-10-12 19:50 E22 scored (job 39851973): call type 0.8538 L3, n.d. from E19 and AVES -> criterion not met.
  Finding 068 confirmed after critic (ACCEPT-WITH-CHANGES, applied). E22 done.
  TODO (resume waste): last.ckpt is saved only at epoch end, so each lowprio preemption restarts from the last epoch
  boundary (E22 lost >=6k steps over 3 preemptions). Fix candidate: save_last on the every-2500-step ModelCheckpoint
  (or resume from the newest stepstep=*.ckpt). Code change in the training script, not flags -> propose, don't
  self-approve; check how run16's resumes behaved first.
- 2026-10-12 20:25 E11 done (idle analysis): label-free bird-held-out stability for 29 models; finding 069 (open)
  after critic (ACCEPT-WITH-CHANGES, applied). No model beats AVES on it; weak agreement with call type among close
  models, +0.90 on 050's five encoders; call type's own seed reliability is ~0.44. Added to score_run.sh (--only TAG)
  and the board (Label-free k=30 column). Optional later: score both directions per halving.
- 2026-10-12 22:20 Controlled-variables audit (user asked). AVES-bio cfg read from its checkpoint: mask 0.8/10
  (ours 0.4/3; measured 54% vs 34% of frames masked), lr 2e-4, warmup 32k/100k, 700 s/update, iteration-2 targets,
  seed 1337. Same as ours: size, input, optimiser, dropout, k=200. run17 already matched 86% of AVES compute.
  User approved E25-E27 and closing variables -> E28 (DAPT + AVES masking, 2 seeds) and E29 (AVES recipe from
  scratch, 2 seeds, judged on BEANS) pre-registered and pushed (eb788ae). train.py --mask-prob/--mask-length added
  (defaults unchanged), committed in pytorchAudio (c8e02f58, not yet pushed to Lab_Remote), code on Savio.
  dtn import check: first attempt hung (train.py --help slow/hung on dtn; stray procs killed, dtn master reset);
  relaunched detached -> ~/jobs/maskcheck.out. Submit E28/E29 after it passes.
  E25 started: AVES-nonbio (360 h, no animal audio) downloaded; call type 0.8470 = AVES-bio (+0.0018 [-0.0154,
  +0.0189]), beats run11 +0.0352 [+0.0110, +0.0621] -> animal data buys nothing on ZF call type. Detection running.
- 2026-10-12 22:45 E7 scored -> finding 070 (confirmed after critic): run16 seeds 0.8379/0.8224/0.8417; 052 holds at the
  best layer on 2/3 seeds, at L3 on 1/3; 052 stays open. E25 batch 1 (HF, same size, speech only): hubert-ls960
  0.8288 (n.d. from run16 -0.0091 [-0.0340, +0.0134] and AVES -0.0164 [-0.0357, +0.0037]), wavlm+ 0.8262 (below AVES
  resolved), data2vec 0.8118, wav2vec2 0.8022; all peak at L0-L1. AVES-nonbio = AVES-bio on call type and detection.
  E28 running (ETA 23:07), E29 pending. BRC cert expires 23:12 -> notified user (sv login).
- 2026-10-12 23:50 E25 batch 2: AST 0.8338 (n.d. vs AVES/run16). WHISPER-BASE ENCODER 0.8643 at L0 (20.6M params):
  above aves-base-bio +0.0191 [+0.0042, +0.0370] and run16 +0.0264 [+0.0048, +0.0489], resolved; NOT above
  birdaves-biox-large (+0.0091 [-0.0098, +0.0292]) -> E25 criterion not met. Baselines: duration-only 0.393, Whisper's
  own log-mel mean+std 0.667, +duration 0.723 -> not an artefact of length/padding. Replication (tiny, small) running.
  Detection for HF models is not wired (detection_variants handles torchaudio/DAPT only) -> TODO before any claim.
  E28a: 0.8417, vs E18 -0.0085 [-0.0219, +0.0025] n.d. E28b scoring.
- 2026-10-13 01:40 Night status: E25 done (072, open; Whisper = point-estimate leader), E28 done (071), E7 done (070).
  E29 x2 pending (4-GPU lowprio). E30-E32 proposed, awaiting user. E26 design drafted in experiments.yaml (not
  pre-registered; needs teacher features on Savio GPU and two user questions: donor source, dose). E27 needs a
  large-model branch in lightning_modules (only hubert_pretrain_base is built) -- not started.
- 2026-10-13 10:03 Digest written (harness/digest/2026-10-13.md). E29 x2 + E8 still PENDING (Priority). Task-list doc: no state changes. c8e02f58 is on Lab_Remote (contained in updated-hyperparams). Waiting on user: E30-E32, E26 donor/dose, resume fix.
- 2026-10-13 12:45 E25 detection for same-size speech SSL (idle unit, within E25's approval): e25_hf_frames.py + hf-ssl branch in detection_variants (HF HuBERT-ls960 frames == torchaudio HUBERT_BASE exactly). Pre-committed ZF AUC: hubert 0.9646, wav2vec2 0.9628, wavlm+ 0.9658, data2vec 0.9649; all below aves-base-bio at 12/12 arms (resolved for hubert, wav2vec2), all resolved below run16 (paired_vs.json). Added to 072 evidence (stays open); dashboard models dataset -> asset 8fa27184 (doc v5). Queue: E8 est. 02:04, E29 est. 14:05 tomorrow; cert to 20:59.
- 2026-10-13 16:46 E8 (run19 seed 2, job 39781626) RUNNING since 15:54 on n0387, resumed from epoch=5-step=15039 (preempted 10-10); step 22557 at 16:38, ~2,500 steps / 14 min -> ETA ~23:20 for 93,750. E29 est. 02:04 / 03:54 tomorrow. Cert expires 20:59 -> user must sv login to score E8.
- 2026-10-13 18:48 E8 PREEMPTED 17:55 (2nd time; last.ckpt = epoch 13 step 35092), requeued, est. 18:02 Oct 14. E29a (job 39859251) RUNNING since 18:02 on n0387: script passes --mask-prob 0.8 --mask-length 10 (no hparams.yaml written); 1,254 steps/epoch, ~14 min/epoch -> ~18.5 h for 100k, ETA ~12:40 Oct 14 (my earlier ~12 h estimate was wrong). Step 3761 loss 3.64. E29b est. 02:04. Nothing to score tonight.
- 2026-10-13 20:51 E29b (job 39859252, seed 2) RUNNING since 20:49 on n0388; ETA ~15:20 Oct 14 if not preempted. E29a step 16295 (on pace, ETA ~12:40). E8 requeued (est. 18:02 Oct 14). Cert expires 20:59 -> Savio checks pause until user sv login.
- 2026-10-14 00:20 Cert renewed by user (valid 10-13 20:27 -> 10-14 08:27, verified locally). E29a step 36352, E29b step 20055 at 00:15 (~97 steps/min; ETA ~11:10 / ~14:00) -> another sv login needed before scoring. Peer session notes branch harness-upgrades (9b74023) unmerged; still following current PI.md.
- 2026-10-14 04:40 E29 scoring plan (prepared; DRY=1 score_run.sh OK for run29_avesrecipe). Per seed, after its final
  step-100000 checkpoint exists (A: tag run29_avesrecipe, exp temp_train_run29_avesrecipe, job 39859251; B: tag
  run29_avesrecipe_s2, exp temp_train_run29_avesrecipe_s2, job 39859252):
  1. SCORE_HOST=savio-dtn bash harness/score_run.sh <tag> /global/scratch/users/jonathanswang/<exp> 200 <job>
  2. Mac: zfeval/experiments/beans_probe.py --task {bats,esc50} --models <tag> aves-base-bio run16_compute4x --baseline aves-base-bio
  3. Savio: cd beans_eval && sbatch --export=ALL,TASKS="hiceas enabirds dcase",MODELS="<tag> aves-base-bio run16_compute4x",BASELINE=aves-base-bio zfeval/slurm/beans_detect.sh
     (MODELS env added to beans_detect.sh 04:35, pushed; beans_probe.py on Savio updated to the merge-safe version)
  E29 criterion: <tag> vs_aves-base-bio not resolved negative on >=4 of the 5 tasks, both seeds.
  E29a step 62675 at 04:15 (ETA ~10:15), E29b 47631. Cert expires 08:27.
- 2026-10-14 10:10 E29a (39859251) COMPLETED 09:53 rc 0, max_steps=100000 reached; step-100000 ckpt NOT saved (not in top-5 by train_loss), so scoring uses epoch=78-step=99030 (last 970 steps, ~1%, at near-zero lr, absent). E8 restarted (RUNNING on n0387). Cert expired 08:27: dtn refuses new connections; login node works over the open master and its python now imports torch fine -> score_run.sh with default SCORE_HOST=savio-login, started 10:10.
- 2026-10-14 10:35 E29a scored (score_run.sh rc 0; export verify max|diff| 0). Call type 0.8300 at L2 (L3 0.8200;
  run29_avesrecipe_calltype.json); calltype_pairs.json observed: vs aves-base-bio -0.0152 [-0.0388, +0.0072], vs run16
  -0.0079 [-0.0217, +0.0058], vs run17 -0.0021 [-0.0201, +0.0164], all n.d. Detection precommit native_L0 ZF AUC 0.9713;
  paired_vs.json: vs aves-base-bio +0.0039 [+0.0010, +0.0061] (ahead 9/12 arms), vs run16 -0.0005 [-0.0016, +0.0015]
  (3/12), vs run17 -0.0003 [-0.0021, +0.0010] (2/12). BirdPark AP 12-arm median 0.7598 (aves 0.8115, run16 0.7864,
  run17 0.7713). Label-free k30 0.4842 (aves 0.4783). BEANS: detection job 39913752 (hiceas enabirds dcase, vs
  aves-base-bio) submitted; bats + esc50 probes running on the Mac. E29b's BEANS job must go after 39913752 (same JSONs).
- 2026-10-14 10:55 E29a BEANS classification (beans_bats.json, beans_esc50.json; clip bootstrap, observed): bats 0.6615 (L0) vs aves-base-bio 0.6860 -> -0.0245 [-0.0455, -0.0040] RESOLVED NEGATIVE; vs run16 (0.6145) +0.0470 [+0.0260, +0.0670]. esc50 0.6975 (L6) vs aves 0.7150 -> -0.0175 [-0.0600, +0.0250] n.d.; vs run16 (0.6425) +0.0550 [+0.0150, +0.0950]. E29 criterion so far on seed 1337: 1 of 5 resolved negative (bats); needs hiceas/enabirds/dcase (job 39913752) all not-negative. Note run29 differs from run16 in several knobs (accum, mask, lr, schedule); run17 has no BEANS score.
- 2026-10-14 11:40 E29a BEANS detection (job 39913752 rc 0, beans_detect_39913752.log; observed mAP diff, clip-block bootstrap): vs aves-base-bio hiceas -0.0994 [-0.1757, -0.0234], enabirds -0.0681 [-0.0831, -0.0410], dcase -0.0603 [-0.0892, -0.0022], all RESOLVED NEGATIVE. With bats: 4 of 5 resolved negative on seed 1337 -> E29 pre-registered criterion FAILS (needed >=4 of 5 not negative on BOTH seeds). E29b COMPLETED 11:19 (epoch=79-step=100000.ckpt saved); scoring started 11:36. E8 running again (n0388). Next: E29b BEANS (vs aves) + both seeds vs run16 on the 3 detection tasks (one job after another, same JSONs); sync beans_{hiceas,enabirds,dcase}.json to the Mac.
- 2026-10-14 12:00 E29b scored (export verify 0 diff; final ckpt epoch=79-step=100000). Call type 0.8368 at L1 (L3 0.8294); calltype_pairs.json observed: vs aves-base-bio -0.0085 [-0.0313, +0.0121], vs run16 -0.0012 [-0.0204, +0.0167], vs run17 +0.0047 [-0.0113, +0.0224], vs E29a +0.0067 [-0.0082, +0.0218], all n.d. Detection precommit native_L3 ZF AUC 0.9707; paired_vs: vs aves +0.0034 [+0.0015, +0.0055], vs run16 -0.0011 [-0.0019, +0.0015], vs run17 -0.0009 [-0.0021, +0.0007], vs E29a -0.0005. Label-free k30 0.4953. BEANS: Savio j1 39914954 (E29b vs aves), j2 39914955 afterany (both seeds vs run16); bats/esc50 running on Mac.
- 2026-10-14 12:15 E29b BEANS classification: bats 0.6460 (L2) vs aves -0.0400 [-0.0620, -0.0190] RESOLVED NEG, vs run16 +0.0315 [+0.0100, +0.0520]; esc50 0.6750 (L4) vs aves -0.0400 [-0.0850, +0.0026] n.d., vs run16 +0.0325 [-0.0100, +0.0725] n.d. Savio j1 39914954 running, j2 39914955 waits on it.
- 2026-10-14 12:40 BEANS detection vs run16 (job 39914955): hiceas E29a +0.0231 [-0.0187, +0.0722], E29b +0.0158 [-0.0409, +0.0689]; enabirds -0.0090 [-0.0195, +0.0113], -0.0002 [-0.0136, +0.0144]; dcase +0.0127 [-0.0090, +0.0391], +0.0110 [-0.0168, +0.0431]; all n.d. E29b vs aves (job 39914954): hiceas -0.1066, enabirds -0.0593, dcase -0.0620, all resolved negative. Both seeds 4/5 resolved negative -> E29 fails. Finding 073 drafted, critic running. Dashboard: E29 rows added (build_comparison META), assets models 24d210c3 / beans 6df2efcd.

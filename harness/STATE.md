# Lead state -- OVERWRITE every wake (<= 40 lines). Volatile facts only; history goes to pi_notes.md.
Updated: 2026-10-15 22:14 (lead: session_01EGDWEvYFd6zSmWQfUugJnH; written as the resume point before a /clear)

## Where we stand (goal: beat AVES-bio on ZF call type AND ZF detection, 2 seeds; real aim: label-free bat units)
- Call type: nothing of ours is resolved above aves-base-bio (0.8453). Best points: E23 DAPT 0.8564, run16 0.8379,
  E29 0.8300/0.8368; Whisper-small 0.8599 (n.d.). Same-recipe seeds move call type by 0.01-0.02 (070).
- ZF detection AUC: ours resolved above AVES (run16 0.9718, E29 0.9713/0.9707); Whisper-small 0.9733 is top (072).
- BEANS (073): AVES-bio leads us resolved on 4-5 of 9 tasks; giving our code AVES's masking/lr/warmup/length (E29)
  lifts bats/esc50 over run16 but stays below AVES on 4/5 on both seeds. Never matched: AVES feature penalty (10,
  ours 0) and clip length (ours fixed 20 s, AVES <=15.6 s crops). Leading suspect: targets -- run19 (AVES's own
  clusters) tied AVES on 7/8 BEANS tasks, one seed (063); E8 is its second seed.
- Bats (074): ~91% of bat energy is above 8 kHz; slowing 4x raises BEANS bats run16 +0.080, AVES +0.067 (mostly the
  stretch itself); label-free n.d. -> E16 criterion not met. Options doc: docs/t7_bat_sample_rate_options.md.

## In flight
- E8 run19 seed 2: job 39945354, resumes from epoch=34-step=87734 (~6k of 93,750 left). PENDING, est. 2026-10-17 05:06.
- E9 run20 seed 2 (FSD50K only): job 39945376. PENDING, est. 2026-10-17 17:10. Nothing running on the Mac.

## Next action (in order)
1. When E8 finishes: score_run.sh run19_seed2 /global/scratch/users/jonathanswang/temp_train_run19_seed2 200 39945354
   (newest epoch ckpt if step 93750 is missing); then BEANS for run19_seed2 vs aves-base-bio (recipe: pi_notes
   2026-10-14 04:40). This is the replication test of "targets close the BEANS gap".
2. When E9 finishes: score it the same way (E9's question in experiments.yaml).
3. Act on the user's decisions below; build + pre-register + critic each before submitting.
4. Idle (<= ~2 h/day): PI.md backlog item 1 (roadmap section 04 figures).

## Waiting on the user
- New (073): test AVES's feature penalty (10) and/or clip cropping, one knob each on top of E29? (~15 h/seed, 2 seeds)
- Go/no-go: E30 (log-mel front end, recommended), E31 (DAPT on Whisper-small), E32 (AVES mask embedding).
- E26: donor source (curated 3412 clips vs detector-mined) and dose (~25 h). T7: which bat recordings/approach.
- Resume fix: save last.ckpt every 2,500 steps (E8 needed 6 preemptions + a REQUEUE_HOLD).
- Older: speech-commands, Kaggle for cbi/rfcx/Perch, 615 GB deletion, intern details, 174-clip label audit.

## Sync state (verified 2026-10-15 21:55-22:14; resume: /loop follow harness/PI.md)
- GitHub: project UMAP pushed (ttt5600/NeruoBatLab); pytorchAudio updated-hyperparams db5151b4 = Lab_Remote.
- Savio: all 121 scp_to_savio.sh files identical; loss/ identical; BEANS eval code (beans_eval/) synced.
- Dashboard https://claude.ai/artifact/RRGaF3dkeK3mNd3gjX5GQ7: 43 models, 0 mismatches vs source JSONs.

## Clocks
- BRC cert EXPIRED 20:58 Oct 15: `sv login` before E8 starts (~Oct 17 05:00). Next digest after 09:00 Oct 16.
- /clear at ~22:20 Oct 15: a heartbeat from session_01EGDWEvYFd6zSmWQfUugJnH is THIS lead before the clear, not another lead.

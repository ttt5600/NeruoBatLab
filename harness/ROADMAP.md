# Zebra finch encoder roadmap (canonical)

Updated 2026-11-05. This file replaces docs/roadmap.html (frozen at Oct 17, superseded). The same roadmap, with charts, is the
"Roadmap" section of the report doc https://claude.ai/artifact/Q3Bf46TNUyRHh8cZZEdH1M. Keep the two in step.
On resume the user wants three things first: **where we are, possible leads, next steps.** Give them from this file
(after checking the jobs under "In flight"), in under 15 lines, then ask which next step to take.

Goal (hub goal zf): specialise a HuBERT-architecture encoder to zebra finch, judged on 11-class call type
(CPU board, leave-birds-out). BEANS and other species are out of scope (user, 2026-10-15).

## Where we are
- **Vanilla HuBERT ties our best recipes.** The stock recipe on our 116 h of zebra finch audio, with lr 1e-4 (the stock 5e-4
  collapses at our batch size, as run5 did), scores 0.8356 vs run16 0.8353 (E36, seed 1 of 2). No recipe arm of ours is
  distinguishable from it: run16 -0.0035, run19 (AVES labels) +0.0147, E13 (BirdAVES-large labels) +0.0170, E30 -0.0094
  (seed_mean_pairs_cpu*.json, "<arm> | vanilla_lr1e4"). The one change that clearly mattered was the learning rate.
- **The pipeline does learn from its labels.** Shuffled labels collapse call type to 0.6334, -0.2019 [-0.2364, -0.1664] vs
  run16 (finding 088). So the near-ties are about the task, not a broken pipeline.
- **The test is the bottleneck.** Untrained networks score 0.588-0.612, every pretrained encoder 0.809-0.865, and a new seed
  moves a model about 0.02. The arm-level test resolves only differences of about 0.03 (findings 079, 087-089).
- **Point-estimate leaders:** downloads: Whisper base 0.8646, Whisper small 0.8599, BirdAVES bioxn-large 0.8590. Ours, from
  scratch: run19 seed 2 0.8546, E13 seed 2 0.8538. Nothing we trained beats the best download.
- **Closed this round:** E30 log-mel front end (087, kill fired), E13 stronger teacher (089, not met, kill not fired).
- Detection: every pretrained model reaches ZF AUC ~0.97; the shuffled-label model 0.891.

## In flight (paid A40 on fc_birdpow, waiting on a QoS cap shared with other users; check with `sw status` or squeue)
- 40654081 E36 vanilla lr 1e-4 seed 2 (runs first), then 40654080 E36 stock seed 2.
- 40654077 E37 random-projection labels seed 2 (seed 1 scored 0.8447, not distinguishable from run16 or run23).

## Next steps, in order
0. Savio first: the BRC certificate expired 2026-11-05 13:31; the user runs `sv login`, then `sw status` / squeue.
1. Score 40654081: `SCORE_HOST=savio-dtn bash harness/score_run.sh run31_vanilla_lr1e4_s2
   /global/scratch/users/jonathanswang/temp_train_run31_vanilla_lr1e4_s2 100 40654081`. Rerun the arm contrasts against vanilla with both seeds
   (seed_mean_pairs_cpu.py vanilla_lr1e4=run31_vanilla_lr1e4,run31_vanilla_lr1e4_s2, best and --layer 3), write the E36
   finding through the critic, and redraw the doc's forest plot (widget a557a9c6-097f): rows from `python zfeval/experiments/forest_vs_ref.py
   run31_vanilla_lr1e4` (single pairs), plus the arm-level table under it.
2. Score 40654077 (`... run37_randproj_s2 /global/scratch/users/jonathanswang/temp_train_run37_randproj_s2 200 40654077`) and judge randproj by its frozen E37 rule (2 seeds vs run16's 3 and vs run23). If it ties, the k-means
   step is not what makes our targets work.
3. Score 40654080 (`... run31_vanilla_s2 .../temp_train_run31_vanilla_s2 100 40654080`; stock seed 2) to confirm the stock rate fails at our batch size.
4. Clean up Savio (needs the user's yes): E13 relabel features (~165 GB, combined_teacher_bioxnl12_eval/data/hubert_12/feat)
   and the 14 GB BEANS copy.
5. With E36 settled, the user picks the next phase from the decisions below.

## Decisions waiting on the user
- **E36b:** vanilla HuBERT on the mixed 224 h corpus, the fair baseline for our mixed-corpus models (proposed 2026-10-29).
- **E31:** continue training Whisper small on zebra finch: can the strongest download be specialised? (proposed)
- **E35:** eval-mode AVES labels, to separate the teacher-dropout bug (085) from the teacher (not yet proposed).
- **E32** (AVES mask embedding restored in DAPT, proposed); **run23 seed 2** (MFCC-label tie, critic asked for it);
  **E15** (bigger general-audio corpus, approved, corpus not chosen); **E26** (synthetic mixtures, approved, design open);
  **E33** voice-invariant DAPT (not recommended at +/-2 semitones).

## Leads (ideas, not pre-registered; check the findings before proposing)
- **A sharper test.** Recipe effects under 0.03 are invisible, so more training runs cannot rank recipes. Options: a power
  analysis (how many birds/clips resolve 0.01), the label audit (174 clips wrong for every model, Mac-era count: a labelling
  ceiling?), or tasks where models do differ (per-call-type clusters/ABX, individual identity, the chick holdout).
- **Where Whisper wins.** Its call-type lead over run16 is mostly Tet and Nest (finding 081). A per-type look at what our
  models confuse could point at the missing information (timing? harmonics?).
- **Decision ensemble on call type.** Averaging two encoders' probabilities helped detection (031); it is untested on call
  type for the top downloads + run19 (043 found distillation does nothing on call type). CPU only, no training.
- **AVES-family labels are the only consistent positive sign** (+0.015 to +0.021 over vanilla at every readout, run19 and E13,
  not resolved). If E36 seed 2 holds that gap, a powered test of "teacher labels vs MFCC labels" is the one recipe lead left.

## Closed lines (do not reopen without a new reason)
Log-mel front end (087) · stronger teacher among AVES-family models (089) · compute past 4x (054) · AVES's own recipe knobs
(073) · continued pretraining from AVES/BirdAVES for call type (075, 084) · iteration-2 own labels (056) · BEANS (retired 2026-10-22).

## Pointers
STATE: harness/STATE.md · notes: harness/pi_notes.md · register + frozen rules: harness/experiments.yaml · findings:
knowledge/findings/ (index knowledge/index.md) · analysis JSONs: ~/zf_labelset/zf_detection_dataset_v1/analysis/ · scorer:
harness/score_run.sh · report doc: https://claude.ai/artifact/Q3Bf46TNUyRHh8cZZEdH1M (doc id ba903e67-b3f0-47aa-9de1-fd7903adac44).

# Lead state -- OVERWRITE every wake (<= 40 lines). Volatile facts only; history goes to pi_notes.md.
Updated: 2026-11-05 (lead session_01EGDWEvYFd6zSmWQfUugJnH; the side-hustle session stood down on zf)
Next: when 40654081 (E36 lr1e-4 seed 2) finishes, score it (harness/score_run.sh, tag run31_vanilla_lr1e4_s2, 100 classes),
rerun the "<arm> | vanilla_lr1e4" contrasts with both vanilla seeds, write the E36 finding (critic), refresh Doc ba903e67.

## USER PRIORITY: ZF specialization judged on CALL TYPE; skip BEANS (data deleted locally 2026-10-22). Deliver visually.
Call type is CPU ONLY (076). Model table: `python zfeval/experiments/our_models_table.py`.
## LOOPS OFF (2026-10-21): no /loop, harness agent, escalation or auto-resume unless the user asks.

## In flight (paid A40, fc_birdpow; QOSGrpCpuLimit shared with other FCA users)
- 40654081 E36 lr1e-4 seed 2 (moved first 2026-11-05) -> 40654080 E36 stock seed 2 (afterany:40654081).
- 40654077 E37 randproj seed 2 (pending on the QoS cap). Then judge randproj by its E37 rule (arm vs run16 x3 and vs run23).
- E13 relabel feat/ on Savio (combined_teacher_bioxnl12_eval/data/hubert_12/feat, ~165 GB) is due for deletion per the
  prereg; not done -- confirm the path holds only features, then delete.

## Latest results (2026-11-05; analysis JSONs in ~/zf_labelset/zf_detection_dataset_v1/analysis)
- 088 E37 shuffle: 0.6334, -0.2019 [-0.2364,-0.1664] vs run16 -> pipeline uses its labels. Scored at step 72,709.
- 089 E13: arm +0.0023 [-0.0125,+0.0172] best, -0.0022 L3; not met, kill not fired; closed.
- E36 interim: stock lr failed the baseline rule (loss -> 4.29 = label entropy); lr1e-4 s1 0.8356 = run16 0.8353.
  Arms vs vanilla (seed_mean_pairs_cpu*.json): run16 -0.0035, run19 +0.0147, E13 +0.0170, E30 -0.0094, all n.d.
  57 models vs vanilla in seed_pairs_cpu[_L3].json; forest plot in the Doc (widget a557a9c6-097f).
- E37 randproj s1 0.8447; vs run16 arm +0.0126 [-0.0186,+0.0438]; vs run23 +0.0026 [-0.0137,+0.0208] (seed_pairs).
- 087 E30 closed. Floors: untrained 0.5876-0.6120, log-mel 0.6679.

## Waiting on the user
- Which lead session continues (peer agreed zf is mine). $150 question answered: election-night Kalshi bot (quant session).
- Optional: E36b (vanilla on mixed corpus), E31, lab share read-only, delete 14 GB BEANS on Savio, demo re-record.

## Traps
- score_run.sh now prefers $CKDIR/final.ckpt and WARNS when the newest epoch=*-step=* is short of max_steps (2026-11-05).
  Runs before pytorchAudio 1bc6d7ec have no final.ckpt; report their scored step.
- Logged masked accuracy is a running total since process start; it cannot show the end state of a collapsed run.
- Teacher must be .eval(). Single seed pairs ignore seed noise (sd ~0.011); a one-seed arm's seed term is borrowed (df 1-2).
- calltype_split_sensitivity --only merges. Local tests: my_audio_env python (torch 2.2.2 = Savio).

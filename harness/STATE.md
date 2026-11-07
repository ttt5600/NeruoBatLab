# Lead state -- OVERWRITE every wake (<= 40 lines). Volatile facts only; history goes to pi_notes.md.
Updated: 2026-11-05 evening (lead session_01EGDWEvYFd6zSmWQfUugJnH; handoff before the user's /clear)
ON RESUME: read harness/ROADMAP.md and give the user, in under 15 lines: where we are, possible leads, next steps.
Next: jobs 40654077/40654081 est. start 2026-11-13 (A40 queue, Priority); when done follow
ROADMAP.md "Next steps" 1-3 (score finished jobs, E36 finding with both vanilla seeds, E37 randproj verdict).

## USER PRIORITY: ZF specialization judged on CALL TYPE; skip BEANS (data deleted locally 2026-10-22). Deliver visually.
Call type is CPU ONLY (076). Model table: `python zfeval/experiments/our_models_table.py`.
## LOOPS OFF (2026-10-21): no /loop, harness agent, escalation or auto-resume unless the user asks.

## In flight (paid A40, fc_birdpow; pending on a QoS cap shared with other FCA users, est. ~Nov 7)
- 40654081 E36 lr1e-4 seed 2 (EXP temp_train_run31_vanilla_lr1e4_s2) -> 40654080 E36 stock seed 2 (temp_train_run31_vanilla_s2).
- 40654077 E37 randproj seed 2 (temp_train_run37_randproj_s2). Exact score commands in ROADMAP.md.

## Latest results (analysis JSONs in ~/zf_labelset/zf_detection_dataset_v1/analysis)
- 088 E37 shuffle: 0.6334, -0.2019 [-0.2364,-0.1664] vs run16 -> the pipeline uses its labels (scored at step 72,709).
- 089 E13 bioxn-large labels vs run19: +0.0023 best / -0.0022 L3, not met, kill not fired; closed.
- E36 interim (no finding yet): stock lr collapsed to the label prior; vanilla lr1e-4 seed 1 0.8356 = run16 0.8353. Arms vs
  vanilla (seed_mean_pairs_cpu*.json "<arm> | vanilla_lr1e4"): run16 -0.0035, run19 +0.0147, E13 +0.0170, E30 -0.0094, all n.d.
  57 models vs vanilla: forest_vs_run31_vanilla_lr1e4.json (zfeval/experiments/forest_vs_ref.py); Doc widget a557a9c6-097f.
- E37 randproj s1 0.8447: vs run16 arm +0.0126 [-0.0186,+0.0438]; vs run23 +0.0026 [-0.0137,+0.0208].
- Per call type (calltype_per_class_cpu.json, 34 models, "vs_vanilla"): Tet carries the model gaps; Doc heatmap 160dd46e-e837.
- Roadmap: harness/ROADMAP.md (canonical) = Doc ba903e67 section "Roadmap" (https://claude.ai/artifact/Q3Bf46TNUyRHh8cZZEdH1M).
  docs/roadmap.html (artifact K2NJhuv7hqnZj9FVBbwRo7) is frozen at Oct 17 and marked superseded.

## Waiting on the user
- `sv login`. Yes/no to deleting E13 relabel feat (~165 GB) and the 14 GB BEANS copy on Savio.
- Next phase after E36: E36b, E31, E35, E32, run23 seed 2 (ROADMAP.md "Decisions"). Which lead session continues.

## Authorizations (unchanged)
- fc_birdpow FCA for E36/E37 only ("as long as we're not chewing through it"); E13/E30-type runs on free savio_lowprio.

## Traps
- score_run.sh prefers $CKDIR/final.ckpt and WARNS when the newest epoch=*-step=* is short of max_steps (2026-11-05).
  Runs before pytorchAudio 1bc6d7ec have no final.ckpt; report their scored step. 1bc6d7ec is on Savio and GitHub.
- Logged masked accuracy is a running total since process start; it cannot show the end state of a collapsed run.
- Single seed pairs ignore seed noise (sd ~0.011); a one-seed arm's seed term is borrowed (df 1-2), so optimistic.
- Teacher must be .eval(). calltype_split_sensitivity --only merges. Local tests: my_audio_env python (torch 2.2.2).
- savio_artifacts/analysis/*.json are scorer copies left uncommitted on purpose; the canonical JSONs live in ~/zf_labelset/.../analysis.

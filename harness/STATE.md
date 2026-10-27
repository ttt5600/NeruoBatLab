# Lead state -- OVERWRITE every wake (<= 40 lines). Volatile facts only; history goes to pi_notes.md.
Updated: 2026-10-27 (lead session_01EGDWEvYFd6zSmWQfUugJnH; hub lease released)
Next: once the user picks E13's teacher and runs `sv login`: push (pytorchAudio/scp_to_savio.sh + scp
release/zf_hubert_run11/export_weights.py to Savio ~), import-check build_teacher on savio-dtn, then sbatch
slurm/relabel_teacher_combined.sh (TAG/LAYER/CKPT[/TEACHER_CONFIG] per the choice) and two
slurm/train_teacher_labels.sh with --dependency=afterok (SEED=2 on the second); ledger rows in ~/jobs/ledger.tsv.

## USER PRIORITY: ZF specialization judged on CALL TYPE; skip BEANS (data deleted locally 2026-10-22). Deliver visually.
Call type is CPU ONLY (076). Model table: `python zfeval/experiments/our_models_table.py`.
## LOOPS OFF (2026-10-21): no /loop, harness agent, escalation or auto-resume unless the user asks.

## In flight: nothing on Savio. Cert expired 2026-10-26 21:33 -> `sv login` before any Savio work.

## Latest results (2026-10-27)
- 085: build_teacher never called .eval() -> every model-teacher label set (run12/13/18/19/22, DAPT E17-E28, E14, E27)
  read with dropout 0.1; 17% of L6 labels move at fixed codebook, but partition AMI 0.578 ~= k-means seed refit 0.585.
  Fixed + tests (uncommitted until this handoff; NOT on Savio). analysis/teacher_dropout_diagnostic.json, _replication.json
- 086: call-type alignment of rebuilt labels fails validation (gets AVES>spectrogram student contrast backwards);
  not a layer-chooser. analysis/e13_teacher_proxy.json
- Arm contrast tool zfeval/experiments/seed_mean_pairs_cpu.py (birds + seeds interval): run19 (2) - run16 (3) = +0.0182
  best / +0.0209 L3, NOT resolved [-0.0076,+0.0439] / [-0.0051,+0.0469]; 6/6 seed pairs positive. seed_mean_pairs_cpu*.json
- E30 log-mel front end built + tested (utils/logmel_frontend.py, --frontend logmel, export/scoring bit-identical,
  learnable: tests/_artifacts/logmel_overfit_compare_relaxed.json). Script slurm/train_run30_logmel.sh.

## Waiting on the user
- E13 teacher: (A) bioxn-large L12 or (B) biox-base L6 (experiments.yaml E13 prereg_2026_10_27). Power poor either way.
- E30 go/no-go (~12 GPU-h lowprio; passes only if log-mel is worth ~+0.03). E35 (eval-mode AVES control) not proposed.
- Older: E26 design, E15 corpus, E33 (rec no), 10-split headline, label audit; Google Doc 123rAGhW T10/T11 (ask first).
- Savio scratch: 14 GB of BEANS detection-task audio no longer needed (offered to delete; not touched).

## Traps
- Teacher must be .eval(); inference_mode() does not turn dropout off. Test: same input twice -> identical output.
- Single seed pairs / bird-only bootstraps ignore seed noise (sd ~0.011); use seed_mean_pairs_cpu.py for arm claims.
- Local python lacks lightning: use /Applications/anaconda3/envs/my_audio_env/bin/python (torch 2.2.2 = Savio) for tests.
- Login-node python EIO is intermittent: import-check on savio-dtn, then plain sbatch + manual ledger row.

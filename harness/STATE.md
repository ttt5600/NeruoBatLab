# Lead state -- OVERWRITE every wake (<= 40 lines). Volatile facts only; history goes to pi_notes.md.
Updated: 2026-10-29 (lead session_01EGDWEvYFd6zSmWQfUugJnH; E36 vanilla baseline submitted)
Next: check job 40578099's log for "[teacher] eval mode, 24 layers" and "PASS: labels written" (else scancel
40578100 40578101); when trainings finish, export with ~/export_weights.py and score with harness/score_run.sh,
then judge E13/E30 by their frozen prereg_2026_10_27 criteria (seed_mean_pairs_cpu.py).

## USER PRIORITY: ZF specialization judged on CALL TYPE; skip BEANS (data deleted locally 2026-10-22). Deliver visually.
Call type is CPU ONLY (076). Model table: `python zfeval/experiments/our_models_table.py`.
## LOOPS OFF (2026-10-21): no /loop, harness agent, escalation or auto-resume unless the user asks.

## In flight (submitted 2026-10-28 ~15:45, savio_lowprio, logs ~/jobs/, ledger rows added)
- E13: relabel 40578099 (bioxn-large L12 eval) -> run24 40578100 + run24s2 40578101 (afterok; scancel if gate fails)
- E30: run30 40578102 + run30s2 40578103 (log-mel front end). Slurm est. start Nov 10-11 (2026-10-29 check).
- E36 vanilla HuBERT on ZF: relabel 40631383 (MFCC k=100) -> run31 40631384/85 (stock lr 5e-4) + 40631386/87 (lr 1e-4).
  When done: fill the pending block in Claude Doc ba903e67 (section 'Against vanilla HuBERT') with all-models forest plot.

## Latest results (2026-10-27)
- 085: build_teacher never called .eval() -> every model-teacher label set (run12/13/18/19/22, DAPT E17-E28, E14, E27)
  read with dropout 0.1; 17% of L6 labels move at fixed codebook, but partition AMI 0.578 ~= k-means seed refit 0.585.
  Fixed + tests, committed (pytorchAudio 6ef94ca8) and on Savio (md5-checked). analysis/teacher_dropout_diagnostic.json
- 086: call-type alignment of rebuilt labels fails validation (gets AVES>spectrogram student contrast backwards);
  not a layer-chooser. analysis/e13_teacher_proxy.json
- Arm contrast tool zfeval/experiments/seed_mean_pairs_cpu.py (birds + seeds interval): run19 (2) - run16 (3) = +0.0182
  best / +0.0209 L3, NOT resolved [-0.0076,+0.0439] / [-0.0051,+0.0469]; 6/6 seed pairs positive. seed_mean_pairs_cpu*.json
- E30 log-mel front end built + tested (utils/logmel_frontend.py, --frontend logmel, export/scoring bit-identical,
  learnable: tests/_artifacts/logmel_overfit_compare_relaxed.json). Script slurm/train_run30_logmel.sh.

## Waiting on the user
- E35 (eval-mode AVES control) not proposed. User decided 2026-10-27: E13 teacher = bioxn-large L12; E30 approved.
- Older: E26 design, E15 corpus, E33 (rec no), 10-split headline, label audit; Google Doc 123rAGhW T10/T11 (ask first).
- Savio scratch: 14 GB of BEANS detection-task audio no longer needed (offered to delete; not touched).

## Traps
- Teacher must be .eval(); inference_mode() does not turn dropout off. Test: same input twice -> identical output.
- Single seed pairs / bird-only bootstraps ignore seed noise (sd ~0.011); use seed_mean_pairs_cpu.py for arm claims.
- Local python lacks lightning: use /Applications/anaconda3/envs/my_audio_env/bin/python (torch 2.2.2 = Savio) for tests.
- Login-node python EIO is intermittent: import-check on savio-dtn, then plain sbatch + manual ledger row.

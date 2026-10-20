# Lead state -- OVERWRITE every wake (<= 40 lines). Volatile facts only; history goes to pi_notes.md.
Updated: 2026-10-19 21:10 (lead: session_01EGDWEvYFd6zSmWQfUugJnH; user asked: docs, MFCC control, model sizes)

## USER PRIORITY: ZF specialization judged on CALL TYPE; skip BEANS. Deliver visually (notebooks/artifacts).
Memory: feedback_specialization_not_beans.md. Call type is CPU ONLY (076): never quote ct11_<tag>_emb.npy readers.

## 2026-10-19 morning (user request)
- Model sizes from the weights: analysis/model_param_counts.json (zfeval/experiments/model_param_counts.py).
  HuBERT base = AVES base = BirdAVES biox-base = ours = 94,370,944, same tensors; BirdAVES *-large = 315,427,968.
- Call Type Breakdown dashboard (https://claude.ai/artifact/86skJnBrVBgqnQDZe3wsku) rebuilt on CPU (was MPS, Oct 9):
  calltype_per_class.py --cpu -> calltype_per_class_cpu.json; harness/build_calltype_data.py; 4-model side-by-side.
- Claude Docs updated: tasks doc 1e00135c rev 46 (T10 run23, T11 run22), Savio guide 9e2d877b rev 6 (eval/embed).
  Google Docs updated in place via Playwright (memory reference_google_docs_editing.md); luckyshark668 has writer.

## In flight
- E34 = run23 MFCC-target control: relabel 40144545 PASSED 14:20 (763/763, purity vs spectrogram 0.247) ->
  train 40144546 PENDING, est. start 2026-10-21 21:05. Pass mark
  pre-registered in experiments.yaml. Score: `bash harness/score_run.sh run23_mfcc
  /global/scratch/users/jonathanswang/temp_train_run23_mfcc 200 40144546`. Check relabel log for "PASS: labels written".
- E12 = run22 (40090445): preempted 2x, requeued at step 41,018; Slurm est. restart now 2026-10-21 09:00.
- Savio login-node python EIO (02:47-~14:00) is gone: hubert_env imports torch there again at 21:07.

## Waiting on the user
- E13 teacher, E26 success + donors, E27 24-layer branch, E15 corpus, E33 (rec: no at +-2 st), 10-split headline,
  BeVo split, AVEX, WiSE-FT. Older: E30-32, label audit. Savio cert valid to 14:45 (jobs don't need it).

## Resume: /loop follow harness/PI.md  (first wake: read PI.md, AGENT.md, this file, knowledge/BRIEF.md, pi_status.sh)

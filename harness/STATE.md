# Lead state -- OVERWRITE every wake (<= 40 lines). Volatile facts only; history goes to pi_notes.md.
Updated: 2026-10-20 (lead: session_01EGDWEvYFd6zSmWQfUugJnH; user asked: DAPT on BirdAVES, size/data comparison)

## USER PRIORITY: ZF specialization judged on CALL TYPE; skip BEANS. Deliver visually (notebooks/artifacts).
Memory: feedback_specialization_not_beans.md. Call type is CPU ONLY (076): never quote ct11_<tag>_emb.npy readers.

## 2026-10-19 morning (user request)
- Model sizes from the weights: analysis/model_param_counts.json (zfeval/experiments/model_param_counts.py).
  HuBERT base = AVES base = BirdAVES biox-base = ours = 94,370,944, same tensors; BirdAVES *-large = 315,427,968.
- Call Type Breakdown dashboard (https://claude.ai/artifact/86skJnBrVBgqnQDZe3wsku) rebuilt on CPU (was MPS, Oct 9):
  calltype_per_class.py --cpu -> calltype_per_class_cpu.json; harness/build_calltype_data.py; 4-model side-by-side.
- Claude Docs updated: tasks doc 1e00135c rev 46 (T10 run23, T11 run22), Savio guide 9e2d877b rev 6 (eval/embed).
  Google Docs updated in place via Playwright (memory reference_google_docs_editing.md); luckyshark668 has writer.

## Results 2026-10-21 (all scored on CPU; experiments.yaml E12/E27/E34 = done; table: zfeval/experiments/our_models_table.py)
- E34 run23 MFCC labels: 0.8420, +0.0067 n.d. vs run16 (L3 -0.0059 n.d.) -> spectrogram targets buy nothing measurable.
- E27 BirdAVES bioxn-large DAPT: 0.8403 / 0.8473 vs parent 0.8590 (-0.0188 resolved, -0.0117 n.d.) -> not met, DAPT lowers it.
- E12 run22 AVES labels on FSD50K only: 0.8511; beats run20 (+0.0229 resolved), ties run19 -> not additive.
- Training hours verified from Savio TSVs: ZF 120 files 116.03 h, FSD50K 643 files 108.39 h, mix 763 / 224.42 h.
- Findings for E12/E27/E34 NOT written yet (no critic pass). Nothing in flight on Savio.

## Waiting on the user
- E13 teacher, E26 success + donors, E15 corpus, E33 (rec: no at +-2 st), 10-split headline,
  BeVo split, AVEX, WiSE-FT. Older: E30-32, label audit. Savio cert valid to 14:45 (jobs don't need it).

## LOOPS OFF (2026-10-21, user: "turn off all loops unless we're actually getting somewhere")
- Lead /loop stopped; savio-watch escalate + harness agent disabled (jobs.json); usage-watch auto-resume disabled.
- Do NOT restart /loop or re-enable them unasked. Score finished runs on request with score_run.sh (commands above).
- Still running (no Claude): savio-watch shell polling, usage warnings, quant kalshi-paper / gap-logger / live-signals / paper-step.

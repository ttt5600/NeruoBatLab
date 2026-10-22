# Lead state -- OVERWRITE every wake (<= 40 lines). Volatile facts only; history goes to pi_notes.md.
Updated: 2026-10-22 (lead: session_01EGDWEvYFd6zSmWQfUugJnH; pre-/clear handoff at the user's request)

## USER PRIORITY: ZF specialization judged on CALL TYPE; skip BEANS. Deliver visually (notebooks/artifacts).
Call type is CPU ONLY (076). Model table: `python zfeval/experiments/our_models_table.py` (reads score JSONs only).

## LOOPS OFF (2026-10-21): no /loop, harness agent, escalation or auto-resume unless the user asks (feedback_loops_off.md)
Still running, no Claude: savio-watch shell poll, usage warnings, quant kalshi-paper / gap-logger / live-signals / paper-step.

## Latest results (2026-10-21, all CPU; experiments.yaml E12/E27/E34 status done with result lines)
- E34 run23 MFCC labels 0.8420 vs run16 0.8353: +0.0067 n.d. (L3 -0.0059 n.d.) -> our spectrogram targets buy nothing.
- E27 DAPT from BirdAVES bioxn-large: 0.8403 / 0.8473 vs parent 0.8590 (-0.0188 resolved, -0.0117 n.d.) -> not met.
- E12 run22 AVES labels on FSD50K only: 0.8511; beats run20 +0.0229 resolved, ties run19 -> not additive.
- Pattern: all 3 AVES-label runs (run19, run19s2, run22) top every other from-scratch run on the 10-split mean.
- Size/hours do nothing (biox-large - biox-base -0.0056 n.d.; AVES all - core +0.0015 n.d.); at fixed L3 nothing beats run16.
- Hours verified from Savio TSVs: ZF 120 files 116.03 h, FSD50K 643 files 108.39 h, mix 763 / 224.42 h.
- Nothing in flight on Savio.

## Open tasks (not started)
1. Task docs: T10 (run23) / T11 (run22) still say "In progress"; add E27. Claude Doc 1e00135c + Google Doc 123rAGhW
   (Google = Playwright route, memory reference_google_docs_editing.md; needs Chrome-profile copy, delete after).
2. Findings for E12, E27, E34 in knowledge/findings + critic pass (CRITIC.md). Not written.
3. Call Type Breakdown dashboard (86skJnBr): add run22, run23, E27 s1/s2 (calltype_per_class.py --cpu CPU_MODELS,
   build_calltype_data.py RECIPE, re-upload datasets). Offered, not asked.
4. luckyshark668@gmail.com has writer on both Google Docs (granted for the Playwright edit); user may revoke.
5. Open question, no experiment yet: why Whisper (speech, supervised, log-mel, 680k h) beats speech HuBERT by ~0.035.

## Waiting on the user
- E13 teacher, E26 (approved; design open: donors, dose), E15 corpus, E33 (rec: no at +-2 st), 10-split headline,
  BeVo split, AVEX, WiSE-FT. Older: E30-32, label audit.

## Savio notes
- Login-node python EIO is intermittent: import-check on savio-dtn, then plain sbatch + manual ~/jobs/ledger.tsv row.
- Certificate lasts 12 h (`sv login`); BatchMode SSH only.

# Lead state -- OVERWRITE every wake (<= 40 lines). Volatile facts only; history goes to pi_notes.md.
Updated: 2026-10-25 (after /clear: open tasks 1 (Claude Doc half) and 2 done)

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

## Done 2026-10-25
- Findings 082 (E34), 083 (E12), 084 (E27) written; critic ACCEPT-WITH-CHANGES on all three, changes applied.
- Claude Doc 1e00135c (Encoder Tasks): T10/T11 -> Done with results, T12 row for E27, as-of 2026-10-25.

## Open tasks
1. Google Doc 123rAGhW still says T10/T11 "In progress", no E27 row. Only route is Playwright paste-over-body on a
   Chrome-profile copy (reference_google_docs_editing.md); user hand-edits it -> ASK before doing it.
2. Call Type Breakdown dashboard (86skJnBr): add run22, run23, E27 s1/s2. Offered, not asked.
3. luckyshark668@gmail.com has writer on both Google Docs (granted for the Playwright edit); user may revoke.
4. Open question, no experiment yet: why Whisper beats speech HuBERT by ~0.035.
5. Critic-flagged follow-ups (optional): second seed of run23 (E34 says needed before any claim); E14 vs E27 bootstrap pair.
6. Savio cert expired 2026-10-22: `sv login` before any Savio work.

## Waiting on the user
- E13 teacher, E26 (approved; design open: donors, dose), E15 corpus, E33 (rec: no at +-2 st), 10-split headline,
  BeVo split, AVEX, WiSE-FT. Older: E30-32, label audit.

## Savio notes
- Login-node python EIO is intermittent: import-check on savio-dtn, then plain sbatch + manual ~/jobs/ledger.tsv row.
- Certificate lasts 12 h (`sv login`); BatchMode SSH only.

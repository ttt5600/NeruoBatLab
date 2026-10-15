# Lead state -- OVERWRITE every wake (<= 40 lines). Volatile facts only; history goes to pi_notes.md.
Updated: 2026-10-14 23:10 (seeded from digest 2026-10-14 + pi_notes + pi_status.sh; items marked VERIFY are unchecked)

## In flight
- E8 run19 seed 2, job 39781626: REQUEUE_HOLD "launch failure limit exceeded"; last ckpt epoch=34-step=87734
  (~6k steps left). Will not run until released or resubmitted (resume picks up last.ckpt).
- Nothing else in the queue. max_concurrent 2 -> at least one slot free (two, while E8 is held).

## Done since last digest
- E29 (AVES-bio recipe in our code, 2 seeds): FAILS -- 4/5 BEANS tasks resolved below aves-base-bio on both
  seeds; call type n.d. vs aves/run16. Finding 073 confirmed (critic applied).
- E25 speech SSL: Whisper-small/base above AVES on call type (resolved), not above biox-large; finding 072 OPEN.

## Next action (in order)
1. E8: read the hold reason (`sacct -j 39781626 --duplicates`, job log); if node-side, `scontrol release
   39781626` (our job, lowprio) or resubmit the same script; then score when step 93750 lands.
2. VERIFY experiments.yaml: E7 says `scored` but notes say done as finding 070 -> fix status.
3. VERIFY why approved E9/E12/E13/E15/E16 were not submitted (requires? blocked?) before filling the free slot.
   E26 and E27 are approved but blocked (E26: user's donor/dose; E27: no large-model branch in lightning_modules).

## Waiting on the user
- Go/no-go: E30 (log-mel front end, recommended), E31 (DAPT on Whisper-small), E32 (AVES mask embedding).
- E26: donor source (curated 3412 clips vs detector-mined) and dose (~25 h proposed).
- Resume fix: save last.ckpt every 2,500 steps (preemptions lost up to 6k steps; final-step ckpt often unsaved).
- Usage-watch auto-resume still collides with the lead (2 duplicate submissions). Undesignate?
- Older: speech-commands, Kaggle for cbi/rfcx/Perch, 615 GB deletion, intern details, 174-clip label audit.

## Clocks
- BRC cert valid to 2026-10-15 11:03. Daily digest due after 09:00 Oct 15 (last: 2026-10-14).

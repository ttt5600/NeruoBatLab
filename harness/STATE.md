# Lead state -- OVERWRITE every wake (<= 40 lines). Volatile facts only; history goes to pi_notes.md.
Updated: 2026-10-15 00:45

## In flight
- E8 run19 seed 2: resubmitted by tick.py as job 39945354 (old 39781626 hit REQUEUE_HOLD after 6 preemptions,
  cancelled). Resumes from last.ckpt epoch=34-step=87734 (~6k of 93,750 left). PENDING (Priority).
- E9 run20 seed 2 (FSD50K only): job 39945376, submitted by tick.py 23:38. PENDING.
- E16 (approved analysis, bats time expansion). BEANS half DONE (beans_bats.json): F=4 resolved gain for both
  run16 (+0.0800) and aves-base-bio (+0.0670); F=2 resolved for run16 only; run16 still below aves at F=4 (-0.0585).
  Control @lpte4 (low-pass 8 kHz, then x4; pre-registered, commit ce2bbbf) running on the Mac since 00:46.
  Label-free half not yet designed (needs an individual-held-out E11 variant; pre-register before running).

## Done since last digest (2026-10-14)
- E29 FAILS (finding 073, confirmed): 4/5 BEANS tasks resolved below aves-base-bio on both seeds; feature penalty
  (AVES 10, ours 0) and clip length (ours 20 s chunks) were never matched.
- E25 speech SSL detection: all four below aves-base-bio at 12/12 arms (072 evidence). 072 stays OPEN: Whisper-base
  is above AVES on call type only at its best layer (L0); Whisper-small (selection-free L3) is n.d.
- Bats energy: ~91% of a typical bat clip's energy is above 8 kHz (mostly 8-16 kHz); see pi_notes 2026-10-14 23:49.

## Next action (in order)
1. When the control finishes: read beans_bats.json [<m>@lpte4].vs_<m>@te4 and vs_<m>; draft finding 074 (open
   until critic); then pre-register the label-free half of E16.
2. Score E8 when step 93,750 lands (score_run.sh run19_seed2 ...; final ckpt may be missing -> newest epoch ckpt).
3. Digest after 09:00 Oct 15.

## Waiting on the user
- New, from 073: test AVES's feature penalty (10) and/or clip cropping, one knob each on top of E29?
- Go/no-go: E30 (log-mel front end, recommended), E31 (DAPT on Whisper-small), E32 (AVES mask embedding).
- E26: donor source (curated 3412 clips vs detector-mined) and dose (~25 h proposed).
- Resume fix: save last.ckpt every 2,500 steps (E8 hit 6 preemptions + REQUEUE_HOLD; E29a's step-100000 ckpt unsaved).
- Usage-watch auto-resume still collides with the lead (2 duplicate submissions). Undesignate?
- Older: speech-commands, Kaggle for cbi/rfcx/Perch, 615 GB deletion, intern details, 174-clip label audit.

## Clocks
- BRC cert valid to 2026-10-15 11:03 (notify ~09:30). Digest due after 09:00 Oct 15 (last: 2026-10-14).

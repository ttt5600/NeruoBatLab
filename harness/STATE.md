# Lead state -- OVERWRITE every wake (<= 40 lines). Volatile facts only; history goes to pi_notes.md.
Updated: 2026-10-15 21:47 (lead: session_01EGDWEvYFd6zSmWQfUugJnH, chosen by the user 21:35; the fork 7115f6c0 stopped)

## In flight
- E8 run19 seed 2: resubmitted by tick.py as job 39945354 (old 39781626 hit REQUEUE_HOLD after 6 preemptions,
  cancelled). Resumes from last.ckpt epoch=34-step=87734 (~6k of 93,750 left). PENDING; Slurm est. start 2026-10-17 05:06.
- E9 run20 seed 2 (FSD50K only): job 39945376, submitted by tick.py 23:38. PENDING; est. start 2026-10-17 17:10.
- E16 DONE: label-free half run as pre-registered; no arm distinguishable from plain -> criterion NOT met.
  Finding 074 confirmed (two parallel critics, both applied and merged). Nothing running on the Mac.
- T7 groundwork written: docs/t7_bat_sample_rate_options.md (4 options + costs; decision is the user's).

## Done since last digest (2026-10-14)
- E29 FAILS (finding 073, confirmed): 4/5 BEANS tasks resolved below aves-base-bio on both seeds; feature penalty
  (AVES 10, ours 0) and clip length (ours 20 s chunks) were never matched.
- E25 speech SSL detection: all four below aves-base-bio at 12/12 arms (072 evidence). 072 stays OPEN: Whisper-base
  is above AVES on call type only at its best layer (L0); Whisper-small (selection-free L3) is n.d.
- E16/074: bats ~91% energy above 8 kHz; slowing 4x raises BEANS bats run16 0.6145->0.6945, AVES 0.6860->0.7530;
  low-pass-then-slow control keeps 0.69 / 0.78 of the gain.

## Next action (in order)
1. Score E8 when step 93,750 lands (score_run.sh run19_seed2 ...; final ckpt may be missing -> newest epoch ckpt).
2. Idle: next approved analysis by priority, else backlog item 1 (roadmap 04 figures).

## Waiting on the user
- New, from 073: test AVES's feature penalty (10) and/or clip cropping, one knob each on top of E29?
- Go/no-go: E30 (log-mel front end, recommended), E31 (DAPT on Whisper-small), E32 (AVES mask embedding).
- E26: donor source (curated 3412 clips vs detector-mined) and dose (~25 h proposed).
- Resume fix: save last.ckpt every 2,500 steps (E8 hit 6 preemptions + REQUEUE_HOLD; E29a's step-100000 ckpt unsaved).
- Usage-watch auto-resume still collides with the lead (2 duplicate submissions). Undesignate?
- Older: speech-commands, Kaggle for cbi/rfcx/Perch, 615 GB deletion, intern details, 174-clip label audit.

## Clocks
- BRC cert EXPIRED 20:58 Oct 15 (ControlMaster still up); run `sv login` before E8 starts ~Oct 17. Digest 2026-10-15 written; next due after 09:00 Oct 16.

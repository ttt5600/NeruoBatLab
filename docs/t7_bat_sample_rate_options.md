# T7: bat audio above 8 kHz: options and costs

Written 2026-10-15 from E16 (finding 074). The decision is the user's; this lays out what each option buys and costs.

**The problem.** Every encoder here runs at 16 kHz, so anything above 8 kHz is discarded. On BEANS's Egyptian fruit
bats (250 kHz recordings) the median clip has 9% of its energy at 4-8 kHz, 76% at 8-16 kHz and 12% at 16-32 kHz:
plain resampling throws away about 91% of it (pi_notes 2026-10-14 23:49).

**What E16 measured (finding 074, frozen probe, BEANS bats = identify which of 10 bats is calling).**

| Input | run16 | aves-base-bio |
|---|---|---|
| plain 16 kHz resampling | 0.6145 | 0.6860 |
| slowed 2x (content up to 16 kHz kept) | 0.6405 | 0.7050 |
| slowed 4x (content up to 32 kHz kept) | 0.6945 | 0.7530 |
| control: low-pass 8 kHz, then slowed 4x | 0.6695 | 0.7385 |

Slowing 4x is a resolved gain for both models (+0.080, +0.067). Most of it survives the control, so it comes mainly
from the stretching (4x more frames per call, content 4x lower in pitch), not only from the recovered high band. On
the label-free score (do clusters recur across individuals?) no arm beats plain resampling; the read-out is weak with
10 individuals.

## Options

1. **Time-expand into the existing 16 kHz encoders** (what E16 did). No training. Read the file as if recorded at
   sr/F and resample to 16 kHz; F = 4 covers up to 32 kHz for a 250 kHz source. Cost: clips get F times longer, so
   encoding is roughly F times slower (measured on the Mac: 6,000 bats clips in 376 s at F=2 and 650 s at F=4 for
   run16). Each 20 ms frame then covers 5 ms of real time. Ready now: `beans_probe.py` tags `<model>@te4`.
2. **Pretrain our recipe on time-expanded bat audio.** Same code, same 16 kHz model; only the data is slowed before
   k-means and training. Cost: one pretraining run per seed (E29-sized runs took ~15 h on 4 L40s on lowprio), plus a
   bat corpus, which we do not have on Savio yet (the user's call: which recordings, and whose).
3. **Retrain at a higher sample rate** (e.g. 64 kHz input, with the conv front end's total stride raised from 320 to
   1,280 samples, two extra stride-2 layers, so frames stay 20 ms). Changes the front end, so no existing checkpoint
   transfers; cost as option 2 plus code and validation.
4. **Frequency division or heterodyning** (shift down without stretching). Keeps timing, distorts harmonic structure;
   not measured here.

**Open questions for the user.** Which bat recordings (species, sample rate, whether echolocation or only social
calls matter); whether the label-free unit-discovery goal should be judged on call types (needs labels to check) or
on individual-held-out stability as in E16; and whether option 1 is good enough to start with.

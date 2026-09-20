# Red team: every attack I can make on the detection results

Written against the 32 entries in `knowledge/findings/` plus the 2026-09-13 runs. Each attack gets
a status: **OPEN** (would damage the paper today), **MITIGATED** (answered, needs to stay visible in
the text), or **FATAL TO A FRAMING** (the result is fine, the story built on it is not).

Ordered by how much damage it does, not by how easy it is to answer.

---

## A1. The released encoder is not the contribution — an off-the-shelf model ties it, and beats it out of distribution
**Status: FATAL TO A FRAMING. New evidence 2026-09-13.**

AVES (Hagiwara 2023) is HuBERT-base self-supervised on the animal-sound subset of AudioSet/VGGSound.
It has never heard a zebra finch colony. It is the same architecture as run11 down to the 320-sample
hop, so it is comparable frame for frame and the only variable is the pretraining corpus.

In distribution, on the 30 min of exhaustively annotated colony audio (90,061 frames):

| encoder | best layer | AUC | AP |
|---|---|---|---|
| run11 (120 ZF recordings) | L0 | 0.9687 | 0.8681 |
| AVES (generic animal) | L6 | 0.9674 | 0.8593 |

Difference **+0.0014 AUC**. 120 recordings of colony-specific pretraining bought approximately
nothing over a 377 MB file you can `curl`.

On the ZF → BirdPark holdout — the only genuinely encoder-level test available (finding 019) — it is
worse than nothing:

| model | AUC | AP |
|---|---|---|
| AVES L3 | **0.9009** | **0.8390** |
| AVES L9 | 0.8969 | 0.8173 |
| AVES L6 | 0.8830 | 0.8057 |
| AVES L0 | 0.8814 | 0.8263 |
| run11 L0 | 0.8649 | 0.8123 |
| log-energy | 0.8542 | 0.7472 |
| run11 L6 | 0.7760 | 0.6897 |

**AVES beats run11 at every one of its four layers.** run11's best is +0.011 AUC over a log-energy
detector; AVES's best is +0.047. run11's mid-depth layer collapses to *below* energy.

Be precise about what the statistics support. At the **pre-committed** layer (AVES L6, chosen on ZF),
AVES − run11 is **+0.0161 AUC, not distinguishable**, winning 3 of 4 blocks. The larger +0.0345 comes
from AVES L3, which is its best layer *on BirdPark* — selection on the test set, and a sensitivity
check only. 118.5 s of BirdPark is ~4 independent 30 s blocks, so nothing short of a catastrophe is
detectable at that n. The informative signal is the *consistency of sign* — AVES ahead at 4/4 layers,
and run11's deep layers degrading — not any single interval.

In distribution the tie is measured properly and holds: AVES − run11 AUC **−0.0002
[−0.0030, +0.0025]**.

Reading: run11 overfit its acoustic environment, which is exactly what 120 recordings from one
colony on one rig should be expected to do. Deeper layers specialise harder, which is why L6 falls
apart off-domain while L0 survives.

**What this kills:** "we release a zebra finch encoder" as the paper's contribution.
**What survives:** pretraining on *some* animal audio matters enormously (A3), the evaluation
methodology, the contamination finding, and the timing work.
**The remaining question is now settled, and it went the same way.** Call type is the
species-specific task — "which of 8 types is this" is not a judgement a generic encoder should be
good at. On run11's own published evaluation (2814 clips, 26 birds, leave-birds-out, majority 0.207),
**AVES beats run11 at all 12 layers**: best AVES L1 **0.8308** vs best run11 L2 **0.8003**, bootstrap
over birds **−0.0306 [−0.0543, −0.0102], significant**. Here, unlike the 118.5 s BirdPark holdout,
there is real power — 2814 clips across 26 birds — and the result is unambiguous.

So **run11 is a negative result**: there is no measured task on which colony-specific pretraining
beats an off-the-shelf animal-sound encoder. It should be written up as one, not buried.

**What this attack produced, constructively.** If the two encoders were redundant, combining them
would do nothing. They are not: error correlation is 0.858, and averaging their two probabilities
beats run11 alone both in distribution (AUC 0.9732 vs 0.9687, bootstrap **+0.0047 [+0.0034, +0.0061]**;
AP 0.8791 vs 0.8681, **+0.0118 [+0.0081, +0.0156]**) and on the BirdPark holdout (0.8914 vs 0.8649,
**+0.0249 [+0.0061, +0.0533]**, 4/4 blocks). Concatenating their *features* does not help and is
significantly worse on AP — the third time concatenation has failed on this project, after multi-layer
concat and HuBERT+log-mel. **Combine decisions, not features.** Cost: two encoder passes, zero new
parameters.

One caveat that matters for the recommendation: this does **not** generalise to call type. There the
mean (0.8273) beats run11 (+0.0272 [+0.0115, +0.0443]) but is *not* distinguishable from AVES alone
(−0.0034 [−0.0132, +0.0058]). Ensemble for detection; for call type just use AVES.

---

## A2. Every dense number in the paper comes from one recording annotated by one person
**Status: OPEN. This is the single biggest structural weakness.**

Frame-level AUC/AP, the layer plateau, the resolution sweep, the 125 ms crossover, onset and offset
error, event F1, the decoder tuning — all of it is `111021-000`, 30.02 min, 2540 segments, annotated
in SoundSep by one annotator. Inter-annotator agreement on that dense annotation has never been
measured. The only external check is Julie Elie on **30 windows** (28/30 binary agreement).

Two specific ways this bites:
- The p5 inter-call gap is **16 ms**, below the 20 ms frame grid. Some annotated boundaries are not
  expressible on the grid the model outputs, so a nonzero part of the residual error is unresolvable
  in principle rather than a model failure.
- The measured offset bias is **+20 ms with duration ratio 1.20** — predictions run long. That is
  equally consistent with a model that over-extends and an annotator who releases the boundary early.
  Real data cannot separate them.

**What would settle it:** a second annotator on a stratified subsample (the boundary-dense regions
matter most), and dense annotation of a second recording. The synthetic benchmark (A11) removes the
annotator from the *timing* numbers but not from the detection numbers.

---

## A3. "Pretraining does the work" is argued against a baseline nobody doubted
**Status: MITIGATED, but the emphasis is wrong.**

The random-init control gives +0.195 to +0.208 at every transformer layer and +0.244 at the CNN, all
intervals clear of zero. That is real and it was worth running. But a randomly-initialised deep CNN
is a straw man: its activations came out ~20,000× off in scale (absmax 0.0007 vs 15.82), and the
result "trained weights beat untrained weights" was never in question.

The baseline that matters is the hand-designed one, and it loses by far less: log-mel is 0.9003
window-level and **0.9467 frame-level** against HuBERT's 0.9687. A skeptical reader is entitled to
say the informative comparison is +0.022, not +0.244.

**Keep both, lead with the mel one.** And note what A1 adds: the +0.195 is a statement about
pretraining in general, not about *this* pretraining.

---

## A4. The margin over a spectrogram shrinks exactly as the evaluation gets harder to game
**Status: OPEN framing problem.**

| unit | HuBERT | log-mel | Δ AUC |
|---|---|---|---|
| 1 s windows, eval A | 0.9774 | 0.9267 | +0.051 |
| 1 s windows, eval B (held-out rec.) | 0.9557 | 0.8364 | +0.119 |
| 20 ms frames | 0.9687 | 0.9467 | **+0.022** |

The advantage is largest where the unit is coarsest and smallest at the model's native resolution.
A reviewer will read that as pooling and prevalence doing work that the representation is being
credited for. The honest defence is **AP, not AUC**: at frame level the AP gap is +0.068 against
+0.022 AUC, and at 12% prevalence AP is the metric that reflects operational cost. Lead with AP and
state the AUC gap in the same breath.

---

## A5. It is a colony-background contrast detector, not a vocalization detector
**Status: MITIGATED and documented (findings 010, 011) — must survive into the abstract.**

Mean P(voc) on synthetic probes: harmonic stack 1.000, FM sweep 1.000, real call 0.996,
**brown noise 0.993**, pure tone 0.892, AM noise 0.840, click train 0.538, pink noise 0.325,
white noise 0.120, real background 0.384. And it fires on **digital silence at P=1.000** — the probe
extrapolating 9.9 sd outside the training hull.

Brown noise accepted while white and pink are rejected points at spectral tilt, not harmonicity. The
energy floor at −80 dB patches the silence case and touches no real window; nothing patches the
brown-noise case. Any claim of the form "detects vocalizations" is unsupported; the supported claim
is "separates call from *this colony's* background."

---

## A6. Best-layer claims are selection over 12 layers on a plateau that is explicitly not resolvable
**Status: OPEN, cheap to fix.**

L0 is best for frames, L6 for events, L3 as the iteration-2 teacher, L6 for AVES, L3 for AVES on
BirdPark. Finding 003 says layers 0–8 are a **plateau** and L0–L6 are not distinguishable. Both
statements cannot be load-bearing at once. "L0 is the precise one (P 0.922) and L6 the sensitive one
(R 0.806)" may be a real precision/recall trade along depth or may be two draws from a plateau.

**Fix:** choose the layer inside the CV loop, out of fold, and report the *distribution* of chosen
layers rather than a single winner. Cheap — no new features needed.

---

## A7. The contamination result re-labels a published benchmark by our own ears
**Status: OPEN, and it is load-bearing for the dataset contribution.**

"The published negative class is 62.6% contaminated" (2358 of 3768 judged windows contain a call)
is our judgment against someone else's released labels. The expert check is n=30, with both
disagreements in our favour, which makes 62.6% a *floor* — but 30 windows is thin for a claim this
consequential, and a hostile reader will note the people who benefit from the number are the ones who
produced it.

**What would settle it:** a larger expert sample, or two independent labelers with agreement
reported, on a stratified draw.

---

## A8. No bird and (except BirdPark) no recording is held out of pretraining
**Status: MITIGATED by disclosure (finding 028), unresolvable without new audio.**

99.5% of curated clips came from the 120 pretrained recordings, so "leave-birds-out" constrains the
probe and not the encoder. The corpus *is* the public Elie & Theunissen release (finding 001), so no
online ZF dataset can supply a holdout — it is the same audio. BirdPark is the only encoder-level
holdout and it is 118.5 s, which is why A1's bootstrap has no power.

**The single highest-value thing to collect:** a hard, noisy, colony-like recording from birds
outside the corpus. It closes A1, A2, and A8 simultaneously. Nothing else on this list has that
leverage.

---

## A9. The claim that our false positives are really the annotator's misses is currently unsupported
**Status: OPEN — built but not run.**

The blind adjudication (52 high-confidence disputed FPs + 18 TP + 18 TN loudness-matched controls,
key held locally, read out as a rate between controls) exists and is published. Until it is
completed, "the detector finds calls the annotator missed" is a hypothesis, and any error-rate
number that assumes it is circular.

---

## A10. Decoder gains are inside the noise a 2160-point grid generates
**Status: MITIGATED by the findings' own honesty — must stay that way.**

Hysteresis gave collar-50 F1 +0.009 at L6, −0.001 at L0, with no bootstrap. Finding 023 already says
"+0.009 must not be called an improvement yet." With 2160 configs × 5 folds, a fold-level best is
biased upward by selection. Report the tuning budget next to the gain, or do not report the gain.

---

## A11. Offsets: measured, but only at grid resolution, and confounded with the annotator
**Status: partially addressed today; correcting my own earlier statement.**

I told Jonathan offsets were "essentially unevaluated." That was wrong: `onset_granular.json`
(finding 022) reports offset median 20 ms, bias +20 ms, p90 60 ms, duration ratio 1.20 at every
layer. The accurate complaint is narrower and still real:
- every offset value is a multiple of 20 ms — they are quantised to the frame grid
- the sub-frame interpolation that cut onset error 13.3 → 8.8 ms was never applied to offsets, even
  though `events_from_curve` has always interpolated both edges
- the +20 ms bias cannot be attributed to model or annotator from real data (see A2)

`synth_timing.py` addresses all three, and the answer is clean: with truth exact to the sample the
model localises **both** edges to 5.3 / 6.6 ms and shows **no duration bias** (ratio 0.98, offset bias
+2.6 ms). Re-run with the calls' quiet tails deliberately retained and truth still at the loud core,
it is ratio 1.06 / bias +5.2 ms. Neither construction reproduces the +20 ms / ×1.20 seen against hand
labels, so that bias is a property of **where the boundary is drawn**, not of the model — which is
also why finding 023's offset-shrink was rejected by the tuner in every fold. There was nothing
intrinsic to shrink.

A second result falls out: **crowding costs more than SNR.** At the same +20 dB, moving from 994 ms
gaps to realistic 95 ms gaps drops recall 0.87 → 0.70 and AUC 0.988 → 0.910. Real recordings are
crowded, which is a better explanation of the real-data numbers than any deficiency in the encoder.

---

## A12. What is the scientific claim?
**Status: the question the paper has to answer before anything else on this list matters.**

After A1, the deliverables are: an encoder that ties an off-the-shelf model in distribution and loses
to it out of distribution; a detection benchmark on one recording; and a set of controls and negative
results. "We pretrained an encoder for zebra finches" is not supportable.

The framing the evidence *does* support is methodological, and it is not a weak paper:

1. **A published bioacoustic negative set is 62.6% contaminated**, with an expert check, and the
   evaluation consequences are quantified (A7, finding 014).
2. **What self-supervision actually buys, decomposed.** Pretraining vs random init +0.244 at the CNN
   and +0.011 across twelve transformer blocks — the learning is almost entirely in the convolutional
   front end. Domain-specific vs generic pretraining: nothing in distribution, negative out of it.
   Twelve blocks are not earning their keep, which is a practical recommendation.
3. **A timing benchmark with sample-exact ground truth**, separating model acuity from label noise,
   plus the free sub-frame fix that matches 4× compute (13.3 → 8.8 ms by interpolating a threshold
   crossing, against 8.6 ms for four shifted encoder passes).
4. **Failure modes stated plainly**: fires on digital silence and brown noise; energy VAD collapses
   to chance (0.532) on exhaustively annotated audio while the representation holds 0.956.

That is a methods-and-benchmarks paper with several genuinely useful negative results. It is a
different paper from the one the release bundle implies, and it is more likely to be true.

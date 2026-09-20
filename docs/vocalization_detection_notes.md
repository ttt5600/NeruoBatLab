# Detecting Zebra Finch Vocalizations with Self-Supervised Speech Features

*Working notes toward a paper. Written to be readable from scratch — no prior knowledge of this
project assumed.*

> **Status (2026-07-29).** The detection numbers previously in this document (AUC 0.948, later
> 0.984) are **retracted** — an audit found two bugs in the ground-truth builder (§6). The
> pipeline is fixed, tested, and rerun; §5 reports the corrected measurement. The corrected
> headline is **AUC 0.921**, and the earlier "detection peaks at deep layers" claim **reverses**.

---

## 1. The question

We have long continuous audio recordings of zebra finches (each roughly **58 minutes**, up to
94 minutes; 120 recordings in total). Somewhere inside each are **vocalizations** (calls, song)
buried in **noise** (cage sounds, wing flaps, other birds, silence, equipment hum).

The core question: **can we automatically find the vocalizations in these long recordings, and
how accurate can we get?**

A sharper second question: **does a modern self-supervised audio model actually help, or would a
simple "detect the loud parts" rule do just as well?** The honest baseline for "find the calls"
is loudness — calls tend to be loud. If a large model doesn't beat loudness, it isn't earning
its keep.

---

## 2. Background concepts (skip if familiar)

- **HuBERT** is a self-supervised speech model: trained on raw audio *without human labels*, by
  predicting masked (hidden) portions of the sound from surrounding context. We pretrained our
  own on ~116 hours of zebra finch audio (internally **run11**). It was never trained to detect
  calls; it just learns the structure of the sound.

- **Embedding / feature.** Feeding audio into HuBERT yields, at each internal layer, a vector of
  768 numbers per time-frame summarizing what the model "hears." HuBERT has **12 transformer
  layers**, so there are 12 embeddings to choose from, shallow (0) to deep (11).

- **Linear probe.** To test whether the information we want is *present* in an embedding, we
  train the simplest possible classifier — logistic regression — on top of the frozen
  embeddings. We do **not** fine-tune HuBERT. If a linear classifier can separate calls from
  noise, the information is already in the representation. This is the standard way to measure
  representation quality.

- **AUC (Area Under the ROC Curve).** How well a detector *ranks* calls above noise, independent
  of any yes/no threshold. Equivalently: pick a random call window and a random noise window;
  AUC is the probability the call scores higher. **0.5 = coin flip, 1.0 = perfect.** We lead
  with AUC because it doesn't depend on where the decision cutoff is placed.

- **Leave-recordings-out cross-validation.** We never test on a recording we trained on, so the
  classifier can't cheat by memorizing a recording's background hum or a particular bird's voice.

---

## 3. The hard problem: we have no labels

Measuring detection accuracy normally requires someone to hand-mark every call in the 58-minute
recordings. **No such annotations exist here** (the only marked files on hand belong to an
unrelated 2019 dataset). Hand-labeling hours of audio is exactly the expensive work we want to
avoid.

### The trick: ground truth by search

We *do* have a separate collection of **curated call clips** — short, clean recordings of
individual calls (2,867 of them), each named with bird and date, e.g.
`BlaBla0506_110302-DC-01.wav`. The date code (`110302`) identifies which continuous recording it
came from.

Crucially these clips are **near-exact excerpts cut out of the continuous recordings**. So
instead of hand-labeling, we **search for each clip inside its source recording** and mark where
it's found as a known vocalization.

The search uses **normalized cross-correlation**: slide the clip across the recording and measure
how well it matches at each position. Because the clip really is a piece of the recording, a true
match produces a sharp, unambiguous spike. Concretely, a true match scores **0.85–0.99** on a
0-to-1 scale, while a clip from a different recording scores **~0.1–0.3** — a wide margin.

Every sample inside a located clip becomes **vocalization**; everything else is treated as
**noise**.

### Making the search affordable

Correlating thousands of clips against 90-million-sample recordings at full resolution is slow.
The pipeline therefore searches on a **decimated (downsampled) copy** to propose candidate
locations cheaply, then **re-scores each candidate at full resolution** to decide whether to
accept it. The coarse pass says *where to look*; the full-resolution pass makes the *decision*.
That split matters — see §6, where conflating the two is exactly what went wrong.

### The caveat we always report

Our "noise" label means "not a *curated* call." The recordings contain many **uncurated** calls
that were never clipped, and those get labeled noise. So a detector is **penalized for correctly
firing on a real but unlabeled call**, which makes reported precision a **pessimistic lower
bound**.

> **Update (2026-08-02): this caveat was true but badly understated, and "lower bound" was the
> wrong frame.** It is measured in §5.5. Two things we did not appreciate:
>
> 1. **The scale.** 65% of the negative class contains calls — the negative class is
>    majority-positive. This is not a small correction to precision, it invalidates every
>    fixed-threshold metric.
> 2. **The direction.** Contamination inflates *recall* as much as it deflates precision, so it
>    is not a bound at all — the errors point opposite ways and one of them is optimistic.
>
> The mechanism is that curation selected the **best-sounding** calls for the clip library. That
> makes this a **positive-unlabeled** problem with **non-random** selection: the unlabeled calls
> are systematically the faint and overlapping ones. See §5.5 for why that also manufactures an
> apparent blind spot in the detector.

---

## 4. Method

1. **Build ground truth** by locating curated clips in each recording (§3), searching *every*
   recording index of a date so the true source file is found rather than assumed.
2. **Sample windows.** Cut 0.5-second windows: positives centered on located calls, negatives
   drawn at random from outside call regions (with a guard band so labels stay clean).
3. **Two detectors on the same windows:**
   - **HuBERT probe** — logistic regression on the mean-pooled embedding, at each of the 12
     layers separately.
   - **Energy baseline** — logistic regression on the window's log-RMS loudness. This is "just
     threshold loudness."
4. **Evaluate** with leave-recordings-out cross-validation: accuracy, precision, recall, F1, AUC.

---

## 5. Results

### 5.1 Ground truth recovered

The corrected localization pass (jobs 36016563 → 36020417) verifiably located **2,450 call
intervals across 96 recordings**, versus 851 under the old buggy code. Recovery is largest
exactly where the old code failed:

| recording | intervals (fixed) | intervals (old) |
|---|---|---|
| 130416-001 (286 clips) | **266** | 6 |
| 110620-000 | **128** | 12 |
| 110406-001 | 31 | 0 |
| 110304-000 | 34 | 2 |

`110615-000` stays low (5 of 40 sampled), legitimately: its verified scores are cleanly bimodal
(median 0.265, max 0.982) — a few real matches, and many clips whose source file simply isn't in
our set. Rejecting those is correct.

### 5.2 Detection

7,350 windows (2,450 vocalization / 4,900 noise) from 71 contributing recordings,
leave-recordings-out CV, model run11:

| detector | acc | P | R | F1 | **AUC** |
|---|---|---|---|---|---|
| Energy (loudness baseline) | 0.695 | 0.596 | 0.267 | 0.368 | **0.666** |
| **HuBERT layer 0 (best)** | **0.845** | 0.786 | 0.734 | 0.759 | **0.921** |
| HuBERT layer 3 | 0.810 | 0.740 | 0.666 | 0.701 | 0.896 |
| HuBERT layer 9 | 0.796 | 0.718 | 0.638 | 0.676 | 0.879 |
| HuBERT layer 11 | 0.799 | 0.722 | 0.644 | 0.681 | 0.883 |

> ⚠️ **The acc / P / R / F1 columns above are not reportable.** A 300-window hand-label audit
> (§5.5) found the negative class is **65% contaminated** with real calls. Every threshold-based
> metric here is wrong in *both* directions — precision far too low, recall far too high — and
> accuracy is below the always-say-voc baseline once labels are corrected. **AUC survives**
> (0.921 curated vs 0.878 human-labelled, overlapping CIs), because ranking is robust to this
> label noise in a way that a fixed decision threshold is not. Cite AUC only.

**HuBERT still beats loudness decisively: AUC 0.921 vs 0.666, ΔAUC +0.256.** The energy baseline
recovers only 27% of calls, confirming that loudness alone cannot separate calls from the many
loud non-call sounds in these recordings.

### 5.3 Two corrections to earlier claims

**The headline dropped.** AUC 0.984 → **0.921** (and energy 0.709 → 0.666). This is the expected
direction: the old positives were partly digital silence, which is trivially separable, so the
old figure was optimistic.

**The layer story reverses.** Detection now peaks at the **shallowest** layer (0: AUC 0.921) and
declines monotonically with depth (layer 9: 0.879). The contaminated run had shown the opposite —
deep layers 9–11 winning at 0.983–0.984. A plausible mechanism: silence-pinned positives made the
task partly "silence vs. audio," which deeper, more abstract layers separated better than
shallow spectral ones. On clean labels, detecting whether a call is present is a largely
*low-level spectral* judgment, so the first layer suffices and deeper layers — which specialize
toward HuBERT's self-supervised objective — add nothing.

This does **not** overturn the separate call-type finding (§9), which used clean labels
throughout: *identity* still peaks mid-stack (layer 3). The combined, corrected statement is that
**shallow layers detect, middle layers identify, and the last layer is best for neither.**

### 5.4 Preprocessing ablation

Training normalizes each ~20 s chunk; earlier eval runs fed raw audio. Measured, this is *not*
the no-op it was assumed to be — features differ by ~30% relative. Downstream, though, the effect
is small and the scope barely matters:

| normalization | best AUC |
|---|---|
| `context` (20 s, matches training) | 0.921 |
| `window` (per 0.5 s window) | 0.921 |
| `none` (old behavior) | 0.905 |

So *some* normalization is worth ≈ +0.016 AUC; which scope is used is immaterial here. `context`
is kept as the default because it is the faithful match to training and preserves the loudness
cue that per-window scaling would remove.

### 5.5 The label audit: 300 hand-labelled windows (2026-08-02)

`eval_voc_detection.py --review-export` dumped 300 windows from the run11 eval; all 300 were
labelled by ear with `label_review.py`, **blind** (set membership and model score hidden) and in
**shuffled** order, so the labeler could not be primed by the model's opinion or drift in a way
that aligned with the variable being measured. Labels: `voc` / `voc_noise` (call plus a competing
sound) / `noise` / `quiet` / `unsure`; the first two count as positive, matching how ground truth
sets `y=1` on *any* overlap with a curated call.

Three strata, each answering a different question:

| stratum | what it is | result |
|---|---|---|
| `flagged_neg` (150) | **census** of the 150 highest-scoring negatives (all p ≥ 0.83) | **149 / 150 are real calls** (99.3%, CI 96.3–99.9) |
| `random_neg` (100) | **uniform sample** of all 4,900 negatives | **65 / 100 contain calls** (CI 55.3–73.6) |
| `pos_check` (50) | uniform sample of the 2,450 curated positives | 50 / 50 — the labeler's criterion agrees with curation |

**Essentially every "false positive" was the model being right.** This replicates a known result
in bioacoustics: de Wolff et al. (2024) had 357 AST false-positive manatee detections re-verified
by an expert and confirmed as genuine vocalizations, on a dataset with *partial positive labels*
— the same failure mode, independently arrived at.

**Corrected precision**, three scopes:

| scope | precision |
|---|---|
| as reported | 0.786 |
| correcting only the 150 censused windows | 0.851 |
| correcting all 490 FPs at the rate in the unbiased subsample (7/8) | **0.973** (0.899–0.995) |
| measured directly on the unbiased sample | **0.977** |

#### An unbiased 150-window evaluation set, for free

`random_neg` drew 100 of 4,900 negatives and `pos_check` drew 50 of 2,450 positives — both
exactly **1-in-49**. Their union is therefore a *uniform sample of all 7,350 windows*, and the
model can be scored against human labels with no reweighting. (`flagged_neg` must be excluded: it
is a top-score sample and biased by construction.)

```
human-labelled prevalence   0.767      vs curated prevalence 0.333
AUC vs human labels         0.878  [0.812, 0.931]    vs 0.921 curated — no significant change
threshold 0.50              P 0.977   R 0.374   acc 0.513
threshold 0.05              P 0.977   R 0.730   F1 0.836
threshold 0.005             P 0.849   R 0.930   F1 0.888   <- peak F1
always-say-voc baseline               acc 0.767   (beats the model at threshold 0.5)
```

Two independent problems compounded in the original numbers. **The labels were wrong**, and
**0.5 was never chosen** — it is a default, and it is the wrong reading of a classifier trained
on positive-unlabeled data. Correcting only the first still leaves recall at 0.374; correcting
both puts the model at P 0.85 / R 0.93 with *no retraining*.

#### The masked-call gap, and why it is an artifact of the same selection

Splitting positives by whether a competing sound is present:

| | n | mean score | median | AUC vs non-calls |
|---|---|---|---|---|
| clean calls | 90 | 0.464 | 0.454 | **0.921** [0.865, 0.966] |
| masked calls (`voc_noise`) | 25 | 0.075 | 0.019 | **0.724** [0.588, 0.841] |
| non-calls | 35 | 0.033 | 0.006 | — |

Masked calls are **21.7%** of all calls in this corpus and the model is nearly deaf to them —
those CIs do not overlap, so the gap is real.

**But it was taught, not inherent.** Curation selected the best-sounding calls, so every training
positive is a *clean* call and every masked call was labeled negative. The probe was explicitly
trained to treat a call under a wing-flap as a non-call. Two consequences:

1. The gap is **not independent evidence** about HuBERT features. It is downstream of the same
   selection bias that produced the 65% contamination.
2. The features look fine: 0.724 is **above chance** (CI lower bound 0.588) from a probe trained
   *against* that signal. The information survives in the representation; the probe was told to
   discard it. Predicts that retraining on corrected labels recovers much of the gap — this is
   the experiment to run, not a pretraining change.

#### Caveats on this audit

- **One labeler, one pass, no re-test.** `pos_check` at 50/50 shows the criterion is *sensitive*
  enough to catch everything curation calls a call. It does **not** show the criterion isn't
  *wider* than curation's. The confound is partly closed by the curator confirming that clips
  were selected for sound quality, which predicts exactly this contamination — but a second
  labeler on a subset would close it properly.
- **Only 35 non-calls** in the unbiased sample. The shape of the PR curve is trustworthy; low-
  threshold precision figures rest on 2–19 windows.
- **The threshold was selected on the same 150 windows it is scored on**, so F1 0.888 is
  optimistically biased. Needs a held-out split before it is reported.
- `voc_noise` is a binary presence flag, not a graded SNR. It cannot say *how* masked.

### 5.6 Synthesizing masked calls (`synthesize_masked.py`, 2026-08-02)

§5.5 left one real gap — masked-call AUC 0.724 vs clean 0.921 — and no way to study it. The 25
masked calls we can point to were found by ear, and we cannot mine more by thresholding the
detector, because *the detector not hearing them is the problem*. Mining by score is circular.

Mixing breaks the circle: a call we know is a call, plus background we know is call-free, added at
a chosen SNR. Onset, offset and SNR are exact because we chose them, so a detection-vs-SNR curve
costs zero further human labelling. Same construction as DESED's synthetic partition.

Four design points that are not cosmetic:

1. **Background must be human-verified call-free.** Random negatives here are ~65% real calls
   (§5.5). Mixing into one makes the nominal SNR meaningless *and* writes a "negative" control
   that contains a vocalization. The bank is therefore the 35 windows labelled `noise`/`quiet` in
   the audit — small, but certified. `voc_noise` windows are excluded from both banks: a masked
   call is neither a clean foreground nor a clean background.
2. **Scenes, not independent draws.** One (call, background, placement) is rendered at *every*
   SNR plus its background-only control, all sharing one gain. The sweep is then a within-scene
   paired comparison — call, background, placement and absolute level held fixed, only the mixing
   ratio moving. Drawing independently per SNR (the obvious implementation, and the first one
   written here) adds between-item variance on top of the SNR effect.
3. **Level control belongs to the scene, not the mixture.** Clipping each mixture individually to
   fit in [-1, 1] makes the applied gain correlate with SNR, because loud mixtures clip more
   often — absolute level then leaks the answer. Caught by a check that subtracted each mixture
   from its paired negative: the residual should be the injected call and nothing else, and was
   not until the control was scaled by the same factor. It is now exactly 0 outside the call.
4. **Report in-band SNR, sweep broadband.** Restricted to 1–8 kHz where zebra-finch call energy
   lives, SNR runs **+4 to +8 dB above the broadband figure**, because this colony's background is
   dominated by sub-1 kHz rumble the call never competes with. A nominal −10 dB mixture is really
   about −4 dB where it counts. Quoting the broadband number alone would overstate the difficulty.

**Validation before any model runs.** Verified on real audio: train/test source-disjoint;
mixture − paired negative = the injected call with residual exactly 0.00 outside its support; SNR
re-measured from the written PCM16 files matches target to mean +0.004 dB (max 0.16). The
difficulty gradient is real — a plain energy baseline on 250 test scenes:

| SNR (dB) | +20 | +15 | +10 | +5 | 0 | −5 | −10 | −15 |
|---|---|---|---|---|---|---|---|---|
| in-band SNR | 26.3 | 21.3 | 16.3 | 11.3 | 6.3 | 1.3 | −3.7 | −8.7 |
| AUC, broadband energy | 0.972 | 0.914 | 0.832 | 0.739 | 0.629 | 0.567 | 0.531 | 0.514 |
| AUC, in-band energy | 0.992 | 0.959 | 0.890 | 0.811 | 0.726 | 0.647 | 0.585 | 0.548 |

Loudness degrades to chance by −15 dB and is already near-useless from 0 dB down, so the sweep
contains a wide regime where a detector must genuinely hear the call. That regime is the
experiment.

**The inference this enables.** Real masked calls scored AUC 0.724 (§5.5). Running the probe over
this sweep locates the SNR at which it also scores 0.724 — that is the *effective SNR of the real
masked calls*, estimated without ever having to measure SNR on real audio (which we tried and
could not do: an in-situ estimator scored genuine `noise` windows at 14.6 dB, i.e. it was
measuring background non-stationarity, not calls, and was discarded).

Two modes: `--mode benchmark` (SNR sweep + paired controls, for eval) and `--mode augment`
(uniform SNRs, no controls, for the noise-augmentation training experiment). Known gap: additive
mixing is not a distant call — no reverberation, no air absorption, no altered mic response.
`--masker-dir` covers overlap by using another *call* as the masker, which is the honest model of
"masked" in a colony; the rest remains a limitation.

---

## 6. The audit — what went wrong, and how it was caught

This section is worth keeping for the paper's methods/validation discussion: it is a concrete
argument for why label-generation code needs its own tests.

### Bug 1: digital silence broke the "normalized" correlation

A normalized cross-correlation is mathematically bounded between −1 and 1. The pipeline was
producing values like **76.2**, and nobody had checked.

The recordings contain long runs of **exact zero samples** (494,484 of them in `110411-001`).
The correlation is normalized by dividing by the energy of the recording window under the clip.
For a window of pure digital silence that energy is exactly `0.0`, so the code's fixed
`+1e-12` safety term was all that remained in the denominator — while the numerator still
carried floating-point round-off from the FFT (about `7.6e-11`, and that round-off scales with
the *whole recording's* magnitude, not the silent window's). Dividing one by the other gave
`76.2` where the true correlation was `0.016`.

Consequence: the search's "best match" was pinned to a silent region, and it sailed through the
acceptance thresholds (`76 ≥ 0.6`). **Some fraction of our "vocalization" windows were silence.**
That is doubly bad, because silence is trivially easy to tell apart from random audio — so the
contamination would tend to make the detector look *better* than it is.

Why it hid for so long: it only reproduces at realistic signal amplitude *and* full recording
length. The original validation used a 120-second synthetic recording of quiet noise, where the
round-off-to-floor ratio is far too small to trigger it. Even a second synthetic test *with*
exact zeros missed it, because the amplitude was too low.

It was finally pinned down by **cross-checking the fast FFT path against a brute-force direct
computation at the reported location** — `brute force = 0.016 vs FFT = 76.2`.

### Bug 2: the search was too coarse, and the accept rule was scale-dependent

The coarse search decimated to 1 kHz, which lowpasses the audio at **500 Hz** — below where most
zebra finch call energy lives. Real matches were simply invisible. Searching at 4 kHz instead:
`110406-001` goes from **0/12 clips to 12/12**; `130416-001` (the largest date, 286 clips) from
1/12 to 11/12.

Compounding it, acceptance used a "sharpness" ratio (peak height versus background) that **is
not scale-invariant**: the same perfect match scores 8.45 at full resolution but 1.32 at a 1 kHz
search, because decimation smooths the signal and raises the background. A threshold tuned at
full resolution therefore rejected genuine matches once the coarse search was introduced for
speed.

### The fix

Acceptance is now decided **only** by the full-resolution verified correlation, which is
scale-invariant and physically meaningful ("was this clip cut from here?"). The coarse pass only
proposes candidates (top-k, with non-maximum suppression, so the true match isn't lost when it
isn't the coarse argmax). Degenerate windows are excluded by an energy floor relative to the
clip, and any value outside [−1, 1] is discarded as the numerical artifact it must be.

`test_localize.py` now pins all of this: bounded correlation in the presence of silence, exact
onset recovery, rejection of absent clips, **invariance of the accept/reject decision to the
search rate**, interval merging, and boundary/zero-clip safety.

### The transferable lesson

A metric computed on unvalidated ground truth is not a result. The label-generation step deserves
the same testing as the model — and the cheapest, strongest test is to **assert the invariants
the math guarantees** (here: a normalized correlation cannot exceed 1). That single assertion
would have caught this on day one.

---

## 7. Limitations

- **The negative class is 65% contaminated** (§5.5), because curation kept the best-sounding
  calls. No fixed-threshold metric on these labels is reportable; AUC is. This is the dominant
  limitation and it subsumes the old "precision is a lower bound" phrasing, which was both
  understated and directionally wrong.
- **Masked calls are 21.7% of the corpus and near-undetected** (AUC 0.724 vs 0.921 on clean
  calls), but that gap is downstream of the same curation bias, so it is not yet a clean
  statement about the representation.
- **Coverage is uneven.** Some dates have no verifiable source recording in our set, so they
  contribute nothing; positives concentrate in the recordings that do match.
- **Window-level, not temporal.** We classify 0.5 s windows rather than emitting call
  onsets/offsets over a continuous stream. A reviewer asking "can you find calls in continuous
  audio" will want the latter.
- **Single model.** Numbers come from run11; run12 (a second pretraining iteration) has not been
  run through the corrected detection pipeline.

---

## 8. Next steps

1. ~~Finish the corrected rerun and report honest numbers.~~ Done (§5).
2. ~~Re-run on run12 to confirm the layer story holds.~~ Done (§9) — it holds, and run12 is
   slightly worse everywhere.
3. ~~Hand-label the review bundles.~~ Done (§5.5) — 300/300 on `voc_detect_review_36020417`.
   Outcome exceeded the hypothesis: not a modest precision correction but a finding that the
   benchmark's negative class is majority-positive. `mined_negatives_35707863/` is still
   unlabelled, and its label vocabulary (`silent`/`voc`/`unsure`) needs aligning with
   `label_review.py`'s before it is used.
4. ~~Sweep the threshold.~~ Done (§5.5) — recall 0.374 → 0.930 across the sweep at precision
   ≥ 0.85, no retraining. **But the threshold was chosen in-sample**; redo on a held-out split
   before quoting an operating point.
4b. ~~**Synthesize masked calls at controlled SNR.**~~ Built and validated —
   `synthesize_masked.py`, see §5.6. Turns the single number (AUC 0.724) into a detection-vs-SNR
   curve, with ground truth by construction and no new labelling. Still to run on Savio against
   the real `adultvoc_16k` foregrounds and to score with the run11 probe.
4c. **Retrain the probe on corrected labels** and test whether masked-call AUC moves toward 0.92
   (§5.5 predicts it will). 150 human labels is enough to *demonstrate* the effect, not to train
   on; the synthetic set from 4b is what makes this trainable.
4d. **Report PSDS, not F1 at a threshold.** Bilen et al.'s polyphonic sound detection score
   integrates over operating points, which is exactly the failure this section documents.
   `sed_scores_eval` computes it over all thresholds without the approximation error of a
   threshold grid.
5. ~~Rank candidate k without retraining.~~ Done (§9) — k=500 was not too many; every
   chance-corrected metric still rises at k=2000. **Stop tuning k.**
6. **Test the teacher layer instead.** Iteration 2 clustered run11's **layer 6** because that is
   HuBERT BASE's published choice — where phonetic information peaks *in human speech*. It was
   never validated on zebra finch, and on run11 layer 6 is below average by both readouts we have:

   | | best layer | layer 6 (the teacher) | worst |
   |---|---|---|---|
   | call-type probe (acc) | 0.821 @ L3 | 0.804 | 0.797 @ L11 |
   | detection (AUC) | 0.921 @ L0 | 0.884 | 0.879 @ L9 |

   So iteration 2 may have failed simply because it distilled a weak layer. Relabelling from
   layer 3 costs ~15 min (`sbatch --export=ALL,LAYER=3,NUM_CLUSTER=500
   slurm/preprocess_iter2_hubert.sh`), and the same target-scoring sweep can compare layer 3's
   clusters against layer 6's at matched k *before* committing to a 12-hour retrain. Caveat: the
   layer profile is shallow on call type (0.797–0.821 across all 12 layers, comparable to fold
   noise); the detection profile has the real dynamic range (0.879–0.921) and is what makes
   layer 6 look like a poor choice.
7. ~~Evaluate run13.~~ Done (§9) — 0.913 / 0.824, better than run12 but short of run11's 0.921.
   **Stop iterating the recipe.** Cluster count and teacher layer are both eliminated.
8. **Attack the background-dominance problem instead.** The standing hypothesis for why
   bootstrapping stalls: 79.3% of iteration-1's frames landed in low-energy clusters and
   preprocessing ran `--skip-vad`, so masked prediction spends most of its capacity on ambient
   audio. Cheapest test that does not need a retrain: re-run the target-scoring sweep restricted
   to frames above an energy floor, and see whether AMI against call type jumps. If it does, a
   VAD-filtered or energy-reweighted pretraining pass becomes the first thing worth 12 GPU-hours.
9. Move from window classification to true **temporal detection** (onsets/offsets over a whole
   recording, scored against the located intervals).

---

## 9. Related result: a second HuBERT training iteration did not help

HuBERT is often trained iteratively — retrain using better pseudo-labels derived from the first
model's own features. We did this (run11 → run12). **It gave no downstream gain**: on call-type
classification with clean labels, across all 12 layers and leave-birds-out CV, run11 peaked at
**0.821** (layer 3) and run12 at **0.819** (layer 4) — a tie within noise (job 35653431).

**Confirmed on a second, independent task (job 36076151).** The call-type probe could have been
saturated — a tie proves nothing if the measuring stick has no headroom. So we re-ran run12
through the corrected detection benchmark, which is a different task on different data. run12 is
not a tie there; it is **slightly worse at every one of the 12 layers**:

| layer | 0 | 3 | 6 | 9 | 11 |
|---|---|---|---|---|---|
| run11 AUC | **0.921** | 0.896 | 0.884 | 0.879 | 0.883 |
| run12 AUC | 0.905 | 0.892 | 0.880 | 0.870 | 0.874 |

Best-layer AUC 0.921 → 0.905 (ΔAUC −0.016), and the deficit holds across the whole depth profile
rather than appearing at one layer. Both models still beat the energy VAD (0.666) decisively, and
both peak at layer 0, so the shallow-detects finding is model-independent.

Interpretation: two readouts now agree that iteration 2 bought nothing, and the second suggests it
cost a little. A legitimate negative result worth a paragraph. Two candidate causes, only one of
which survives testing: the labels were too fine-grained (**ruled out** below), or the teacher
layer was badly chosen (§8, item 6 — layer 6 is below average on both readouts, and it was
inherited from a human-speech recipe rather than measured on birdsong).

### Was k=500 the problem?

The natural suspicion is that iteration 2's k-means used **k=500** — HuBERT's published iter-2
default, chosen for 960 h of human speech with ~40 phonemes — and that this is too fine for a
116 h zebra-finch corpus. The cluster statistics saved at relabel time say otherwise:

| | iter-1 (k=100, spectrogram) | iter-2 (k=500, layer 6) |
|---|---|---|
| label entropy | 3.459 nats | 6.103 nats |
| effective perplexity | 31.8 / 100 (**31.8%** of budget) | 447.0 / 500 (**89.4%**) |
| empty clusters | 0 | 0 |
| largest cluster | 17.9% of frames | 3.6% |
| clusters holding 50% of frames | 5 | 188 |
| final train loss (masked CE/frame) | 1.149 | 2.522 |
| information learned = H − loss | 2.310 nats | **3.580 nats** |

k=500 is not the problem — by every measure above, the iter-2 label set is the *healthier* of the
two. It uses 89% of its label budget against iter-1's 32%, has no dead clusters, spreads its mass
across 188 clusters rather than 5, and the model extracted **more** absolute information from it
(3.58 vs 2.31 nats). If anything these numbers indict **iteration 1**: k=100 on spectrogram
features collapsed to only ~32 effective clusters.

> **A statistic that does not support a comparison here.** `cluster_stats.json` also records
> `silence_frame_frac` — the share of frames landing in the bottom-energy quartile of clusters —
> and it reads 79.3% for iter-1 versus 21.7% for iter-2. That looks like "iter-2 is far less
> silence-dominated", but the two numbers are **not comparable**. The energy proxy is
> `centers.mean(dim=1)`, the mean over feature dimensions of each cluster centre. For iter-1's
> 4000-D log-spectrogram that genuinely is a log-energy. For iter-2's 768-D HuBERT activations it
> is just a mean activation — normalised roughly to zero (the recorded threshold is −0.0006 versus
> iter-1's −7.42) and carrying no energy interpretation. iter-2's 21.7% is close to the ~25% you
> would get from an arbitrary quartile of a well-spread assignment, i.e. it measures nothing.
> Only the iter-1 figure is meaningful, and on its own it says the k=100 spectrogram label set
> really was dominated by low-energy frames. Related: `n_silence_clusters` is 25/100 and 125/500 —
> exactly 25% in both, because it is defined as the bottom quartile. That is a tautology, not a
> finding.

The remaining rows are all properties of the assignment *distribution* (entropy, perplexity, empty
count, top share) and are feature-space-agnostic, so those comparisons do hold.

One number worth keeping: iter-1's label entropy is **3.459 nats** and the runs that plateaued
(run5–run9) converged to a train loss of **3.4566**. A model whose loss equals the marginal
entropy of its targets has learned the marginal and nothing else — that plateau was literally the
model predicting the label histogram. (The *cause* was the 5e-4 learning rate; this is what the
symptom looks like from the information side.)

What the table does *not* settle is whether the k=500 clusters split on the right *axis* —
learnable is not the same as biologically meaningful. That needs a direct test.

### The k sweep: k=500 was, if anything, too FEW (job 36076486)

Retraining at each k is 12 GPU-hours, so we scored the *targets* instead: refit k-means on the
already-dumped layer-6 features (2 M-frame pool, same as production) at each k, then measure how
much the resulting cluster id tells you about the 8 human call types across 45 200 frames of
curated clips.

Ranked on **AMI** (adjusted mutual information, chance-corrected) and **homogeneity**
(= I/H(calltype), which does not punish a merely finer split) — deliberately *not* on NMI or
purity, which rise with k by construction:

| k | AMI | homogeneity | eff = I/H(cluster) | ARI | budget used |
|---|---|---|---|---|---|
| 50 | 0.1380 | 0.1991 | 0.1072 | 0.0466 | 93.9% |
| 100 | 0.1615 | 0.2621 | 0.1190 | 0.0413 | 91.9% |
| 200 | 0.1777 | 0.3168 | 0.1268 | 0.0327 | 91.3% |
| 500 | 0.1849 | 0.3728 | 0.1291 | 0.0234 | 90.4% |
| 1000 | 0.1931 | 0.4276 | 0.1337 | 0.0177 | 88.1% |
| 2000 | **0.1981** | **0.4830** | **0.1375** | 0.0110 | 87.5% |

AMI, homogeneity **and** eff all rise monotonically through k=2000 **on layer 6**. `eff` is the
decisive one: it is the share of each label's information budget spent on call type rather than
nuisance, so if the extra clusters were splitting on amplitude or bird identity it would *fall*.
On layer 6 it rises — no sign of over-clustering anywhere in the range, gains flattening
(AMI +0.024 from 50→100, +0.005 from 1000→2000) but never turning over.

> This is a **layer-6 result and does not generalise** — layer 3 has a clear knee at k≈200. See
> the correction below before quoting "500 was if anything too few".

ARI is the one dissenter, falling 0.0466 → 0.0110. ARI counts same-class *pairs* landing in the
same cluster, so it collapses under any refinement regardless of quality — it is the metric most
hostile to large k, which is precisely why it cannot distinguish "misaligned" from "finer but
still aligned." The three refinement-tolerant metrics agree with each other.

**Why the metric choice was load-bearing.** Validated on synthetic labels before the real sweep
ran: on clusters that are *pure noise* with respect to the target, NMI climbs 0.0002 → 0.066 as k
goes 8 → 5000 while AMI stays at 0.000; and under a perfectly-aligned nested split, purity is
pinned at 1.0 for every k. Ranking on either would have "shown" that the largest k was best,
for free.

**What this does and does not license.** It is tempting to say "the k=500 targets scored better
than k=100 and still trained a worse model, so target scores don't predict anything." That
inference is **invalid**, and it was made once in an earlier draft of this document. run11's
targets were **spectrogram** features at k=100; run12's were **layer-6** features at k=500. Those
differ in feature space *and* in k, so the comparison cannot be attributed to k. No proxy score
was ever computed for run11's spectrogram targets — the sweep only scores HuBERT-feature targets.
The AMI column above compares k=100 to k=500 *within layer-6 features*, and neither of those was
ever trained.

The proxy's one clean test is run12 vs run13 (§ below): k fixed at 500, only the teacher layer
changes. It predicted layer 3 > layer 6, and downstream confirmed it on both tasks. So the proxy
is validated where it was actually testable — but only as a *within-feature-space* ranking. It has
never been shown to rank across feature spaces, and it should not be used that way.

### The teacher layer looks like the real mistake (jobs 36076689, 36077191)

Iteration 2 clustered **layer 6** because that is HuBERT BASE's published choice — the layer where
phonetic information peaks *in human speech*. It was never validated on birdsong, and on run11 it
is a below-average layer by both readouts we have (call-type 0.804 vs 0.821 @ L3; detection AUC
0.884 vs 0.921 @ L0). So we dumped layer-3 features over the whole corpus (~18 min) and ran the
identical target-scoring sweep on them.

**Layer 3 beats layer 6 at every single k**, on both chance-corrected metrics:

| k | AMI L6 → L3 | homogeneity L6 → L3 | eff L6 → L3 |
|---|---|---|---|
| 50 | 0.1380 → **0.1578** | 0.1991 → **0.2313** | 0.1072 → **0.1212** |
| 100 | 0.1615 → **0.1742** | 0.2621 → **0.2804** | 0.1190 → **0.1286** |
| 200 | 0.1777 → **0.1968** | 0.3168 → **0.3533** | 0.1268 → **0.1397** |
| 500 | 0.1849 → **0.1988** | 0.3728 → **0.4039** | 0.1291 → **0.1375** |
| 1000 | 0.1931 → **0.1985** | 0.4276 → **0.4400** | 0.1337 → **0.1367** |
| 2000 | 0.1981 → 0.1995 | 0.4830 → 0.4895 | 0.1375 → 0.1373 |

6/6 on AMI and homogeneity. The margin is widest in the mid range (k=200: +0.019 AMI) and
essentially closes by k=2000, which reads as layer 3 reaching with 200–500 clusters what layer 6
needs 2000 to reach. The label distributions agree: at k=500 layer 3 uses **96.0%** of its budget
versus layer 6's 89.4%, and its largest cluster holds **0.72%** of frames versus 3.63%.

### Correction: on layer 3, k=500 *is* past the knee

The "every metric still rises at k=2000" conclusion above is a statement about **layer 6**, and an
earlier draft of this document over-generalised it to k in general. Layer 3 behaves differently,
and it is the layer that matters:

| k | AMI | eff | ARI |
|---|---|---|---|
| 100 | 0.1742 | 0.1286 | 0.0472 |
| **200** | 0.1968 | **0.1397** ← peak | 0.0456 |
| 500 | 0.1988 | 0.1375 | **0.0236** ← halves |
| 1000 | 0.1985 | 0.1367 | 0.0157 |
| 2000 | 0.1995 | 0.1373 | 0.0104 |

Three of the four metrics locate a knee at **k ≈ 200**:

- **AMI saturates.** 100 → 200 gains +0.0226; 200 → **2000** gains +0.0027. Everything past 200 is
  within noise of everything else.
- **eff has an interior maximum** at 200 and declines after — precisely the "the extra clusters are
  buying nuisance, not signal" signature this metric exists to detect. Layer 6 shows no such peak.
- **ARI falls off a cliff between 200 and 500**, roughly flat at 50–200 (0.056, 0.047, 0.046) then
  halving. Layer 6's ARI declines smoothly with no step. ARI over-punishes refinement in general,
  but a *discontinuity* in it is more than the usual monotone decay.

Only homogeneity keeps climbing, and it is the least informative of the four here: it rises under
any refinement by construction (pinned at 1.0 across all k in the synthetic nested-split test).

So the honest statement is layer-specific. On layer 6, no evidence of over-clustering. On layer 3,
k=500 is **past the point of diminishing returns**, though nothing suggests it is actively harmful.

**And none of this is a downstream test.** Every k number here is target-side. No model has ever
been trained at any k other than 500, so "would fewer clusters train a better model?" is
unanswered, and the proxy has only been validated as a *within-feature-space* ranking — never as a
predictor of what k does downstream. run14 (layer 3, k=200, job 36094545) is that test: if it beats
run13's 0.913 / 0.824, k was costing us; if not, k is not the lever.

### run13: the proxy was right, and it still wasn't enough (jobs 36091633, 36091649)

run13 = retrain from the layer-3 k=500 labels, everything else identical to run11/run12
(7 h 23 m, full 93 750 steps, `verify_bridge` ALL CHECKS PASSED).

**Prediction confirmed: layer 3 is a better teacher than layer 6.** On detection, run13 beats
run12 at **9 of 12 layers** (ties at 2, never worse by more than 0.001), mean AUC 0.8871 vs
0.8831; best-layer 0.913 vs 0.905. On call type, 0.824 vs 0.819. The pretraining objective agrees:
run13 extracted 3.727 nats vs run12's 3.580, at higher masked accuracy (0.346 vs 0.341) on a
*harder* target (label entropy 6.173 vs 6.103). This is the target-scoring sweep's one clean test
and it passed.

**But it still does not beat iteration 1.** run11 remains the best detector:

| | detection AUC (best layer) | call type (best layer) |
|---|---|---|
| **run11** (iter 1, spectrogram k=100) | **0.921** @ L0 | 0.821 @ L3 |
| run12 (iter 2, layer 6 k=500) | 0.905 @ L0 | 0.819 @ L4 |
| run13 (iter 2, layer 3 k=500) | 0.913 @ L0 | **0.824** @ L0 |

run13 beats run11 at only **1 of 12** detection layers. The call-type edge (+0.003) is well inside
fold noise. Fixing the teacher layer recovered about half the ground run12 lost, and no more.

**Verdict: iterative refinement does not help on this corpus.** Three passes, two candidate causes
tested and eliminated — cluster count (k=500 was if anything too few; every chance-corrected
metric still rises at k=2000) and teacher layer (layer 3 genuinely better than layer 6, confirmed
downstream, still insufficient). **run11 is the model to use.** That is a cleaner negative result
than "we tried iteration 2 and it didn't work," because the two obvious explanations are now ruled
out rather than merely unexamined.

Where to look next is the *data*, not the recipe. These are continuous recordings dominated by
background: iteration 1's spectrogram clustering put **79.3% of frames in low-energy clusters**
(a valid figure — see the box above for why the layer-6 analogue is not), and preprocessing ran
with `--skip-vad`. A masked-prediction objective spending most of its capacity on ambient frames
is the standing hypothesis for why bootstrapping stalls here.

**A flaw in the "pick the best-probing layer" rule worth keeping in mind.** Layer 0 probes best for
detection precisely because it is closest to raw acoustics — which is what iteration 1 already
clustered, so distilling it would add nothing new. HuBERT uses a middle layer deliberately, to get
targets that *differ* from the previous round's. Layer 3 is a defensible middle ground, but the
probe ranking alone would have argued for layer 0, and that would likely have been a mistake.

A related observation from run13: its best call-type layer moved to **layer 0** (0.824) and its
whole layer profile flattened, the same flattening run12 showed. Both iteration-2 models push
useful information toward the shallow layers rather than building a stronger middle. Consistent
with the shallow-detects / middle-identifies picture, and with iteration 2 adding no new abstraction.

A related finding from that same sweep, independent of the detection bugs: **the best layer
depends on the task.** Call-type identity peaks at a *middle* layer (3), not the last layer
(0.821 vs 0.797) — so probing the final layer by default leaves accuracy on the table. Combined
with the corrected detection sweep (§5.3): **shallow detects, middle identifies, last is best at
neither.**

---

## Appendix: code & data pointers

- `pytorchAudio/examples/hubert/eval_voc_detection.py` — detection benchmark; `localize_clips()`
  is the shared ground-truth builder.
- `pytorchAudio/examples/hubert/test_localize.py` — localization regression tests.
- `pytorchAudio/examples/hubert/mine_negatives.py` — mines silence/loud negatives for review.
- `slurm/voc_localize.sh` (stage 1, CPU: build interval cache) → `slurm/eval_voc_detection.sh`
  (stage 2, GPU: probe). Localization is FFT-bound and CPU-only, so it is kept off the GPU queue
  and cached for reuse across models.
- Model: run11 checkpoint `temp_train_run11/.../epoch=18-step=93750.ckpt` (Savio scratch).
- Curated clips: `adultvoc_16k/` — note the directory also contains one macOS `._*` sidecar per
  clip, which `pathlib.glob` matches and shell globbing doesn't; they must be filtered.
- Continuous recordings: `preprocessed_audio/<datecode>-<NNN>.wav` (120 files).

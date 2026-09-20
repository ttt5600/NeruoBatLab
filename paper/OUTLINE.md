# Outline — working title: *What self-supervised audio representations actually buy for bioacoustic detection*

Every quantitative claim below names the `results.json` key that backs it, in `{braces}`.
`audit_numbers.py` enforces that nothing else numeric enters the prose. Figures are `figures.py`.

**Status key:** ✅ evidence in hand · ⚠️ evidence in hand but caveated · ❌ needs work before it can be written

---

## The framing decision this outline commits to

`ADVERSARIAL.md A1` establishes that run11 — 120 recordings of colony-specific HuBERT pretraining —
ties an off-the-shelf generic animal-sound encoder in distribution {aves.zf.run11_minus_aves.auc}
and loses to it on the only encoder-level holdout {aves.bp.run11_L0.auc} vs {aves.bp.aves_L3.auc}.

So this is **not** an encoder-release paper. It is a measurement paper: what does self-supervision
buy, where does it come from inside the network, what does it cost to evaluate it honestly, and what
breaks. The negative results are the contribution, not an embarrassment to be buried.

---

## 1. Introduction
Detection is the gate every downstream bioacoustic analysis passes through, and it is usually
evaluated on labels that were never verified. Two claims motivate the paper:
- published negative sets are contaminated at a rate that changes conclusions ⚠️ (§2)
- the field is adopting self-supervised encoders without a decomposition of what they contribute ✅

## 2. A verified detection benchmark
- One recording annotated exhaustively: {frame.n} frames, prevalence {frame.prevalence}. ✅
- **The published negative class is 62.6% contaminated** (2358/3768 judged windows contain a call),
  expert-checked on 30 windows at 28/30 agreement, both disagreements in our favour. ⚠️
  *Blocking:* n=30 is thin for the load this claim carries (`A7`). Needs a second labeler.
- Why `labeled=0` may never be defaulted to negative — the positive-unlabeled trap, which bit twice. ✅
- Energy VAD reads 0.532 on exhaustively annotated audio {win.evalB.energy.auc} while the
  representation holds {win.evalB.hubert_L0.auc} — the contamination is what made energy look usable. ✅

## 3. What pretraining buys, and where it lives
**Figure 1** {f1}, **Figure 2** {f2}
- Pretrained vs random init at the CNN output: {randinit.cnn.gap}, every interval clear of zero. ✅
- Twelve transformer blocks add +0.011 over the CNN output: **the learning is in the convolutional
  front end.** ✅ Practical consequence: a much smaller detector should match this one. ❌ untested —
  this is the most valuable cheap experiment left.
- Weight-shuffling (value multiset preserved) is *worse* than fresh random init
  {randinit.cnn.shuf_mean} — so it is not weight scale. ✅
- Honest counterweight: log-mel {randinit.logmel} is far closer than random init
  {randinit.cnn.rand_mean}. Lead with the mel comparison (`A3`). ✅

## 4. Domain-specific vs generic pretraining
**Figure 5** {f5}
- In distribution: a tie {aves.zf.run11_minus_aves.auc}. ✅
- ZF → BirdPark holdout: AVES ahead at all four layers; {aves.bp.aves_L3.auc} vs
  {aves.bp.run11_L0.auc}, against energy {aves.bp.logenergy.auc}. run11's mid-depth layer collapses
  to {aves.bp.run11_L6.auc}, below energy. ✅
- ⚠️ **Report the pre-committed layer choice as primary.** L6 is AVES's best *ZF* layer; L3 is its
  best *BirdPark* layer, so the L3 comparison is selection on the test set and is a sensitivity
  check only (`ensemble_detect.py --aves-layer`).
- ⚠️ The holdout is {aves.bp.n_frames} frames ≈ 4 independent 30 s blocks. Intervals are wide;
  the evidence is the consistency of sign across layers, not any one interval.
- **Call type settles it** (Figure 8 {f8}). {ct.n_clips} clips, {ct.n_birds} birds, majority
  {ct.majority}, leave-birds-out: best AVES {ct.aves.best_acc} vs best run11 {ct.run11.best_acc},
  bootstrap over birds {ct.boot.run11_vs_aves} [{ct.boot.run11_vs_aves.lo},
  {ct.boot.run11_vs_aves.hi}] — significant, with AVES ahead at all 12 layers. ✅
  Reproduction guard against the released table: max |diff| {ct.reproduction.max_diff}.
  **run11 is a negative result and the paper should say so.**

## 5. Combining encoders — the one improvement that survived
**Figure 6** {f6}
- Averaging two probabilities beats either encoder: {ens.zf.mean.auc} vs {ens.zf.run11.auc},
  bootstrap {ens.boot.mean_vs_run11.auc} [{ens.boot.mean_vs_run11.lo}, {ens.boot.mean_vs_run11.hi}]. ✅
- Concatenating the features does **not** {ens.zf.concat.auc} — echoing the earlier result that
  HuBERT+mel concatenation is worse than HuBERT alone. ✅
- Error correlation {ens.error_correlation} bounds what any combiner can recover. ✅
- Zero new parameters, two encoder passes. State the compute cost honestly.

## 6. Resolution, onsets, offsets
**Figure 3** {f3}, **Figure 4** {f4}
- Windowed vs continuous crossover at ~125 ms {res.w125.windowed_L0} / {res.w125.continuous_L0}. ✅
- Label definition is not innocent: the same 1 s audio reads {res.w1000.windowed_L0} under a centre
  label and {res.w1000.overlap_windowed_L0} under overlap. Report the definition or report nothing. ✅
- The nearest-frame rule is late by exactly HOP/2 {onset.coarse_nearest.bias_ms}; interpolating the
  threshold crossing fixes it for free {onset.coarse_interp.mae_ms} and matches four shifted encoder
  passes {onset.shifted_K4_nearest.mae_ms}. ✅ A readout bug, not a model limit.
- Matching rule changes F1 by more than any model choice: {event.L6.overlap_f1} overlap vs
  {event.L6.collar50_f1} under a 50 ms onset collar. ✅

## 7. Sample-exact timing: separating the model from the label
**Figure 7** {f7} — the section that could not be written before this week.
- Construction: real calls windowed to zero outside their own energy support
  ({synth.taper_ms} ms taper), mixed into verified call-free background at known sample offsets;
  probe trained only on the first 60% of the recording, donors from the last 40%. ✅
- With truth exact to the sample: onset {synth.core.L0.onset_mae_ms} ms, **offset
  {synth.core.L0.offset_mae_ms} ms** — the first offset measurement free of label noise. ✅
- **The +20 ms late-offset bias and 1.20 duration ratio measured against hand labels do not
  reproduce**: {synth.core.L0.offset_bias_ms} ms / ×{synth.core.L0.duration_ratio}, and with quiet
  call tails deliberately retained, {synth.tails.L0.offset_bias_ms} ms / ×{synth.tails.L0.duration_ratio}.
  So it is a property of where the boundary is drawn, not of the model — which explains why the
  offset-shrink correction was rejected by the tuner in every fold. ✅
- Psychometric curve with exact truth: recall {synth.core.L0.recall} at +20 dB falling to
  {synth.core.L0.r_at_0db} at 0 dB, with duration ratio collapsing to {synth.core.L0.dur_at_0db} —
  at low SNR only the loud core is recovered. ✅
- **Crowding costs more than SNR**: same +20 dB, realistic gaps ({synth.natural.median_gap_ms} ms
  median) drops recall to {synth.natural.L0.recall}. ✅ This is why real-data numbers are worse.
- Controls: false events on call-free background {synth.bg.L0.false_per_min}/min; 2.1% of false
  positives within 50 ms of a stitching seam. ✅
- ⚠️ Donors are selected clean (local SNR ≥ 10 dB) and 40–250 ms; this is an upper bound on acuity.

## 8. Failure modes
- Brown noise accepted {fail.brown_noise} while white {fail.white_noise} and pink {fail.pink_noise}
  are rejected → spectral tilt, not harmonicity. ✅
- Digital silence at {fail.digital_silence} — the probe extrapolating outside its training hull;
  an energy floor fixes it and touches no real window. ✅
- Conclusion to state plainly: this separates call from *this colony's background*. It is not a
  general vocalization detector. ✅

## 9. Limitations (a real section, not a ritual)
- One recording, one annotator, no measured inter-annotator agreement (`A2`). ⚠️
- No bird and almost no recording is held out of pretraining (`A8`); the corpus is the public
  Elie & Theunissen release, so no online ZF dataset can supply a holdout. ✅
- Best-layer claims sit on a plateau that is explicitly not resolvable (`A6`) — choose the layer
  out of fold. ❌
- Decoder gains are inside the noise a 2160-point grid produces (`A10`). ✅ disclosed.
- The 52 disputed false positives are not yet adjudicated (`A9`). ❌ blocking for any claim that
  our false positives are the annotator's misses.

---

## Work queue, in value order
1. ❌ **Small-model ablation** — CNN + k blocks vs full stack. §3 already implies a much smaller
   detector suffices; this turns an observation into a recommendation. Cheap, features exist.
2. ❌ **Out-of-fold layer selection** — closes `A6` across the whole paper. Cheap.
3. ❌ **Finish the blind adjudication** — unblocks `A9`.
4. ✅ **AVES on the call-type probe** — DONE 2026-09-13. AVES wins at all 12 layers; run11 is a
   clean negative result. Remaining sibling question: individual identity, where the confound
   structure differs.
5. ❌ **Second annotator on a stratified subsample** — the only fix for `A2`.
6. ❌ **A hard, noisy, colony-like recording from birds outside the corpus** — closes `A1`, `A2`,
   `A8` at once. Highest leverage, longest lead time. Start the collection now.

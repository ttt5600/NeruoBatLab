# Project context: zebra finch HuBERT

Generated 2026-09-21. 50 findings: 45 live, 4 closed.

Read the REFUTED section. Several of these ideas look obviously correct and are not; they have each cost a day.


## What is established

### [001] The corpus is the public Elie and Theunissen release  ·  **CONFIRMED**

All 120 pretraining recordings come from the publicly released Elie & Theunissen zebra finch corpus (CC BY 4.0, figshare).

**Evidence.** 120 recordings, 26 birds, 2867 curated AdultVocalization clips across 64 dates.

**Caveats.** Consequence: no online zebra finch dataset derived from this release can supply a holdout, because it is the same audio. An external holdout must come from a different lab. See [016].

### [002] 111021-000 is inside the pretraining manifest  ·  **CONFIRMED**

The recording used as the detection holdout was itself in the run11 pretraining file list, so every detection number measured on it is a probe-level holdout and not an encoder-level one.

**Evidence.** Line 31 of 100 in the derived holdout manifest: preprocessed_audio/111021-000.wav with 77329020 frames, matching the file exactly.

**Caveats.** The encoder heard the audio unlabeled via masked prediction; it never saw any annotation, and the probe was trained on entirely different recordings. So the transfer result is real, but optimistic relative to genuinely unseen audio by an unknown margin. Resolved by [016] and [017].

### [004] Energy VAD collapses to chance on exhaustively annotated windows  ·  **CONFIRMED**

A log-energy detector is at chance on 1 s windows of exhaustively annotated colony audio, while HuBERT holds AUC 0.955 on the identical windows.

**Evidence.** Energy AUC 0.532 (accuracy 0.685, exactly the 0.676 majority rate); HuBERT 0.955. Measured loudness gap between voc and non-voc windows is only +1.66 dB.

**Caveats.** Window-level and specific to the window length. At 20 ms frame level the same feature reaches AUC 0.794, because a frame is either inside a call or not, whereas a 1 s window containing an 80 ms call is ~92% background. Never quote 0.532 as 'energy does not work' without the frame-level number beside it.

### [005] Unannotated is not negative, and it has bitten twice  ·  **CONFIRMED**

Treating unlistened audio as negative has corrupted this project's numbers twice, the second time during the very session documenting the first.

**Evidence.** (1) The original benchmark's 4900 negatives were never listened to; 62.6% of the judged ones contain a call. (2) The cross-dataset job initially scored all 80.6 min of 111021-000 against annotations covering only the first 30.0 min, dropping ZF frame AUC from 0.968 to 0.849 and voiced fraction from 15.2% to 5.7%.

**Caveats.** Standing rule: every dataset records which span was actually listened to, and scoring truncates to it. labeled=0 means nobody listened and must never default to y=0.

### [006] Iteration 2 helps in-distribution only  ·  **CONFIRMED**

run12 (iteration-2, layer-6 teacher, k=500) is reliably better than run11 on the neg-pool windows and not better on a held-out recording, so the gain does not transfer.

**Evidence.** Layer-matched paired bootstrap: eval A run12 better at 9/12 layers, worse at 0/12 (+0.003 to +0.006). Eval B better at 0/12, worse at 3/12 (deep layers, to -0.018). Layer chosen on A and reported on B: run12(L6)-run11(L0) = -0.0065 [-0.0164, +0.0029], not distinguishable. run13 (layer-3 teacher): 4/12 better on A, 0/12 either way on B.

**Caveats.** run11 remains the release model; there is no held-out evidence for switching.

### [008] StandardScaler costs 0.005 AUC on these features  ·  **CONFIRMED**

Standardizing HuBERT features before logistic regression consistently hurts detection accuracy.

**Evidence.** Layer 0 mean, eval A: 0.9774 unscaled vs 0.9728 scaled. float16 storage accounts for 0.0001 of that; the scaler accounts for the rest.

**Caveats.** Mechanism: under a shared L2 penalty, leaving the natural variance in place gives high-variance dimensions effectively weaker regularization, which is the better prior here. Recheck if the head changes.

### [010] The detector fires on digital silence with P=1.000  ·  **CONFIRMED**

All-zero audio is classified as a vocalization with maximum confidence, because it lands far outside the training distribution and the linear probe extrapolates.

**Evidence.** 20 all-zero clips all score P(voc)=1.000. Their feature vector is deterministic (spread 1.2e-6) and sits 2.1 sd from the real-data mean on average, up to 9.9 sd, with 7% of layer-0 dimensions beyond 5 sd.

**Caveats.** FIX, verified: an energy floor at -80 dB rejects silence and touches zero real windows (quietest real window is -68.6 dB). Recordings with dropouts or muted channels would otherwise produce confident false detections.

### [011] The detector is not a general bird-vocalization detector  ·  **CONFIRMED**

Probed with signals of known structure, it accepts call-like harmonic and FM structure but also brown noise, AM noise and pure tones, so it encodes a contrast against this colony's background rather than a general notion of vocalization.

**Evidence.** Mean P(voc): harmonic stack 1.000, FM sweep 1.000, real call in silence 0.996, brown noise 0.993, pure tone 0.892, AM noise 0.840, click train 0.538, pink noise 0.325, white noise 0.120, real background 0.384.

**Caveats.** White and pink noise correctly rejected while brown noise is accepted suggests sensitivity to spectral tilt, not harmonicity alone. Do not deploy on a new acoustic environment without re-measuring the false-alarm rate.

### [012] Detection sensitivity reaches 50 percent at about -10 dB SNR  ·  **CONFIRMED**

With real calls mixed into real background from the same recording, detection stays above the background false-alarm floor down to roughly -10 dB call-to-background SNR.

**Evidence.** Fraction detected at P>0.5: +20 dB 0.97, +10 dB 1.00, 0 dB 0.88, -5 dB 0.68, -10 dB 0.55, -15 dB 0.38, -20 dB 0.35. Background alone sits at 0.32.

**Caveats.** The 0.32 background false-alarm rate at threshold 0.5 is high and matches the independent eval-B false-positive rate (169/584 = 29%). Read the curve as 'above floor', not as absolute detection rate.

### [014] Eval B is a cross-domain transfer, not a same-domain test  ·  **CONFIRMED**

The two halves of the detection dataset are almost perfectly separable from the features alone, so eval B measures transfer across a real domain gap.

**Evidence.** A probe predicting SOURCE (neg-pool 0.5 s windows vs 111021-000 1.0 s windows) from layer 6 reaches AUC 0.9989.

**Caveats.** Makes eval B's 0.955 stronger than it reads, and explains the miscalibrated 0.5 threshold: accuracy 0.883 at 0.5, 0.892 prior-matched, 0.907 oracle, with AUC 0.955 throughout.

### [015] Eval-B false positives are two distinct populations  ·  **CONFIRMED**

The false positives split by confidence into loud broadband transients (genuine errors) and tonal events indistinguishable from true calls (possibly unannotated calls).

**Evidence.** Marginal FPs (0.5<p<=0.944, n=117): spectral flatness 0.220 vs 0.111 for true positives, p=2.5e-10; loud and non-tonal. High-confidence FPs (p>0.944, n=52): flatness 0.122, and no measure separates them from true positives (flatness p=0.41, pitch p=0.25, peak dB p=0.13).

**Caveats.** n=52 is underpowered; indistinguishable from true positives is not is a call. All 211 errors exported as audio with the 52 high-confidence ones first, so a human can settle it.

### [016] BirdPark is a genuine encoder-level holdout  ·  **CONFIRMED**

Zenodo record 20608098 (BirdPark, Hahnloser lab ETH Zurich) is zebra finch audio with exact onset/offset annotations that cannot be in our pretraining corpus, giving the project its first encoder-level holdout.

**Evidence.** 720 valid annotations (1 NaN row dropped) merging to 532 events over 118.5 s; 52.2% of 20 ms frames voiced. Recorded 2020-2021, published 2026-06-09, CC-BY-4.0.

**Caveats.** Channel identification matters and is counterintuitive: channels 0-1 are the LOUDEST but have spectral centroid ~500-650 Hz with 99% of energy below 4 kHz -- they are backpack accelerometers. Channels 2-6 are the microphones (centroid ~4 kHz). A lag scan confirms it: annotated regions exceed gaps by 18.9 dB on the mic mix and only 2.0 dB on the accelerometer mix. Channel 6 is excluded because it is silent 52% of the time. Annotations verified approximately exhaustive: only 11.9% of unannotated 120 ms windows exceed the 25th percentile of annotated ones.

### [018] Frame-level detection produces usable onsets and offsets  ·  **CONFIRMED**

A 50 Hz frame probe on HuBERT features recovers 80% of annotated vocalization events at 88% precision with onset timing at the frame grid, vastly outperforming a log-energy detector through the identical decoder.

**Evidence.** Event level, layer 6: P=0.882 R=0.806 F1=0.842, 2039/2540 events found from 2317 emitted, onset |error| 20 ms. Layer 0: P=0.922 R=0.755 F1=0.829 at 12 ms. Energy through the same pipeline: P=0.448 R=0.736 F1=0.553 from 4084 emitted. Frame level AUC/AP: L0 0.969/0.890, L6 0.960/0.871, energy 0.794/0.390.

**Caveats.** Within-recording: trained and tested on different 6-minute blocks of 111021-000, so it measures localization ability, not generalization to a new recording. The precision/recall trade runs along depth -- layer 0 is the precise one (P 0.922) and layer 6 the sensitive one (R 0.806). A 20 ms onset error IS the frame grid, so these are as well aligned as the representation can express. Two bugs were caught by assertions here: each 20 s chunk yielded 999 frames not 1000, silently leaving 90 all-zero rows, and the energy baseline's grid search twice sat on a grid edge which made its F1 a floor rather than a maximum.

### [019] ZF to BirdPark is the clean direction, and I reported the wrong one  ·  **CONFIRMED**

Of the two cross-dataset directions only ZF->BP tests on audio the encoder never heard, and it is the direction where HuBERT barely beats energy; the direction with the large margin (BP->ZF) tests on a recording that IS in the pretraining manifest.

**Evidence.** ZF->BP: tests on BirdPark (unseen), AUC 0.888 vs energy 0.881 (+0.008), AP 0.893 vs 0.866 (+0.028). BP->ZF: tests on 111021-000 (in the manifest, see [002]), AUC 0.865 vs energy 0.781 (+0.084), AP 0.615 vs 0.335 (+0.280).

**Caveats.** We currently have NO direction that is both clean and convincing. ZF->BP is clean but BirdPark is close-miked and quiet (18.9 dB separation) so energy nearly solves it; BP->ZF has the large margin but a contaminated test set. Closing this needs a HARD, noisy, colony-like recording that is not in the pretraining corpus. Also note the training-set asymmetry: ZF->BP fits on 30.0 min / 90061 frames, BP->ZF on 118.5 s / 5925 frames.

### [020] Loudness is the dominant embedding axis and removing it is catastrophic  ·  **CONFIRMED**

The single largest direction in the embedding correlates with loudness and carries most of the detection signal; projecting it out destroys performance, so 'correlated with loudness' must not be mistaken for 'is a loudness detector'.

**Evidence.** PC0 explains 52.8% of feature variance and correlates 0.724 with log-energy. Unsupervised k-means aligns far better with loudness quintile (AMI 0.335 at k=8) than with voc/noise (0.169) or recording identity (0.043). Removing it: drop PC0 costs -0.194 eval A AUC, top-3 -0.204, top-8 -0.243, while PCA-64 with nothing dropped costs only -0.004. Partialling out the linear loudness axis costs -0.078. Decomposed: PC0 alone gives AUC 0.851, log-energy alone 0.791, PC0 with its energy component regressed out 0.745, both together 0.851.

**Caveats.** This does NOT contradict [004]. Energy is at chance on eval B (0.532) but reaches 0.791 on eval A, so loudness is genuinely informative on the neg-pool windows. PC0 beats the scalar it correlates with by +0.060, i.e. it carries spectral structure the scalar throws away. Practical: do not 'normalize away' loudness, and do not read the UMAP's left-right gradient as a vocalization axis -- it is a loudness axis.

### [021] Linear separability and neighbourhood purity peak at different depths  ·  **CONFIRMED**

Flat linear AUC across layers 0-7 hides a real geometric change: local neighbourhood purity and cluster separation improve with depth while linear separability slowly degrades.

**Evidence.** Across L0->L11: linear AUC 0.9774 -> 0.9714 (flat then declining), kNN-10 accuracy 0.883 -> 0.906 (peaking L5), silhouette 0.195 -> 0.220 (peaking L5) -> 0.205, Fisher ratio 0.573 -> 0.533 (minimum L7), PCA dim at 90% variance 15 -> 51 (monotone). The 4-way human label (call / call+noise / noise / silence) is recoverable at 0.806 accuracy against a 0.515 majority, best at L0. Clean call vs call+noise is only AUC 0.77 at every depth.

**Caveats.** Explains why AUC picks L0 and accuracy picks L5 ([003]): they measure different geometry. The monotone growth of PCA dimensionality also rules out representational collapse with depth, which was a live concern earlier in the project.

### [022] Onset and offset detection in detail, and where it actually fails  ·  **CONFIRMED**

Overlap-F1 flatters the detector; under a strict onset collar performance drops sharply, and the dominant errors are deletions of short calls and merges of consecutive ones.

**Evidence.** Layer 6 F1 by onset tolerance: overlap 0.840, 500 ms 0.865, 200 ms 0.852, 100 ms 0.843, 50 ms 0.814, 20 ms 0.736. Boundary error: onset median 20 ms with bias 0 and p90 40 ms; offset median 20 ms but bias +20 ms and p90 60 ms, with predicted/true duration ratio 1.20. Error taxonomy at L6: 305 deletions, 265 insertions, 154 merges, 21 fragmentations from 2317 predictions against 2540 events. Recall by true duration: 0.685 for 0-50 ms (n=504), 0.877 for 50-80 ms, 0.938 for 80-120 ms, 0.981 above 200 ms. Median IoU 0.667.

**Caveats.** Energy has HIGHER recall in every duration bin (0.764 on 0-50 ms vs 0.685) but only by emitting 4084 predictions against HuBERT's 2317 -- its precision is 0.45 vs 0.88 and it makes 2252 insertions. Short calls are the real weakness: a 50 ms call is 2.5 frames at 20 ms resolution.

### [024] The detector is not reading loudness  ·  **CONFIRMED**

When loudness is held constant by design, the detector still separates vocalizations from background almost as well as it does unrestricted, while loudness itself falls to chance.

**Evidence.** Within narrow dB bands (quintiles), weighted-mean AUC: eval A HuBERT 0.9629 vs energy 0.5641; eval B HuBERT 0.9542 vs energy 0.5580, against unrestricted 0.9774 / 0.9557. On 1:1 pairs matched within 0.5 dB (843 pairs eval A, 528 eval B, mean signed gap +0.008 / +0.005 dB), HuBERT ranks the call above its loudness twin 0.9609 / 0.9394 of the time while energy sits at 0.5302 / 0.5114.

**Caveats.** The first version of the pair matcher scanned upward from j-200 and took the FIRST negative within tolerance, which systematically paired positives with quieter negatives and gave energy a spurious 0.926 win rate. Caught because the printed label said 'near chance by construction' and the number was not. Always check the control's own null.

### [025] HuBERT significantly beats a strong log-mel spectrogram baseline  ·  **CONFIRMED**

The self-supervised representation is worth a large, significant margin over hand-designed spectral features, and the margin more than doubles on the held-out recording.

**Evidence.** Best log-mel: eval A 0.9267, eval B 0.8364. HuBERT layer 0: 0.9774 / 0.9557. Paired bootstrap, HuBERT untuned vs mel tuned: eval A +0.0507 [+0.0406, +0.0613] SIGNIFICANT; eval B +0.1189 [+0.0937, +0.1477] SIGNIFICANT. Log-energy alone for reference: 0.7911 / 0.5324.

**Caveats.** Alignment was verified before scoring: window energy recomputed from local audio matches the Savio-extracted Xen at corr 1.000000, median |diff| 0.000 dB. Concatenating HuBERT with log-mel is WORSE than HuBERT alone (A 0.9717, B 0.9396), so the mel features add nothing the representation lacks. Still outstanding: a random-init encoder control, which would separate 'the architecture' from 'the pretraining'.

### [026] Pipeline sanity controls pass  ·  **CONFIRMED**

The detection result is not a leak and is not carried by a handful of easy recordings.

**Evidence.** Shuffled-label control: 0.4914 +- 0.0140 over 5 permutations against 0.9774 on real labels. Per-recording AUC across 53 recordings with both classes: median 0.9839, mean 0.9734, IQR [0.9591, 0.9994], min 0.8571, and 0 of 53 below 0.80. The five largest recordings hold 30% of windows and average 0.9751 versus 0.9732 for the rest.

**Caveats.** The shuffled control validates the splitting and prediction path only; it cannot detect a problem in the LABELS themselves, which is why [005] and [015] matter separately.

### [027] Leave-recordings-out is bird-leaky, but detection does not care  ·  **CONFIRMED**

The recording-grouped CV lets 96.8% of test-fold birds also appear in training, yet switching to a genuinely bird-disjoint split costs nothing measurable for detection.

**Evidence.** Recording-grouped 5-fold: 96.8% of test birds also in train (3 of 5 folds at 100%). Same 3564 windows: group=recording AUC 0.9741, group=bird-component AUC 0.9721. Delta -0.0020, paired bootstrap over components -0.0017 [-0.0040, +0.0009], not distinguishable.

**Caveats.** Contrast with [project_probe_split_inflation]: the CALL-TYPE probe inflates by +0.114 under a random split. Detection does not, which is coherent -- 'is a call present' is not a bird-specific judgement while 'which call type' partly is. Note 29 birds after merging HPiHPi4748/HpiHpi4748, which differ only by case, plus an 'Unknown000' placeholder that is not a bird. This is a PROBE-level split only; see [028] for the encoder.

### [028] No zebra finch bird is held out of pretraining, and none can be  ·  **CONFIRMED**

Every bird in the corpus appears in the pretraining manifest, so no ZF result is a bird-level encoder holdout; the only birds the encoder has never heard are BirdPark's.

**Evidence.** All 120 recordings in run5-4-26-full/data/spectrogram/preprocessed_audio are in the training tsv (verified in [002]), and those recordings cover all 29 birds across 60 datecodes. BirdPark contributes 16 birds (8 pairs) from a different lab.

**Caveats.** A bird-level encoder holdout could in principle be built by retraining on one of the 7 bird-disjoint components -- the holdout_lblred0613 experiment already did this for one bird, at the cost of 16.7% of pretraining audio, which confounds 'never heard this bird' with 'less data'. Until such a run exists, bird-level generalization for ZF is untested and only BirdPark speaks to it.

### [029] What the AUC numbers mean in operational terms  ·  **CONFIRMED**

At a fixed 5% false-alarm budget on the held-out recording, the representation finds 1.8x as many calls as a strong spectrogram and 10x as many as an energy detector.

**Evidence.** Eval B, 5% false alarms: HuBERT recovers 1016/1217 calls (83.5%), log-mel 573/1217 (47.1%), log-energy 102/1217 (8.4%), each with 29 false alarms out of 584 non-voc windows. As ranking-error rate (1-AUC): HuBERT 4.43%, log-mel 16.36%, energy 46.76% -- HuBERT removes 73% of the spectrogram's errors. On eval A at 5%: 92.6% / 73.8% / 15.4%.

**Caveats.** AUC 0.9557 means: pick one random call window and one random non-call window, and the probe ranks the call higher 95.57% of the time. It is threshold-free -- the operating points above are what a chosen threshold turns it into, and the right threshold depends on whether misses or false alarms cost more for the study.

### [030] Generic animal pretraining ties run11 in distribution and beats it on the holdout  ·  **CONFIRMED**

AVES, a HuBERT-base encoder self-supervised on generic animal sound and never exposed to a zebra finch colony, matches run11 on in-distribution detection and outperforms it at every layer on the only encoder-level holdout. Colony-specific pretraining is not what makes the detector work.

**Evidence.** In distribution (90,061 frames, 20 ms, prevalence 0.1242): run11 L0 AUC 0.9687 / AP 0.8681, best AVES (L6) 0.9674 / 0.8593, difference +0.0014 AUC. Paired block bootstrap (1500-frame blocks) on the pre-committed layers: AVES - run11 AUC -0.0002 [-0.0030, +0.0025] NOT DISTINGUISHABLE. ZF -> BirdPark holdout (5925 frames, prevalence 0.4290): AVES L3 0.9009/0.8390, L9 0.8969/0.8173, L6 0.8830/0.8057, L0 0.8814/0.8263; run11 L0 0.8649/0.8123; log-energy 0.8542/0.7472; run11 L6 0.7760/0.6897, BELOW energy. AVES is ahead at 4 of 4 layers. Within-BirdPark 5-fold for reference: run11 L0 0.9637, AVES L6 0.9650.

**Caveats.** The holdout is 118.5 s, about 4 independent 30 s blocks, so intervals are very wide and the pre-committed run11-vs-AVES contrast reads not_distinguishable; the evidence is the consistency of sign across all four layers, not any single interval. AVES L3 is its best BirdPark layer and using it would be selection on the test set -- quote L6. BirdPark is close-miked so energy alone reaches 0.854 and all differences compress. This does NOT refute [003]/[randinit]: pretraining still beats random init by +0.244 at the CNN. It refutes only that the pretraining had to be colony-specific. Still untested, and the last place run11 could win: the call-type and identity probes.

### [031] Averaging two encoders beats either; concatenating them does not  ·  **CONFIRMED**

Mean of the two probes' probabilities is a significant improvement over run11 alone at zero added parameters, while feature concatenation is not an improvement and is significantly worse on AP.

**Evidence.** In distribution, out-of-fold, run11 L0 + AVES L6: run11 0.9687/0.8681, AVES 0.9674/0.8593, concat (1536-d) 0.9683/0.8610, stack 0.9698/0.8751, MEAN 0.9732/0.8791. Paired block bootstrap vs run11: mean AUC +0.0047 [+0.0034, +0.0061] SIGNIFICANT, AP +0.0118 [+0.0081, +0.0156] SIGNIFICANT; concat AUC +0.0001 [-0.0016, +0.0018] not distinguishable, AP -0.0059 [-0.0106, -0.0012] WORSE. Error correlation between the two encoders 0.8575.

**Caveats.** Costs two encoder passes at inference. The pattern matches [013] (multi-layer concat buys nothing) and the earlier HuBERT+log-mel result (concatenation was WORSE than HuBERT alone): adding dimensions to one linear probe dilutes it, while combining decisions does not. The gain also holds on the ZF -> BirdPark holdout at the pre-committed layer: run11 0.8649, AVES 0.8830, concat 0.8481, MEAN 0.8914, energy 0.8542; mean - run11 AUC +0.0249 [+0.0061, +0.0533] SIGNIFICANT with 4/4 block wins, while concat is significantly WORSE on AP (-0.0326 [-0.0470, -0.0195]). At that layer AVES alone - run11 is +0.0161, NOT distinguishable (3/4 block wins), so the ensemble gain is better evidenced than either encoder's individual edge.

### [032] The late-offset bias is a property of the labels, not the model  ·  **CONFIRMED**

Measured against ground truth that is exact to the sample, the detector shows no duration bias at all. The +20 ms late offsets and 1.20 predicted/true duration ratio seen against hand-drawn boundaries do not reproduce under two independent exact constructions, so they describe where the annotator draws the boundary rather than how the model behaves.

**Evidence.** Layer 6, isolated calls at +20 dB. Truth = the inserted signal's own edges: onset |err| 5.3 ms, offset |err| 6.6 ms, offset bias +2.6 ms, duration ratio 0.98. Truth = the loud core while the signal KEEPS its quiet tails: offset bias +5.2 ms, duration ratio 1.06, onset bias -1.9 ms. Against hand labels the same layer reads offset |err| 20 ms, bias +20 ms, ratio 1.20. Psychometric curve (isolated, exact truth): recall 0.87 at +20 dB, 0.85 at +10, 0.75 at +5, 0.49 at 0, 0.20 at -5, 0.02 at -10, with duration ratio falling 0.97 -> 0.70 -> 0.56 as only the loud core survives.

**Caveats.** Explains why [023]'s offset-shrink correction was rejected by the tuner in every fold: there was nothing intrinsic to shrink. Donors are selected clean and 40-250 ms, so ~5-7 ms is an upper bound on acuity, not typical performance. Crowding matters more than SNR at high SNR: at the same +20 dB, realistic inter-call gaps (median 95 ms) drop recall from 0.87 to 0.70 and AUC from 0.988 to 0.910 -- which is why real-recording numbers are worse than this. Construction controls: 2.1% of false positives fall within 50 ms of a background stitching seam, and call-free background yields 2.0-4.0 false events per minute.

### [033] Generic pretraining also wins on call type, so run11 is a negative result  ·  **CONFIRMED**

On 8-way call-type classification -- the species-specific task where colony pretraining should help if it helps anywhere -- AVES beats run11 at every one of the 12 layers, and the margin at each encoder's best layer is significant under a bootstrap over birds. There is now no measured task on which zebra-finch-specific pretraining beats a generic animal-sound encoder.

**Evidence.** run11's own published evaluation, unchanged: 2814 curated clips, 26 birds, 8 classes, majority 0.2072, leave-birds-out StratifiedGroupKFold(5). Best run11 L2 0.8003; best AVES L1 0.8308. Cluster bootstrap over the 26 birds: run11 - AVES -0.0306 [-0.0543, -0.0102] SIGNIFICANT. AVES is ahead at all 12 layers (deltas -0.0092 to -0.0373). Both encoders decline with depth and peak shallow (AVES L1, run11 L2). Combiners: mean of the two probes 0.8273, which beats run11 (+0.0272 [+0.0115, +0.0443]) but is NOT distinguishable from AVES alone (-0.0034 [-0.0132, +0.0058]); concat 1536-d 0.8152, worse than either best single layer.

**Caveats.** AMENDED 2026-09-13 -- see the matched-extraction caveat below; this was previously recorded in the status field, which the schema does not allow. AMENDMENT, same day. The headline above compares run11's RELEASED embeddings against an AVES extraction of our own. Re-extracting BOTH encoders through one identical code path changes the picture at 8 classes: run11 rises to 0.8193 (L7) against AVES 0.8359 (L1), a gap of -0.0166 [-0.0400, +0.0041], NOT distinguishable. So part of the original -0.0306 was a difference between extraction pipelines, not between encoders, and the 8-class claim must be downgraded to "AVES ahead, not significant". The 11-class result (calltype11.json, both encoders extracted identically) does hold: run11 L3 0.8118 vs AVES L3 0.8453, -0.0338 [-0.0609, -0.0069] SIGNIFICANT over 48 birds, with AVES ahead at all 12 layers. Always extract both arms yourself; never compare a released feature file against a fresh extraction. A reproduction guard ran first: re-deriving run11's per-layer accuracies from its released embeddings gave max |diff| 0.0152 against the published table (published best L3 0.8109, ours L2 0.8003), the gap being fold assignment since the published random_state is not recorded. So the comparison is internally fair but the absolute values sit ~0.01 below the released ones. Both encoders' best layers were selected on the same metric being reported, so both are optimistic by a similar amount and the best-vs-best bootstrap inherits that. Unlike detection, where averaging the two probes beat BOTH encoders, here the ensemble buys nothing over AVES alone -- on call type the recommendation is simply to use AVES. Concatenation has now failed a fourth time.

### [034] The 2023 AVES call-type number was a padding artifact, not a fact about AVES  ·  **CONFIRMED**

The ~60 percent that AVES scored on 11-way call type in the April 2023 notebook measures that notebook's preprocessing, not the encoder. Run with the audio handled correctly, the same checkpoint on the same 11 classes reaches 0.845.

**Evidence.** datasets/11905533/AVESZF.ipynb padded every clip to 250,606 samples (15.66 s) with pad_sequence and then mean-pooled the encoder output over all ~782 frames. The median clip in that corpus is 0.142 s, so a typical clip embedding was a mean over a frame axis that is 99.1 percent zeros. The same pipeline also half-wave rectified the waveform (wav[wav < 0] = 0 inside wav_resample) and trained a single nn.Linear with SGD lr 0.01, batch size 1, 5 epochs, on a random 80/20 split rather than leave-birds-out. Re-run with per-clip encoding (no padding), mean-pooling over real frames only, a converged logistic probe and leave-birds-out: AVES L3 0.8453 on 11 classes (3412 clips, 48 birds, majority 0.1797).

**Caveats.** The fairseq task config records normalize: False, so AVES expects raw un-normalised audio -- which is what both this run and run11's own zf_hubert.embed_file path use, so preprocessing is not smuggling in an advantage. Watch the case trap when parsing types: blind title-casing rewrites DC to Dc and LT to Lt and silently drops 838 clips, which happened once here before it was caught by the class count coming back 9 instead of 11.

### [035] run11 wins nowhere once layer choice is made out of fold  ·  **CONFIRMED**

Across four pre-enumerated axes chosen to favour colony-specific pretraining, run11 is significantly ahead on none of them once the layer (or layer subset) is selected inside the training fold instead of on the reported metric. The identity advantage reported earlier shrinks by more than half and stops being distinguishable.

**Evidence.** 11-class call type with multi-layer concatenation, layer set chosen out of fold: run11 0.8036 vs AVES 0.8403, bootstrap over 48 birds -0.0368 [-0.0643, -0.0097], AVES BETTER. Bird identity, leave-session-out, layer set chosen out of fold: run11 0.3280 vs AVES 0.3154, +0.0125 [-0.0151, +0.0410], NOT distinguishable -- against +0.0286 when each model's layer was picked on the reported metric. Per class: Be (begging) run11 0.971 vs AVES 0.934, +0.035 [+0.007, +0.080]; Ag +0.011 [-0.027, +0.051] not distinguishable. Chick clips: run11 0.8836 vs AVES 0.8966. Adult clips: 0.8005 vs 0.8372.

**Caveats.** The Be result is the one survivor, and it should not be treated as a finding yet: Be was chosen for follow-up BECAUSE it led in an exploratory pass over 11 classes, so its interval is conditioned on that selection. At 11 classes one significant result at alpha 0.05 is roughly what chance produces. It is a hypothesis for a pre-registered test on new begging-call data, not a win. Equally, 'not distinguishable' on identity is not proof of equality -- run11 was numerically ahead at all 12 single layers, which is suggestive; what died is the claim that the margin is established. Multi-layer concatenation helped AVES (0.8453 -> 0.8403 is within noise) and did not rescue run11, consistent with concatenation having failed four previous times on this project.

### [036] Fine-tuning does not flip the ranking  ·  **CONFIRMED**

The hypothesis that run11's colony weights are a better INITIALISATION even though their frozen features are less linearly separable is not supported. Unfreezing the encoder raises both models by a similar amount and leaves the gap where it was.

**Evidence.** 11-class call type, leave-birds-out, identical folds/hyperparameters/seeds for both. AVES minus run11: +0.0265 frozen L3, +0.0296 full fine-tune, +0.0259 transformer-only. Full fine-tune means: run11 0.8306 / 0.8304 (seeds 0/1), AVES 0.8607 / 0.8594. Seed spread run11 0.0002, AVES 0.0013 -- the gap is about 20x the seed noise. run11 fine-tuned (0.8305) still lands BELOW AVES frozen at its best layer (0.8429). Gains from fine-tuning: run11 +0.014, AVES +0.017. Freezing the CNN and tuning only the transformer is worth +0.002 / +0.001.

**Caveats.** The transformer-only arm ran at one seed. That the transformer-only arm buys almost nothing for either model is independent support for [randinit]: the CNN front end carries most of the pretraining benefit, so unfreezing only the blocks above it cannot recover much. This closes the last protocol under which run11 could have won call type -- frozen probe, multi-layer, out-of-fold layer choice, and now full fine-tuning all rank AVES ahead.

### [037] torchaudio batched feature extraction is not padding-invariant for these models  ·  **CONFIRMED**

Batching variable-length clips through hubert_base / wav2vec2_model corrupts the CNN features of the SHORT clips, at their valid frames, and passing `lengths` does not fix it. Any variable-length batch through this encoder is silently wrong.

**Evidence.** Verified directly on run11. Clips of 0.12 s, 1.00 s and 15.60 s batched together against the same clips encoded one at a time, comparing only VALID frames: max abs difference 121.41 for the 0.12 s clip, 60.76 for the 1.00 s clip, and 0.0000 for the 15.60 s clip -- which is the one that defines the batch length and therefore carries no padding. Identical numbers with lengths=None and with lengths passed. The subagent measured up to 123.0 in features and 0.468 in final logits on a real batch of 8, and recovered 1.9e-6 agreement with a masked recomputation.

**Caveats.** No stored result in this repo is affected, which was checked rather than assumed. The only batching site is resolution_extract.windowed_pass, where every window is exactly `win` samples so no padding exists -- and np.stack would have raised on ragged input rather than pad silently. Every other path (frame_pass, embed_all in aves_calltype/calltype11, window_features) encodes one clip or one chunk at a time. This is the same failure mode as the 2023 AVES notebook's mean-over-padding bug ([034]) arriving by a quieter route: there the padding polluted the POOLING, here it pollutes the NORMALISATION. Rule: never batch variable-length audio through these encoders without recomputing the group norm over real frames, and verify batched == single to ~1e-5 before trusting any batched extraction.

### [038] The run11 vs AVES comparison confounds corpus with target quality  ·  **CONFIRMED**

run11 and AVES differ in more than their pretraining corpus. AVES used ITERATION-2 k-means targets taken from a trained HuBERT's layer 6 at k=200; run11 used ITERATION-1 targets from log-mel spectrogram at k=100. Target quality and corpus diversity are therefore confounded with the domain-match variable the comparison set out to test, and the target difference is separately testable at a fraction of the cost of any new pretraining run.

**Evidence.** From the local 2022 fairseq checkpoint cfg (datasets/11905533/aves-base-bio.pt): label_dir = /mnt/dev/hubert/data/faav150k/hblab.c200 -- "hblab" identifies HuBERT-layer labels, not MFCC, and c200 gives 200 clusters; label_embs_concat is (204, 256) = 200 clusters + 4 fairseq specials. Paper supplies the layer (6). AVES-bio: 360 h of audio (153 h FSD50K + AudioSet-balanced general, plus ~207 h animal), 100,000 updates, lr 2e-4, 700.0 s of audio per update (max_tokens 1.4e6 / 16000 x update_freq 8 x world_size 1), distributed_world_size = 1, wall clock 88,709.81 s = 24.6 h on ONE GPU. Total audio budget is 6% of HuBERT-base's. warmup_updates 32000 against total 100000 -- 32% of the run, fairseq stock warmup left in place after cutting 400k to 100k. normalize = False. The 360 h figure was corroborated independently from the checkpoint's own epoch/iteration counters: (100000-306)/54 x 700 s = 359.0 h vs 360 published.

**Caveats.** This does not overturn [030]/[033]/[035]/[036] -- run11 still loses under every protocol tested. It bounds what those results can be attributed to: "colony-specific pretraining does not pay" is supported, "domain-matched CORPORA do not pay" is only partly supported, because the corpora were not the sole difference. The cheap control is to re-derive run11 targets from a trained encoder layer at k=200 and rerun one HuBERT iteration, holding the corpus fixed. Also load-bearing for the write-up: AVES's own aves-base-nonbio control (animal-filtered vs size-matched non-animal corpus) scores 61.3 vs 61.4 overall and 61.8 vs 61.2 on classification -- their own paper contains evidence that animal-filtering the corpus does not help, which makes run11 a sharper instance of a phenomenon already visible in the literature rather than an anomaly.

### [039] run11's targets were not the problem; the corpus is the live hypothesis  ·  **CONFIRMED**

Adopting AVES's target recipe on this corpus makes the pseudo-labels WORSE at respecting call boundaries, not better. Combined with run12/run13 already having lost to run11 on the full retrain, the recipe explanation for AVES beating run11 is closed, and corpus composition is what is left.

**Evidence.** AMI against call-boundary identity on the same 90,061 frames: run11's actual recipe (log-mel, k=100) 0.0854; log-mel at k=200 0.0646; run11 layer-6 at k=200 0.0488; AVES layer-6 at k=200 0.0482. Decomposition: raising k from 100 to 200 with the feature held fixed costs -0.0208, and switching log-mel to layer 6 with k held at 200 costs a further -0.0158. Against voc/noise the ordering reverses slightly (layer-6 0.0757/0.0758 vs log-mel 0.0733), so layer-6 clusters know call-vs-background marginally better while keeping individual calls together much worse. The teacher ENCODER barely matters: run11-layer-6 0.0488 vs AVES-layer-6 0.0482.

**Caveats.** This is a CROSS-feature-space comparison, and [project_iter2_stage2] establishes the k-sweep proxy is only valid WITHIN a feature space. So treat it as directional, not proof. What makes it load-bearing anyway is that it agrees with a full retrain that was already run: run12 (layer-6 teacher, k=500) reached detection 0.905 and call type 0.819, run13 (layer-3, k=500) 0.913 and 0.824, both against run11's 0.921 and 0.821. Teacher layer and cluster count were each eliminated separately there. The remaining asymmetry against AVES is the CORPUS: iteration-1 put 79.3% of frames in low-energy clusters and preprocessing ran --skip-vad, so roughly four frames in five of run11's masked-prediction budget went on ambient colony noise, against AVES's 360 h of curated sound events (FSD50K + AudioSet-balanced). The testable version is to re-pretrain with VAD or an energy floor so capacity lands on vocalisations.

### [040] The spectrogram target choice was right, and it is not an architecture difference  ·  **CONFIRMED**

run11's use of a spectrogram rather than MFCC to build its k-means pseudo-labels produces measurably better targets than AVES's iteration-1 MFCC on this corpus, at matched cluster count and on identical frames. But the choice lives in the LABEL pipeline, not the network. run11 and AVES are the same architecture, so there is no architecture variable to hold training constant against; the spectrogram-vs-MFCC variable is real and is a target-quality variable.

**Evidence.** AMI against call-boundary identity on 90,061 shared frames, MiniBatchKMeans, seed 0. run11 actual feature (soundsig Gaussian STFT, 50 Hz spacing, 25 ms window flattened to 4000-d): k=100 0.0712, k=200 0.0642. AVES iteration-1 feature (fairseq stock 13 Kaldi MFCC + delta + deltadelta, 39-d): k=100 0.0487, k=200 0.0401. Spectrogram minus MFCC at matched k = +0.0226 (k=100) and +0.0241 (k=200) -- same sign and near-identical size at both k, so it is a feature effect and not a k artifact. run11 as shipped (spectrogram, k=100) vs AVES iteration-1 as shipped (MFCC, k=200) = +0.0311. Against voc/noise the ordering is the same: 0.0726/0.0704 vs 0.0626/0.0563. A 384-d log-mel summary scores HIGHER than run11's own 4000-d soundsig feature on call id (0.0854 at k=100), so run11's target feature is not the best available even inside the spectrogram family. k=100 beats k=200 for all three features on both references, which reproduces [039]'s k effect in three independent feature spaces.

**Caveats.** Cross-feature-space AMI comparison, so directional rather than proof -- the same caveat [039] carries. The within-feature k=100 vs k=200 contrast does not have that problem. sklearn on this Mac emits tens of thousands of spurious overflow/divide-by-zero matmul RuntimeWarnings: verified spurious by reproducing 33,015 of them on clean synthetic Gaussian input that yielded finite centroids and the full cluster count, so they do not indicate corrupted results here. This finding does NOT reverse [030]/[033]/[035] -- AVES still wins downstream. It narrows why: the target feature was not the weakness, which strengthens [039]'s corpus hypothesis.

### [042] Attentive probing does not close the gap, and learned pooling loses to mean pooling  ·  **CONFIRMED**

The published result that attentive probing lifts frozen audio encoders by 3-13 AUROC points does not transfer to this corpus. Every LEARNED readout tested -- attentive pooling, 4-head attentive pooling, a 2-layer transformer head -- scores BELOW plain mean pooling on both encoders. A better readout does raise both models slightly, via cheap statistical pooling rather than learned pooling, and it narrows the run11-AVES gap without closing it.

**Evidence.** 11-class call type, 3412 clips, 48 birds, leave-birds-out StratifiedGroupKFold(5), seed 0, majority 0.1797. Best overall is aves|L3|meanmax 0.8526, then aves|L3|multi 0.8517 and aves|L3|meanstd 0.8496 -- all unlearned pooling. Best run11 is run11|L1|multi 0.8280, then run11|L1|max 0.8262. Learned heads on AVES L3: mean_adamw 0.8376/0.8403 (2 seeds), attn4 0.8297/0.8195, attn_cat 0.8209/0.8168, trf 0.8200/0.8209, attn 0.8107/0.8054 -- all below the 0.8455 that plain mean pooling gets on the same features. Same ordering on run11. Against the linear-probe baselines (run11 0.8118, AVES 0.8453) the best readout buys run11 +0.0162 and AVES +0.0073, so the gap narrows from -0.0334 to -0.0246 and AVES still wins.

**Caveats.** Single hyperparameter setting per head (lr 1e-3, batch 64, attn_hidden 128, trf_dim 256, 2 layers, 4 heads, max 80 epochs, patience 12); a wider sweep could lift the learned heads, though the deficit is large and consistent across two seeds and two layers. Only L1 and L3 were swept. The BEATs comparison that motivated this is a different architecture and benchmark, so this refutes the transfer of the claim to this corpus, not the original result. 3412 clips is small for a learned pooling head, and that is the most likely mechanism for the loss.

### [043] Distilling the two-encoder ensemble into run11 alone beats the ensemble on detection  ·  **CONFIRMED**

The +0.0047 AUC / +0.0118 AP that [031] buys by averaging run11 and AVES probabilities can be recovered -- and slightly exceeded on AP -- while running only run11 at inference. An MLP head trained on run11 features against the ensemble's soft probabilities matches the two-encoder teacher on AUC and beats it on AP, at half the encoder cost. On call type, distillation does nothing.

**Evidence.** Detection, 90,061 frames, prevalence 0.1242, contiguous 60 s time blocks, bootstrap over 1500-frame blocks, 2000 resamples. Teachers reproduced to max|diff| 3.1e-05 against the published figures before anything was distilled. run11|mlp|kl reaches AUC 0.9740 / AP 0.8842 against the mean ensemble teacher 0.9732 / 0.8792: AP +0.0042 [+0.0013, +0.0072] (a_better) and AUC +0.0005 [-0.0004, +0.0015] (not distinguishable). Against the run11 linear probe alone it is AP +0.0159 [+0.0129, +0.0190] and AUC +0.0052 [+0.0040, +0.0063]. The KL objective beats plain CE on the same architecture: AP +0.0060 [+0.0029, +0.0096]. Distilling into AVES instead is worse everywhere (best aves|mlp|kl 0.9709 / 0.8742). On 11-class call type, cross-encoder feature distillation moves run11 from 0.8118 to at best 0.8265 (map_mlp_rev) and every arm remains below AVES 0.8453, six of eight significantly so.

**Caveats.** The detection split is contiguous time blocks WITHIN one recording, not a recording holdout, so this is an in-distribution result and does not speak to the BirdPark transfer in [030]. Soft targets are in-sample teacher probabilities, which are overconfident relative to held-out ones. The AUC gain over the teacher is not distinguishable -- the defensible claim is parity on AUC plus a small real AP gain at half the inference cost, not that the student beats the ensemble outright. Single seed on the student. Consistent with [031]: averaging helps, concatenating does not, and this is now the fifth time concatenation has failed on this project.

### [044] run11 is under-trained on both axes; the corrected compute budget  ·  **CONFIRMED**

AVES saw 9.30x more audio than run11 AND made 3.00x more passes over its own corpus. run11 is under-trained on both the total-compute axis and the data-diversity axis. This REPLACES finding 041, which claimed the opposite (that run11 made 1.7x more passes and so was under-diversified rather than under-trained); 041 is refuted, not merely refined.

**Evidence.** run11 93,750 updates x 80.3 realised audio-seconds/update = 2,091.3 h of audio seen, over a 116.03 h corpus = 18.0 epochs. aves-base-bio 100,000 updates x 700 audio-seconds/update = 19,444.4 h, over a 360 h corpus = 54.0 epochs. Ratios AVES/run11: steps 1.07x, audio-per-step 8.72x, total audio 9.30x, corpus 3.10x, epochs 3.00x. Peak LR 1e-4 vs 2e-4. Warmup fraction 0.033 vs 0.320. k-means clusters 100 vs 200. AVES wall clock 24.6 h on one GPU; run11 also used ONE GPU (having reserved four), under a 12 h limit, so <= 12 GPU-hours.

**Caveats.** Audio-seconds seen counts repeats, so it is a compute measure, not an information measure; the epochs row is the one that speaks to diversity, and here BOTH rows favour AVES, which is why the conclusion is unambiguous where 041's was not. The 360 h AVES corpus figure is still from Hagiwara 2023 Table 1 rather than machine-read, but the epoch self-check constrains it. This finding says run11 had less compute and less data -- it does NOT establish that more of either would close the gap, and it does not disturb the separate result that iterative target refinement did not help (039) or that the spectrogram target beat MFCC (040). The live test of whether more/better-matched training helps is the DAPT pair 38991843/38996893.

### [045] DAPT from AVES wins in-distribution and FAILS the holdout stop rule  ·  **CONFIRMED**

Continuing HuBERT pretraining on zebra finch from AVES weights beats run11 in-distribution but collapses on the encoder-level holdout, falling BELOW the energy baseline at the pre-committed arm. The pre-registered stop rule fired. This is the SONAR catastrophic-forgetting failure mode, observed in our own corpus, and it is reported as a failure rather than rescued by post-hoc arm selection.

**Evidence.** Jobs 38991843 (lr 5e-5) and 38996893 (lr 1e-4): AVES-init, run11's exact corpus and k=100 spectrogram labels, 15,000 steps, 1 L40S, ~37 min each. Pretraining objective reached run11's level in 6.25x fewer steps (masked accuracy 0.567 / 0.570 at 2.9 epochs vs run11 ~0.58 at 18 epochs), and the two LRs agree to 0.003. PRE-COMMITTED arm (max out-of-fold ZF AUC), both selected native_L6: in-distribution ZF AUC 0.9715 / 0.9720 vs run11 0.9687 -- deltas +0.0032 [+0.0013, +0.0053] and +0.0033 [+0.0013, +0.0055], both a_better; ZF AP +0.0080 [+0.0029, +0.0138] and +0.0050 [-0.0009, +0.0112]. ZF->BirdPark holdout AP 0.6942 / 0.7052 vs run11 0.8121 -- deltas -0.1002 [-0.1939, -0.0406] and -0.1032 [-0.2433, -0.0247] on 1500-frame blocks, both b_better. The log-energy baseline is AP 0.7472, so both DAPT arms sit BELOW it (-0.0530, -0.0420): the stop rule fired. No DAPT arm anywhere in the 12-arm grid dominates run11 on both axes.

**Caveats.** IMPORTANT and unresolved. The post-hoc best BirdPark arm for these same checkpoints is matched_L9: AP 0.8447 (5e-5) and 0.8623 (1e-4) -- the latter is the highest BirdPark AP of ANY model measured here, above birdaves-bioxn-large's 0.8442. That arm is NOT headlined because selecting it requires the test set. But its existence means the honest conclusion is narrower than "DAPT does not work": DAPT moves the model along an in-distribution/holdout TRADE-OFF, and the selection criterion picks the in-distribution end. DAPT's ZF-selected arms are consistently `native` (inheriting AVES's raw-waveform training) while its transferring arms are `matched` and deep. Whether DAPT helps cannot be settled without an arm-selection signal that is neither the ZF training set nor the BirdPark test set. See [046] for why the criterion itself is the bottleneck. EXTENDED BY [047] 2026-09-17: all 12 step-spaced checkpoints have now been scored. The endpoint result here holds and strengthens -- across the full trajectory ZERO checkpoints beat run11, 10 of 12 are significantly worse, and only 2 clear the energy floor. But the trajectory SHAPE is not measurable: the two learning rates trace opposite, crossing curves, so no early-stopping rule is licensed and "when the holdout collapsed" has no answer this holdout can give.

### [046] The layer-selection criterion has ~100x less resolution than its holdout consequence  ·  **CONFIRMED**

Picking the (normalisation, layer) arm by out-of-fold in-distribution ZF AUC is close to blind. Across the top-3 arms of a model, ZF AUC varies by 0.001-0.004 while their BirdPark AP varies by 0.05-0.15 -- a 20x to 98x amplification. The criterion cannot see the thing it is being used to decide, so every pre-committed holdout number in this project carries a selection lottery inside it.

**Evidence.** Top-3 arms by out-of-fold ZF AUC, spread in ZF AUC -> spread in ZF->BirdPark AP: run11 0.0015 -> 0.1501 (98x); aves-base-bio 0.0010 -> 0.0521 (51x); dapt5e5_step15000 0.0033 -> 0.1071 (32x); dapt1e4_step15000 0.0040 -> 0.0810 (20x). Concretely for run11: matched_L0 (ZF 0.9687, BP AP 0.8121), matched_L1 (0.9686, 0.7696), matched_L3 (0.9672, 0.6620) -- 0.0015 of ZF AUC separates arms whose holdout AP differs by 0.15. run11's selected arm happens to be its BEST BirdPark arm; DAPT's selected arm happens to be near its worst. Neither outcome was earned by the criterion.

**Caveats.** This does NOT invalidate the pre-commitment protocol -- pre-committing is still strictly better than choosing on the test set, and [project_run11_wins_nowhere] shows what happens without it. It says the protocol is under-powered here, which is a different and fixable problem. The fix needs a THIRD signal for arm selection, independent of both the ZF training set and the BirdPark test set: candidates are a held-out ZF RECORDING (the current split is contiguous 60 s blocks within one recording, explicitly "NOT a recording holdout"), or the chick corpus, or BirdPark internal CV restricted to birds not used in the reported score. Until then, holdout comparisons between models should report the ARM SPREAD alongside the point estimate, because the spread is larger than every between-model difference this project has reported.

### [047] No DAPT checkpoint beats run11 on the holdout, and the trajectory shape is not measurable  ·  **CONFIRMED**

Continued pretraining from AVES on ZF-only audio never improved held-out detection, at any of 6 checkpoints, at either of 2 learning rates. Ten of twelve are significantly WORSE than run11 and the best two are merely not distinguishable from it. Separately, the SHAPE of the degradation over training steps is not resolvable on BirdPark and must not be described as a curve.

**Evidence.** Paired cluster bootstrap (2000 resamples, 1500-frame blocks), BirdPark AP, each DAPT checkpoint minus run11 at the arm pre-committed on out-of-fold ZF AUC: 10 of 12 return verdict b_better (run11 better, CI excludes zero), worst dapt1e4_step2500 at -0.1499 [-0.2587, -0.0662]; 2 of 12 return not_distinguishable, dapt5e5_step2500 at -0.0360 [-0.0774, +0.0148] and dapt1e4_step10000 at -0.0553 [-0.1676, +0.0090]. ZERO return a_better. Only 2 of 12 clear the ZF->BP log-energy floor of AP 0.7472. The trajectories are NON-MONOTONIC and mutually contradictory: lr 5e-5 runs 0.7725 0.7259 0.6768 0.6808 0.6882 0.6942 (falls then recovers) while lr 1e-4 runs 0.6105 0.6884 0.7067 0.7515 0.7240 0.7052 (rises then falls), so the two learning rates CROSS and neither ordering holds across steps. In-distribution ZF AUC spans just 0.0018 (5e-5) and 0.0027 (1e-4) across the same checkpoints while BirdPark AP spans 0.0958 and 0.1410 -- a 50-70x amplification of the selection criterion.

**Caveats.** RETRACTION ON RECORD. An earlier reading of this same experiment, taken while only the 5e-5 arm had finished scoring, described a clean monotone forgetting curve and concluded that the lower learning rate was safer. The completed data refutes both: the arms trace opposite shapes and cross. The mistake was reading six points as a trend on a test set whose own bootstrap calls a 0.049 AP difference not distinguishable. What survives is only the between-model comparison, which is large enough to measure. Nothing here licenses an early-stopping rule -- picking step 2500 because it happens to clear the floor is selection on four independent 30 s blocks. The binding constraint is holdout size: BirdPark is 5925 frames. See finding 046 for the selection-resolution problem this compounds, and 045 for the endpoint-only result this supersedes.

### [048] The chick holdout does not adjudicate DAPT, and that is itself informative  ·  **CONFIRMED**

A second encoder-level holdout was built from the chick recordings to relieve BirdPark's low power. It does not resolve any DAPT comparison -- every model is not distinguishable from run11 on both readouts. But the DIRECTION is consistent and non-trivial: DAPT never degrades chick discrimination, while it significantly degrades BirdPark detection. The damage is specific to cross-LAB transfer, not a wholesale loss of representation.

**Evidence.** Paired bootstrap RESAMPLING CHICKS (2000 replicates for Be/LT AUC, 400 for AMI(k=4)), each model minus run11 at the arm pre-committed on out-of-fold ZF detection AUC. Be/LT AUC deltas span +0.0044 to +0.0083 and EVERY interval straddles zero (tightest dapt1e4_step15000 +0.0060 [-0.0004, +0.0178]; aves-base-bio +0.0076 [+0.0000, +0.0215]). AMI(k=4) deltas span -0.0228 to -0.1057, every interval straddles zero, and the intervals are enormous -- widths 0.20 to 0.43 (dapt5e5_step2500 -0.1057 [-0.3381, +0.0878]) -- so the unsupervised readout has essentially no power at 15 chicks. Point estimates that looked like results before bootstrapping and are NOT: every DAPT checkpoint appeared to beat run11 on Be/LT (0.9902-0.9943 vs 0.9858) and run11 appeared to beat everything on AMI (0.5361 vs 0.3857-0.4958, AVES 0.3873). Validation before use: this script reproduces the published run11 chick numbers -- native L1 Be/LT AUC 0.9904 vs 0.991 published, duration-only baseline 0.8108 vs 0.820.

**Caveats.** DO NOT treat this as a second axis that can settle DAPT questions -- it cannot, and the intervals say so. The likely reason it shows no damage is that chicks are NOT a domain holdout in the sense BirdPark is: they are zebra finches from the same corpus family, differing in individual and recording date, whereas BirdPark is a different lab, different room, different microphones. Continued pretraining on ZF colony audio should be expected to PRESERVE or help same-corpus discrimination while eroding cross-lab transfer, which is exactly the pattern observed. So this supports rather than contradicts [047]. The cohort filter depends on a pretrain-manifest file that no longer exists on disk, so the unseen-date selection is not locally re-verifiable; the cached name list is reused to preserve the published cohort. Getting real power on domain transfer needs a LARGER cross-lab holdout, not another same-corpus one.

### [049] Waveform normalisation is the largest preprocessing lever on transfer, and the selection rule discards it  ·  **CONFIRMED**

Matched chunk normalisation beats native raw waveform on the BirdPark holdout for EVERY model measured -- 23 of 23, with no exceptions -- at a median of +0.0405 AP. The same choice is worth essentially nothing in distribution (median +0.0000 AP across pretrained baselines). Because the pre-committed arm is selected on in-distribution ZF AUC, and native wins in distribution for the DAPT checkpoints, the selection rule systematically discards the preprocessing choice that carries transfer. For DAPT this is a 13x bad trade -- it buys 0.0067 of ZF AUC and pays 0.0861 of BirdPark AP.

**Evidence.** From the pre-commit block of detection_variants.json, which records both normalisations at each model's own best layer. matched minus native on BirdPark AP, all 23 entries positive: median +0.0405, min +0.0013 (birdaves-bioxn-large), max +0.1916 (dapt1e4_step2500). run11 itself +0.0841 (0.8121 matched vs 0.7281 native). Split by family: the 7 pretrained baselines show median dZF AUC +0.0000 and median dBP AP +0.0190; the 16 DAPT checkpoints show median dZF AUC -0.0067 and median dBP AP +0.0861. The sign flip in the first column is the mechanism -- DAPT inherits AVES's raw-waveform training condition, so native genuinely fits it better in distribution, and the criterion duly picks native. dapt1e4_step15000 is the sharpest case: its matched arm reaches BirdPark AP 0.8479, ABOVE run11's 0.8121, while its selected native arm scores 0.7052, below the 0.7472 log-energy floor. The same checkpoint is either the best model measured or a failed one depending only on this preprocessing switch.

**Caveats.** This is a per-model paired comparison without a bootstrap on the median, so "23 of 23" is the strength of the claim, not an interval -- individual deltas near +0.001 are well inside the +/-0.06 resolution of the BirdPark holdout and should not be read as real on their own. The claim is about the CONSISTENCY of the sign, which a sign test on 23 paired observations makes very unlikely by chance. It does not establish WHY matched normalisation transfers better; the obvious hypothesis -- that per-chunk normalisation removes a recording-level gain offset that differs between colonies -- is untested here. Complements [046], which shows the selection criterion cannot resolve arm differences at all; this finding shows the direction of the resulting bias for one specific variable. Does not retract [045]: DAPT still fails its pre-registered stop rule, but part of that failure is now attributable to preprocessing rather than to forgetting alone.

### [050] No label-free criterion picks k, but held-out-individual stability ranks encoders correctly  ·  **CONFIRMED**

Choosing k without labels is not a solvable problem on this corpus, because there is no k to find: cluster-then-vote accuracy rises monotonically with k until it reaches the exact 1-NN ceiling (peak minus 1-NN = -0.0006 to +0.0067 across five encoders). Every internal criterion that selects k by argmax is therefore answering a question with a degenerate answer. The well-posed label-free question is the OTHER one -- given a fixed k, which representation is better -- and there reproducibility criteria work while compactness criteria fail. Across the five encoders, mean Spearman rho against true cluster-vote accuracy over 12 values of k is +0.41 for bird-held-out stability and +0.37 for bootstrap stability (positive at 10/12 and 9/12 values of k), versus -0.50 for silhouette, -0.66 for Calinski-Harabasz, -0.30 for Davies-Bouldin, +0.03 for the gap statistic and -0.18 for prediction strength. Silhouette, CH and DB are not merely uninformative -- they rank the encoders BACKWARDS, positive at only 1 of 12 k each. The labelled reference, AMI, scores +0.73 and is positive at 12/12.

**Evidence.** Cohort 3412 clips / 11 call types / 48 birds, layer 3, ward linkage, leave-birds-out StratifiedGroupKFold(5). Accuracy-vs-k to the ceiling: run11 peak 0.7693 @k=650 vs 1-NN 0.7699; aves 0.7828 @k=1600 vs 0.7843; aves-base-all 0.8028 @k=1600 vs 0.7995; aves-base-core 0.7954 @k=450 vs 0.7887; birdaves-biox-base 0.7688 @k=300 vs 0.7649. The column-shuffled null is the only curve with an interior peak (0.1858 @k=30, falling to 0.1243 at 1-NN, against a majority floor of 0.1797) -- the peak-then-fall shape is itself the no-structure signature. Null separation at k=30, real median vs shuffled null: bootstrap stability 0.548 vs 0.0016 (346x), bird-held-out stability 0.489 vs 0.0014 (353x), CH 101 vs 1.94 (52x), silhouette 0.0853 vs -0.0022, prediction strength 0.193 vs 0.0397 (4.9x), gap 1.91 vs 1.45 (1.3x). On the extended grid bird-held-out stability tracks true accuracy for k in roughly 13-140 (rho +0.3 to +0.9) and loses the signal above k~200, where the accuracy spread between encoders collapses to 0.03-0.05. Stability's own argmax lands at k=4-25 (median 8), below the 11 human call types.

**Caveats.** The ranking result rests on FIVE encoders whose true accuracies span only 0.052, so any single rho is one swap from non-significance and individual values of +-0.3 are noise; the claim is carried by the sign consistency across k (10/12 and 9/12), not by any one correlation. All five encoders share an architecture, a dimensionality and a layer, so this tests ranking within a family, not across model classes. Stability's argmax at k=4-25 is partly methodological -- ARI declines with k for generic combinatorial reasons -- so it is suggestive of coarse super-groups consistent with the k=4 HDBSCAN result, NOT evidence that zebra finches have 8 call types. The 1-NN ceiling makes the cluster-vote metric a compression curve, which retrospectively reframes the oracle-k numbers in [project_calltype_clustering] as a statement about how much compression the geometry tolerates rather than about a correct number of clusters. Nothing here says stability would rank encoders correctly on a species with a different call repertoire; it says the procedure survived its one available calibration.


## Open questions

### [013] Multi-layer concatenation gives a small unconfirmed gain  ·  **OPEN**

Concatenating layers 0,2,4,6 with stronger regularization is the best configuration found, but the improvement is not statistically distinguishable from the published baseline.

**Evidence.** Eval A 0.9774 -> 0.9789, eval B 0.9557 -> 0.9602. Paired bootstrap: eval A +0.0016 [-0.0003, +0.0037]; eval B +0.0047 [-0.0007, +0.0100]. Both intervals include zero.

**Caveats.** Better or equal on both evals and never worse, so a reasonable default, but must not be reported as a significant improvement. Confirming it needs more held-out recordings, not more tuning.

### [023] Hysteresis helps onsets marginally, offset shrinking not at all  ·  **OPEN**

Replacing the single threshold with a dual high/low threshold and tuning for the 50 ms collar buys about +0.01 F1 at mid depth and nothing at shallow depth; the offset-shrink correction was rejected by the tuner in every fold.

**Evidence.** Collar-50 F1, v1 -> v2: L6 0.814 -> 0.823 (+0.009), L9 0.810 -> 0.816 (+0.006), L3 0.809 -> 0.804 (-0.004), L0 0.798 -> 0.797 (-0.001). Overlap F1 L6 0.842 -> 0.849. Chosen parameters were thr_hi 0.5-0.7 with thr_lo 0.3 (hysteresis genuinely used), merge_gap 0 in every block, shrink 0 in every block.

**Caveats.** The offset shrink was motivated by a real measured bias (+20 ms, duration ratio 1.20) and still did not help -- one frame is within the resolution, and shrinking breaks marginal overlap matches. No bootstrap has been run, so +0.009 must not be called an improvement yet. The likely bigger lever is a dedicated onset probe trained on 'is this frame within one frame of an onset', which needs the frame features (currently only on Savio).


## Refuted — do not retry without new evidence

- **[003] Detection does NOT decline with encoder depth** — The earlier claim that detection peaks at layer 0 and declines monotonically with depth does not survive human labels: layers 0-7 are a statistical plateau and only 8-11 are reliably worse. _Paired bootstrap, 2000 resamples. AUC L0-L6 = +0.0015 [-0.0014, +0.0043] on eval A and +0.0009 [-0.0036, +0.0056] on eval B. Accuracy L5-L0 = +0.0042 [-0.0025, +0.0105], L5-L6 = +0.0006 [-0.0043, +0.0052]. L5 vs L9/10/11 all significant._
- **[007] No pooling statistic beats the plain mean** — The hypothesis that mean-pooling dilutes the 80 ms call inside a 1 s window, and that a max or percentile statistic would recover it, is wrong: mean wins at every layer. _At all 6 tested layers, max / p90 / std / top-20% mean all score below mean on both evals (L0: mean A=0.9728 vs max 0.9691, p90 0.9604, std 0.9670, top20 0.9716). Concatenating all five gives only +0.0017 on A and +0.0022 on B._
- **[009] Nonlinear heads do not help** — MLP and gradient-boosted heads are worse than plain logistic regression on these features. _Eval B AUC: LR 0.9557, LinearSVC+calibration 0.9575, MLP-256 0.9547, MLP-512-128 0.9510, HistGradientBoosting 0.9497._
- **[041] REFUTED (see 044): run11 is not under-trained, it is under-diversified** — run11 and AVES got comparable optimiser budgets. AVES saw 2.13x more audio because its batch was twice as large, but it spread that over a corpus 3.6x bigger, so run11 actually made 1.7x MORE passes over its own data. More training on the same 100 h is therefore not the missing ingredient. _run11 93,750 updates x 350 audio-seconds/update = 9,114.6 h of audio seen, over a ~100 h corpus = 91.1 epochs. aves-base-bio 100,000 updates x 700 audio-seconds/update = 19,444.4 h, over a 360 h corpus = 54.0 epochs. Ratios AVES/run11: steps 1.07x, audio 2.13x, corpus 3.60x, epochs 0.59x. Peak LR 1e-4 vs 2e-4. Warmup fraction 0.033 vs 0.320 (fairseq stock 32k warmup left in place after cutting the run to 100k). k-means clusters 100 vs 200. AVES wall clock 24.6 h on ONE GPU; run11 requested 12 h on 4 GPUs so <= 48 GPU-hours, actual not recorded locally._
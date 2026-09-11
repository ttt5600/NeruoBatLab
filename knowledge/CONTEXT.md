# Project context: zebra finch HuBERT

Generated 2026-09-10. 18 findings: 15 live, 3 closed.

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

### [017] The representation transfers across labs, but the easy direction hides it  ·  **CONFIRMED**

A frame-level probe transfers between the Berkeley colony corpus and ETH BirdPark in both directions, but the gain over a log-energy baseline is only visible on the acoustically hard dataset.

**Evidence.** Frame AUC/AP. ZF->ZF L0 0.968/0.889 vs energy 0.794/0.390. BP->BP L0 0.938/0.933 vs energy 0.891/0.865. ZF->BP L0 0.888/0.893 vs energy 0.881/0.866 (delta only +0.008 AUC). BP->ZF L6 0.865/0.615 vs energy 0.781/0.335 (delta +0.084 AUC, +0.280 AP).

**Caveats.** BirdPark is close-miked and quiet (18.9 dB separation) so energy alone nearly solves it; ZF->BP therefore overstates what the encoder contributes. The informative direction is BP->ZF: trained on two minutes of another lab's audio and tested on noisy colony recordings, AP 0.615 vs energy 0.335. Prevalences differ (15.2% vs 52.2%) so AP, not AUC, is the comparable metric. Note the depth asymmetry: ZF->BP is best at layer 0 and decays to 0.757 by layer 6, while BP->ZF is best at layer 6 -- unexplained.

### [018] Frame-level detection produces usable onsets and offsets  ·  **CONFIRMED**

A 50 Hz frame probe on HuBERT features recovers 80% of annotated vocalization events at 88% precision with onset timing at the frame grid, vastly outperforming a log-energy detector through the identical decoder.

**Evidence.** Event level, layer 6: P=0.882 R=0.806 F1=0.842, 2039/2540 events found from 2317 emitted, onset |error| 20 ms. Layer 0: P=0.922 R=0.755 F1=0.829 at 12 ms. Energy through the same pipeline: P=0.448 R=0.736 F1=0.553 from 4084 emitted. Frame level AUC/AP: L0 0.969/0.890, L6 0.960/0.871, energy 0.794/0.390.

**Caveats.** Within-recording: trained and tested on different 6-minute blocks of 111021-000, so it measures localization ability, not generalization to a new recording. The precision/recall trade runs along depth -- layer 0 is the precise one (P 0.922) and layer 6 the sensitive one (R 0.806). A 20 ms onset error IS the frame grid, so these are as well aligned as the representation can express. Two bugs were caught by assertions here: each 20 s chunk yielded 999 frames not 1000, silently leaving 90 all-zero rows, and the energy baseline's grid search twice sat on a grid edge which made its F1 a floor rather than a maximum.


## Open questions

### [013] Multi-layer concatenation gives a small unconfirmed gain  ·  **OPEN**

Concatenating layers 0,2,4,6 with stronger regularization is the best configuration found, but the improvement is not statistically distinguishable from the published baseline.

**Evidence.** Eval A 0.9774 -> 0.9789, eval B 0.9557 -> 0.9602. Paired bootstrap: eval A +0.0016 [-0.0003, +0.0037]; eval B +0.0047 [-0.0007, +0.0100]. Both intervals include zero.

**Caveats.** Better or equal on both evals and never worse, so a reasonable default, but must not be reported as a significant improvement. Confirming it needs more held-out recordings, not more tuning.


## Refuted — do not retry without new evidence

- **[003] Detection does NOT decline with encoder depth** — The earlier claim that detection peaks at layer 0 and declines monotonically with depth does not survive human labels: layers 0-7 are a statistical plateau and only 8-11 are reliably worse. _Paired bootstrap, 2000 resamples. AUC L0-L6 = +0.0015 [-0.0014, +0.0043] on eval A and +0.0009 [-0.0036, +0.0056] on eval B. Accuracy L5-L0 = +0.0042 [-0.0025, +0.0105], L5-L6 = +0.0006 [-0.0043, +0.0052]. L5 vs L9/10/11 all significant._
- **[007] No pooling statistic beats the plain mean** — The hypothesis that mean-pooling dilutes the 80 ms call inside a 1 s window, and that a max or percentile statistic would recover it, is wrong: mean wins at every layer. _At all 6 tested layers, max / p90 / std / top-20% mean all score below mean on both evals (L0: mean A=0.9728 vs max 0.9691, p90 0.9604, std 0.9670, top20 0.9716). Concatenating all five gives only +0.0017 on A and +0.0022 on B._
- **[009] Nonlinear heads do not help** — MLP and gradient-boosted heads are worse than plain logistic regression on these features. _Eval B AUC: LR 0.9557, LinearSVC+calibration 0.9575, MLP-256 0.9547, MLP-512-128 0.9510, HistGradientBoosting 0.9497._
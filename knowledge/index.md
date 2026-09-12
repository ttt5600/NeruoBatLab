# Findings index

Generated 2026-09-12 by `build.py`. Do not edit by hand; edit `findings/*.yaml`.

29 findings.


## CONFIRMED (23)

### [001] The corpus is the public Elie and Theunissen release  ·  **CONFIRMED**

All 120 pretraining recordings come from the publicly released Elie & Theunissen zebra finch corpus (CC BY 4.0, figshare).

**Evidence.** 120 recordings, 26 birds, 2867 curated AdultVocalization clips across 64 dates.

**Method.** Provenance traced from the preprocessed_audio directory back to the published release.

**Caveats.** Consequence: no online zebra finch dataset derived from this release can supply a holdout, because it is the same audio. An external holdout must come from a different lab. See [016].

**Provenance.** run5-4-26-full/data/spectrogram/preprocessed_audio; memory project_corpus_is_public_elie_theunissen

### [002] 111021-000 is inside the pretraining manifest  ·  **CONFIRMED**

The recording used as the detection holdout was itself in the run11 pretraining file list, so every detection number measured on it is a probe-level holdout and not an encoder-level one.

**Evidence.** Line 31 of 100 in the derived holdout manifest: preprocessed_audio/111021-000.wav with 77329020 frames, matching the file exactly.

**Method.** Proven by derivation, not inference. slurm/00_build_tsv.sh reads run11's own spectrogram/tsv/ZF_test_pipeline_train.tsv line by line and drops 11 dates; 111021 is not among them, so anything in the output was in the input. train_lowprio.sh --dataset-path confirms which manifest the trainer reads.

**Caveats.** The encoder heard the audio unlabeled via masked prediction; it never saw any annotation, and the probe was trained on entirely different recordings. So the transfer result is real, but optimistic relative to genuinely unseen audio by an unknown margin. Resolved by [016] and [017].

**Provenance.** 00_build_tsv.sh; temp_files/holdout_lblred0613/data/spectrogram/tsv/; verified 2026-09-10

### [004] Energy VAD collapses to chance on exhaustively annotated windows  ·  **CONFIRMED**

A log-energy detector is at chance on 1 s windows of exhaustively annotated colony audio, while HuBERT holds AUC 0.955 on the identical windows.

**Evidence.** Energy AUC 0.532 (accuracy 0.685, exactly the 0.676 majority rate); HuBERT 0.955. Measured loudness gap between voc and non-voc windows is only +1.66 dB.

**Method.** Eval B: probe trained on the 3768 neg-pool windows, tested on 1801 one-second windows tiling 111021-000.

**Caveats.** Window-level and specific to the window length. At 20 ms frame level the same feature reaches AUC 0.794, because a frame is either inside a call or not, whereas a 1 s window containing an 80 ms call is ~92% background. Never quote 0.532 as 'energy does not work' without the frame-level number beside it.

**Provenance.** job 38678496

### [005] Unannotated is not negative, and it has bitten twice  ·  **CONFIRMED**

Treating unlistened audio as negative has corrupted this project's numbers twice, the second time during the very session documenting the first.

**Evidence.** (1) The original benchmark's 4900 negatives were never listened to; 62.6% of the judged ones contain a call. (2) The cross-dataset job initially scored all 80.6 min of 111021-000 against annotations covering only the first 30.0 min, dropping ZF frame AUC from 0.968 to 0.849 and voiced fraction from 15.2% to 5.7%.

**Method.** Case 2 was caught only because the voiced-fraction printout did not match the known 15.2%.

**Caveats.** Standing rule: every dataset records which span was actually listened to, and scoring truncates to it. labeled=0 means nobody listened and must never default to y=0.

**Provenance.** export_negatives_for_labeling.py; eval_crossdataset.py limit_to_annotated; job 38704761 (bad) vs 38705017 (fixed)

### [006] Iteration 2 helps in-distribution only  ·  **CONFIRMED**

run12 (iteration-2, layer-6 teacher, k=500) is reliably better than run11 on the neg-pool windows and not better on a held-out recording, so the gain does not transfer.

**Evidence.** Layer-matched paired bootstrap: eval A run12 better at 9/12 layers, worse at 0/12 (+0.003 to +0.006). Eval B better at 0/12, worse at 3/12 (deep layers, to -0.018). Layer chosen on A and reported on B: run12(L6)-run11(L0) = -0.0065 [-0.0164, +0.0029], not distinguishable. run13 (layer-3 teacher): 4/12 better on A, 0/12 either way on B.

**Method.** Same layer compared to same layer, all 12, rather than each run's argmax. The argmax version said +0.0044 SIGNIFICANT and was layer-selection bias.

**Caveats.** run11 remains the release model; there is no held-out evidence for switching.

**Provenance.** jobs 38704167, 38704168; analysis/run_comparison_fair.json

### [008] StandardScaler costs 0.005 AUC on these features  ·  **CONFIRMED**

Standardizing HuBERT features before logistic regression consistently hurts detection accuracy.

**Evidence.** Layer 0 mean, eval A: 0.9774 unscaled vs 0.9728 scaled. float16 storage accounts for 0.0001 of that; the scaler accounts for the rest.

**Method.** Same features, same CV, scaler toggled; confirmed across C in {0.03, 0.3, 1.0, 10.0}.

**Caveats.** Mechanism: under a shared L2 penalty, leaving the natural variance in place gives high-variance dimensions effectively weaker regularization, which is the better prior here. Recheck if the head changes.

**Provenance.** analysis/head_sweep.json

### [010] The detector fires on digital silence with P=1.000  ·  **CONFIRMED**

All-zero audio is classified as a vocalization with maximum confidence, because it lands far outside the training distribution and the linear probe extrapolates.

**Evidence.** 20 all-zero clips all score P(voc)=1.000. Their feature vector is deterministic (spread 1.2e-6) and sits 2.1 sd from the real-data mean on average, up to 9.9 sd, with 7% of layer-0 dimensions beyond 5 sd.

**Method.** Probe trained on real data only, applied frozen to synthetic clips.

**Caveats.** FIX, verified: an energy floor at -80 dB rejects silence and touches zero real windows (quietest real window is -68.6 dB). Recordings with dropouts or muted channels would otherwise produce confident false detections.

**Provenance.** job 38704952; analysis/synthetic_probe.json

### [011] The detector is not a general bird-vocalization detector  ·  **CONFIRMED**

Probed with signals of known structure, it accepts call-like harmonic and FM structure but also brown noise, AM noise and pure tones, so it encodes a contrast against this colony's background rather than a general notion of vocalization.

**Evidence.** Mean P(voc): harmonic stack 1.000, FM sweep 1.000, real call in silence 0.996, brown noise 0.993, pure tone 0.892, AM noise 0.840, click train 0.538, pink noise 0.325, white noise 0.120, real background 0.384.

**Method.** 940 synthetic 1 s clips; probe trained on real data only and never updated. Window-scope normalization on both sides.

**Caveats.** White and pink noise correctly rejected while brown noise is accepted suggests sensitivity to spectral tilt, not harmonicity alone. Do not deploy on a new acoustic environment without re-measuring the false-alarm rate.

**Provenance.** job 38704952; analysis/synthetic_probe.png

### [012] Detection sensitivity reaches 50 percent at about -10 dB SNR  ·  **CONFIRMED**

With real calls mixed into real background from the same recording, detection stays above the background false-alarm floor down to roughly -10 dB call-to-background SNR.

**Evidence.** Fraction detected at P>0.5: +20 dB 0.97, +10 dB 1.00, 0 dB 0.88, -5 dB 0.68, -10 dB 0.55, -15 dB 0.38, -20 dB 0.35. Background alone sits at 0.32.

**Method.** 40 mixtures per SNR level; calls 60-250 ms from annotated events, background from verified unvoiced stretches of the same recording.

**Caveats.** The 0.32 background false-alarm rate at threshold 0.5 is high and matches the independent eval-B false-positive rate (169/584 = 29%). Read the curve as 'above floor', not as absolute detection rate.

**Provenance.** job 38704952; analysis/synthetic_probe.json

### [014] Eval B is a cross-domain transfer, not a same-domain test  ·  **CONFIRMED**

The two halves of the detection dataset are almost perfectly separable from the features alone, so eval B measures transfer across a real domain gap.

**Evidence.** A probe predicting SOURCE (neg-pool 0.5 s windows vs 111021-000 1.0 s windows) from layer 6 reaches AUC 0.9989.

**Method.** 5-fold CV on the source label using the same features the detection probe consumes.

**Caveats.** Makes eval B's 0.955 stronger than it reads, and explains the miscalibrated 0.5 threshold: accuracy 0.883 at 0.5, 0.892 prior-matched, 0.907 oracle, with AUC 0.955 throughout.

**Provenance.** analysis/evalB_threshold.json

### [015] Eval-B false positives are two distinct populations  ·  **CONFIRMED**

The false positives split by confidence into loud broadband transients (genuine errors) and tonal events indistinguishable from true calls (possibly unannotated calls).

**Evidence.** Marginal FPs (0.5<p<=0.944, n=117): spectral flatness 0.220 vs 0.111 for true positives, p=2.5e-10; loud and non-tonal. High-confidence FPs (p>0.944, n=52): flatness 0.122, and no measure separates them from true positives (flatness p=0.41, pitch p=0.25, peak dB p=0.13).

**Method.** Acoustic statistics on the loudest 120 ms of each window. Over the full 1 s the same statistics cannot separate even true positives from true negatives (p=0.84), so the window-level version of this test has no power.

**Caveats.** n=52 is underpowered; indistinguishable from true positives is not is a call. All 211 errors exported as audio with the 52 high-confidence ones first, so a human can settle it.

**Provenance.** analysis/evalB_fp_two_populations.json; analysis/evalB_errors.wav

### [016] BirdPark is a genuine encoder-level holdout  ·  **CONFIRMED**

Zenodo record 20608098 (BirdPark, Hahnloser lab ETH Zurich) is zebra finch audio with exact onset/offset annotations that cannot be in our pretraining corpus, giving the project its first encoder-level holdout.

**Evidence.** 720 valid annotations (1 NaN row dropped) merging to 532 events over 118.5 s; 52.2% of 20 ms frames voiced. Recorded 2020-2021, published 2026-06-09, CC-BY-4.0.

**Method.** Downloaded via the open Zenodo API. Audio is the mean of wav channels 2-5 resampled 24414 -> 16000 Hz.

**Caveats.** Channel identification matters and is counterintuitive: channels 0-1 are the LOUDEST but have spectral centroid ~500-650 Hz with 99% of energy below 4 kHz -- they are backpack accelerometers. Channels 2-6 are the microphones (centroid ~4 kHz). A lag scan confirms it: annotated regions exceed gaps by 18.9 dB on the mic mix and only 2.0 dB on the accelerometer mix. Channel 6 is excluded because it is silent 52% of the time. Annotations verified approximately exhaustive: only 11.9% of unannotated 120 ms windows exceed the 25th percentile of annotated ones.

**Provenance.** ~/zf_labelset/external/birdpark; prep_birdpark.py

### [018] Frame-level detection produces usable onsets and offsets  ·  **CONFIRMED**

A 50 Hz frame probe on HuBERT features recovers 80% of annotated vocalization events at 88% precision with onset timing at the frame grid, vastly outperforming a log-energy detector through the identical decoder.

**Evidence.** Event level, layer 6: P=0.882 R=0.806 F1=0.842, 2039/2540 events found from 2317 emitted, onset |error| 20 ms. Layer 0: P=0.922 R=0.755 F1=0.829 at 12 ms. Energy through the same pipeline: P=0.448 R=0.736 F1=0.553 from 4084 emitted. Frame level AUC/AP: L0 0.969/0.890, L6 0.960/0.871, energy 0.794/0.390.

**Method.** Frames are 20 ms. Split is contiguous 6-minute blocks, NOT random frames -- adjacent frames are near-identical so a random split leaks across the fold boundary. Decoder is median-smooth then threshold then min-duration then gap-bridge, all four knobs tuned out of fold. Event matching is one-to-one by overlap via a two-pointer sweep over sorted non-overlapping intervals.

**Caveats.** Within-recording: trained and tested on different 6-minute blocks of 111021-000, so it measures localization ability, not generalization to a new recording. The precision/recall trade runs along depth -- layer 0 is the precise one (P 0.922) and layer 6 the sensitive one (R 0.806). A 20 ms onset error IS the frame grid, so these are as well aligned as the representation can express. Two bugs were caught by assertions here: each 20 s chunk yielded 999 frames not 1000, silently leaving 90 all-zero rows, and the energy baseline's grid search twice sat on a grid edge which made its F1 a floor rather than a maximum.

**Provenance.** job 38704640; analysis/onset_events.json; analysis/onset_timeline.png; analysis/predicted_intervals_L6.csv

### [019] ZF to BirdPark is the clean direction, and I reported the wrong one  ·  **CONFIRMED**

Of the two cross-dataset directions only ZF->BP tests on audio the encoder never heard, and it is the direction where HuBERT barely beats energy; the direction with the large margin (BP->ZF) tests on a recording that IS in the pretraining manifest.

**Evidence.** ZF->BP: tests on BirdPark (unseen), AUC 0.888 vs energy 0.881 (+0.008), AP 0.893 vs 0.866 (+0.028). BP->ZF: tests on 111021-000 (in the manifest, see [002]), AUC 0.865 vs energy 0.781 (+0.084), AP 0.615 vs 0.335 (+0.280).

**Method.** Probe trained on one corpus, tested on the other; the TEST set is what must be unseen for an encoder-level claim.

**Caveats.** We currently have NO direction that is both clean and convincing. ZF->BP is clean but BirdPark is close-miked and quiet (18.9 dB separation) so energy nearly solves it; BP->ZF has the large margin but a contaminated test set. Closing this needs a HARD, noisy, colony-like recording that is not in the pretraining corpus. Also note the training-set asymmetry: ZF->BP fits on 30.0 min / 90061 frames, BP->ZF on 118.5 s / 5925 frames.

**Provenance.** job 38705017; analysis/crossdataset.json

Supersedes: 017

### [020] Loudness is the dominant embedding axis and removing it is catastrophic  ·  **CONFIRMED**

The single largest direction in the embedding correlates with loudness and carries most of the detection signal; projecting it out destroys performance, so 'correlated with loudness' must not be mistaken for 'is a loudness detector'.

**Evidence.** PC0 explains 52.8% of feature variance and correlates 0.724 with log-energy. Unsupervised k-means aligns far better with loudness quintile (AMI 0.335 at k=8) than with voc/noise (0.169) or recording identity (0.043). Removing it: drop PC0 costs -0.194 eval A AUC, top-3 -0.204, top-8 -0.243, while PCA-64 with nothing dropped costs only -0.004. Partialling out the linear loudness axis costs -0.078. Decomposed: PC0 alone gives AUC 0.851, log-energy alone 0.791, PC0 with its energy component regressed out 0.745, both together 0.851.

**Method.** Three independent removal methods (per-dimension partial-out, PC dropping, within-recording rank normalization), each fit on training data only and applied to both evals.

**Caveats.** This does NOT contradict [004]. Energy is at chance on eval B (0.532) but reaches 0.791 on eval A, so loudness is genuinely informative on the neg-pool windows. PC0 beats the scalar it correlates with by +0.060, i.e. it carries spectral structure the scalar throws away. Practical: do not 'normalize away' loudness, and do not read the UMAP's left-right gradient as a vocalization axis -- it is a loudness axis.

**Provenance.** analysis/loudness_removal.json, analysis/what_dominates.json, analysis/umap_labels_clusters.png

### [021] Linear separability and neighbourhood purity peak at different depths  ·  **CONFIRMED**

Flat linear AUC across layers 0-7 hides a real geometric change: local neighbourhood purity and cluster separation improve with depth while linear separability slowly degrades.

**Evidence.** Across L0->L11: linear AUC 0.9774 -> 0.9714 (flat then declining), kNN-10 accuracy 0.883 -> 0.906 (peaking L5), silhouette 0.195 -> 0.220 (peaking L5) -> 0.205, Fisher ratio 0.573 -> 0.533 (minimum L7), PCA dim at 90% variance 15 -> 51 (monotone). The 4-way human label (call / call+noise / noise / silence) is recoverable at 0.806 accuracy against a 0.515 majority, best at L0. Clean call vs call+noise is only AUC 0.77 at every depth.

**Method.** Leave-recordings-out CV on the 3768 neg-pool windows; silhouette on a 3000-point cosine subsample.

**Caveats.** Explains why AUC picks L0 and accuracy picks L5 ([003]): they measure different geometry. The monotone growth of PCA dimensionality also rules out representational collapse with depth, which was a live concern earlier in the project.

**Provenance.** analysis/embedding_stats.json

### [022] Onset and offset detection in detail, and where it actually fails  ·  **CONFIRMED**

Overlap-F1 flatters the detector; under a strict onset collar performance drops sharply, and the dominant errors are deletions of short calls and merges of consecutive ones.

**Evidence.** Layer 6 F1 by onset tolerance: overlap 0.840, 500 ms 0.865, 200 ms 0.852, 100 ms 0.843, 50 ms 0.814, 20 ms 0.736. Boundary error: onset median 20 ms with bias 0 and p90 40 ms; offset median 20 ms but bias +20 ms and p90 60 ms, with predicted/true duration ratio 1.20. Error taxonomy at L6: 305 deletions, 265 insertions, 154 merges, 21 fragmentations from 2317 predictions against 2540 events. Recall by true duration: 0.685 for 0-50 ms (n=504), 0.877 for 50-80 ms, 0.938 for 80-120 ms, 0.981 above 200 ms. Median IoU 0.667.

**Method.** Decoder parameters tuned out of fold on contiguous 6-minute blocks. Collar matching requires the predicted onset within the tolerance; overlap matching accepts any overlap.

**Caveats.** Energy has HIGHER recall in every duration bin (0.764 on 0-50 ms vs 0.685) but only by emitting 4084 predictions against HuBERT's 2317 -- its precision is 0.45 vs 0.88 and it makes 2252 insertions. Short calls are the real weakness: a 50 ms call is 2.5 frames at 20 ms resolution.

**Provenance.** analysis/onset_granular.json; job 38704640

### [024] The detector is not reading loudness  ·  **CONFIRMED**

When loudness is held constant by design, the detector still separates vocalizations from background almost as well as it does unrestricted, while loudness itself falls to chance.

**Evidence.** Within narrow dB bands (quintiles), weighted-mean AUC: eval A HuBERT 0.9629 vs energy 0.5641; eval B HuBERT 0.9542 vs energy 0.5580, against unrestricted 0.9774 / 0.9557. On 1:1 pairs matched within 0.5 dB (843 pairs eval A, 528 eval B, mean signed gap +0.008 / +0.005 dB), HuBERT ranks the call above its loudness twin 0.9609 / 0.9394 of the time while energy sits at 0.5302 / 0.5114.

**Method.** Two designs that remove loudness by construction rather than by projection: stratify AUC within dB bands so no comparison crosses loudness, and greedily pair each positive with the NEAREST unused negative in dB. Energy's own paired win rate is the validity check -- it must be ~0.5, and it is.

**Caveats.** The first version of the pair matcher scanned upward from j-200 and took the FIRST negative within tolerance, which systematically paired positives with quieter negatives and gave energy a spurious 0.926 win rate. Caught because the printed label said 'near chance by construction' and the number was not. Always check the control's own null.

**Provenance.** analysis/energy_matched.json, analysis/energy_matched_pairs.json

### [025] HuBERT significantly beats a strong log-mel spectrogram baseline  ·  **CONFIRMED**

The self-supervised representation is worth a large, significant margin over hand-designed spectral features, and the margin more than doubles on the held-out recording.

**Evidence.** Best log-mel: eval A 0.9267, eval B 0.8364. HuBERT layer 0: 0.9774 / 0.9557. Paired bootstrap, HuBERT untuned vs mel tuned: eval A +0.0507 [+0.0406, +0.0613] SIGNIFICANT; eval B +0.1189 [+0.0937, +0.1477] SIGNIFICANT. Log-energy alone for reference: 0.7911 / 0.5324.

**Method.** 384-d baseline built to be strong, not a strawman: 64 log-mel bands (25 ms / 10 ms hop, to 8 kHz) summarized by mean, std, max, p90 plus delta mean and std; no normalization, so it KEEPS absolute loudness that HuBERT's context normalization partly removes. Regularization swept for the baseline and not for HuBERT.

**Caveats.** Alignment was verified before scoring: window energy recomputed from local audio matches the Savio-extracted Xen at corr 1.000000, median |diff| 0.000 dB. Concatenating HuBERT with log-mel is WORSE than HuBERT alone (A 0.9717, B 0.9396), so the mel features add nothing the representation lacks. Still outstanding: a random-init encoder control, which would separate 'the architecture' from 'the pretraining'.

**Provenance.** analysis/mel_baseline.json, analysis/mel_significance.json

### [026] Pipeline sanity controls pass  ·  **CONFIRMED**

The detection result is not a leak and is not carried by a handful of easy recordings.

**Evidence.** Shuffled-label control: 0.4914 +- 0.0140 over 5 permutations against 0.9774 on real labels. Per-recording AUC across 53 recordings with both classes: median 0.9839, mean 0.9734, IQR [0.9591, 0.9994], min 0.8571, and 0 of 53 below 0.80. The five largest recordings hold 30% of windows and average 0.9751 versus 0.9732 for the rest.

**Method.** Labels permuted inside the same StratifiedGroupKFold plumbing; per-recording AUC computed from the out-of-fold predictions.

**Caveats.** The shuffled control validates the splitting and prediction path only; it cannot detect a problem in the LABELS themselves, which is why [005] and [015] matter separately.

**Provenance.** analysis/sanity_controls.json

### [027] Leave-recordings-out is bird-leaky, but detection does not care  ·  **CONFIRMED**

The recording-grouped CV lets 96.8% of test-fold birds also appear in training, yet switching to a genuinely bird-disjoint split costs nothing measurable for detection.

**Evidence.** Recording-grouped 5-fold: 96.8% of test birds also in train (3 of 5 folds at 100%). Same 3564 windows: group=recording AUC 0.9741, group=bird-component AUC 0.9721. Delta -0.0020, paired bootstrap over components -0.0017 [-0.0040, +0.0009], not distinguishable.

**Method.** Bird identity recovered from curated clip filenames (bird_datecode-CallType-idx.wav). Recordings are colony recordings: median 2 birds per date (max 7), median 5 dates per bird, so holding out a recording does not hold out its birds. Dates sharing a bird must stay together, which yields 7 bird-disjoint components (19/16/11/7/5/1/1 dates); 5 of them contain eval-A windows.

**Caveats.** Contrast with [project_probe_split_inflation]: the CALL-TYPE probe inflates by +0.114 under a random split. Detection does not, which is coherent -- 'is a call present' is not a bird-specific judgement while 'which call type' partly is. Note 29 birds after merging HPiHPi4748/HpiHpi4748, which differ only by case, plus an 'Unknown000' placeholder that is not a bird. This is a PROBE-level split only; see [028] for the encoder.

**Provenance.** analysis/leave_birds_out.json; bird_map.json

### [028] No zebra finch bird is held out of pretraining, and none can be  ·  **CONFIRMED**

Every bird in the corpus appears in the pretraining manifest, so no ZF result is a bird-level encoder holdout; the only birds the encoder has never heard are BirdPark's.

**Evidence.** All 120 recordings in run5-4-26-full/data/spectrogram/preprocessed_audio are in the training tsv (verified in [002]), and those recordings cover all 29 birds across 60 datecodes. BirdPark contributes 16 birds (8 pairs) from a different lab.

**Method.** Manifest membership from [002] combined with the bird-to-date mapping from the curated clip filenames.

**Caveats.** A bird-level encoder holdout could in principle be built by retraining on one of the 7 bird-disjoint components -- the holdout_lblred0613 experiment already did this for one bird, at the cost of 16.7% of pretraining audio, which confounds 'never heard this bird' with 'less data'. Until such a run exists, bird-level generalization for ZF is untested and only BirdPark speaks to it.

**Provenance.** 00_build_tsv.sh; bird_map.json; analysis/leave_birds_out.json

### [029] What the AUC numbers mean in operational terms  ·  **CONFIRMED**

At a fixed 5% false-alarm budget on the held-out recording, the representation finds 1.8x as many calls as a strong spectrogram and 10x as many as an energy detector.

**Evidence.** Eval B, 5% false alarms: HuBERT recovers 1016/1217 calls (83.5%), log-mel 573/1217 (47.1%), log-energy 102/1217 (8.4%), each with 29 false alarms out of 584 non-voc windows. As ranking-error rate (1-AUC): HuBERT 4.43%, log-mel 16.36%, energy 46.76% -- HuBERT removes 73% of the spectrogram's errors. On eval A at 5%: 92.6% / 73.8% / 15.4%.

**Method.** ROC operating points interpolated from the same out-of-fold and held-out predictions used for the AUC comparisons.

**Caveats.** AUC 0.9557 means: pick one random call window and one random non-call window, and the probe ranks the call higher 95.57% of the time. It is threshold-free -- the operating points above are what a chosen threshold turns it into, and the right threshold depends on whether misses or false alarms cost more for the study.

**Provenance.** analysis/mel_baseline.json; computed alongside analysis/mel_significance.json


## REFUTED (3)

### [003] Detection does NOT decline with encoder depth  ·  **REFUTED**

The earlier claim that detection peaks at layer 0 and declines monotonically with depth does not survive human labels: layers 0-7 are a statistical plateau and only 8-11 are reliably worse.

**Evidence.** Paired bootstrap, 2000 resamples. AUC L0-L6 = +0.0015 [-0.0014, +0.0043] on eval A and +0.0009 [-0.0036, +0.0056] on eval B. Accuracy L5-L0 = +0.0042 [-0.0025, +0.0105], L5-L6 = +0.0006 [-0.0043, +0.0052]. L5 vs L9/10/11 all significant.

**Method.** Cluster bootstrap over the 71 recordings for eval A; 30 s moving-block bootstrap for eval B, because adjacent 1 s windows are temporally correlated.

**Caveats.** The original claim came from cross-correlation ground truth in which silence-pinned positives made the task partly 'silence vs audio'. Practical consequence: use whatever layer is convenient within 0-7.

**Provenance.** job 38678496; analysis/layer_bootstrap.txt; acc_vs_auc.py

### [007] No pooling statistic beats the plain mean  ·  **REFUTED**

The hypothesis that mean-pooling dilutes the 80 ms call inside a 1 s window, and that a max or percentile statistic would recover it, is wrong: mean wins at every layer.

**Evidence.** At all 6 tested layers, max / p90 / std / top-20% mean all score below mean on both evals (L0: mean A=0.9728 vs max 0.9691, p90 0.9604, std 0.9670, top20 0.9716). Concatenating all five gives only +0.0017 on A and +0.0022 on B.

**Method.** Five pooling statistics extracted per window per layer from the 24-49 HuBERT frames each window contains; identical CV protocol.

**Caveats.** The hypothesis was motivated by a correct observation and still turned out wrong. Frame-level modelling is what actually exploits the short call, not a different window-level statistic.

**Provenance.** job 38704687; analysis/pooling_comparison.json

### [009] Nonlinear heads do not help  ·  **REFUTED**

MLP and gradient-boosted heads are worse than plain logistic regression on these features.

**Evidence.** Eval B AUC: LR 0.9557, LinearSVC+calibration 0.9575, MLP-256 0.9547, MLP-512-128 0.9510, HistGradientBoosting 0.9497.

**Method.** Best single layer, same splits, early stopping for the MLPs.

**Caveats.** Not an exhaustive architecture search; it says the easy nonlinear wins are absent, not that none exist.

**Provenance.** analysis/head_sweep.json


## SUPERSEDED (1)

### [017] The representation transfers across labs, but the easy direction hides it  ·  **SUPERSEDED**

A frame-level probe transfers between the Berkeley colony corpus and ETH BirdPark in both directions, but the gain over a log-energy baseline is only visible on the acoustically hard dataset.

**Evidence.** Frame AUC/AP. ZF->ZF L0 0.968/0.889 vs energy 0.794/0.390. BP->BP L0 0.938/0.933 vs energy 0.891/0.865. ZF->BP L0 0.888/0.893 vs energy 0.881/0.866 (delta only +0.008 AUC). BP->ZF L6 0.865/0.615 vs energy 0.781/0.335 (delta +0.084 AUC, +0.280 AP).

**Method.** Probe trained on all of one recording set, tested on the other; within-set references use contiguous 6-minute blocks (ZF) and leave-one-pair-out over 8 pairs (BirdPark). Both truncated to their annotated spans.

**Caveats.** BirdPark is close-miked and quiet (18.9 dB separation) so energy alone nearly solves it; ZF->BP therefore overstates what the encoder contributes. The informative direction is BP->ZF: trained on two minutes of another lab's audio and tested on noisy colony recordings, AP 0.615 vs energy 0.335. Prevalences differ (15.2% vs 52.2%) so AP, not AUC, is the comparable metric. Note the depth asymmetry: ZF->BP is best at layer 0 and decays to 0.757 by layer 6, while BP->ZF is best at layer 6 -- unexplained.

**Provenance.** job 38705017; analysis/crossdataset.json

> Superseded by [019].


## OPEN (2)

### [013] Multi-layer concatenation gives a small unconfirmed gain  ·  **OPEN**

Concatenating layers 0,2,4,6 with stronger regularization is the best configuration found, but the improvement is not statistically distinguishable from the published baseline.

**Evidence.** Eval A 0.9774 -> 0.9789, eval B 0.9557 -> 0.9602. Paired bootstrap: eval A +0.0016 [-0.0003, +0.0037]; eval B +0.0047 [-0.0007, +0.0100]. Both intervals include zero.

**Method.** Configuration selected on eval A only; eval B reported but never used for selection. ~60 configurations swept over pooling, scaling, regularization, layer sets and head type.

**Caveats.** Better or equal on both evals and never worse, so a reasonable default, but must not be reported as a significant improvement. Confirming it needs more held-out recordings, not more tuning.

**Provenance.** analysis/head_sweep.json, analysis/improvement_significance.json

### [023] Hysteresis helps onsets marginally, offset shrinking not at all  ·  **OPEN**

Replacing the single threshold with a dual high/low threshold and tuning for the 50 ms collar buys about +0.01 F1 at mid depth and nothing at shallow depth; the offset-shrink correction was rejected by the tuner in every fold.

**Evidence.** Collar-50 F1, v1 -> v2: L6 0.814 -> 0.823 (+0.009), L9 0.810 -> 0.816 (+0.006), L3 0.809 -> 0.804 (-0.004), L0 0.798 -> 0.797 (-0.001). Overlap F1 L6 0.842 -> 0.849. Chosen parameters were thr_hi 0.5-0.7 with thr_lo 0.3 (hysteresis genuinely used), merge_gap 0 in every block, shrink 0 in every block.

**Method.** 2160 configurations per fold over smoothing, dual thresholds, min duration, gap bridging and offset shrink; all tuned out of fold and scored on the held-out block.

**Caveats.** The offset shrink was motivated by a real measured bias (+20 ms, duration ratio 1.20) and still did not help -- one frame is within the resolution, and shrinking breaks marginal overlap matches. No bootstrap has been run, so +0.009 must not be called an improvement yet. The likely bigger lever is a dedicated onset probe trained on 'is this frame within one frame of an onset', which needs the frame features (currently only on Savio).

**Provenance.** analysis/onset_events_v2.json

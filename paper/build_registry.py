#!/usr/bin/env python
"""Single source of truth for every number the manuscript is allowed to state.

Numbers in a paper drift. A value gets copied into the text, the analysis is rerun with a fixed bug,
and the text keeps the stale figure -- and nothing complains, because prose has no type checker.
This builds `results.json` by pulling each citable value out of the analysis JSON it came from, by
explicit path, and `audit_numbers.py` then refuses any numeral in the manuscript that does not match.

The spec is deliberately verbose: one line per citable number, naming the file and the dotted path.
A missing path is a hard failure, not a skip, so that when an analysis script changes shape the
registry breaks loudly instead of going quietly stale.
"""
from __future__ import annotations
import json, sys, datetime
from pathlib import Path

ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
HERE = Path(__file__).resolve().parent
OUT = HERE / "results.json"

# key                                   file                      dotted path                                      fmt    note
SPEC = [
    # ---- target features: what the spectrogram-over-MFCC choice bought (finding 040)
    ("tgt.spec_k100.ami_call",           "target_features.json",  "grid.soundsig4000_k100.ami_call",                "{:.4f}", "run11's ACTUAL iteration-1 feature, 4000-d soundsig STFT"),
    ("tgt.spec_k200.ami_call",           "target_features.json",  "grid.soundsig4000_k200.ami_call",                "{:.4f}", ""),
    ("tgt.mfcc_k100.ami_call",           "target_features.json",  "grid.mfcc39_k100.ami_call",                      "{:.4f}", "AVES's iteration-1 feature, fairseq stock 39-d MFCC"),
    ("tgt.mfcc_k200.ami_call",           "target_features.json",  "grid.mfcc39_k200.ami_call",                      "{:.4f}", ""),
    ("tgt.logmel_k100.ami_call",         "target_features.json",  "grid.logmel384_k100.ami_call",                   "{:.4f}", "beats run11's own 4000-d feature"),
    ("tgt.spec_k100.ami_voc",            "target_features.json",  "grid.soundsig4000_k100.ami_voc",                 "{:.4f}", ""),
    ("tgt.mfcc_k100.ami_voc",            "target_features.json",  "grid.mfcc39_k100.ami_voc",                       "{:.4f}", ""),
    ("tgt.spec_minus_mfcc.k100",         "target_features.json",  "effects.spectrogram_minus_mfcc_at_k100_ami_call", "{:+.4f}", "the spectrogram choice, isolated at run11's k"),
    ("tgt.spec_minus_mfcc.k200",         "target_features.json",  "effects.spectrogram_minus_mfcc_at_k200_ami_call", "{:+.4f}", "same sign and size at AVES's k, so it is a feature effect not a k artifact"),
    ("tgt.run11_minus_aves_iter1",       "target_features.json",  "effects.run11_actual_minus_aves_iter1_ami_call",  "{:+.4f}", "as-shipped vs as-shipped"),

    # ---- training budget, machine-read from the configs (finding 041)
    ("budget.run11.updates",             "compute_budget.json",   "run11.updates",                                  "{:d}",   "from slurm/train_iter7_long_lowprio.sh"),
    ("budget.aves.updates",              "compute_budget.json",   "aves.updates",                                   "{:d}",   "from the fairseq cfg in the 2022 checkpoint"),
    ("budget.run11.sec_per_update",      "compute_budget.json",   "run11.audio_seconds_per_update",                 "{:.1f}", "REALISED: corpus_seconds / steps_per_epoch. NOT 4 x 87.5 -- --gpus 4 was never honoured"),
    ("budget.aves.sec_per_update",       "compute_budget.json",   "aves.audio_seconds_per_update",                  "{:.1f}", "max_tokens 1.4e6/16000 x update_freq 8"),
    ("budget.run11.audio_hours",         "compute_budget.json",   "run11.total_audio_hours",                        "{:,.1f}", "audio-seconds seen, counting repeats"),
    ("budget.aves.audio_hours",          "compute_budget.json",   "aves.total_audio_hours",                         "{:,.1f}", ""),
    ("budget.run11.epochs",              "compute_budget.json",   "run11.epochs_over_corpus",                       "{:.1f}", "corroborated by Lightning's own counter, 18"),
    ("budget.aves.epochs",               "compute_budget.json",   "aves.epochs_over_corpus",                        "{:.1f}", "corroborated by fairseq's own counter, 55"),
    ("budget.audio_ratio",               "compute_budget.json",   "comparison.audio_hours_ratio_aves_over_run11",   "{:.2f}", "AVES saw 9.30x more audio"),
    ("budget.epochs_ratio",              "compute_budget.json",   "comparison.epochs_ratio",                        "{:.2f}", "AND 3.00x the passes -- run11 under-trained on both axes"),
    ("budget.run11.corpus_hours",        "compute_budget.json",   "run11.corpus_hours",                             "{:.2f}", "from the training TSV (120 files), NOT the README's ~100 h"),
    ("budget.aves.corpus_hours",         "compute_budget.json",   "aves.corpus_hours",                              "{:.1f}", "Hagiwara 2023 Table 1"),
    ("budget.run11.world_size",          "compute_budget.json",   "run11.world_size",                               "{:d}",   "what srun launched; the script reserved 4 GPUs and used 1"),
    ("budget.run11.warmup_frac",         "compute_budget.json",   "run11.warmup_fraction",                          "{:.3f}", ""),
    ("budget.aves.warmup_frac",          "compute_budget.json",   "aves.warmup_fraction",                           "{:.3f}", "fairseq stock 32k warmup left in after cutting the run to 100k"),
    ("budget.aves.wall_hours",           "compute_budget.json",   "aves.wall_clock_seconds",                        "{:.0f}", "seconds; 24.6 h on ONE GPU"),

    # ---- readout: attentive probing does not transfer (finding 042)
    ("readout.aves_best.acc",            "attentive_probe.json",  "results[aves|L3|meanmax].acc",                   "{:.4f}", "best of 52 arms, and it is UNLEARNED pooling"),
    ("readout.run11_best.acc",           "attentive_probe.json",  "results[run11|L1|multi].acc",                    "{:.4f}", ""),
    ("readout.aves_attn.acc",            "attentive_probe.json",  "results[aves|L3|attn|s0].acc",                   "{:.4f}", "learned attentive pooling, below mean pooling"),
    ("readout.aves_trf.acc",             "attentive_probe.json",  "results[aves|L3|trf|s0].acc",                    "{:.4f}", "2-layer transformer head, also below"),
    ("readout.aves_mean.acc",            "attentive_probe.json",  "results[aves|L3|mean].acc",                      "{:.4f}", ""),
    ("readout.run11_mean.acc",           "attentive_probe.json",  "results[run11|L3|mean].acc",                     "{:.4f}", ""),

    # ---- distillation (finding 043)
    ("distil.student_best.auc",          "distillation.json",     "exp1_ensemble_distillation_detection.students[run11|mlp|kl].auc", "{:.4f}", "run11 features, ensemble soft targets"),
    ("distil.student_best.ap",           "distillation.json",     "exp1_ensemble_distillation_detection.students[run11|mlp|kl].ap",  "{:.4f}", "beats the two-encoder teacher"),
    ("distil.teacher_mean.auc",          "distillation.json",     "exp1_ensemble_distillation_detection.meta.published_reference.mean", "{:.4f}", ""),
    ("distil.vs_teacher.ap_delta",       "distillation.json",     "exp1_ensemble_distillation_detection.bootstrap[run11|mlp|kl__vs__teacher_mean__ap].delta", "{:+.4f}", ""),
    ("distil.vs_teacher.ap_lo",          "distillation.json",     "exp1_ensemble_distillation_detection.bootstrap[run11|mlp|kl__vs__teacher_mean__ap].lo",    "{:+.4f}", ""),
    ("distil.vs_teacher.ap_hi",          "distillation.json",     "exp1_ensemble_distillation_detection.bootstrap[run11|mlp|kl__vs__teacher_mean__ap].hi",    "{:+.4f}", ""),
    ("distil.ct_best.acc",               "distillation_exp2.json", "exp2_cross_encoder_feature_distillation.arms.map_mlp_rev.acc",   "{:.4f}", "call type: still below AVES"),
    # ---- frame-level detection, the core benchmark
    ("frame.n",                          "frame_baselines.json",  "n_frames",                                      "{:d}",   "20 ms frames, 30.02 min of 111021-000"),
    ("frame.prevalence",                 "frame_baselines.json",  "prevalence",                                     "{:.4f}", "fraction of frames whose centre is in a call"),
    ("frame.run11_L0.auc",               "frame_baselines.json",  "full.hubert_L0.auc",                             "{:.4f}", "best run11 layer at frame level"),
    ("frame.run11_L0.ap",                "frame_baselines.json",  "full.hubert_L0.ap",                              "{:.4f}", ""),
    ("frame.run11_L6.auc",               "frame_baselines.json",  "full.hubert_L6.auc",                             "{:.4f}", ""),
    ("frame.logmel100.auc",              "frame_baselines.json",  "full.logmel_100ms.auc",                          "{:.4f}", "mel given 100 ms context, 4x HuBERT's receptive field"),
    ("frame.logmel100.ap",               "frame_baselines.json",  "full.logmel_100ms.ap",                           "{:.4f}", ""),
    ("frame.logmel25.auc",               "frame_baselines.json",  "full.logmel_25ms.auc",                           "{:.4f}", "mel matched to the 25 ms receptive field"),
    ("frame.logenergy.auc",              "frame_baselines.json",  "full.logenergy.auc",                             "{:.4f}", ""),
    ("frame.logenergy.ap",               "frame_baselines.json",  "full.logenergy.ap",                              "{:.4f}", ""),
    ("frame.gap_vs_mel.auc",             "frame_baselines.json",  "hubert_minus_mel100_auc",                        "{:+.4f}", ""),
    ("frame.gap_vs_mel.ap",              "frame_baselines.json",  "hubert_minus_mel100_ap",                         "{:+.4f}", "the honest headline: AP, not AUC, at 12% prevalence"),

    # ---- AVES: does domain-specific pretraining buy anything
    ("aves.zf.run11_L0.auc",             "aves_baseline.json",    "results.run11_L0.auc",                           "{:.4f}", ""),
    ("aves.zf.best_aves.auc",            "aves_baseline.json",    "best_aves.auc",                                  "{:.4f}", ""),
    ("aves.zf.best_aves.name",           "aves_baseline.json",    "best_aves.name",                                 "{}",     ""),
    ("aves.zf.run11_minus_aves.auc",     "aves_baseline.json",    "run11_minus_aves_auc",                           "{:+.4f}", "in-distribution: essentially a tie"),
    ("aves.zf.run11_minus_aves.ap",      "aves_baseline.json",    "run11_minus_aves_ap",                            "{:+.4f}", ""),
    ("aves.bp.run11_L0.auc",             "aves_holdout.json",     "zf_to_bp.run11_L0.auc",                          "{:.4f}", "ZF->BirdPark, the clean encoder-level holdout"),
    ("aves.bp.run11_L0.ap",              "aves_holdout.json",     "zf_to_bp.run11_L0.ap",                           "{:.4f}", ""),
    ("aves.bp.run11_L6.auc",             "aves_holdout.json",     "zf_to_bp.run11_L6.auc",                          "{:.4f}", "deep ZF layers transfer worst"),
    ("aves.bp.aves_L3.auc",              "aves_holdout.json",     "zf_to_bp.aves_L3.auc",                           "{:.4f}", "best AVES layer on the holdout"),
    ("aves.bp.aves_L3.ap",               "aves_holdout.json",     "zf_to_bp.aves_L3.ap",                            "{:.4f}", ""),
    ("aves.bp.logenergy.auc",            "aves_holdout.json",     "zf_to_bp.logenergy.auc",                         "{:.4f}", "BirdPark is close-miked; energy nearly solves it"),
    ("aves.bp.logenergy.ap",             "aves_holdout.json",     "zf_to_bp.logenergy.ap",                          "{:.4f}", ""),
    ("aves.bp.n_frames",                 "aves_holdout.json",     "birdpark.n_frames_scored",                       "{:d}",   "why the holdout bootstrap has no power"),
    ("aves.bp.prevalence",               "aves_holdout.json",     "birdpark.prevalence",                            "{:.4f}", ""),
    ("aves.bp.n_events",                 "aves_holdout.json",     "birdpark.n_events",                              "{:d}",   ""),

    # ---- what pretraining buys (random-init control)
    ("randinit.n",                       "randinit_control.json", "n",                                              "{:d}",   "1 s windows, soundsep_111021"),
    ("randinit.prevalence",              "randinit_control.json", "prevalence",                                     "{:.4f}", "majority rate, NOT 0.5"),
    ("randinit.cnn.pretrained",          "randinit_control.json", "layer_table.cnn.pretrained",                     "{:.4f}", ""),
    ("randinit.cnn.rand_mean",           "randinit_control.json", "layer_table.cnn.rand_mean",                      "{:.4f}", "3 seeds"),
    ("randinit.cnn.gap",                 "randinit_control.json", "layer_table.cnn.gap_vs_rand",                    "{:+.4f}", "the CNN front end is where the pretraining lives"),
    ("randinit.L6.pretrained",           "randinit_control.json", "layer_table.L6.pretrained",                      "{:.4f}", ""),
    ("randinit.L6.gap",                  "randinit_control.json", "layer_table.L6.gap_vs_rand",                     "{:+.4f}", ""),
    ("randinit.logmel",                  "randinit_control.json", "baselines.logmel",                               "{:.4f}", ""),
    ("randinit.logenergy",               "randinit_control.json", "baselines.logenergy",                            "{:.4f}", ""),

    # ---- temporal resolution
    ("res.w1000.windowed_L0",            "resolution_sweep.json", "w1000.center.windowed_L0",                       "{:.4f}", "centre label: length-neutral"),
    ("res.w1000.continuous_L0",          "resolution_sweep.json", "w1000.center.continuous_L0",                     "{:.4f}", ""),
    ("res.w1000.overlap_windowed_L0",    "resolution_sweep.json", "w1000.overlap.windowed_L0",                      "{:.4f}", "same audio, 'any overlap' label -- 0.185 higher"),
    ("res.w125.windowed_L0",             "resolution_sweep.json", "w125.center.windowed_L0",                        "{:.4f}", "the crossover"),
    ("res.w125.continuous_L0",           "resolution_sweep.json", "w125.center.continuous_L0",                      "{:.4f}", ""),
    ("res.w40.windowed_L0",              "resolution_sweep.json", "w40.center.windowed_L0",                         "{:.4f}", ""),
    ("res.w40.continuous_L0",            "resolution_sweep.json", "w40.center.continuous_L0",                       "{:.4f}", "context still helps at 40 ms"),
    ("res.frames.L0",                    "resolution_sweep.json", "frames_20ms.L0",                                 "{:.4f}", ""),

    # ---- sub-frame onsets
    ("onset.coarse_nearest.mae_ms",      "subframe_onset.json",   "arms.coarse_nearest.med_abs_err_ms",             "{:.1f}", ""),
    ("onset.coarse_nearest.bias_ms",     "subframe_onset.json",   "arms.coarse_nearest.bias_ms",                    "{:+.1f}", "exactly HOP/2 = 10 ms: the nearest-frame rule is late by construction"),
    ("onset.coarse_interp.mae_ms",       "subframe_onset.json",   "arms.coarse_interp.med_abs_err_ms",              "{:.1f}", "interpolating the crossing is free"),
    ("onset.coarse_interp.bias_ms",      "subframe_onset.json",   "arms.coarse_interp.bias_ms",                     "{:+.1f}", ""),
    ("onset.shifted_K4_nearest.mae_ms",  "subframe_onset.json",   "arms.shifted_K4_nearest.med_abs_err_ms",         "{:.1f}", "4x the encoder compute, no better than interpolation"),
    ("onset.shifted_K4_interp.mae_ms",   "subframe_onset.json",   "arms.shifted_K4_interp.med_abs_err_ms",          "{:.1f}", ""),
    ("onset.n_true",                     "subframe_onset.json",   "n_true",                                         "{:d}",   ""),

    # ---- events and offsets at grid resolution
    ("event.L6.collar50_f1",             "onset_events_v2.json",  "L6.overall.collar_f1",                                 "{:.3f}", ""),
    ("event.L6.overlap_f1",              "onset_events_v2.json",  "L6.overall.overlap_f1",                                  "{:.3f}", "overlap matching flatters long predictions"),
    ("event.energy.collar50_f1",         "onset_events_v2.json",  "energy_prob.overall.collar_f1",                        "{:.3f}", ""),
    ("offset.L6.mae_ms",                 "onset_granular.json",   "L6.offset_ms.median_abs",                        "{:.0f}", "quantised to the 20 ms grid"),
    ("offset.L6.bias_ms",                "onset_granular.json",   "L6.offset_ms.bias",                              "{:+.0f}", "predictions run long -- model or annotator, real data cannot say"),
    ("offset.L6.duration_ratio",         "onset_granular.json",   "L6.duration_ratio_median",                       "{:.2f}", ""),
    ("recall.L6.short_calls",            "onset_granular.json",   "L6.recall_by_duration.0-50 ms.recall",           "{:.3f}", "a 50 ms call is 2.5 frames"),
    ("recall.L6.long_calls",             "onset_granular.json",   "L6.recall_by_duration.200-inf ms.recall",        "{:.3f}", ""),

    # ---- failure modes
    ("fail.brown_noise",                 "synthetic_probe.json",  "categories.noise_brown.mean",                    "{:.3f}", "accepted, while white noise is rejected: spectral tilt, not harmonicity"),
    ("fail.white_noise",                 "synthetic_probe.json",  "categories.noise_white.mean",                    "{:.3f}", ""),
    ("fail.digital_silence",             "synthetic_probe.json",  "categories.digital_silence.mean",                "{:.3f}", "the probe extrapolating outside its training hull"),
    ("fail.real_background",             "synthetic_probe.json",  "categories.real_background.mean",                "{:.3f}", "false-alarm floor at threshold 0.5"),
    ("fail.real_call",                   "synthetic_probe.json",  "categories.real_call_in_silence.mean",           "{:.3f}", ""),

    # ---- operating points
    ("op.evalB.hubert_recall_at_5fa",    "mel_baseline.json",     None,                                             "{}",     "see finding 029; not stored as a field"),
    # ---- window-level, the published comparison (eval A in-distribution, eval B held-out recording)
    ("win.evalA.hubert_L0.auc",          "mel_baseline.json",     "hubert_L0.A_auc",                                "{:.4f}", ""),
    ("win.evalB.hubert_L0.auc",          "mel_baseline.json",     "hubert_L0.B_auc",                                "{:.4f}", "held-out recording"),
    ("win.evalA.logmel.auc",             "mel_baseline.json",     "[logmel_C0.03].A_auc",                             "{:.4f}", "regularisation swept for the baseline, not for HuBERT"),
    ("win.evalB.logmel.auc",             "mel_baseline.json",     "[logmel_C0.03].B_auc",                             "{:.4f}", ""),
    ("win.evalA.energy.auc",             "mel_baseline.json",     "energy_only.A_auc",                              "{:.4f}", ""),
    ("win.evalB.energy.auc",             "mel_baseline.json",     "energy_only.B_auc",                              "{:.4f}", "energy collapses to near chance on exhaustive labels"),
    ("win.evalA.hubert_plus_mel.auc",    "mel_baseline.json",     "hubert_plus_mel.A_auc",                          "{:.4f}", "concatenating mel makes it WORSE"),
    ("win.gap.evalA",                    "mel_significance.json", "eval A (in-distribution).delta",                 "{:+.4f}", ""),
    ("win.gap.evalA.lo",                 "mel_significance.json", "eval A (in-distribution).lo",                    "{:+.4f}", ""),
    ("win.gap.evalA.hi",                 "mel_significance.json", "eval A (in-distribution).hi",                    "{:+.4f}", ""),
    ("win.gap.evalB",                    "mel_significance.json", "eval B (held-out recording).delta",              "{:+.4f}", ""),
    ("win.gap.evalB.lo",                 "mel_significance.json", "eval B (held-out recording).lo",                 "{:+.4f}", ""),
    ("win.gap.evalB.hi",                 "mel_significance.json", "eval B (held-out recording).hi",                 "{:+.4f}", ""),

    # ---- AVES, every layer, so the pre-committed layer choice is auditable
    ("aves.bp.aves_L0.auc",              "aves_holdout.json",     "zf_to_bp.aves_L0.auc",                           "{:.4f}", ""),
    ("aves.bp.aves_L6.auc",              "aves_holdout.json",     "zf_to_bp.aves_L6.auc",                           "{:.4f}", ""),
    ("aves.bp.aves_L9.auc",              "aves_holdout.json",     "zf_to_bp.aves_L9.auc",                           "{:.4f}", ""),
    ("aves.bp.aves_L0.ap",               "aves_holdout.json",     "zf_to_bp.aves_L0.ap",                            "{:.4f}", ""),
    ("aves.bp.aves_L6.ap",               "aves_holdout.json",     "zf_to_bp.aves_L6.ap",                            "{:.4f}", ""),
    ("aves.bp.aves_L9.ap",               "aves_holdout.json",     "zf_to_bp.aves_L9.ap",                            "{:.4f}", ""),
    ("aves.bp.run11_L6.ap",              "aves_holdout.json",     "zf_to_bp.run11_L6.ap",                           "{:.4f}", ""),
    ("aves.bp.internal_run11_L0.auc",    "aves_holdout.json",     "bp_internal.run11_L0.auc",                       "{:.4f}", "within-BirdPark CV: not a holdout, shows BP is separable"),
    ("aves.bp.internal_aves_L6.auc",     "aves_holdout.json",     "bp_internal.aves_L6.auc",                        "{:.4f}", ""),
    ("aves.zf.aves_matched_L3.auc",      "aves_baseline.json",    "results.aves_matched_L3.auc",                    "{:.4f}", ""),
    ("aves.zf.aves_native_L3.auc",       "aves_baseline.json",    "results.aves_native_L3.auc",                     "{:.4f}", "AVES's own preprocessing, not ours"),
    ("aves.zf.aves_native_L3.ap",        "aves_baseline.json",    "results.aves_native_L3.ap",                      "{:.4f}", ""),

    # ---- the ensemble: the one improvement that survived a bootstrap
    ("ens.zf.run11.auc",                 "ensemble_detect.json",  "in_distribution.run11.auc",                      "{:.4f}", ""),
    ("ens.zf.aves.auc",                  "ensemble_detect.json",  "in_distribution.aves.auc",                       "{:.4f}", ""),
    ("ens.zf.concat.auc",                "ensemble_detect.json",  "in_distribution.concat.auc",                     "{:.4f}", "1536-d concat is WORSE than run11 alone"),
    ("ens.zf.mean.auc",                  "ensemble_detect.json",  "in_distribution.mean.auc",                       "{:.4f}", "averaging two probabilities: no new parameters"),
    ("ens.zf.mean.ap",                   "ensemble_detect.json",  "in_distribution.mean.ap",                        "{:.4f}", ""),
    ("ens.zf.stack.ap",                  "ensemble_detect.json",  "in_distribution.stack.ap",                       "{:.4f}", ""),
    ("ens.error_correlation",            "ensemble_detect.json",  "error_correlation",                              "{:.4f}", "how much decorrelated error there is to exploit"),
    ("ens.boot.mean_vs_run11.auc",       "ensemble_detect.json",  "bootstrap.zf_mean_vs_run11_auc.delta",           "{:+.4f}", ""),
    ("ens.boot.mean_vs_run11.lo",        "ensemble_detect.json",  "bootstrap.zf_mean_vs_run11_auc.lo",              "{:+.4f}", ""),
    ("ens.boot.mean_vs_run11.hi",        "ensemble_detect.json",  "bootstrap.zf_mean_vs_run11_auc.hi",              "{:+.4f}", ""),

    # ---- failure modes, remaining categories
    ("fail.pure_tone",                   "synthetic_probe.json",  "categories.pure_tone.mean",                      "{:.3f}", ""),
    ("fail.am_noise",                    "synthetic_probe.json",  "categories.am_noise.mean",                        "{:.3f}", ""),
    ("fail.click_train",                 "synthetic_probe.json",  "categories.click_train.mean",                     "{:.3f}", ""),
    ("fail.pink_noise",                  "synthetic_probe.json",  "categories.noise_pink.mean",                      "{:.3f}", ""),
    ("fail.harmonic_stack",              "synthetic_probe.json",  "categories.harmonic_stack.mean",                  "{:.3f}", ""),
    ("fail.fm_sweep",                    "synthetic_probe.json",  "categories.fm_sweep.mean",                        "{:.3f}", ""),
    ("snr.frac_at_0db",                  "synthetic_probe.json",  "snr_sweep[0.0].frac",                              "{:.2f}", "window-level, against a 0.32 background floor"),
    ("snr.frac_at_-10db",                "synthetic_probe.json",  "snr_sweep[-10.0].frac",                            "{:.2f}", ""),

    # ---- random-init, remaining
    ("randinit.L0.pretrained",           "randinit_control.json", "layer_table.L0.pretrained",                      "{:.4f}", ""),
    ("randinit.L0.rand_mean",            "randinit_control.json", "layer_table.L0.rand_mean",                       "{:.4f}", ""),
    ("randinit.L0.gap",                  "randinit_control.json", "layer_table.L0.gap_vs_rand",                     "{:+.4f}", ""),
    ("randinit.cnn.shuf_mean",           "randinit_control.json", "layer_table.cnn.shuf_mean",                      "{:.4f}", "value-preserving weight shuffle"),
    ("randinit.cnn.knn",                 "randinit_control.json", "layer_table.cnn.knn_pretrained",                 "{:.4f}", ""),

    # ---- events, remaining
    ("event.L0.collar50_f1",             "onset_events_v2.json",  "L0.overall.collar_f1",                           "{:.3f}", ""),
    ("event.L0.overlap_f1",              "onset_events_v2.json",  "L0.overall.overlap_f1",                          "{:.3f}", ""),
    ("event.L6.collar_p",                "onset_events_v2.json",  "L6.overall.collar_p",                            "{:.3f}", ""),
    ("event.L6.collar_r",                "onset_events_v2.json",  "L6.overall.collar_r",                            "{:.3f}", ""),
    ("event.energy.overlap_f1",          "onset_events_v2.json",  "energy_prob.overall.overlap_f1",                  "{:.3f}", ""),
    ("offset.L6.p90_ms",                 "onset_granular.json",   "L6.offset_ms.p90_abs",                           "{:.0f}", ""),
    ("onset.probe.L0.boundary_auc",      "onset_probe.json",      "layers.L0.boundary_frame_auc",                    "{:.4f}", "dedicated 'is this frame near an onset' probe"),
    ("onset.probe.L6.boundary_auc",      "onset_probe.json",      "layers.L6.boundary_frame_auc",                    "{:.4f}", ""),
# ---- synthetic timing: ground truth exact to the sample, no annotator in the loop
    ("synth.core.L0.onset_mae_ms",       "synth_timing.json",     "conditions.isolated[20.0].layers.L0.onset_mae_ms",        "{:.1f}", "isolated, +20 dB: the encoder's intrinsic acuity"),
    ("synth.core.L0.offset_mae_ms",      "synth_timing.json",     "conditions.isolated[20.0].layers.L0.offset_mae_ms",       "{:.1f}", "first offset measurement free of label noise"),
    ("synth.core.L0.offset_bias_ms",     "synth_timing.json",     "conditions.isolated[20.0].layers.L0.offset_bias_ms",      "{:+.1f}", "vs +20 ms measured against hand labels"),
    ("synth.core.L0.duration_ratio",     "synth_timing.json",     "conditions.isolated[20.0].layers.L0.duration_ratio_median", "{:.2f}", "vs 1.20 against hand labels"),
    ("synth.core.L0.recall",             "synth_timing.json",     "conditions.isolated[20.0].layers.L0.recall",              "{:.3f}", ""),
    ("synth.core.L0.auc",                "synth_timing.json",     "conditions.isolated[20.0].layers.L0.frame_auc",           "{:.3f}", ""),
    ("synth.core.L0.r_at_0db",           "synth_timing.json",     "conditions.isolated[0.0].layers.L0.recall",               "{:.3f}", ""),
    ("synth.core.L0.r_at_-5db",          "synth_timing.json",     "conditions.isolated[-5.0].layers.L0.recall",              "{:.3f}", ""),
    ("synth.core.L0.dur_at_0db",         "synth_timing.json",     "conditions.isolated[0.0].layers.L0.duration_ratio_median", "{:.2f}", "at low SNR only the loud core is found"),
    ("synth.natural.L0.recall",          "synth_timing.json",     "conditions.natural[20.0].layers.L0.recall",               "{:.3f}", "same SNR, realistic gaps: crowding costs more than SNR"),
    ("synth.natural.L0.auc",             "synth_timing.json",     "conditions.natural[20.0].layers.L0.frame_auc",            "{:.3f}", ""),
    ("synth.natural.median_gap_ms",      "synth_timing.json",     "conditions.natural[20.0].median_gap_ms",                  "{:.0f}", ""),
    ("synth.isolated.median_gap_ms",     "synth_timing.json",     "conditions.isolated[20.0].median_gap_ms",                 "{:.0f}", ""),
    ("synth.bg.L0.false_per_min",        "synth_timing.json",     "conditions.isolated[20.0].background_only.L0.per_min",    "{:.1f}", "false events per minute on call-free background"),
    ("synth.n_donors",                   "synth_timing.json",     "config.n_donors",                                         "{:d}",   "clean donor calls from the held-out 40% of the recording"),
    ("synth.taper_ms",                   "synth_timing.json",     "config.taper_ms",                                         "{:.1f}", "1/20th of a frame, auditable against the ~5 ms result"),
    ("synth.tails.L0.offset_bias_ms",    "synth_timing_tails.json", "conditions.isolated[20.0].layers.L0.offset_bias_ms",    "{:+.1f}", "quiet tails retained, truth at the core: STILL no late bias"),
    ("synth.tails.L0.duration_ratio",    "synth_timing_tails.json", "conditions.isolated[20.0].layers.L0.duration_ratio_median", "{:.2f}", "so the 1.20 on real data is not tails either"),
    ("synth.tails.L0.onset_bias_ms",     "synth_timing_tails.json", "conditions.isolated[20.0].layers.L0.onset_bias_ms",     "{:+.1f}", ""),
# ---- ensemble on the holdout (pre-committed AVES layer)
    ("ens.bp.run11.auc",                 "ensemble_detect.json",  "zf_to_bp.run11.auc",                             "{:.4f}", ""),
    ("ens.bp.aves.auc",                  "ensemble_detect.json",  "zf_to_bp.aves.auc",                              "{:.4f}", ""),
    ("ens.bp.concat.auc",                "ensemble_detect.json",  "zf_to_bp.concat.auc",                            "{:.4f}", ""),
    ("ens.bp.mean.auc",                  "ensemble_detect.json",  "zf_to_bp.mean.auc",                              "{:.4f}", "the ensemble gain survives out of distribution"),
    ("ens.bp.mean.ap",                   "ensemble_detect.json",  "zf_to_bp.mean.ap",                               "{:.4f}", ""),
    ("ens.bp.boot.mean_vs_run11",        "ensemble_detect.json",  "bootstrap.bp1500_mean_vs_run11_auc.delta",       "{:+.4f}", ""),
    ("ens.bp.boot.mean_vs_run11.lo",     "ensemble_detect.json",  "bootstrap.bp1500_mean_vs_run11_auc.lo",          "{:+.4f}", ""),
    ("ens.bp.boot.mean_vs_run11.hi",     "ensemble_detect.json",  "bootstrap.bp1500_mean_vs_run11_auc.hi",          "{:+.4f}", ""),
    ("ens.bp.boot.aves_vs_run11",        "ensemble_detect.json",  "bootstrap.bp1500_aves_vs_run11_auc.delta",       "{:+.4f}", "pre-committed layer: NOT distinguishable"),
    ("ens.bp.boot.concat_vs_run11.ap",   "ensemble_detect.json",  "bootstrap.bp1500_concat_vs_run11_ap.delta",      "{:+.4f}", "concatenation is significantly worse"),
    ("ens.boot.mean_vs_run11.ap",        "ensemble_detect.json",  "bootstrap.zf_mean_vs_run11_ap.delta",            "{:+.4f}", ""),
    ("ens.boot.mean_vs_run11.ap_lo",     "ensemble_detect.json",  "bootstrap.zf_mean_vs_run11_ap.lo",               "{:+.4f}", ""),
    ("ens.boot.mean_vs_run11.ap_hi",     "ensemble_detect.json",  "bootstrap.zf_mean_vs_run11_ap.hi",               "{:+.4f}", ""),
    ("ens.boot.aves_vs_run11.auc",       "ensemble_detect.json",  "bootstrap.zf_aves_vs_run11_auc.delta",           "{:+.4f}", "in-distribution tie, measured"),
    # ---- synthetic, layer 6 (the layer the figure and the real comparison use)
    ("synth.core.L6.onset_mae_ms",       "synth_timing.json",     "conditions.isolated[20.0].layers.L6.onset_mae_ms",        "{:.1f}", ""),
    ("synth.core.L6.offset_mae_ms",      "synth_timing.json",     "conditions.isolated[20.0].layers.L6.offset_mae_ms",       "{:.1f}", ""),
    ("synth.core.L6.offset_bias_ms",     "synth_timing.json",     "conditions.isolated[20.0].layers.L6.offset_bias_ms",      "{:+.1f}", ""),
    ("synth.core.L6.duration_ratio",     "synth_timing.json",     "conditions.isolated[20.0].layers.L6.duration_ratio_median", "{:.2f}", ""),
    ("synth.tails.L6.offset_mae_ms",     "synth_timing_tails.json", "conditions.isolated[20.0].layers.L6.offset_mae_ms",     "{:.1f}", ""),
    ("synth.tails.L6.offset_bias_ms",    "synth_timing_tails.json", "conditions.isolated[20.0].layers.L6.offset_bias_ms",    "{:+.1f}", ""),
    ("synth.tails.L6.duration_ratio",    "synth_timing_tails.json", "conditions.isolated[20.0].layers.L6.duration_ratio_median", "{:.2f}", ""),
    ("synth.tails.L6.onset_mae_ms",      "synth_timing_tails.json", "conditions.isolated[20.0].layers.L6.onset_mae_ms",      "{:.1f}", ""),
    ("synth.natural.L6.recall",          "synth_timing.json",     "conditions.natural[20.0].layers.L6.recall",               "{:.3f}", ""),
    ("synth.core.L6.recall",             "synth_timing.json",     "conditions.isolated[20.0].layers.L6.recall",              "{:.3f}", ""),
    ("synth.core.L6.auc",                "synth_timing.json",     "conditions.isolated[20.0].layers.L6.frame_auc",           "{:.3f}", ""),
    ("synth.natural.L6.auc",             "synth_timing.json",     "conditions.natural[20.0].layers.L6.frame_auc",            "{:.3f}", ""),
    # ---- randinit range across transformer layers
    ("randinit.L8.gap",                  "randinit_control.json", "layer_table.L8.gap_vs_rand",                     "{:+.4f}", ""),
    ("randinit.L3.gap",                  "randinit_control.json", "layer_table.L3.gap_vs_rand",                     "{:+.4f}", ""),
    ("randinit.L7.pretrained",           "randinit_control.json", "layer_table.L7.pretrained",                      "{:.4f}", ""),
    ("aves.zf.aves_matched_L6.ap",       "aves_baseline.json",    "results.aves_matched_L6.ap",                     "{:.4f}", ""),
    ("ens.boot.aves_vs_run11.lo",        "ensemble_detect.json",  "bootstrap.zf_aves_vs_run11_auc.lo",              "{:+.4f}", ""),
    ("ens.boot.aves_vs_run11.hi",        "ensemble_detect.json",  "bootstrap.zf_aves_vs_run11_auc.hi",              "{:+.4f}", ""),
    # ---- call-type classification: the species-specific task
    ("ct.n_clips",                       "aves_calltype.json",    "n_clips",                                        "{:d}",   "curated clips, Unknown* placeholder birds dropped"),
    ("ct.n_birds",                       "aves_calltype.json",    "n_birds",                                        "{:d}",   ""),
    ("ct.majority",                      "aves_calltype.json",    "majority",                                       "{:.4f}", "8 classes, so chance is 0.207 not 0.125"),
    ("ct.run11.best_acc",                "aves_calltype.json",    "run11.best.acc",                                 "{:.4f}", ""),
    ("ct.run11.best_layer",              "aves_calltype.json",    "run11.best.layer",                               "{:d}",   ""),
    ("ct.aves.best_acc",                 "aves_calltype.json",    "aves.best.acc",                                  "{:.4f}", ""),
    ("ct.aves.best_layer",               "aves_calltype.json",    "aves.best.layer",                                "{:d}",   ""),
    ("ct.boot.run11_vs_aves",            "aves_calltype.json",    "bootstrap_run11_vs_aves.delta",                  "{:+.4f}", "cluster bootstrap over the 26 birds"),
    ("ct.boot.run11_vs_aves.lo",         "aves_calltype.json",    "bootstrap_run11_vs_aves.lo",                     "{:+.4f}", ""),
    ("ct.boot.run11_vs_aves.hi",         "aves_calltype.json",    "bootstrap_run11_vs_aves.hi",                     "{:+.4f}", ""),
    ("ct.ensemble_mean.acc",             "aves_calltype.json",    "ensemble_mean.acc",                              "{:.4f}", ""),
    ("ct.boot.mean_vs_run11",            "aves_calltype.json",    "bootstrap_mean_vs_run11.delta",                  "{:+.4f}", ""),
    ("ct.boot.mean_vs_run11.lo",         "aves_calltype.json",    "bootstrap_mean_vs_run11.lo",                     "{:+.4f}", ""),
    ("ct.boot.mean_vs_run11.hi",         "aves_calltype.json",    "bootstrap_mean_vs_run11.hi",                     "{:+.4f}", ""),
    ("ct.concat.acc",                    "aves_calltype.json",    "concat.acc",                                     "{:.4f}", ""),
    ("ct.boot.mean_vs_aves",             "aves_calltype.json",    "bootstrap_mean_vs_aves.delta",                   "{:+.4f}", "the ensemble does NOT beat AVES alone here"),
    ("ct.boot.mean_vs_aves.lo",          "aves_calltype.json",    "bootstrap_mean_vs_aves.lo",                      "{:+.4f}", ""),
    ("ct.boot.mean_vs_aves.hi",          "aves_calltype.json",    "bootstrap_mean_vs_aves.hi",                      "{:+.4f}", ""),
    ("ct.reproduction.max_diff",         "aves_calltype.json",    "reproduction.max_abs_diff",                      "{:.4f}", "guard: our probe vs the released per-layer accuracies"),
    # ---- 11-class call type, both encoders extracted through one identical path
    ("ct11.n_clips",                     "calltype11.json",       "n_clips",                                        "{:d}",   "adults + chicks, the 2023 notebook's cohort"),
    ("ct11.n_birds",                     "calltype11.json",       "n_birds",                                        "{:d}",   ""),
    ("ct11.majority",                    "calltype11.json",       "majority",                                       "{:.4f}", ""),
    ("ct11.n_chick",                     "calltype11.json",       "n_chick",                                        "{:d}",   ""),
    ("ct11.run11.best_acc",              "calltype11.json",       "run11.best.acc",                                 "{:.4f}", ""),
    ("ct11.run11.best_layer",            "calltype11.json",       "run11.best.layer",                               "{:d}",   ""),
    ("ct11.aves.best_acc",               "calltype11.json",       "aves.best.acc",                                  "{:.4f}", "vs ~0.60 in the 2023 notebook, which was 99% padding"),
    ("ct11.aves.best_layer",             "calltype11.json",       "aves.best.layer",                                "{:d}",   ""),
    ("ct11.boot.run11_vs_aves",          "calltype11.json",       "bootstrap_run11_vs_aves.delta",                  "{:+.4f}", ""),
    ("ct11.boot.run11_vs_aves.lo",       "calltype11.json",       "bootstrap_run11_vs_aves.lo",                     "{:+.4f}", ""),
    ("ct11.boot.run11_vs_aves.hi",       "calltype11.json",       "bootstrap_run11_vs_aves.hi",                     "{:+.4f}", ""),
    ("ct8m.n_clips",                     "calltype11.json",       "adults_8class.n_clips",                          "{:d}",   "8-class arm with BOTH encoders extracted identically"),
    ("ct8m.majority",                    "calltype11.json",       "adults_8class.majority",                         "{:.4f}", ""),
    ("ct8m.run11.best_acc",              "calltype11.json",       "adults_8class.run11.best.acc",                   "{:.4f}", "higher than the released features give"),
    ("ct8m.aves.best_acc",               "calltype11.json",       "adults_8class.aves.best.acc",                    "{:.4f}", ""),
    ("ct8m.boot.run11_vs_aves",          "calltype11.json",       "adults_8class.bootstrap_run11_vs_aves.delta",    "{:+.4f}", "NOT distinguishable once extraction is matched"),
    ("ct8m.boot.run11_vs_aves.lo",       "calltype11.json",       "adults_8class.bootstrap_run11_vs_aves.lo",       "{:+.4f}", ""),
    ("ct8m.boot.run11_vs_aves.hi",       "calltype11.json",       "adults_8class.bootstrap_run11_vs_aves.hi",       "{:+.4f}", ""),
]


def split_path(path):
    """Dots separate, except inside [brackets] -- needed for keys that are themselves numbers
    like snr_sweep[-10.0], which a naive split would tear in half."""
    out, buf, depth = [], "", 0
    for ch in path:
        if ch == "[":
            if buf:
                out.append(buf); buf = ""
            depth += 1
        elif ch == "]":
            depth -= 1
            out.append(buf); buf = ""
        elif ch == "." and depth == 0:
            if buf:
                out.append(buf); buf = ""
        else:
            buf += ch
    if buf:
        out.append(buf)
    return out


def dig(o, path):
    cur = o
    for part in split_path(path):
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            if part not in cur:
                raise KeyError(part)
            cur = cur[part]
    return cur


def main():
    cache, entries, missing = {}, {}, []
    # Numbers whose only machine-readable home is a findings YAML or the dataset README. They are
    # load-bearing, so they get declared provenance here rather than floating free in the prose --
    # but they are NOT auto-derived, and the `manual` flag says so.
    man = HERE / "manual_numbers.json"
    if man.exists():
        for k, e in json.loads(man.read_text()).items():
            entries[k] = dict(value=e["value"], display=e.get("display", str(e["value"])),
                              source=e["source"], note=e.get("note", ""), manual=True)
    for key, fname, path, fmt, note in SPEC:
        if path is None:
            continue
        p = ANA / fname
        if not p.exists():
            missing.append(f"{key}: file {fname} not found")
            continue
        if fname not in cache:
            cache[fname] = json.loads(p.read_text())
        try:
            v = dig(cache[fname], path)
        except (KeyError, IndexError, ValueError) as e:
            missing.append(f"{key}: path '{path}' not in {fname} ({type(e).__name__} {e})")
            continue
        entries[key] = dict(value=v, display=fmt.format(v), source=f"analysis/{fname}:{path}",
                            note=note)
    reg = dict(generated=datetime.datetime.now().isoformat(timespec="seconds"),
               n_entries=len(entries), entries=entries)
    OUT.write_text(json.dumps(reg, indent=2))
    print(f"wrote {OUT}  ({len(entries)} entries)")
    if missing:
        print(f"\n{len(missing)} SPEC entries could not be resolved:")
        for m in missing:
            print(f"  {m}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

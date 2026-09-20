# AVES methodology audit + transfer-learning / distillation literature survey

Written 2026-09-15. Purpose: establish exactly how AVES was trained (so run11 can be compared
against it on equal terms), and survey what the literature says about continued pretraining and
distillation — specifically to answer *"would continued pretraining from AVES weights, or
distillation, beat either run11 or AVES?"*

## Reading rules used in this document

* Every number carries a source. `[cfg]` means it was read out of the local fairseq checkpoint
  `datasets/11905533/aves-base-bio.pt` — that file is ground truth for the recipe.
* **NOT VERIFIED** means I looked and could not confirm it from a source I actually read. It does
  not mean "probably true".
* **DERIVED** means I computed it from two verified numbers; the arithmetic is shown.
* Where the paper, the repo, the blog post and the config disagree, the disagreement is stated
  rather than silently resolved.

---

# Part 1 — AVES methodology audit

## 1.1 Primary sources actually read

| Tag | Source | What it covers |
|---|---|---|
| `[paper]` | Hagiwara, "AVES: Animal Vocalization Encoder based on Self-Supervision", arXiv:2210.14493v1 (26 Oct 2022); ICASSP 2023. Full PDF text extracted locally. | Corpora, hours, segments, clusters, LR, batch, steps, all eval tables |
| `[repo]` | `github.com/earthspecies/aves` README (raw, retrieved 2026-09-15) | Checkpoint table, hours per checkpoint, BirdAVES table, download URLs |
| `[cfg]` | `datasets/11905533/aves-base-bio.pt` — fairseq `cfg` / `extra_state` / `optimizer_history` / `model` | Ground-truth recipe for the released `aves-base-bio` |
| `[tacfg]` | `…/ported_aves/aves-base-*.torchaudio.model_config.json` and `…/birdaves/birdaves-*.torchaudio.model_config.json` | Architecture of every released checkpoint |
| `[blog]` | earthspecies.org, "Introducing BirdAVES", 20 Jun 2024 | BirdAVES data/model/compute prose |
| `[fsq]` | `fairseq/examples/hubert/config/pretrain/hubert_base_librispeech.yaml` (raw, retrieved 2026-09-15) | The stock HuBERT-base defaults AVES deviated from |
| `[beans]` | Hagiwara et al., "BEANS: The Benchmark of Animal Sounds", arXiv:2210.12300 | Benchmark composition |

Note the repo is **frozen**: `[repo]` states AVES has migrated to
`github.com/earthspecies/avex` and "this repo will be frozen. Any issues should be posted on the
AVEX repository". The weights remain downloadable at the URLs in `[repo]`.

## 1.2 Methodology table

Hours and segment counts are per-checkpoint pretraining corpus size. "Iterations" = HuBERT
acoustic-unit-discovery iterations.

| Checkpoint | Corpora | Hours | Segments | Iter. | k-means source & clusters | Steps | Hardware | Source |
|---|---|---|---|---|---|---|---|---|
| `aves-base-core` | FSD50K + AudioSet **balanced** 20k subset | 153 | 67k | 2 | it1: 39-d MFCC, k=200; it2: layer 6 of it1 model, k=200 | 100k | NOT VERIFIED | `[paper]` Tab.1 + §3.1; `[repo]` |
| `aves-base-bio` ← **the local file** | core + AudioSet/VGGSound **animal** subset | 360 | 142k | 2 | it1: 39-d MFCC, k=200; it2: layer 6, k=200 (`[cfg]` confirms k=200 and that targets came from a HuBERT model, not MFCC — see §2.3) | 100k `[cfg]` `max_update=100000`, `num_updates=100000` | **1 GPU**, ≥30 GB free VRAM, grad-accum ×8, fp16; 24.6 h wall clock `[cfg]` | `[paper]`, `[repo]`, `[cfg]` |
| `aves-base-nonbio` | core + AudioSet/VGGSound **non-animal**, size-matched to `bio` | 360 | 142k | 2 | same | 100k | NOT VERIFIED | `[paper]` Tab.1; `[repo]` |
| `aves-base-all` | core + **all** AudioSet + **all** VGGSound | 5054 | 1846k | 2 | same | 100k | NOT VERIFIED | `[paper]` Tab.1; `[repo]` |
| `aves-*-large` (not released) | same four mixes | same | same | 2 | it1: MFCC k=200; it2: **layer 12**, k=200 | **150k** | NOT VERIFIED | `[paper]` §3.1 |
| `birdaves-biox-base` | `bio` + Xeno-canto | 2570 | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED ("increased the number of training steps", no figure) | NOT VERIFIED ("significantly scaled up the training compute") | `[repo]`, `[blog]` |
| `birdaves-biox-large` | `bio` + Xeno-canto | 2570 | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | `[repo]`, `[blog]` |
| `birdaves-bioxn-large` | `bio` + Xeno-canto + iNaturalist | 3076 | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | `[repo]`, `[blog]` |

Per-checkpoint step counts for `core` / `nonbio` / `all`: `[paper]` §3.4 says "we pretrained all
models for the same number of steps", and §3.1 gives 100k for base — so 100k for all four base
configs is supported, but no config file for those three was read. The step count for the
BirdAVES models is **NOT VERIFIED**; `[blog]` only says it was increased.

### Architecture, from the released configs `[tacfg]`

| | `aves-base-*` (all four) | `birdaves-biox-base` | `birdaves-*-large` (both) |
|---|---|---|---|
| CNN extractor | 7 layers, `(512,10,5)+(512,3,2)×4+(512,2,2)×2`, no bias, group_norm | identical | identical |
| Transformer | 12 layers, 768-d, 12 heads, FFN 3072 | 12 / 768 / 12 / 3072 | **24 layers, 1024-d, 16 heads, FFN 4096** |
| layer_drop | 0.05 | 0.05 | 0.05 |
| Params | ~95 M `[blog]`; 94,620,800 in the fairseq file incl. heads `[cfg]` | ~95 M `[blog]` | ~316 M `[blog]` |
| Frame rate | 50 Hz = **20 ms** (`label_rate=50.0` `[cfg]`; "50 frames per second" `[paper]`; "1 second → 49 steps" `[repo]`) | same | same |

`[blog]` says "per-frame (50ms) embeddings". That contradicts `[paper]`, `[cfg]` and `[repo]`,
all of which give 50 frames/s = 20 ms. **Treat the blog's "50ms" as an error.**

### Derived corpus splits (arithmetic only, not stated anywhere)

* Animal AudioSet/VGGSound portion of `bio` = 360 − 153 = **207 h** — DERIVED.
* All AudioSet/VGGSound in `all` = 5054 − 153 = **4901 h** — DERIVED.
* Xeno-canto portion of `biox` = 2570 − 360 = **2210 h** — DERIVED, *and only valid if
  `[repo]`'s "biox = bio + xeno-canto" is right* (see §1.3 conflict).
* iNaturalist portion of `bioxn` = 3076 − 2570 = **506 h** — DERIVED, same caveat.

## 1.3 What the letters mean

From `[paper]` §3.1 and `[repo]`:

* **core** = FSD50K + the *balanced* 20k subset of AudioSet. Included in *every* other config.
* **bio** = core + "all audio segments that have corresponding labels under the **Animal (ID:
  /m/0jbk)** concept in the AudioSet ontology, and the **animals class group** in VGGSound"
  `[paper]`. This is the only filtering step: ontology-label selection, no acoustic filtering,
  no VAD, no energy gate mentioned.
* **nonbio** = core + "randomly selected segments taken from AudioSet and VGGSound", **size-matched
  to bio** `[paper]`. This is an explicit domain-shift control — see §3.3, it matters a lot.
* **all** = core + every segment in AudioSet and VGGSound.
* **biox** = bio + Xeno-canto `[repo]` table.
* **bioxn** = bio + Xeno-canto + iNaturalist `[repo]` table.

**CONFLICT inside the BirdAVES sources.** `[repo]`'s table says biox/bioxn build on **`bio`**.
`[blog]`'s prose says "In addition to the **`core`** configuration used for AVES, we added a large
amount of bird recordings from Xeno-canto and iNaturalist". These are different corpora (bio
includes 207 h of animal AS/VS that core does not). The table is the more specific statement;
the letter "bio" in the checkpoint name supports the table. Flagging it rather than resolving it.

**Second minor repo inconsistency:** `[repo]`'s embedding-dimension table lists
`BirdAVES-bion-large`, while its download table lists `BirdAVES-bioxn-large`. Same model;
`bioxn` is the name on the actual files.

Also worth recording from `[blog]`: BirdAVES "uses only a portion (**33%**) of the available
Xeno-canto recordings and does not use any other large-scale datasets such as the Macaulay
library, and does not rely on advanced data augmentation techniques."

Data-provenance caveat, quoted from `[paper]` footnote 3: "Neither AudioSet nor VGGSound
distributes the audio segments and the actual data we used and the statistics are as of when we
obtained the corresponding YouTube data in 2022." **AVES's corpus is not exactly reproducible.**

## 1.4 The training recipe, as published

Verbatim from `[paper]` §2.1 and §3.1:

* Stage 1 targets: "cluster labels are obtained by applying k-means clustering on
  **39-dimensional MFCC features**".
* Stage 2 targets: "the clustering is repeated on the features extracted from some internal layer
  (**6th for base**, 12th for large architectures) of the pretrained model itself… **we used the
  model of the second iteration**." So the released checkpoints are iteration-2 models.
* "**200 clusters** (at both the first and the second stage)".
* "a learning rate of **2.0×10⁻⁴**".
* "an effective batch size of **700 seconds** (450 seconds for large) of audio".
* "**100k training steps** (150k steps for large)".
* "We determined these hyperparameters on the basis of the results from preliminary experiments
  with a smaller subset of the AudioSet dataset. Other hyperparameters followed the original
  HuBERT base … configurations. We used the implementation from fairseq."
* Fine-tuning for downstream tasks: mean-pool over time → one linear layer; **CNN encoder frozen**,
  everything above it trained. Adam, β=(0.9, 0.999), ε=1e-8, batch 32, LR swept over
  {1e-5, 5e-5, 1e-4}, 50 epochs, best-on-validation. Input resampled to 16 kHz.

Nothing in `[paper]` states the hardware, wall clock, warmup schedule, mask parameters, crop
length, or normalisation. `[cfg]` supplies all of those — §2.

## 1.5 Published evaluation numbers

**`[paper]` Table 2** — accuracy for classification and auxiliary, mAP for detection. Only
`AVES-bio` is reported; the other configs appear as aggregate T-scores in Table 3. Best baselines
shown for context.

| Model | wtkn | bat | cbi | hbdb | dogs | dcase | enab | hiceas | rfcx | gib | esc | sc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **AVES-bio** | **0.879** | **0.748** | 0.598 | **0.810** | **0.950** | **0.392** | **0.555** | **0.629** | 0.130 | 0.284 | 0.773 | **0.964** |
| vggish | 0.847 | 0.743 | 0.440 | 0.808 | 0.906 | 0.335 | 0.535 | 0.463 | 0.140 | 0.150 | 0.705 | 0.948 |
| rn152p-s (sup. topline) | 0.835 | 0.606 | 0.583 | 0.700 | 0.799 | 0.300 | 0.520 | 0.326 | 0.097 | 0.303 | 0.788 | 0.947 |
| rn152p | 0.720 | 0.544 | 0.573 | 0.662 | 0.741 | 0.198 | 0.429 | 0.273 | 0.085 | 0.230 | 0.540 | 0.946 |
| rn18p | 0.735 | 0.532 | 0.509 | 0.649 | 0.705 | 0.223 | 0.462 | 0.262 | 0.079 | **0.316** | 0.590 | 0.936 |
| svm | 0.870 | 0.720 | 0.139 | 0.779 | 0.914 | 0.146 | 0.299 | 0.218 | 0.038 | 0.039 | 0.478 | 0.572 |

(On cbi, AVES 0.598 is best, rn152p-s 0.583 second, rn152p 0.573 third. The two datasets where
AVES was *not* best or second are the sparse-vocalisation detection sets **rfcx** — AVES 0.130 vs
vggish-s 0.143 and vggish 0.140 — and **gib** — AVES 0.284 vs rn18p 0.316 and rn152p-s 0.303.
`[paper]`: "These datasets are challenging due to the sparsity of the vocalizations in the
training data.")

**`[paper]` Table 3 — T-scores by pretraining config** (normalised per dataset to mean 50, sd 10):

| Config | Classification | Detection | Auxiliary | **Total** |
|---|---|---|---|---|
| AVES-core (153 h) | 59.2 | 62.6 | 58.6 | 60.5 |
| **AVES-bio (360 h)** | 61.2 | 62.6 | 59.0 | **61.4** |
| **AVES-nonbio (360 h, no animal audio)** | **61.8** | 61.7 | 58.8 | **61.3** |
| AVES-all (5054 h) | 60.9 | 60.0 | 59.3 | 60.3 |

**This is the single most important table in the paper for run11.** The size-matched *non-animal*
control is within 0.1 T-score of `bio` overall and **beats it on classification**. And the 14×
larger `all` corpus is the *worst* config. `[paper]`'s own wording: "all configurations performed
somewhat similarly, because we pretrained all models for the same number of steps, although
AVES-bio outperforms all the other configurations with **a small margin**."

**`[paper]` Table 4 — model size:** AVES-bio-base total 61.4 vs AVES-bio-large **57.2**
(class. 56.8, det. 57.9, aux. 56.4). Large *lost*, "consistently … in all configurations".
`[paper]` attributes it to "overfitting and an undertuning of model-specific hyperparameters".
`[paper]` footnote 5 also records that **data2vec was tried and "the preliminary results were not
promising"**.

**`[repo]` BirdAVES table** (BEANS averages; birds = cbi, dcase, enabirds, rfcx):

| Config | Hours | BEANS avg (all) | BEANS avg (birds) |
|---|---|---|---|
| AVES-bio (baseline) | 360 | 0.643 | 0.419 |
| BirdAVES-biox-base | 2570 | 0.678 | 0.476 |
| BirdAVES-biox-large | 2570 | **0.686** | 0.511 |
| BirdAVES-bioxn-large | 3076 | 0.679 | **0.512** |

So scaling data ~7× and params ~3.3× moved bird-task BEANS from 0.419 → 0.512 (+0.093 absolute,
+22% relative — consistent with `[blog]`'s "over a 20% improvement"). Note `biox-base` alone
(same 95 M params, 7× data) gets +0.057 of that +0.093; the *large* upgrade adds +0.035 more.

`[blog]` also reports a Table 2 comparing AVES/BirdAVES against BirdNET 2.4 and Perch v8 with
cbi excluded, but that table is an image and I could not read its values — **NOT VERIFIED**. The
accompanying prose: BirdAVES is "still lagging slightly behind on bird datasets compared against
supervised models".

**Third-party numbers for AVES** (useful because they are measured under someone else's protocol):

| Source | Protocol | AVES result | Best in that study |
|---|---|---|---|
| Perch 2.0, arXiv:2508.04665v2, Tab.3 | BEANS; AVES row is **FT** (full fine-tune), Perch row is **LP** (linear probe) | AVES-Bio **0.817 acc / 0.398 mAP** | Perch 2.0 (Phase I, LP) 0.835 / 0.426; Perch 1.0 (LP) 0.809 / 0.353 |
| AVEX, arXiv:2508.11845v3, Tab.3 | BEANS-Class linear probe / retrieval-AUC | **Bird-AVES-biox-base 0.705 / 0.646** | sl-BEATS-bio 0.840 / sl-BEATS-all 0.813 |
| AVEX, same table | BirdSet probe / R-AUC | Bird-AVES-biox-base **0.092 / 0.670** | sl-BEATS-all 0.294 / 0.732 |
| FM review, arXiv:2508.01277 | BEANS AUROC, linear probe | BirdAVES **87.89** (second-worst of 16 models) | BEATs-NLM 98.57 (attentive) |
| FM review, same | BirdSet AUROC, linear / attentive | AVES **63.80 / 74.48**; BirdAVES **65.58 / 78.87** (the two bottom performers on linear) | Perch 2.0 90.78 (restricted) |

`[beans]`: BEANS is "12 datasets", 5 classification + 5 detection + 2 auxiliary (ESC-50,
SpeechCommands). Metrics: accuracy for classification, mAP for detection.

---

# Part 2 — Reconciling the local checkpoint against the paper

File: `datasets/11905533/aves-base-bio.pt`, 1,135,551,425 bytes, dated 2022-10-26 (same day
arXiv:2210.14493v1 was posted). Top-level keys: `args`, `cfg`, `model`, `criterion`,
`optimizer_history`, `task_state`, `extra_state`, `last_optimizer_state`. **`args` is `None`** —
this is a Hydra-configured run, so `cfg` is the only config record.

## 2.1 Verbatim config values

```
cfg.model._name                 = 'hubert'
cfg.model.label_rate            = 50.0            # 20 ms frames
cfg.model.encoder_layers        = 12
cfg.model.encoder_embed_dim     = 768
cfg.model.encoder_ffn_embed_dim = 3072
cfg.model.encoder_attention_heads = 12
cfg.model.final_dim             = 256
cfg.model.untie_final_proj      = True
cfg.model.extractor_mode        = 'default'
cfg.model.conv_feature_layers   = '[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2'
cfg.model.encoder_layerdrop     = 0.05
cfg.model.feature_grad_mult     = 0.1
cfg.model.logit_temp            = 0.1
cfg.model.mask_prob             = 0.8
cfg.model.mask_length           = 10              # 10 frames = 200 ms
cfg.model.mask_selection        = 'static'
cfg.model.mask_channel_prob     = 0.0             # no channel masking
cfg.model.layer_norm_first      = False

cfg.task._name                  = 'hubert_pretraining'
cfg.task.data                   = '/mnt/dev/hubert/data/faav150k/tsv'
cfg.task.label_dir              = '/mnt/dev/hubert/data/faav150k/hblab.c200'
cfg.task.labels                 = ['km']
cfg.task.label_rate             = 50.0
cfg.task.sample_rate            = 16000
cfg.task.normalize              = False           # AVES expects RAW un-normalised audio
cfg.task.max_sample_size        = 250000          # 15.625 s crops
cfg.task.min_sample_size        = 0
cfg.task.random_crop            = True
cfg.task.pad_audio              = False

cfg.optimization.max_update     = 100000
cfg.optimization.lr             = [0.0002]
cfg.optimization.update_freq    = [8]
cfg.optimization.clip_norm      = 10.0
cfg.dataset.max_tokens          = 1400000         # 87.5 s per micro-batch
cfg.distributed_training.distributed_world_size = 1
cfg.common.fp16                 = True
cfg.common.seed                 = 1337

cfg.optimizer._name             = 'adam'
cfg.optimizer.adam_betas        = '(0.9,0.98)'
cfg.optimizer.adam_eps          = 1e-06
cfg.optimizer.weight_decay      = 0.01
cfg.lr_scheduler._name          = 'polynomial_decay'
cfg.lr_scheduler.warmup_updates = 32000
cfg.lr_scheduler.total_num_update = 100000.0
cfg.lr_scheduler.end_learning_rate = 0.0
cfg.lr_scheduler.power          = 1.0

cfg.criterion._name             = 'hubert'
cfg.criterion.pred_masked_weight = 1.0
cfg.criterion.pred_nomask_weight = 0.0
cfg.criterion.loss_weights      = [10.0]
```

Training state:

```
optimizer_history[-1].num_updates          = 100000
optimizer_history[-1].optimizer_name        = 'FP16Optimizer'
optimizer_history[-1].lr_scheduler_state    = {'best': 3.028}
extra_state.previous_training_time          = 88709.81 s   = 24.64 h
extra_state.train_iterator                  = {'epoch': 55, 'iterations_in_epoch': 2448}
extra_state.val_loss / extra_state.best     = 3.023
metrics train_wall                          = 86887.47 s   = 24.14 h
metrics loss_m_0  (masked-frame loss)       = 3.1351
metrics correct_m_0 (masked-frame accuracy) = 0.4129
metrics correct_u_0 (unmasked accuracy)     = 0.3300
metrics bsz (mean crops per update)         = 71.07
metrics gb_free                             = 29.91 GB
model state_dict: 214 tensors, 94,620,800 params
model['label_embs_concat'].shape            = (204, 256)
model['final_proj'].weight.shape            = (256, 768)
```

## 2.2 Config vs paper — AGREEMENTS

| Quantity | `[paper]` | `[cfg]` | Verdict |
|---|---|---|---|
| Learning rate | 2.0×10⁻⁴ | `lr = [0.0002]` | **Agrees** |
| Effective batch | 700 s of audio | 1,400,000 samples ÷ 16,000 = 87.5 s × `update_freq` 8 × `world_size` 1 = **700.0 s** | **Agrees exactly** |
| Steps | 100k | `max_update=100000`, `num_updates=100000` | **Agrees** |
| Clusters | 200 | `label_dir = …/hblab.**c200**`; `label_embs_concat` is **(204, 256)** = 200 clusters + 4 fairseq special symbols (bos/pad/eos/unk) | **Agrees** |
| Frame rate | 50 frames/s | `label_rate = 50.0` | **Agrees** |
| Architecture | 12-layer, 768-unit | 12 / 768 / 12 heads / 3072 FFN | **Agrees** |
| "Other hyperparameters followed the original HuBERT base" | — | `mask_prob 0.8`, `mask_length 10`, `seed 1337`, `max_tokens 1400000`, `extractor_mode default`, `feature_grad_mult 0.1`, `untie_final_proj true`, `normalize false`, `loss_weights [10.0]`, `clip_norm 10.0`, `adam_betas (0.9,0.98)` — **all identical to `[fsq]`** | **Agrees** |

## 2.3 What the config reveals that the paper does not

1. **This is unambiguously the iteration-2 checkpoint, and the config proves it.**
   `label_dir = '…/hblab.c200'`. `hblab` = HuBERT labels (contrast with a `mfcc`/`mfcclab`
   directory, which is what iteration 1 would read). Combined with `[paper]`'s "we used the model
   of the second iteration", the released `aves-base-bio` is the iteration-2 model.
   *Caveat:* the config does **not** record which layer the targets came from. Layer 6 is
   `[paper]`'s statement, **not** confirmed by `[cfg]`.

2. **AVES-bio was trained on ONE GPU.** `distributed_world_size = 1`, `update_freq = [8]`. The
   700 s effective batch is 87.5 s × 8 gradient-accumulation steps on a single device, not a
   multi-GPU batch. `gb_free = 29.91 GB` at checkpoint time implies a ≥32 GB card; the exact
   model is **NOT VERIFIED**. This is a materially smaller compute footprint than anyone reading
   the paper would assume.

3. **Wall clock: 24.6 hours.** `previous_training_time = 88,709.81 s`; `train_wall = 86,887.47 s`
   (0.869 s/update). Nowhere in `[paper]`. For reference, DistilHuBERT `[distil]` reports HuBERT
   Base pretraining as **2k GPU-hours**; AVES-bio's second iteration cost ~25 GPU-hours.

4. **Warmup is 32% of the run.** `warmup_updates = 32000`, `total_num_update = 100000`,
   `polynomial_decay` with `power = 1.0` and `end_learning_rate = 0.0`. AVES kept `[fsq]`'s stock
   `warmup_updates: 32000` — which was tuned for `max_update: 400000` (8% warmup) — while cutting
   `max_update` to 100k. So LR ramps linearly to 2e-4 over steps 0–32k and then decays linearly
   to 0 by step 100k. Whether that was deliberate or an un-rescaled default is **NOT VERIFIED**,
   but it is a real property of the released weights and it is *not* "the HuBERT base schedule".

5. **AVES expects raw, un-normalised waveforms.** `task.normalize = False`. If you feed AVES
   layer-normalised or peak-normalised audio you are off-distribution. (This matches the lab's
   own finding 034.)

6. **Crops are 15.625 s with random cropping**, `pad_audio = False`, `min_sample_size = 0`.
   Mean crop actually consumed = 700 s ÷ 71.07 crops = **9.85 s** — consistent with 10-second
   AudioSet/VGGSound segments dominating the mix.

7. **Deltas from stock HuBERT-base `[fsq]`** — AVES changed exactly five things:
   | Field | `[fsq]` HuBERT base | `[cfg]` AVES-bio |
   |---|---|---|
   | `max_update` | 400000 | **100000** |
   | `lr` | 0.0005 | **0.0002** |
   | `distributed_world_size` | 32 | **1** |
   | `update_freq` | (unset ⇒ 1) | **8** |
   | `label_rate` | `???` | **50.0** |
   Effective batch therefore dropped from 32 × 87.5 = 2800 s to 700 s, and LR from 5e-4 to 2e-4.
   Total audio processed: 100,000 × 700 s = **19,444 h** (DERIVED) vs HuBERT base's
   400,000 × 2800 s = 311,111 h (DERIVED from `[fsq]`) — AVES used **6% of HuBERT base's
   token budget.**

8. **Final masked-prediction accuracy was 0.4129** with masked-frame loss 3.135 and validation
   loss 3.023 at k=200. This is the number to compare a continued-pretraining run against, but
   only if you match k — see §4.

## 2.4 Independent corroboration of the 360-hour figure

`[paper]` Table 1 says AVES-bio = 360 h / 142k segments. Nothing in `[cfg]` states either
number directly. But `extra_state` lets them be reconstructed:

```
updates completed in whole epochs = 100,000 − (2,448 iterations_in_epoch ÷ 8 update_freq)
                                  = 100,000 − 306 = 99,694   over 54 completed epochs
updates per epoch                 = 99,694 / 54            = 1,846.2
audio per epoch                   = 1,846.2 × 700 s         = 1,292,340 s = 359.0 h
```
**359.0 h computed vs 360 h published.** And on segment count:
```
train segments per epoch = 1,846.2 updates × 71.07 crops/update ≈ 131,218
valid segments per pass  = 245,102 total valid samples ÷ ~20–22 validation passes ≈ 11.1k–12.3k
train + valid            ≈ 142,400 – 143,500   vs 142k published
```
Both reconstructions land within ~1% of Table 1. The 306-update figure is cross-checked
independently: the `train` metric group carries `count = 306`, exactly `2448 / 8`.

**Conclusion: I found no disagreement between `[paper]` and `[cfg]` on any quantity both
report.** All the discrepancies are omissions in the paper (hardware, wall clock, warmup, mask
params, normalisation, crop length) plus one error in `[blog]` (50 ms vs 20 ms frames) and one
conflict inside the BirdAVES sources (§1.3).

## 2.5 Parameter-count reconciliation against run11

```
fairseq file total                                  = 94,620,800   [cfg]
  − final_proj  (256×768 + 256)                     =    196,864
  − label_embs_concat (204×256)                     =     52,224
  = encoder without pretraining heads               = 94,371,712
run11 as reported by the lab                        = 94,370,944
residual                                            =        768
```
The 768 is one 768-d vector's worth of parameters. Most likely a single bias/LayerNorm term that
the lab's torchaudio port counts differently from this arithmetic; it is **not** an architectural
difference. Flagging rather than guessing. (The lab's own release README states 94.6 M for run11,
which matches the with-heads figure — so the 94,370,944 figure is an encoder-only count.)

---

# Part 3 — Literature survey

## 3.1 Continued / domain-adaptive pretraining (DAPT) of speech & audio SSL models

**The canonical framing** comes from NLP: Gururangan et al., *Don't Stop Pretraining*, ACL 2020
(arXiv:2004.10964). Four domains, eight tasks, RoBERTa base. Their numbers are the best-calibrated
answer to "does it pay?" that exists:

| Domain / task | RoBERTa | DAPT | **¬DAPT** (continued on an *irrelevant* domain) | TAPT (continued on the small task corpus) | DAPT+TAPT |
|---|---|---|---|---|---|
| BioMed / ChemProt | 81.9 | 84.2 | **79.4** | 82.6 | **84.4** |
| BioMed / RCT | 87.2 | 87.6 | 86.9 | 87.7 | **87.8** |
| CS / ACL-ARC | 63.0 | **75.4** | 66.4 | 67.4 | **75.6** |
| CS / SciERC | 77.3 | 80.8 | 79.2 | 79.3 | **81.3** |
| News / HyperPartisan | 86.6 | 88.2 | **76.4** | **90.4** | 90.0 |
| News / AGNews | 93.9 | **93.9** | 93.5 | 94.5 | **94.6** |
| Reviews / Helpfulness | 65.1 | 66.5 | 65.1 | **68.5** | 68.7 |
| Reviews / IMDB | 95.0 | 95.4 | 94.1 | 95.5 | **95.6** |

Four things to take from this table:

1. **DAPT gains range from +0.0 to +12.4.** The largest gains are where the target domain is most
   distant from the base corpus *and* the task is low-resource (ACL-ARC, 1.7k examples).
2. **Continued pretraining on the wrong data actively hurts** — `¬DAPT` is below the untouched
   baseline on 6 of 8 tasks, by as much as −10.2 (HyperPartisan).
3. **TAPT — continuing on nothing but the task's own unlabelled data — often matches or beats
   DAPT**, at a fraction of the cost (RCT, HyperPartisan, AGNews, Helpfulness, IMDB).
4. **DAPT+TAPT is the best configuration on all eight tasks.**

Recipe scale: DAPT ran "12.5K steps, which amounts to [a] single pass on each domain dataset, on
a v3-8 TPU"; TAPT ran 100 epochs over the small task corpus. Domain corpora were 47 GB / 48 GB /
39 GB / 11 GB against RoBERTa's 160 GB pretraining corpus — i.e. **DAPT corpora were 7–30% the
size of the original pretraining corpus** and still paid.

**In speech, the same picture, and it is well established:**

* **Nowakowski et al. 2023** (*Information Processing & Management* 60(2):103148; weights at
  `huggingface.co/karolnowakowski/wav2vec2-large-xlsr-53-pretrain-ain`): XLSR-53 (pretrained on
  56k h / 53 languages) continued for **100k steps on 234 h** of Hokkaido + Sakhalin Ainu.
  Quoted from the model card: "234 hours of speech data in Hokkaido Ainu and Sakhalin Ainu",
  "100k steps". Reported as a substantial WER reduction on Sakhalin Ainu. *234 h is the smallest
  in-domain corpus I found with a documented, clearly-positive continued-pretraining result on a
  speech SSL encoder.*
* **Attia et al., CPT-Boosted Wav2vec2.0** (arXiv:2409.14494): CPT on **5,235 h** of
  untranscribed classroom audio. WER without LM, cross-validated: `W2V-LV60K` 22.52 (CPT) vs
  `W2V-SCR` (same architecture pretrained **from scratch** on the same classroom data) **30.25**
  on NCTE and **38.59** on the in-house set. With a 5-gram LM, `W2V-Robust` improves from 27.99 →
  **17.71** WER on NCTE (−10.28 points) and 31.49 → **26.50** on MPT (−4.99). The paper's own
  summary: "CPT improves WER by upwards of 10%", and "CPT is the most effective tool to adapt"
  the model. **Continued pretraining beat from-scratch decisively at equal data.**
* **The choice of starting checkpoint matters:** in the same paper, `XLS-R` (436k h, 128
  languages) after CPT reached 26.53 WER while `W2V-LV60K` (60k h English) after CPT reached
  22.52 — i.e. the more in-domain-adjacent base won, not the biggest base.

**The critical counter-example, and it is in audio, and it involves bioacoustics:**

* **SONAR** (Zhang et al., arXiv:2509.15703) — "Self-distilled cONtinual pre-training for domain
  adaptive Audio Representations", built on BEATs. Their **DCPT** baseline is exactly the naive
  thing: "resumes training from the original BEATs checkpoint on new domain data using the same
  pre-training objectives". Adapting to iNaturalist Sounds (~230k bioacoustic recordings; the
  actual adaptation set was "approximately 30k–50k audio segments per domain"), 10 epochs, Adam,
  LR 1e-4, on RTX6000 Ada GPUs:

  | Downstream CBI (Cornell Bird ID) | frozen probe | fine-tuned |
  |---|---|---|
  | BEATs, **no adaptation** | 43.5 | 64.7 |
  | **DCPT** (naive continued pretraining on bioacoustics) | **11.9** | **46.5** |
  | SONAR (their method) | 44.2 | 65.6 |

  And AudioSet retention after the iNaturalist adaptation: BEATs baseline 34.8 mAP; **DCPT 12.5
  mAP, forgetting rate 73.5%**; SONAR 34.5 mAP, FR 4.2%.

  **Naive continued pretraining on in-domain bioacoustic audio destroyed the encoder** — the
  frozen bird-ID probe fell from 43.5 to 11.9. And their engineered fix recovered only
  **+0.7 frozen / +0.9 fine-tuned** over simply *not adapting at all*.

  *Two honest caveats.* (a) This is BEATs, not HuBERT. BEATs' targets come from a *frozen
  tokenizer*, so DCPT's collapse is plausibly driven by that tokenizer going stale on
  out-of-distribution audio — which is precisely why SONAR adds an online clustered codebook.
  A HuBERT-style DAPT would re-run k-means on the new domain and so would not have that specific
  failure mode. (b) CBI is Cornell Bird ID — hundreds of species of focal bird recordings, not a
  single-species colony. Neither caveat rescues the headline: this is the closest published
  experiment to what the lab is contemplating, and it went badly.

**Standard recipe, distilled from the above.** Initialise from the general checkpoint; keep the
architecture and the objective; **re-derive the discrete targets on the new domain** (for HuBERT:
re-run k-means, either on MFCC/spectrogram of the new audio or, better, on a mid layer of the
*base* checkpoint applied to the new audio); reset the optimiser and use a **short warmup and a
low peak LR** — the CPT literature does not converge on a number, but every DAPT run cited here
uses a peak LR at or below the original pretraining LR, and ran for roughly one to a few passes
over the in-domain corpus (12.5k steps for DAPT, 100k for Ainu's 234 h, 10 epochs for SONAR).
Mixing in a replay fraction of the original corpus is the standard forgetting mitigation, and
SONAR's results are the strongest argument for it.

## 3.2 Knowledge distillation for speech / audio SSL encoders

| Method | Student | What is distilled | Loss | Params | Cost to produce | Key SUPERB numbers |
|---|---|---|---|---|---|---|
| **HuBERT Base** (teacher) | 7-layer CNN + 12 transformer | — | — | 94.68 M | **2k GPU-hours** (32 GPUs) | PR 5.41, ASR-WER 6.42, SID 81.42, IC 98.34 |
| **DistilHuBERT** (Chang et al., arXiv:2110.01900) | CNN + **2** transformer layers, initialised from teacher CNN + first 2 layers | **3 prediction heads predicting teacher layers 4, 8 and 12** | per-layer `L1 + λ·(−log σ(cos))`, λ=1 | 23.49 M (**−75%**, **+73%** faster) | 960 h LibriSpeech, 200k updates, batch 24 utts, **one 32 GB V100, ~55 h** | PR 16.27, WER 13.37, SID 73.54, IC 94.99, KS 95.98 |
| **FitHuBERT** (Lee et al., arXiv:2207.00555) | **deep and thin** — full 12-layer depth, narrowed attention/FFN, + trainable time-reduction layer | **hint-based distillation on every teacher layer**, not just the last | hint + final | 22.49 M (**23.8% of HuBERT**, 35.9% of inference time / 2.8× faster) | 960 h LibriSpeech | PR 13.32, WER 12.09, KS 96.27, IC 91.25, SID **55.71** |
| **LightHuBERT** (Wang et al., arXiv:2203.15610) | once-for-all weight-sharing **supernet** + architecture search; Small supernet 11–45 M, Base supernet 41–95 M | **contextualised latent representation** of the teacher (average of top-k=8 normalised layers), masked-input L1 | L1 on masked steps, p=0.65 | aBase 68 M / aSmall 27 M | **~2k GPU-hours** (62 h × 32 V100 + 19 h × 8 GPU) — as expensive as pretraining | aBase SUPERB overall **80.4** (PR 4.71, WER 6.72, IC 98.00, SID 77.24); aSmall 79.1 (PR 6.60, IC 98.23, SID 69.70); DistilHuBERT 75.9 for comparison |
| **DPHuBERT** (Peng et al., Interspeech 2023) | learned by **joint distillation + structured pruning** — architecture evolves during training | layer-to-layer distillation while pruning to a target sparsity | distillation + L0 with augmented-Lagrangian sparsity constraint | 23.59 M | **18 + 6 = 24 GPU-hours** | PR 9.67, WER 10.47, IC 97.92, SID 76.83 — beats DistilHuBERT and FitHuBERT on almost every task |

Two results in this table matter directly for the lab:

* **Distillation is cheap relative to pretraining.** DistilHuBERT: 55 h on one GPU. DPHuBERT:
  24 GPU-hours. Against 2k GPU-hours to pretrain the teacher. LightHuBERT is the exception and
  its authors' own cost figure (2k GPU-hours) is why DPHuBERT exists.
* **Distillation tolerates small and out-of-domain data.** DistilHuBERT's Table 5:

  | Distillation corpus | Hours | IC acc | SID acc | ASR WER |
  |---|---|---|---|---|
  | LibriSpeech | 960 | 94.99 | 73.54 | 13.34 |
  | LibriSpeech | **100** | 93.17 | 69.46 | 14.77 |
  | WSJ | 81 | 90.22 | 64.14 | 15.59 |
  | AISHELL-1 (**Mandarin** — language mismatch) | 150 | 87.29 | 67.65 | 16.42 |

  100 h costs about 1.8 points of IC and 1.4 WER versus 960 h. And DPHuBERT trained on only
  **100 h** (PR 10.02 / WER 11.38) beats DistilHuBERT trained on **960 h** (16.27 / 13.37) —
  method dominates data volume in this regime.

* **Which teacher layers you target decides what the student is good at.** DistilHuBERT Table 4,
  ablating the predicted layers:

  | Predicted layers | IC | SID | ASR WER |
  |---|---|---|---|
  | 4, 8, 12 | 94.99 | 73.54 | 13.34 |
  | 4 only | 79.09 | **76.85** | 14.90 |
  | 8 only | **96.89** | 61.52 | **12.28** |
  | 12 only | 95.52 | 65.14 | 13.48 |

  Shallow layer → speaker identity; middle layer → content. `[distil]`: "corroborating our
  hypothesis that the bottom layers offered speaker identity."

## 3.3 Bioacoustics-specific: does domain-specific pretraining lose to general pretraining?

**Yes, and it is now a repeatedly reported phenomenon, not an anomaly.** Five independent
sources:

1. **AVES's own ablation.** `[paper]` Table 3: the size-matched *non-animal* control
   (`AVES-nonbio`, 360 h of randomly chosen non-animal AudioSet/VGGSound) scores total **61.3**
   vs `AVES-bio`'s 61.4, and **beats it on classification, 61.8 vs 61.2**. The entire benefit of
   ontology-filtering to animal sounds is within noise of a random size-matched control. And
   `AVES-all` (5054 h, 14× more audio) is the *worst* config at 60.3. The AVES paper is
   therefore itself evidence *against* the "domain-matched pretraining data is what matters"
   story its abstract is usually read as supporting.

2. **Sarkar & Magimai-Doss, arXiv:2501.05987** — the closest analogue to the lab's result, and it
   is the same comparison one level up. AVES-Bio vs speech-pretrained HuBERT Base, identical
   architecture, identical extraction, best-layer UAR %:

   | Model | InfantMarmosetsVox (call type) | Watkins (species) | Abzaliev (dog) |
   |---|---|---|---|
   | AVES-Bio | 62.54 | **94.95** | **54.23** |
   | **HuBERT (LibriSpeech 960 h, human speech)** | **64.35** | 94.18 | 47.96 |
   | WavLM | 58.98 | 94.78 | 43.97 |
   | wav2vec2 | 62.40 | 94.25 | 48.95 |

   **A human-speech encoder beat AVES on marmoset call-type classification.** Paper's own
   conclusion: "pre-training on bioacoustic data provides only marginal improvements over
   speech-pretrained models, with comparable performance in most scenarios." They also report
   that "HuBERT outperforms AVES in the initial and final layers". Separately, ASR fine-tuning on
   top of speech pretraining did **not** help bioacoustics — every best score came from the
   pretrained-only column.

3. **AVEX / "What Matters for Bioacoustic Encoding"** (Miron, Robinson et al., arXiv:2508.11845v3
   — Earth Species Project, i.e. AVES's own lab). 19 models, 26 datasets. Their EAT
   self-supervised arm is a clean three-way data-mix ablation at matched everything else
   (their bio mix = Xeno-canto 10,416 h + iNaturalist 1,539 h + Watkins 27 h + Animal Sound
   Archive 78 h = **12,060 h** DERIVED; AS = AudioSet 5,700 h; all = both = **17,760 h** DERIVED):

   | SSL model | BEANS-Class probe | BEANS-Class R-AUC | BEANS-Det R-AUC | BirdSet R-AUC |
   |---|---|---|---|---|
   | **EAT-bio** (bioacoustics only, 12,060 h) | **0.692** | **0.671** | **0.679** | **0.631** |
   | EAT-AS (AudioSet only, 5,700 h) | 0.704 | 0.714 | 0.704 | 0.685 |
   | EAT-all (both) | 0.709 | 0.704 | 0.694 | 0.677 |

   **Bioacoustics-only self-supervised pretraining is the worst of the three on every column,
   losing even to general-audio-only pretraining on less than half the data.** The paper's
   wording: "a strong effect of including general audio in the data-mix, with the model trained
   with the addition of AudioSet significantly outperforming the bioacoustics-only model across
   tasks."

   The asymmetry is important: for **supervised** training the opposite holds —
   EffNetB0-AudioSet 0.651 vs EffNetB0-bio 0.786 vs EffNetB0-all 0.800 on the same column.
   *General audio helps self-supervision; bioacoustic labels help supervision.*

   AVEX's headline recipe: **SSL pretrain on a mixed bioacoustic + general-audio corpus, then
   supervised post-train on the same mix.** Their sl-BEATS-all reaches 0.832 / 0.813 vs
   Bird-AVES-biox-base's 0.705 / 0.646. They also note SSL encoders generalise better out of
   distribution: moving from BEANS classification (focal) to BEANS detection (soundscape), "the
   self-supervised models drop on average only 0.01 retrieval ROC AUC compared to a drop of 0.09
   … for the supervised models."

4. **Foundation Models for Bioacoustics — a Comparative Review** (arXiv:2508.01277). 16 models,
   linear and attentive probing. Direct quotes:
   * "**Bioacoustic pretraining data does not guarantee better performance**".
   * "Surprisingly, BirdAVES (78.87 AUROC) and ProtoCLR (78.95 AUROC), both using large amounts
     of bird sound data, do not perform particularly well, showing that **training data alone is
     not a guarantee for success**."
   * "general-purpose audio models trained with self-supervised learning on AudioSet outperform
     many specialised bird sound models on BEANS when evaluated with attentive probing."
     Concretely: BEATs (AudioSet only) 97.98 BEANS AUROC / 82.28 BirdSet AUROC with attentive
     probing, against AVES 74.48 and BirdAVES 78.87 on BirdSet.
   * And a structural explanation the lab should take seriously: "**Models processing raw
     waveforms (AVES: 74.48 AUROC, BirdAVES: 78.87 AUROC) consistently underperform compared to
     their spectrogram-based counterparts on both benchmarks**", plus "Models using higher
     sampling rates generally demonstrate superior performance on bird-focused tasks" — the top
     BirdSet models (BirdMAE 86.54, ConvNext-BS 85.75, Perch 85.63, Perch 2.0 90.78) all run at
     **32 kHz** with 5-second windows, against AVES/run11's 16 kHz.
   * "attentive probing is beneficial to extract the full performance of transformer-based
     models" — e.g. BEATs 94.10 → 97.98 on BEANS; AudioMAE 84.47 → 97.19. Linear probing
     systematically *understates* transformer encoders.

5. **Perch 2.0 — "The Bittern Lesson for Bioacoustics"** (arXiv:2508.04665v2). A 12 M-parameter
   supervised EfficientNet-B3 trained on 1.54 M recordings / 14,795 classes (Xeno-Canto 896,255;
   iNaturalist 571,698; Tierstimmenarchiv 33,859; FSD50K 40,966) beats AVES-Bio on BEANS linear
   probe: 0.835 acc / 0.426 mAP vs **0.817 / 0.398**, with 8× fewer parameters. The authors
   report: "we experimented with a variety of self-supervised methods such as MAEs, HuBERT and
   SimCLR but experienced a similar inability to consistently outperform supervised models",
   and offer the standard explanation — "bioacoustics datasets are two orders of magnitude
   smaller than those powering successful self-supervised vision models".

**The explanations offered in the literature**, collected:
* *Data scale* — SSL needs orders of magnitude more data than bioacoustics has: Perch 2.0 contrasts
  DINOv2's 142 M images against Xeno-Canto and iNaturalist being "two orders of magnitude smaller".
* *Diversity, not domain match* — general audio teaches "fundamental sound characteristics that
  aren't unique to specific animal calls" (AVEX); a narrow corpus starves the model of the
  acoustic variety that makes masked prediction a useful pretext task.
* *Matched-step confound* — AVES's own Table 3 similarity across configs is attributed to
  "we pretrained all models for the same number of steps"; at fixed compute, corpus composition
  matters less than one would expect.
* *Front-end and sampling rate*, not pretraining data, may dominate: raw-waveform 16 kHz models
  underperform 32 kHz spectrogram models regardless of what they were pretrained on (FM review).
* *Readout method* — a linear probe on a transformer understates it by several points; part of
  what looks like a pretraining deficit is a probing artefact (FM review).

## 3.4 How much in-domain data does DAPT need to pay off?

There is no single published threshold. What there *is*:

| Setting | In-domain data | Base corpus | Outcome | Source |
|---|---|---|---|---|
| RoBERTa → 4 text domains | 11–48 GB | 160 GB (7–30%) | +0.0 to +12.4, largest where domain most distant | Gururangan 2020 |
| RoBERTa → task corpus only (TAPT) | the labelled task set's text, 100 epochs | 160 GB (≪1%) | matches or beats DAPT on 5/8 tasks | Gururangan 2020 |
| XLSR-53 → Ainu | **234 h** | 56,000 h (0.4%) | substantial WER reduction, 100k steps | Nowakowski 2023 |
| wav2vec2 → classroom | 5,235 h | 60,000 h (8.7%) | −10.3 WER pts; beats from-scratch on same data | arXiv:2409.14494 |
| BEATs → iNaturalist bioacoustics, **naive** | ~30k–50k segments | AudioSet | frozen CBI probe **43.5 → 11.9**; 73.5% forgetting | SONAR, arXiv:2509.15703 |
| BEATs → iNaturalist, with anti-forgetting machinery | same | AudioSet | +0.7 frozen / +0.9 fine-tuned over no adaptation | SONAR |

Practical reading: **~200 h of in-domain audio is enough for continued pretraining to pay when
the domain is genuinely distant from the base corpus and the base corpus is large** (Ainu). Below
that, TAPT-style continuation on the task corpus for many epochs is the documented cheaper
alternative in NLP — but nobody has published it for a speech/audio SSL encoder, so its transfer
is **NOT VERIFIED**. The SONAR row is the one that should govern expectations for *this* domain:
in bioacoustics specifically, naive continuation has been measured to be catastrophic, and the
careful version bought under one point.

---

# Part 4 — What this implies for run11

Lab-side numbers used below, with sources inside this repo:

* run11 recipe: "HuBERT masked-prediction, **iteration 1**, targets from k-means (**k=100**) over
  log-mel spectrogram frames. **19 epochs / 93,750 steps** on 4× NVIDIA L40S. **Learning rate
  1e-4**" — `release/zf_hubert_run11/README.md`. Corpus "~**100 hours** of unlabelled zebra finch
  colony recordings", **120 recordings**. Effective batch size: **NOT VERIFIED** (not recorded in
  the release or anywhere else in this repo).
* Iteration 2 on this corpus "**did not help**… slightly hurt detection (0.913 vs 0.921 AUC) and
  was a wash on call type" — same README.
* run11 vs AVES: 11-class call type leave-birds-out run11 **0.8118** vs AVES **0.8453**,
  −0.0338 [−0.0609, −0.0069] significant over 48 birds — `knowledge/findings/033`.
  In-distribution detection not distinguishable; ZF→BirdPark holdout AVES ahead at 4/4 layers —
  `findings/030`. Fine-tuning does not flip it (+0.0296 for AVES full fine-tune) — `findings/036`.
  run11 wins nowhere once layer choice is out-of-fold — `findings/035`.
* **Averaging the two probes' probabilities beats run11 on detection** (+0.0047 AUC
  [+0.0034, +0.0061], +0.0118 AP) and on the holdout (+0.0249 AUC) — `findings/031`.

**The compute comparison that reframes the whole question.** run11 ran 93,750 steps; AVES-bio ran
100,000. The two models are within 7% of each other on optimiser steps. What differs is: corpus
(100 h of one colony vs 360 h of FSD50K + AudioSet-20k + animal AS/VS), cluster count (100 vs
200), iterations (1 vs 2), and target features (log-mel vs 39-d MFCC then layer 6 of a trained
HuBERT). **run11 is not under-trained relative to AVES; it is under-diversified.** And §3.3 says
diversity is the axis that matters for SSL in this domain.

### Recommendations, in priority order

**R1 — Ship the two-encoder probe average now. Do not train anything for this.**
*Evidence:* `findings/031` — mean of the run11 and AVES probe probabilities beats run11 by
+0.0047 AUC / +0.0118 AP in distribution (both significant) and by +0.0249 AUC on the BirdPark
holdout, at zero added parameters. The two encoders' errors correlate at 0.8575 — high, but far
enough below 1 to leave the complementarity the average exploits. Concatenation has now failed four times on this project and should not be
retried. Note the asymmetry: on call type the ensemble is **not** distinguishable from AVES alone
(`findings/033`), so for call type the recommendation is simply *use AVES*.
*Effort:* hours. It is a change to the inference path, not a training run.
*This is the only recommendation here with a measured positive result behind it.*

**R2 — Before any pretraining, spend a week on the readout and the front end. This is where the
literature says the points actually are.**
*Evidence:* (a) The FM review measures BEATs going 94.10 → 97.98 BEANS AUROC and AudioMAE
84.47 → 97.19 purely by switching linear probing to **attentive probing**; every number the lab
has on run11 vs AVES is a *linear* probe, so both models may be understated and the *gap* may not
be stable under a better readout. (b) The same review finds "Models processing raw waveforms
(AVES: 74.48, BirdAVES: 78.87) consistently underperform compared to their spectrogram-based
counterparts", and every top BirdSet model runs at **32 kHz with 5-second windows** against
run11/AVES's 16 kHz. Zebra finch calls have energy well above 8 kHz, so 16 kHz is a real
information ceiling that no amount of pretraining recovers. (c) AVEX's post-training result —
supervised post-training on top of an SSL backbone gave "consistent improvement vs. raw SSL
backbones" and is what took sl-BEATS from 0.774 to 0.832 on BEANS-Class probe.
*Concretely:* attentive probe on top of AVES; and a supervised post-training pass on the lab's
existing labelled sets before touching the encoder — the 3412-clip / 48-bird 11-class call-type
cohort (`findings/034`) and the 7350-window detection set (2450 vocalization / 4900 background
across 71 recordings, `release/zf_hubert_run11/README.md`). The hand-labelled detection dataset
v1 is larger again; its exact size is not recorded in any file I read for this document, so I have
not quoted a number for it.
*Effort:* attentive probing ~1 week including the leave-birds-out harness. Supervised
post-training ~1–2 weeks. Both reuse existing labels and existing weights.
*Expected:* unknown magnitude for this corpus, but this is the axis with the largest published
effect sizes (+3 to +13 AUROC points), and it is an order of magnitude cheaper than pretraining.

**R3 — Continued pretraining from AVES weights on ZF audio: worth ONE carefully-instrumented
attempt, and I would put it below 50/50. If you run it, run it as DAPT+replay, never as naive
continuation.**
*Evidence for:* CPT beats from-scratch at equal data in speech (W2V-LV60K CPT 22.52 WER vs
W2V-SCR 30.25, arXiv:2409.14494). Gururangan's DAPT paid on corpora 7–30% the size of the base
corpus, and the Ainu result paid at **234 h against a 56k-h base (0.4%)**. The lab's 100 h against
AVES-bio's 360 h is **28%** — comfortably inside the range where DAPT has worked elsewhere.
Iteration 2 already failed on this corpus from a run11 start, but that is a statement about
run11's own targets, not about AVES's initialisation.
*Evidence against, and it is strong:* SONAR's DCPT is the single closest published experiment —
naive continued pretraining of an audio SSL encoder on bioacoustic data dropped a frozen bird-ID
probe from 43.5 to 11.9 and forgot 73.5% of the base task. Their engineered fix recovered only
+0.7 over not adapting at all. Separately, AVEX's EAT ablation says bioacoustics-only SSL data is
the *worst* mix of the three — and 100 h of a single colony is the narrowest possible version of
"bioacoustics-only". And AVES-nonbio ≈ AVES-bio in AVES's own table means the domain-match
hypothesis that motivates this whole experiment is already weakly supported at best.
*If you do it, the protocol that the literature supports:*
  1. Initialise from `aves-base-bio` (**not** the large BirdAVES models — `[paper]` Table 4 shows
     large lost to base on BEANS, and you have 100 h).
  2. **Re-derive the k-means targets on ZF audio from AVES's own layer 6** — `[paper]` says layer 6
     is the base-model iteration-2 source, and `[cfg]`'s `hblab.c200` confirms the released
     checkpoint's targets came from a HuBERT layer, not MFCC. Use **k=200** to match AVES's
     label space, not run11's k=100. This step is the HuBERT analogue of SONAR's online codebook
     and is the mechanism that should prevent DCPT's collapse.
  3. **Mix in replay**: 20–30% of each batch drawn from AudioSet/VGGSound animal segments (or
     FSD50K, which is freely downloadable and is in AVES's `core`). This is the one lever SONAR
     shows to be decisive.
  4. **Low LR, short warmup, few steps.** AVES peaked at 2e-4 with a 700 s batch; run11 needed
     1e-4 and diverged at 5e-4 on this corpus. Start at **5e-5 to 1e-4** with ~5% warmup and
     budget 10k–20k steps, i.e. a handful of passes over 100 h — Gururangan's DAPT was a *single
     pass*. Do not reuse AVES's 32k warmup: at 20k total steps you would never leave warmup.
  5. **Instrument for forgetting explicitly.** Evaluate the adapting checkpoint on BirdPark
     (`findings/016`) and on a general-audio holdout at every save. SONAR's failure was invisible
     from the in-domain loss and only showed up on the retained task. Pre-commit a stop rule.
  6. Pre-commit the comparison: 11-class call type leave-birds-out against AVES's 0.8453 and
     run11's 0.8118, out-of-fold layer selection, bootstrap over the 48 birds — i.e. the exact
     protocol of `findings/033`/`035`, so the result is comparable to what already exists.
*Effort:* 2–4 weeks. The k-means re-derivation over 100 h at 50 fps is ~18 M frames × 768 d, which
per this lab's own chunking lesson must be done in ~16k-row chunks on the time axis. Training
itself is small: AVES-bio's 100k steps took 24.6 h on one GPU `[cfg]`, so 20k steps on 4× L40S is
well under a day.
*Honest expectation:* the most likely outcomes are (a) roughly AVES-level performance, which is
still a win over run11 and would be a publishable negative-to-neutral result, or (b) measurable
forgetting on BirdPark. A clear win over AVES is the least likely of the three.

**R4 — Distillation is the wrong tool for *accuracy*, and the right tool for *deployment*. Frame
it that way or skip it.**
*Evidence:* Every method in §3.2 is a *compression* method, and at the sizes that are cheap to
produce the student loses to the teacher. HuBERT Base PR 5.41 against DistilHuBERT 16.27,
FitHuBERT 13.32, DPHuBERT 9.67 — all at ~23 M params. The one exception is LightHuBERT-aBase at
68 M params, PR **4.71**, which does beat the teacher (its abstract claims it "performs better
than the original HuBERT on ASR and five SUPERB tasks") — but it cost ~2k GPU-hours to produce,
i.e. the same order as pretraining from scratch, which is why DPHuBERT exists. So distillation at
a price the lab would actually pay will not close the run11→AVES gap. What it *will* do, cheaply: give the lab a 23–24 M-parameter encoder
that retains most of AVES's frozen-probe quality, from **24 GPU-hours** (DPHuBERT) or **55 h on
one GPU** (DistilHuBERT), using only **~100 h** of audio — DPHuBERT at 100 h (PR 10.02 / WER
11.38) beat DistilHuBERT at 960 h (16.27 / 13.37). The lab already has 100 h.
*The one genuinely interesting variant:* **distil AVES into a small student using the lab's ZF
audio as the distillation corpus**, choosing teacher layers by the lab's own layer findings —
AVES L3 for call type / L6 for in-distribution detection / L0–L3 for detection (`findings/030`,
`033`, and the release README's "shallow layers detect; middle layers identify"). DistilHuBERT's
Table 4 shows target-layer choice is decisive and task-specific (layer 4 → SID 76.85 but IC
79.09; layer 8 → IC 96.89 but SID 61.52), so a ZF-tuned layer selection is a real design choice
and not just a reimplementation. This is *specialisation without pretraining* — it moves AVES's
representation toward ZF using 100 h, without the forgetting risk of R3, because the teacher is
frozen and the objective is regression onto it rather than masked prediction with new targets.
*Effort:* 1–2 weeks with DPHuBERT's released code, or ~1 week for the simpler DistilHuBERT
recipe. Lower risk than R3 and a more certain (if smaller) payoff: a fast encoder for the
labelling tool and for long-recording sweeps.
*Do not expect an accuracy gain over AVES.* If someone proposes distillation as a way to beat
AVES, that is not supported by anything in §3.2.

**R5 — If the goal is the best possible ZF encoder rather than a defence of run11, the cheapest
large win on the table is 32 kHz + spectrogram + a different backbone — not more HuBERT.**
*Evidence:* Perch 2.0's 12 M-parameter supervised EfficientNet-B3, with only a **linear probe**
on frozen embeddings (0.835 acc / 0.426 mAP on BEANS), beats a **fully fine-tuned** AVES-Bio
(0.817 / 0.398) at 8× fewer parameters — Perch 2.0 Table 3, where the AVES row is marked `FT` and
the Perch row `LP`. The FM review's top BirdSet models
are all 32 kHz spectrogram models and its explicit finding is that raw-waveform 16 kHz models
"consistently underperform". AVEX's best configuration is SSL-on-mixed-data followed by
supervised post-training, reaching 0.832 BEANS-Class probe against Bird-AVES-biox-base's 0.705.
run11 and AVES share a 16 kHz raw-waveform front end, which caps both of them on the
high-frequency content of ZF calls.
*Effort:* large — a new architecture and a new training pipeline, 1–2 months. Listed because
every external benchmark says this is where the headroom is, and it would be dishonest to
present R3 as the highest-ceiling option when it is not.

**R6 — Reframe run11's result as a finding and publish it. It is corroborated, not anomalous.**
*Evidence:* Five independent sources now report domain-specific bioacoustic pretraining failing
to beat general pretraining: AVES's own nonbio control (61.3 vs 61.4 total, and 61.8 vs 61.2 on
classification); Sarkar & Magimai-Doss, where speech-pretrained HuBERT beats AVES on marmoset
call type 64.35 vs 62.54; AVEX's EAT-bio losing to EAT-AS on all four aggregate columns; the FM
review's "bioacoustic pretraining data does not guarantee better performance"; and Perch 2.0's
inability to make HuBERT/MAE/SimCLR beat supervised baselines. What run11 adds that none of these
have is the **narrowest possible domain** — one species, one colony, 100 h — plus an
encoder-level holdout (BirdPark), a leave-birds-out protocol, a fine-tuning arm, and an
out-of-fold layer-selection arm. That is a sharper test of the phenomenon than anything cited
above, and `findings/030`/`033`/`035`/`036` already constitute the experiment.
*Effort:* weeks of writing, no new compute.

### One correction to the framing in the brief

The brief describes AVES as "identical architecture, generic animal-audio pretraining". The
architecture is indeed identical `[tacfg]` vs run11. But "generic animal-audio" understates it:
`aves-base-bio` is **153 h of FSD50K + balanced AudioSet (general audio, mostly not animals)
plus 207 h of animal audio** (DERIVED from `[paper]` Table 1), trained with **iteration-2 targets
from a trained HuBERT's layer 6 at k=200**, against run11's **iteration-1 targets from log-mel at
k=100**. Two of the three most likely causes of the gap — corpus diversity and target quality —
are therefore confounded with the domain-match variable the lab set out to test. If R3 is run, the
k=200-from-layer-6 change (R3 step 2) is arguably the more interesting manipulation than the
continued pretraining itself, and it could be tested on run11's own initialisation as a cheaper
control.

---

## Full source list

* Hagiwara, M. "AVES: Animal Vocalization Encoder based on Self-Supervision." arXiv:2210.14493v1,
  ICASSP 2023. https://arxiv.org/abs/2210.14493
* earthspecies/aves README. https://github.com/earthspecies/aves (frozen; successor
  https://github.com/earthspecies/avex)
* Local checkpoint: `datasets/11905533/aves-base-bio.pt` (fairseq, 2022-10-26).
* Released torchaudio configs: `https://storage.googleapis.com/esp-public-files/ported_aves/`
  and `https://storage.googleapis.com/esp-public-files/birdaves/`
* Hagiwara, M. "Introducing BirdAVES." Earth Species Project blog, 2024-06-20.
  https://earthspecies.org/2024/06/20/introducing-birdaves-self-supervised-audio-foundation-model-for-birds/
* fairseq `examples/hubert/config/pretrain/hubert_base_librispeech.yaml`.
  https://github.com/facebookresearch/fairseq
* Hagiwara et al. "BEANS: The Benchmark of Animal Sounds." arXiv:2210.12300
* Hsu et al. "HuBERT." IEEE/ACM TASLP 29:3451–3460, 2021. arXiv:2106.07447
* Gururangan et al. "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks."
  ACL 2020. arXiv:2004.10964
* Nowakowski et al. "Adapting multilingual speech representation model for a new, underresourced
  language through multilingual fine-tuning and continued pretraining." Information Processing &
  Management 60(2):103148, 2023. Weights:
  https://huggingface.co/karolnowakowski/wav2vec2-large-xlsr-53-pretrain-ain
* "Continued Pretraining for Domain Adaptation of Wav2vec2.0 … Elementary Math Classroom
  Settings." arXiv:2405.13018
* "CPT-Boosted Wav2vec2.0: Towards Noise Robust Speech Recognition for Classroom Environments."
  arXiv:2409.14494
* Zhang et al. "SONAR: Self-Distilled Continual Pre-training for Domain Adaptive Audio
  Representation." arXiv:2509.15703
* Chang et al. "DistilHuBERT." arXiv:2110.01900
* Lee et al. "FitHuBERT." arXiv:2207.00555
* Wang et al. "LightHuBERT." arXiv:2203.15610
* Peng et al. "DPHuBERT: Joint Distillation and Pruning of Self-Supervised Speech Models."
  Interspeech 2023. https://www.isca-archive.org/interspeech_2023/peng23c_interspeech.pdf
* Sarkar & Magimai-Doss. "Comparing Self-Supervised Learning Models Pre-Trained on Human Speech
  and Animal Vocalizations for Bioacoustics Processing." arXiv:2501.05987
* Miron, Robinson et al. "AVEX / What Matters for Bioacoustic Encoding." arXiv:2508.11845v3
* "Foundation Models for Bioacoustics — a Comparative Review." arXiv:2508.01277
* "Perch 2.0: The Bittern Lesson for Bioacoustics." arXiv:2508.04665v2
* Lab-internal: `release/zf_hubert_run11/README.md`, `knowledge/findings/030,031,033,034,035,036`

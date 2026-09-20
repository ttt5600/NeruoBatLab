# Call-type classification: notes

The second of the two open questions (the first is detection — see
`vocalization_detection_notes.md`). Here the labels are trustworthy: human-assigned call types
on curated clips (Elie/Theunissen, 8 classes: DC, Ne, Te, Th, Tu, Ag, So, Wh). The problem is not
label noise, it is **what the existing number actually proves**.

---

## 1. Where we are

The established result is a linear probe on mean-pooled run11 embeddings, leave-birds-out CV:

| | best layer | layer 6 (the iter-2 teacher) | worst |
|---|---|---|---|
| call-type probe (acc) | **0.821** @ L3 | 0.804 | 0.797 @ L11 |

Two things about this are unsatisfying.

**The layer profile is nearly flat.** 0.797–0.821 across all twelve layers is comparable to fold
noise. A representation that genuinely encoded call type more cleanly at some depth should show
more spread than that. The detection task, on the same model, has real dynamic range
(0.879–0.921). One plausible reading: a linear probe is a strong enough learner that it recovers
call type from *any* layer, and so it cannot distinguish "the information is present and
well-organized" from "the information is present and the probe found it anyway."

**A probe can manufacture structure that is not natively there.** With 768 dimensions and a few
thousand clips, logistic regression has enough capacity to carve boundaries through a cloud that
has no call-type geometry at all. Accuracy 0.797 is evidence the information is *decodable*. It
is weaker evidence that the model *organizes* audio by call type.

---

## 2. Pretraining leakage, and what clustering does and does not fix

These curated clips were inside the 100-hour HuBERT pretraining corpus. Worth being precise
about the size of the problem, because it is easy to over- or under-state:

**What is not at risk.** Pretraining was self-supervised — no call-type label existed anywhere in
that objective. The model cannot have memorized a category → label mapping, because it was never
shown one. This is categorically different from a supervised model evaluated on its training set.

**What is at risk.** Representations of these specific waveforms may be unusually crisp relative
to audio the model has never heard, which would inflate *absolute* numbers.

**What is unaffected.** Every relative comparison — layer sweeps, run11 vs run13, HuBERT vs
energy, trained vs random-init — holds all conditions to the same leakage, so it cancels.

**Clustering does not remove this.** It is worth saying plainly, because it is a natural
assumption: k-means on the embeddings still runs on embeddings of audio the model saw. What
clustering removes is a *different* problem — probe capacity (§1). The two are orthogonal, and
clustering is worth doing for the §1 reason, not the leakage reason.

Real controls for the leakage, in descending order of strength:

1. **Hold birds or recordings out of pretraining.** Definitive, and costs a 12-hour retrain.
2. **Leave-birds-out at eval.** Already done (0.797). Controls individual-voice shortcut, which
   is a real and distinct confound, but not waveform memorization.
3. **Disclose it.** Near-universal in bioacoustic SSL — AVES, BirdAVES, and Perch all overlap
   pretraining and evaluation corpora. Not disqualifying; it is disqualifying to leave it unsaid.

Minor point in our favour: the synthetic mixtures from `synthesize_masked.py` are novel
waveforms, so results on them are not pure memorization. Weak mitigation only — the components
were seen.

---

## 3. The clustering eval (`cluster_calltype.py`)

If the model's embedding space has call-type structure, unsupervised k-means should recover
something resembling the human taxonomy without ever seeing a label. That is a strictly stronger
claim than probe accuracy, and it is what the script tests.

`eval_calltype.py` already ran one `KMeans(k=8)` and printed NMI/ARI/purity. That single number
cannot support a claim, for four reasons, each of which is a column or a row in the new output.

### 3.1 Rank on AMI, never NMI or purity

Measured on synthetic embeddings with **no structure whatsoever** (isotropic Gaussian, labels
independent of the data):

| k | AMI | NMI | purity | homogeneity |
|---|---|---|---|---|
| 4 | 0.003 | 0.020 | 0.186 | 0.017 |
| 8 | 0.015 | 0.050 | 0.214 | 0.049 |
| 16 | −0.018 | 0.056 | 0.231 | 0.062 |
| 32 | −0.020 | 0.109 | 0.272 | 0.140 |
| 60 | −0.007 | 0.192 | 0.350 | 0.274 |

NMI, purity and homogeneity all climb monotonically on pure noise; at k = n_clips purity is
exactly 1.0. Ranking a k-sweep on any of them selects the largest k for free. AMI has expectation
0 at every k, so it is the only one of the four that makes the sweep a fair comparison.

**This invalidates one comparison already in `eval_calltype.py`**, now fixed: its Eval A (k=8
clusters on embeddings) was compared against Eval B (k=100 kmeans training targets) on NMI. A
12× cluster-count advantage is worth a large NMI gap on its own. Both now print AMI.

### 3.2 Don't force k=8

The human taxonomy has 8 labels; there is no reason the model's natural partition has 8 parts. It
might split Distance calls by bird, or merge two acoustically adjacent types. Forcing k=8 measures
agreement with a taxonomy rather than the presence of structure. We sweep k=2…60 and report where
AMI peaks — legitimate precisely because AMI is chance-corrected.

### 3.3 Baselines, or the number means nothing

"AMI = 0.35" is uninterpretable alone. Three controls run in the same table:

- **`random@L*`** — identical architecture, **no checkpoint loaded**. Untrained deep nets are
  well known to be decent feature extractors. If trained ≈ random, pretraining contributed
  nothing and we are measuring the convolutional stack's spectral bias. This is the single most
  important row in the table.
- **`spectrogram`** — mean-pooled spectrogram, the raw acoustics the k-means training targets
  were built from. The floor.
- **`duration`** — clip length, one scalar. Call types differ in length, so a clustering that
  merely sorts by duration would look impressive while encoding nothing.

### 3.4 Bird identity is a confound, and the global metric cannot see it

Clusters could track *who* is calling rather than *what* was called — the same shortcut
leave-birds-out was built to catch. Two extra columns: AMI against bird ID (should be low), and
**within-bird AMI**, which recomputes agreement separately inside each bird's clips and averages
weighted by clip count. Structure that survives with voice held fixed is call type.

Validated on four synthetic regimes with known ground truth:

| regime (what the embedding actually encodes) | AMI vs type | AMI vs bird | AMI within-bird |
|---|---|---|---|
| call type | **1.000** (peaks at k=8) | 0.006 | 1.000 |
| nothing (pure noise) | 0.015 | 0.003 | 0.017 |
| bird only, type independent of bird | 0.025 | 0.692 | 0.040 |
| **bird only, but birds have skewed type mixes** | **0.489** | 0.958 | **−0.011** |

The last row is the reason the columns exist. That embedding contains **zero** call-type
information, yet global AMI reads 0.489 (NMI 0.524, purity 0.781) purely because each bird
favours one call type, so a bird-clustering inherits the correlation. A permutation test does not
save you either — 0.489 against a null 95th percentile of 0.021 is wildly "significant" and
entirely wrong. Only the within-bird column (−0.011) exposes it. This is a live risk here, not a
hypothetical: the probe already shows a random-split vs leave-birds-out gap.

### 3.5 Layer sweep is free

`extract_features` returns all twelve layer outputs from one forward pass, so clustering every
layer costs about what one layer costs. Worth doing because depth mattered non-monotonically for
detection (peaks at L0, declines with depth) — and because if the clustering layer profile has
more spread than the probe's flat 0.797–0.821, that itself is the answer to §1.

Valid indices are **0–11**, not 0–12: HuBERT BASE returns 12 outputs, with no separate entry for
the feature-extractor output (verified against torchaudio 2.2.2). The script range-checks the
requested layers against one clip before the main loop, because an off-by-one here would
otherwise surface as an `IndexError` well into a GPU job.

### 3.6 Normalization is not a detail

k-means minimizes Euclidean distance, so dimensions with large variance dominate the partition —
and mean-pooled transformer embeddings have very uneven per-dimension scale. `l2` (spherical
k-means, direction only) and `zscore` (equalize per-dim scale) can give materially different
answers, so both are reported rather than one being silently assumed.

---

## 4. To run

```bash
sbatch pytorchAudio/examples/hubert/slurm/cluster_calltype.sh
```

Needs `adultvoc_16k`, `calltype_labels.csv`, and the run11 checkpoint (all already on scratch).
GPU, ~2h wall limit; the forward pass over the curated clips is the bulk of it.

Read the result in this order:

1. `hubert` vs `random` — if these are close, stop; pretraining did nothing here.
2. `hubert` vs `spectrogram` and `duration` — must clear both to be interesting.
3. AMI(bird) and within-bird AMI — if AMI(bird) is high and within-bird is near zero, the
   clustering is a bird clustering and the headline AMI is the §3.4 trap.
4. Only then, the headline AMI and the cluster × call-type confusion table, which shows *which*
   types merge — the scientifically interesting part.

## 5. Open

- **Augmentation.** Does mixing noise/synthesized masked calls into training (augment mode of
  `synthesize_masked.py`) improve call-type classification, or only detection? Untested.
- **Pretraining holdout.** The only definitive leakage control (§2). 12 GPU-hours.
- **Frame-level vs clip-level.** Everything here mean-pools a whole clip. Call types differ in
  temporal structure, which mean-pooling destroys by construction.

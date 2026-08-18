# Unsupervised call-type discovery from run11 embeddings

Corpus: 2814 curated clips, 26 birds, 8 human call types, layer-3 mean-pooled 768-d
embeddings, L2-normalized. All numbers below reproduce from the scripts in this directory
against `../data/run11_layersweep.npz`; raw output is in `results/`.

---

## Why this exists

The previously reported clustering number was **AMI 0.5475 at k=16**, and k=16 was chosen by
maximizing AMI against the human labels. That is a fine measurement of *how much call-type
structure is present*, but it is not an unsupervised result — the labels picked k. Someone
clustering an unlabeled corpus has no way to know to ask for 16 clusters.

This splits the two questions apart:

1. **How many groups does the embedding contain**, judged only by its own geometry?
2. **Having chosen k without labels**, how well does that partition match the taxonomy?

---

## Headline

**Choosing k with no labels costs almost nothing, and switching off k-means gains a lot.**

| | k | AMI vs human call types |
|---|---|---|
| published (k-means, k chosen by AMI) | 16 | 0.5475 |
| k-means at the label-free k | 13 | 0.5332 |
| **Ward at the label-free k** | **13** | **0.5907** |
| Ward at its AMI-optimal k (uses labels) | 9 | 0.6090 |

Shuffled-label null: 95th percentile **0.0017** (500 permutations). The signal is not chance.

Going fully unsupervised costs **0.018 AMI** (0.5907 vs 0.6090). The honest unsupervised
number is *higher* than the old label-optimized one, because the clusterer mattered more than
k did.

---

## 1. k-means was the wrong clusterer

Ward agglomerative beats k-means at every k tested, by up to 0.13 AMI:

| k | k-means | Ward | Δ |
|---|---|---|---|
| 8 | 0.4676 | 0.5867 | +0.119 |
| 9 | 0.4765 | **0.6090** | +0.133 |
| 13 | 0.5332 | 0.5907 | +0.058 |
| 16 | 0.5475 | 0.5664 | +0.019 |

Ward is deterministic and k-means is not, so this needed a fairness check before it could be
believed (`unsup_kmeans_vs_ward.py`):

| k | k-means, 20 seeds | k-means n_init=200 | Ward |
|---|---|---|---|
| 8 | 0.464 ± 0.011 (max 0.484) | 0.469 | 0.587 |
| 9 | 0.477 ± 0.009 (max 0.486) | 0.485 | 0.609 |
| 16 | 0.533 ± 0.009 (max 0.548) | 0.535 | 0.566 |

Ward beats the *best of 20 seeds* and a 200-restart run. So this is not optimization luck.

It is also not that k-means' objective points the wrong way: `corr(inertia, AMI)` is negative
at every k (−0.36, −0.47, −0.66), meaning a *better* k-means fit does give *better* call-type
recovery. Even k-means' global optimum is simply worse. The spherical, equal-variance cluster
model is what fails — call types differ 3× in frequency (DC 583 clips vs Wh 172) and each is a
curved manifold of renditions rather than a ball.

Ward is also *less* contaminated by bird identity at the same k: at k=9, AMI vs bird is 0.167
for Ward against 0.190 for k-means, while call-type AMI is far higher. It wins on both axes.

> **Note on the published number.** 0.5475 is the maximum over 20 k-means seeds; the mean is
> 0.5326 (sd 0.0092) and a 200-restart run gives 0.5354. Seed 0 with `n_init=10` drew the top
> of the distribution. The old figure is about 0.015 optimistic — not wrong, but it should be
> quoted as 0.535 ± 0.009 if k-means is used at all.

## 2. Choosing k without labels

Six label-free criteria, none of which ever sees `y`:

| criterion | picks k | → Ward AMI | usable? |
|---|---|---|---|
| stability (k≥5) | 13 | 0.5907 | **yes** |
| silhouette | 17 | 0.5623 | yes |
| Ward silhouette | 27 | 0.5474 | weak |
| Calinski-Harabasz | 2 | 0.2617 | no — monotone |
| Davies-Bouldin | 40 | 0.5294 | no — at sweep edge |
| GMM BIC (PCA-30) | 39 | 0.5291 | no — at sweep edge |

CH decreases monotonically from k=2; DB and BIC are still improving at k=40, the edge of the
sweep. All three are effectively uninformative here and should not be quoted as "BIC says 39".

**Stability** (Ben-Hur: cluster two random 80% subsamples, compare their labels on the clips
both saw) is the one that behaves. Its top values for k≥5 are k=13 (0.816), k=7 (0.813),
k=6 (0.812) — an interior optimum landing inside Ward's broad AMI plateau (k=9–13 spans
0.591–0.609). k=3 scores 0.976, but a 3-way split of anything is trivially reproducible; that
is why the k≥5 restriction is stated explicitly rather than hidden.

**Bird identity rises monotonically with k.** AMI vs bird goes 0.063 (k=3) → 0.252 (k=16) →
0.338 (k=40) for k-means. Asking for more clusters increasingly buys individual voices rather
than call types — an argument for small k that owes nothing to the taxonomy.

## 3. Baselines: is HuBERT earning its keep?

Same clips, same protocol, Ward at k=9:

| representation | AMI vs call type | AMI within bird |
|---|---|---|
| duration alone (1 scalar) | 0.3945 | 0.4587 |
| log-band spectrogram (mean+std pooled) | 0.4613 | 0.6127 |
| **HuBERT layer 3** | **0.6090** | **0.7547** |

HuBERT clears the spectrogram by **+0.148** and duration by **+0.215**. It earns its keep.

But **duration alone reaching 0.39 is a bigger free lunch than expected** and should be
reported whenever the clustering number is. Mean durations: So 2.312 s, Wh 0.331, Ag 0.298,
Ne 0.243, DC 0.211, Te 0.105, Th 0.068, Tu 0.068. Song is ~10× longer than anything else and
Thuk/Tuck are indistinguishable by length — which is exactly the split/merge pattern the
embeddings show.

HuBERT's partition is not merely duration in disguise: AMI between the HuBERT clustering and
the duration clustering is only 0.199 at k=8 (vs 0.395 between HuBERT and spectrogram).

Not run: a **random-init HuBERT** control, the sharpest test of whether *pretraining* rather
than the conv stack's spectral bias did the work. It needs torch, absent from the local
analysis env; it belongs on Savio and is the top open item.

## 4. What the clusters actually are

Ward at k=9 (the AMI peak):

| cluster | n | contents | purity | birds |
|---|---|---|---|---|
| 6 | 190 | **So** | 1.00 | 13 |
| 0 | 404 | **Te** | 0.95 | 18 |
| 4 | 209 | **DC** | 0.99 | 9 |
| 8 | 186 | **Ag** | 0.92 | 22 |
| 3 | 364 | **DC** | 0.90 | 21 |
| 5 | 302 | Ne 214, Wh 71 | 0.71 | 24 |
| 7 | 175 | Tu 92, Th 63 | 0.53 | 15 |
| 2 | 584 | Ne 215, Th 210, Tu 139 | 0.37 | 21 |
| 1 | 400 | Ne 123, Te 121, Wh 94 | 0.31 | 19 |

**Four of the eight types are recovered with no labels at all** — Song, Tet, Aggressive and
Distance, each as one or two high-purity clusters drawn from many birds (so they are call
types, not voices). Distance splits in two, both clean, which is a candidate sub-type worth a
listen.

**The other four collapse into one confusable block.** Nest, Thuk, Tuck and Whine all have
their largest cluster dominated by Ne. Nest spreads over three clusters and is never the clean
majority of any.

This is the same failure surface the supervised probe hits — per-class recall Ag 0.949,
DC 0.964, So 0.990, Te 0.832 against Th 0.573 and Tu 0.379. Unsupervised and supervised fail
on the *same* four types, so this is a property of the representation, not of the clustering
algorithm. Two readings, not yet distinguished: the model cannot hear the distinction, or the
distinction is partly annotator convention.

## 5. HDBSCAN: a clean core, with a caveat that matters

HDBSCAN chooses its own cluster count and may refuse to assign a clip:

| min_cluster_size | clusters | noise | AMI on assigned |
|---|---|---|---|
| 10 | 18 | 54.6% | 0.7232 |
| 20 | 9 | 60.3% | 0.7743 |
| 30 | 5 | 66.3% | 0.7757 |
| 50 | 2 | 67.3% | 0.5600 |

At `min_cluster_size=20` it independently lands on **9 clusters** — the same count as Ward's
AMI peak, from a completely different (density-based) model.

AMI 0.774 on the assigned 40% is **not** comparable to the full-set numbers: it is measured on
whatever the method found easy. Two checks:

- **It is not just Song.** Excluding Song, core AMI is still 0.7629 (n=1007).
- **But the small clusters are birds, not call types.** Splitting the core into its two large
  clusters (n=839) and seven small pockets (n=277):

  | | AMI vs call type | AMI vs bird |
  |---|---|---|
  | 2 big clusters | 0.6899 | **0.0425** |
  | 7 small pockets | 0.6908 | **0.7402** |

  The pockets are one bird's Songs, one bird's Tets, and so on. They score well on call type
  only because a single bird's pocket happens to be a single call type. Ward on the *same*
  1116 clips at k=9 gets 0.7385 vs HDBSCAN's 0.7743, so most of the gap is the subset, not the
  density model.

Practically: an auto-labeling tool built on HDBSCAN would need to merge the voice pockets. The
core rate also varies enormously by type — DC 72.7% assigned, Tu only 12.1% — so the "confident
core" is heavily DC-weighted and would not give balanced coverage.

## 6. Layer choice — an unsupervised user cannot get this right

Ward AMI by layer:

| layer | 0 | 1 | 2 | **3** | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k=9 | .552 | .555 | .531 | **.609** | .553 | .515 | .539 | .529 | .574 | .486 | .521 | .543 |
| k=13 | **.598** | .565 | .540 | .591 | .537 | .553 | .571 | .552 | .548 | .522 | .528 | .563 |

Layer 3 is best at k=9 and second at k=13; layers 0–3 are the good region and deep layers are
clearly worse (layer 9 bottoms out). Layer 3 is a defensible default, but the ranking *within*
the shallow group flips with k, so "layer 3" should be quoted as "an early layer, 0–3".

**The label-free criteria do not find it.** Silhouette decreases monotonically with depth
(0.167 at layer 0 → 0.115 at layer 10), so it always picks layer 0 regardless of merit;
stability picks layer 10 at k=9 and layer 0 at k=13. Neither identifies layer 3 at k=9. Layer
choice is something an unsupervised user cannot determine from geometry — it needs a small
labeled validation set, and that limitation should be stated rather than papered over.

---

## Caveats

- **Mean-pooling over a 365× duration range.** Clips run 0.032 s to 11.705 s. A Song is
  averaged over ~585 frames, a Tuck over ~3. Those are not comparable summaries, and it likely
  explains both why duration is such a strong baseline and why Song is trivially separable.
  A duration-matched or segment-level analysis would test this.
- **Pretraining overlap.** These clips were inside the pretraining corpus. Pretraining was
  label-free so no call-type mapping could be memorized, but absolute numbers may be
  optimistic. The duration and spectrogram baselines share the clips, so the *comparisons*
  hold regardless.
- **AMI on a self-selected subset** (the HDBSCAN core) is not comparable to full-set AMI.
- **No random-init control yet** — see §3.

## Reproduce

```bash
cd release/zf_hubert_run11/experiments
python unsup_calltype.py      --kmin 2 --kmax 40 --n-boot 12  # main sweep, ~25 min
python unsup_kmeans_vs_ward.py                                # Ward-vs-kmeans fairness check
python unsup_baselines.py     --ks 8 9 16                     # duration + spectrogram
python unsup_hdbscan_core.py  --min-cluster-size 20           # core composition
python unsup_layer_sweep.py   --ks 9 13 --n-boot 6            # layer choice
```

Needs numpy, scipy, scikit-learn (≥1.3 for `HDBSCAN`). No torch and no GPU — everything reads
the cached `run11_layersweep.npz`, except `unsup_baselines.py`, which also reads the clip wavs
from `datasets/11905533/AdultVocalizations`.

# ZF-HuBERT — a self-supervised speech model for zebra finch vocalizations

A HuBERT BASE encoder pretrained from scratch on ~100 hours of **unlabelled** zebra finch
colony audio. It turns any recording into a sequence of 768-dimensional embeddings that carry
call-type and individual-identity information — without ever having been shown a label.

Everything here runs on a laptop CPU. No GPU, no Savio account, no training code.

```python
from zf_hubert import load_encoder, embed_file

model = load_encoder("weights/zf_hubert_run11_encoder.pt")
emb = embed_file(model, "examples/BlaBla0506_110302-DC-01.wav")   # (12 layers, 768 dims)
```

Requirements: `torch`, `torchaudio`, `numpy`. The notebooks also want `scikit-learn`,
`matplotlib`, and (notebook 3) `umap-learn`.

---

## Start here

| Notebook | What it does | Needs |
|---|---|---|
| **[01_quickstart](notebooks/01_quickstart.ipynb)** | Load the model, embed a clip, look at what the layers respond to | the weights + example wavs |
| **[02_layers_and_probe](notebooks/02_layers_and_probe.ipynb)** | Can a linear classifier read call type out of the embedding? Which layer is best? | cached embeddings only |
| **[03_clustering_and_umap](notebooks/03_clustering_and_umap.ipynb)** | Does the model cluster by call type *without labels*? Plus UMAP | cached embeddings only |

All three ship with outputs already run, so you can read them before installing anything.
Notebooks 2 and 3 use `data/run11_layersweep.npz` (all 12 layers precomputed for all 2867
curated clips; the analyses use the 2814 that map to a named bird), so they need no audio and no model.

---

## What's in the box

```
weights/zf_hubert_run11_encoder.pt   360 MB   the model
zf_hubert.py                                  loader + embedding helpers
notebooks/                                    the three notebooks above
data/run11_layersweep.npz            101 MB   2867 clips x 12 layers x 768 dims, with
                                              call-type labels and bird IDs (26 named
                                              birds + 53 `Unknown*` clips)
data/calltype_labels.csv                      ground-truth call types for those clips
examples/                            1.1 MB   16 example wavs, two per call type
```

The 360 MB file is the training checkpoint with the optimizer state and the pretraining head
stripped out (the original is 1.14 GB). The exported weights were verified to reproduce the
training checkpoint's features **bit-for-bit**, so nothing was lost in the conversion.

---

## Read this before the results

**Every bird in this evaluation was in the pretraining corpus.** The ~100 h of unlabelled audio
is 120 colony recordings, and 99.5% of the curated clips were cut from those same recordings.
No individual is held out — the largest exposure gap for any bird is 5 clips.

Pretraining was label-free, so call-type *labels* cannot have been memorized. But the model has
heard every one of these voices, which means **"leave-birds-out" below constrains the linear
probe, not the encoder.** It means "the classifier never saw this bird's labels", not "the model
never heard this bird". Treat the absolute numbers as an upper bound on what to expect from a
genuinely new individual.

Comparisons *between* layers, runs and baselines are unaffected — all of them see identical
clips. A pretraining holdout is the clean fix and has not been run.

## Results

Eight call types: `Ag` aggressive, `DC` distance, `Ne` nest, `So` song, `Te` tet,
`Th` thuck, `Tu` tuk, `Wh` whine. **2814 curated clips from 26 birds.**

> **Corrected 2026-08-16.** Earlier versions said 2867 clips from 31 birds. That counted
> filename prefixes, not individuals: `HPiHPi4748` and `HpiHpi4748` are one bird under two
> capitalizations, and four `Unknown*` prefixes (53 clips) are catch-alls rather than
> individuals. Merging the first and dropping the second gives 26 birds, and moves the headline
> probe accuracy from 0.821 to **0.811**. Reproduce with `data/corrected_metrics.json`.

### Call-type classification — linear probe, leave-birds-out

Splitting by bird, not at random, so the probe is tested on individuals whose labels it never saw.

| | accuracy |
|---|---|
| majority-class baseline | 0.207 |
| **layer 3 (best)** | **0.811** |
| worst layer (0) | 0.789 |

Per-class recall at layer 3: `So` 0.99, `DC` 0.96, `Ag` 0.95, `Wh` 0.84, `Te` 0.83, `Ne` 0.82,
`Th` 0.57, `Tu` 0.38. Tuk and thuck are the clear weak spots.

**The layer profile is nearly flat** — all 12 layers land between 0.789 and 0.811, a spread of
0.022. The fold-to-fold standard deviation is **0.034**, larger than that entire spread, so
"layer 3 is best" is not resolvable at this sample size. Read it as a weak preference, not a
finding.

**Protocol matters more than any of this.** Swapping leave-birds-out for a random 5-fold split
raises the same probe on the same features to **0.925** (+0.114) — five times the spread between
best and worst layer, and far bigger than the difference between HuBERT iteration 1 and 2
(0.003). Check how a number was split before comparing it to anything.

### Vocalization detection — linear probe, leave-recordings-out

1-second windows, 7350 of them (2450 vocalization / 4900 background) across 71 recordings.

| | AUC | accuracy | F1 |
|---|---|---|---|
| energy VAD baseline | 0.666 | 0.695 | 0.368 |
| pre-transformer (CNN) | 0.852 | | |
| **layer 0 (best)** | **0.921** | 0.845 | 0.759 |
| layer 11 | 0.883 | 0.799 | 0.681 |

**The task is call vs. colony background — NOT call vs. silence.** Negative windows are drawn
from the same continuous recordings and contain cage noise, wing flaps, movement, and
vocalizations that were never curated.

**There is no hand-annotated colony file.** Nobody marked onsets and offsets. The ground truth is
*constructed*: the curated clips are near-exact excerpts of the continuous recordings, so each is
localized back into its source by normalized cross-correlation — a coarse 4 kHz search proposes
candidates (non-max suppressed), each is re-scored at full resolution, and acceptance is decided
only by that scale-invariant verified peak (threshold 0.65). That recovered 2450 intervals across
96 recordings from 2867 clips. Positives are 1 s windows on those intervals; negatives are 1 s
windows elsewhere in the same recordings, 2:1.

**Consequence — precision is a lower bound, not an estimate.** Since "background" operationally
means "not a *curated* call", the detector is penalized whenever it correctly fires on a real
call nobody curated. Review bundles exist but their `human_label` column is still empty, so the
true false-positive rate is unmeasured.

**A retraction.** An earlier version reported **AUC 0.984** with the deepest layers best. It was
computed on corrupted ground truth: normalized cross-correlation blew up inside long runs of
exact digital zeros (denominator collapsing to its `1e-12` floor while FFT round-off remained in
the numerator, giving "correlations" of ~76 where brute force gives 0.016), which pinned some
accepted intervals to **silence** and made the task partly silence-vs-audio. After the fix the
headline fell to 0.921 and the layer ordering reversed. Do not cite 0.984.

Unlike call type, this one has a shape: AUC **declines monotonically with depth**. Shallow layers
keep the acoustic detail that distinguishes a vocalization from other colony sounds. The trend
does not extrapolate backwards, though — the CNN output feeding block 0 scores only 0.852, worse
than layer 0 by more than the entire layer-0-to-layer-11 decline.

Shallow layers detect; middle layers identify.

Full numbers and provenance: `data/detection_results.json`, loaded by `ZF_HuBERT_demo.ipynb`.

### Unsupervised clustering — Ward linkage on embeddings, no labels used

The stronger claim: not just "a classifier can decode call type" but "call type is one of the
axes the representation is organized around." Scored with **adjusted mutual information**,
which has expectation 0 at every k. (NMI and purity inflate with k for free — on structureless
data NMI climbs from 0.02 at k=4 to 0.19 at k=60 — so they cannot be used to compare across k.)

**Use Ward, not k-means.** k-means assumes spherical equal-variance clusters; call types are
elongated and unequal, so k-means has to spend several centroids tiling a single type and the
small types get absorbed into whichever big cluster is nearest. Ward beats k-means at **every**
k tested, by up to ~0.12 AMI.

| k | Ward: AMI call type | k-means, same k | AMI vs bird ID | AMI within bird |
|---|---|---|---|---|
| 8 | 0.587 | 0.468 | 0.162 | 0.734 |
| 9 (AMI peak) | 0.609 | 0.476 | 0.167 | 0.755 |
| **13 (label-free k)** | **0.591** | 0.533 | 0.240 | **0.747** |
| 16 | 0.566 | 0.547 | 0.263 | 0.731 |
| 60 | 0.507 | 0.499 | 0.397 | 0.689 |

Shuffled-label null: 95th percentile 0.002.

**k is chosen without labels.** Selecting k by maximizing AMI against the call-type labels uses
the labels to tune a hyperparameter, which would undercut the whole point of the section — that
was the flaw in the old "k=16, AMI 0.547" headline. k=13 comes from cluster *stability* under
subsampling instead. Going fully unsupervised costs only **0.018 AMI** versus the label-peeking
optimum at k=9. Of the label-free criteria tested, only stability and silhouette give an interior
optimum at all: Calinski-Harabasz is monotone to k=2, and Davies-Bouldin and GMM/BIC are still
improving at k=40, so none of those three can recommend a k on this corpus.

**Baselines:** duration alone **0.395**, log-band spectrogram **0.461**, HuBERT **0.609**
(all Ward, k=9). HuBERT earns its keep, but duration alone is a large free lunch — always report
it alongside.

Three things worth knowing:

1. **It is not a bird-identity clustering in disguise.** This was the main thing that could
   have gone wrong: if the model encoded only voice, and birds differ in which calls they make,
   a pure bird-clustering would still score well against call type. Recomputing AMI *within*
   each bird holds voice constant — and the number goes **up** (0.747 vs 0.591), not down.

   That comparison is not self-evidently fair, because within-bird scoring uses fewer clips and
   fewer call types, which could be easier on its own. A **matched control** settles it:
   resample each bird's clip count and call-type mix but draw the clips from *other* birds, and
   the clustering scores **0.491 ± 0.008** (20 seeds) versus 0.712 within-bird. Size and class
   mix held constant, the gap survives — so the gain comes from holding voice constant, and
   between-bird variation really was acting as noise in the global figure. (That control was run
   on the k-means clustering; the Ward within-bird figure is 0.747 and has not been re-controlled,
   though Ward has *lower* bird AMI than k-means at matched k, so it should only help.)
2. **Identity is encoded but does not dominate** — 0.240 vs 0.591, and it rises steadily with k
   while call type peaks and falls. Extra clusters buy voices, not call types.
3. **RETRACTED: "the model's natural granularity is ~16 clusters, not 8."** An earlier version of
   this README argued that AMI rising past k=8 meant call types contain sub-structure — variants,
   contexts, intensity gradations — that the 8 labels collapse. That conclusion does not survive
   changing the clusterer. It was an artifact of k-means approximating non-spherical clusters
   with many small spheres. **Ward peaks at k=9–13**, close to the 8-label taxonomy.

   The general lesson, worth carrying beyond this dataset: an unsupervised result that holds for
   only one clustering algorithm is a statement about the algorithm, not about the data. Vary the
   clusterer before interpreting a peak.
4. **4 of 8 types are recovered cleanly** — Song, Tet, Distance, and Aggressive, each drawn from
   13–22 birds so they are call types rather than voices. Nest/Thuk/Tuck/Whine collapse into one
   confusable block — the *same* four the supervised probe fails on (recall Tu 0.379, Th 0.573),
   so the limit is the representation, not the clustering algorithm.

---

## Which layer should I use?

| task | layer | why |
|---|---|---|
| detection / segmentation | **0** | AUC declines monotonically with depth |
| call type | **3** | weak peak (0.811); the whole stack spans only 0.789–0.811 |
| clustering / structure discovery | **3** | mild shallow-to-middle preference |
| something new | **sweep all 12** | one forward pass returns all of them; that is how both numbers above were found |

Extracting all layers costs nothing extra — the forward pass dominates:

```python
emb = embed_file(model, "clip.wav")        # (12, 768) - all layers, one pass
emb = embed_file(model, "clip.wav", layer=0)
```

---

## Caveats — please read before quoting numbers

**Pretraining overlap.** See "Read this before the results" above — this is the biggest caveat
in the package, not a footnote. No bird is held out from pretraining, so no number here
describes performance on an unheard individual.

**Individual counts are from filenames.** Bird identity is parsed from the clip filename prefix
and has not been cross-checked against colony records. The 26 figure survives case-folding and
dropping the `Unknown*` catch-alls, but if the true number of individuals is lower — e.g. if
prefixes map to pairs or cages rather than birds — then 5-fold grouped CV puts very few birds in
each test fold and the ±0.034 fold spread badly understates the real uncertainty.

**Mono downmix.** Audio is averaged across channels. Fine for two mics on one bird, wrong if
your channels are different individuals — pick a channel instead.

**Clip length.** Mean-pooling over time works well for short single calls and throws away
temporal structure. For song or anything with internal sequence, use `embed_frames` and keep
the time axis.

**Minimum duration.** Clips under ~0.05 s will fail — the conv feature extractor needs a
minimum receptive field.

---

## How it was trained

HuBERT masked-prediction, iteration 1, targets from k-means (k=100) over log-mel spectrogram
frames. 19 epochs / 93,750 steps on 4× NVIDIA L40S (Berkeley Savio). Learning rate 1e-4 — this
mattered a great deal: earlier runs at 5e-4 plateaued and never recovered, and the fix was the
learning rate, not the features or the normalization.

A second HuBERT iteration (re-clustering the model's own layer-3/6 features to make new
targets, the standard recipe) was tried and **did not help this corpus** — it slightly hurt
detection (0.913 vs 0.921 AUC) and was a wash on call type (0.824 vs 0.821). The weights here
are iteration 1 for that reason.

Model: 94.6 M parameters, 12 transformer layers, 768-d, 16 kHz, one frame per 20 ms.

---

## Getting the files

The two large files (`weights/`, `data/`) are not suitable for git.

**On Savio**, the bundle is already unpacked and world-readable at:

```
/global/home/users/jonathanswang/zf_hubert_run11/
```

Open it through OnDemand → Jupyter Server, or `cp -r` it into your own home. (Note: the copy
under `/global/scratch/users/jonathanswang/release/` is *not* reachable by other users — that
scratch root is `drwx------`. Use the home path above.)

**Off Savio:**

```bash
rsync -avP hpc.brc.berkeley.edu:/global/home/users/jonathanswang/zf_hubert_run11 .

# or the single-file tarball, 431 MB
rsync -avP hpc.brc.berkeley.edu:/global/home/users/jonathanswang/zf_hubert_run11.tar.gz .
tar xzf zf_hubert_run11.tar.gz && cd zf_hubert_run11
```

Then verify nothing was corrupted in transit:

```bash
shasum -a 256 -c SHA256SUMS
```

`export_weights.py` is included for the record — it is the script that produced the weights
file from the training checkpoint, and it refuses to write anything whose features it has not
verified against the original.

"""Build the three release notebooks. Kept as a generator so the notebooks stay diffable as
plain Python here, and so re-running produces executed notebooks with outputs baked in --
a coworker who opens them on GitHub should see results without running anything.
"""
import nbformat as nbf
from pathlib import Path

OUT = Path("/Users/jonathanwang/Desktop/vocalizations_lab/release/zf_hubert_run11/notebooks")
OUT.mkdir(parents=True, exist_ok=True)


def build(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1])
                for c in cells]
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}}
    p = OUT / name
    nbf.write(nb, p)
    print("wrote", p)


# =====================================================================================
# 01 -- quickstart
# =====================================================================================
build("01_quickstart.ipynb", [
    ("md", """# 1 - Quickstart: audio in, embeddings out

This notebook loads the zebra finch HuBERT encoder and embeds a few clips. It runs on a
laptop CPU in well under a minute -- no GPU, no training code, no Savio account.

**What the model is.** A HuBERT BASE encoder pretrained from scratch on ~100 hours of
unlabelled zebra finch colony recordings. It is *self-supervised*: it never saw a call-type
label during training. What it learned, it learned from predicting masked pieces of audio.

**What you get out.** For a piece of audio it returns 12 embeddings -- one per transformer
layer -- each a 768-dimensional vector per 20 ms frame. Different layers are good at
different things, which is the point of notebook 2."""),

    ("code", """import sys, os
sys.path.insert(0, "..")          # so `import zf_hubert` finds the module one level up
import numpy as np
import matplotlib.pyplot as plt
from zf_hubert import load_encoder, load_audio, embed_file, embed_frames, LAYER_NOTES

model = load_encoder("../weights/zf_hubert_run11_encoder.pt")   # ~360 MB, CPU is fine
meta  = model.zf_meta
print(meta["description"])
print()
print(f"pretraining : {meta['training']['objective']}, {meta['training']['steps']:,} steps")
print(f"audio       : {meta['sample_rate']} Hz mono")
print(f"outputs     : {meta['n_layers']} layers x {meta['embed_dim']} dims, "
      f"one frame per {meta['frame_stride_ms']} ms")"""),

    ("md", """## Embed a single clip

`embed_file` mean-pools over time, giving one 768-d vector per layer. That is what every
number in the README was computed from."""),

    ("code", """wav_path = "../examples/BlaBla0506_110302-DC-01.wav"    # DC = distance call

emb = embed_file(model, wav_path)
print("mean-pooled :", emb.shape, "  (layer, dim)")

frames = embed_frames(model, load_audio(wav_path))
print("frame-level :", frames.shape, "  (layer, time, dim)")

dur = load_audio(wav_path).shape[1] / 16000
print(f"\\nclip is {dur:.2f} s -> {frames.shape[1]} frames at 20 ms each")"""),

    ("md", """## What the layers actually look like

Spectrogram on top, then the frame-level activations of a shallow and a deep layer. You are
looking for whether the embedding "lights up" where the call is. Layer 0 tracks the acoustics
closely; deeper layers are smoother and more abstract."""),

    ("code", """wav = load_audio("../examples/BlaBla0506_110304-Song-06.wav")   # song, longer clip
frames = embed_frames(model, wav)

fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
axes[0].specgram(wav.numpy()[0], NFFT=512, Fs=16000, noverlap=384, cmap="magma")
axes[0].set_ylabel("Hz"); axes[0].set_title("spectrogram")

for ax, L in zip(axes[1:], [0, 6]):
    a = frames[L]                                  # (T, 768)
    a = (a - a.mean(0)) / (a.std(0) + 1e-8)        # z-score per dim, else a few dims dominate
    t = np.linspace(0, wav.shape[1] / 16000, a.shape[0])
    ax.imshow(a[:, :120].T, aspect="auto", origin="lower", cmap="RdBu_r",
              vmin=-3, vmax=3, extent=[t[0], t[-1], 0, 120])
    ax.set_ylabel("dim (first 120)"); ax.set_title(f"layer {L} activations")
axes[-1].set_xlabel("time (s)")
plt.tight_layout(); plt.show()"""),

    ("md", """## Do clips of the same call type look alike?

16 example clips, two per call type. Cosine similarity between mean-pooled layer-3 embeddings,
ordered by call type. If the model encodes call type at all, the 2x2 blocks on the diagonal
should be brighter than everything else.

This is a 16-clip eyeball test, not evidence -- notebook 2 does it properly with a probe and
notebook 3 without labels at all."""),

    ("code", """import glob, csv
# Use the shipped ground-truth labels rather than parsing filenames -- the filename convention
# is not quite one-to-one with the 8 class codes.
truth = {r["fname"]: r["label"] for r in csv.DictReader(open("../data/calltype_labels.csv"))}
paths  = sorted(glob.glob("../examples/*.wav"), key=lambda p: truth[os.path.basename(p)])
labels = [truth[os.path.basename(p)] for p in paths]

E = np.stack([embed_file(model, p, layer=3) for p in paths])
E = E / np.linalg.norm(E, axis=1, keepdims=True)
S = E @ E.T

fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(S, cmap="viridis")
ax.set_xticks(range(len(paths))); ax.set_xticklabels(labels, rotation=90, fontsize=8)
ax.set_yticks(range(len(paths))); ax.set_yticklabels(labels, fontsize=8)
ax.set_title("cosine similarity, layer 3 (pairs of the same call type are adjacent)")
plt.colorbar(im); plt.tight_layout(); plt.show()"""),

    ("md", """## Which layer should you use?"""),

    ("code", """print(LAYER_NOTES)"""),

    ("md", """### Embedding your own audio

```python
from zf_hubert import load_encoder, embed_file, embed_files

model = load_encoder("weights/zf_hubert_run11_encoder.pt", device="cuda")  # or "cpu"

emb = embed_file(model, "my_clip.wav")            # (12, 768) - all layers
emb = embed_file(model, "my_clip.wav", layer=0)   # (768,)    - just layer 0

X = embed_files(model, list_of_paths, layer=3)    # (n_clips, 768)
```

Any sample rate and any number of channels works -- audio is resampled to 16 kHz and
downmixed to mono automatically. Clips shorter than ~0.05 s will fail: the conv feature
extractor needs a minimum receptive field."""),
])


# =====================================================================================
# 02 -- layers and probe
# =====================================================================================
build("02_layers_and_probe.ipynb", [
    ("md", """# 2 - Which layer encodes call type? (and how well)

This notebook reproduces the headline call-type result from precomputed embeddings, so it
needs **no GPU and no audio** -- just `data/run11_layersweep.npz`, which holds all 12 layers
for all 2867 curated clips.

The question: *given the model's embedding of a clip, can a simple linear classifier name the
call type?* If yes, the embedding geometry contains call-type information that the model
discovered on its own, since it never saw a label.

**The protocol matters more than the number.** We split by *bird*, not at random. A random
split lets the classifier learn "this individual's voice sounds like this, and this individual
mostly makes distance calls", which inflates the score without any call-type understanding.
Leave-birds-out forces generalization to voices never seen in training."""),

    ("code", """import numpy as np, warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

d = np.load("../data/run11_layersweep.npz", allow_pickle=True)
emb, y, birds, names, classes = d["emb"], d["y"], d["birds"], d["names"], d["classes"]

print(f"{emb.shape[0]} clips | {emb.shape[1]} layers | {emb.shape[2]} dims")
print(f"{len(classes)} call types: {list(classes)}")
print(f"{len(set(birds.tolist()))} individual birds")

counts = np.bincount(y)
majority = counts.max() / counts.sum()
print(f"\\nclass counts: {dict(zip(classes, counts))}")
print(f"majority-class baseline = {majority:.3f}  <- anything at or below this is worthless")"""),

    ("md", """## The sweep

One linear probe per layer, 5-fold cross-validation grouped by bird. Takes a couple of minutes
on a laptop."""),

    ("code", """rows = []
for L in range(emb.shape[1]):
    oof = cross_val_predict(LogisticRegression(max_iter=2000, C=1.0),
                            emb[:, L, :], y, groups=birds,
                            cv=StratifiedGroupKFold(n_splits=5), n_jobs=-1)
    rows.append((L, accuracy_score(y, oof), f1_score(y, oof, average="macro"), oof))
    print(f"layer {L:2d} | acc {rows[-1][1]:.3f} | macroF1 {rows[-1][2]:.3f}")

best = max(rows, key=lambda r: r[1])
print(f"\\nbest layer = {best[0]}  acc = {best[1]:.3f}  (baseline {majority:.3f})")"""),

    ("md", """## Read the shape, not just the peak"""),

    ("code", """L  = [r[0] for r in rows]
acc = [r[1] for r in rows]

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(L, acc, "o-", lw=2, label="call-type probe (leave-birds-out)")
ax.axhline(majority, ls="--", c="gray", label=f"majority baseline ({majority:.3f})")
ax.fill_between(L, min(acc) - .002, max(acc) + .002, alpha=.08, color="C0")
ax.set_xlabel("transformer layer"); ax.set_ylabel("accuracy"); ax.set_xticks(L)
ax.set_ylim(0.18, 0.87); ax.legend(loc="center right"); ax.grid(alpha=.3)
ax.set_title(f"spread across all 12 layers is only {max(acc)-min(acc):.3f}")
plt.tight_layout(); plt.show()

print(f"best  layer {best[0]}: {best[1]:.3f}")
print(f"worst layer {min(rows, key=lambda r: r[1])[0]}: {min(acc):.3f}")
print(f"spread: {max(acc)-min(acc):.3f}")"""),

    ("md", """**This flatness is the honest headline.** Every layer lands between 0.797 and 0.821,
nearly four times the 0.211 baseline. So call-type information is present and linearly
readable — but it is spread evenly through the network rather than concentrated somewhere.

Compare that with the detection task (vocalization vs. background), which has a *shape*:

| layer | 0 | 1 | 2 | 3 | ... | 11 |
|---|---|---|---|---|---|---|
| detection AUC | **0.921** | 0.912 | 0.903 | 0.896 | ... | 0.883 |

Detection declines monotonically with depth — shallow layers keep the acoustic detail that
says "a sound happened here." Call type shows no such gradient. Treat "layer 3 is best" as a
weak preference, not a finding: a 0.024 spread across 12 layers on 2867 clips is well within
what resampling noise can produce."""),

    ("md", """## Where the errors are"""),

    ("code", """oof = best[3]
cm = confusion_matrix(y, oof, normalize="true")

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
ax.set_xlabel("predicted"); ax.set_ylabel("true")
ax.set_title(f"layer {best[0]}, row-normalized")
for i in range(len(classes)):
    for j in range(len(classes)):
        if cm[i, j] > .02:
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > .5 else "black")
plt.colorbar(im); plt.tight_layout(); plt.show()

print("per-class recall:")
for c, r in sorted(zip(classes, cm.diagonal()), key=lambda t: -t[1]):
    print(f"  {c}: {r:.3f}   (n={counts[list(classes).index(c)]})")"""),

    ("md", """## Caveat you should carry into any use of these numbers

The curated clips evaluated here were part of the ~100 h corpus the model was pretrained on.
Pretraining was label-free, so the model cannot have memorized call-type *labels* — but it
has seen this audio, so absolute accuracy may be optimistic relative to a truly fresh
recording session.

What this does **not** affect: comparisons between layers, between runs, or against the
baselines, since every one of those sees exactly the same clips. The clean fix is a pretraining
holdout (retrain with these birds excluded), which has not been run yet."""),
])


# =====================================================================================
# 03 -- clustering
# =====================================================================================
build("03_clustering_and_umap.ipynb", [
    ("md", """# 3 - Does call-type structure show up *without* labels?

Notebook 2 asked whether a classifier can find call type in the embedding. This one asks a
strictly harder question: does the embedding *cluster* by call type on its own?

The difference matters. A linear probe gets 768 dimensions to carve boundaries through the
cloud, and with that much freedom it can find a call-type-separating hyperplane even in a
representation where call type is not a dominant organizing factor. Clustering has no such
freedom -- k-means just finds whatever the dominant geometry is. If the clusters line up with
call type, call type is genuinely a principal axis of the representation.

**We score with AMI (adjusted mutual information).** Not NMI, not purity. NMI and purity both
climb with the number of clusters for free -- on structureless data, NMI goes from 0.02 at k=4
to 0.19 at k=60 -- so any sweep ranked on them just picks the largest k. AMI has expectation 0
at every k, which is the only thing that makes a k-sweep interpretable."""),

    ("code", """import numpy as np, warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.preprocessing import normalize

d = np.load("../data/run11_layersweep.npz", allow_pickle=True)
emb, y, birds, classes = d["emb"], d["y"], d["birds"], d["classes"]
bird_ids = np.unique(birds)
print(f"{emb.shape[0]} clips | {len(classes)} call types | {len(bird_ids)} birds")

LAYER = 3
X = normalize(emb[:, LAYER, :])   # L2 -> spherical k-means.
# Normalization is not cosmetic here: k-means uses Euclidean distance, so without it the few
# embedding dimensions with the largest variance would dictate the partition on their own."""),

    ("md", """## Sweep k

We do *not* force k=8. Forcing the number of clusters to the number of call types measures
agreement with a taxonomy; sweeping measures whether structure exists at all. The model was
never told there are 8 call types, and there is no reason its natural granularity would match
a human labelling scheme."""),

    ("code", """ks = [2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 60]
res = []
for k in ks:
    c = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    res.append((k, ami(y, c), ami(birds, c), c))
    print(f"k={k:3d} | AMI vs call type {res[-1][1]:.3f} | AMI vs bird ID {res[-1][2]:.3f}")

bk = max(res, key=lambda r: r[1])
print(f"\\nbest k = {bk[0]}, AMI vs call type = {bk[1]:.3f}")"""),

    ("md", """## The control that actually matters: is this call type, or is it *the bird*?

Look at the "AMI vs bird ID" column above before celebrating the call-type column.

Here is the failure mode. Suppose the embedding encodes only individual identity -- voice, not
call type. Suppose also that birds differ in which calls they tend to produce (they do). Then a
clustering that is purely a *bird* clustering will still score well against call type, because
each bird's cluster inherits that bird's favoured call type. Global AMI cannot tell the two
apart.

A permutation test does not rescue you: in a simulation of exactly this confound the global AMI
was 0.489 against a null 95th percentile of 0.021 -- wildly "significant", and completely wrong.

The fix is to recompute AMI **within each bird**. That holds voice constant, so whatever
agreement survives is about call type and nothing else."""),

    ("code", """def within_bird_ami(clusters, y, birds, min_clips=12, min_types=2):
    \"\"\"AMI recomputed inside each bird, averaged weighted by clip count.

    Birds with too few clips or only one call type are dropped -- AMI is degenerate there.\"\"\"
    tot_w, tot = 0.0, 0.0
    used = 0
    for b in np.unique(birds):
        m = birds == b
        if m.sum() < min_clips or len(np.unique(y[m])) < min_types:
            continue
        tot += ami(y[m], clusters[m]) * m.sum(); tot_w += m.sum(); used += 1
    return (tot / tot_w if tot_w else np.nan), used

for k, a_type, a_bird, c in res:
    wb, nb = within_bird_ami(c, y, birds)
    print(f"k={k:3d} | call type {a_type:.3f} | bird ID {a_bird:.3f} | within-bird {wb:.3f} ({nb} birds)")"""),

    ("md", """### How to read that table

- **within-bird stays close to the global number** -> the clustering really is tracking call
  type, and the global figure is trustworthy.
- **within-bird collapses toward 0** -> you were looking at a bird-identity clustering wearing
  a call-type costume. The global number is an artifact.

### What we got

| k | AMI vs call type | AMI vs bird | AMI within bird |
|---|---|---|---|
| 8 | 0.466 | 0.182 | 0.644 |
| **20** | **0.547** | 0.273 | **0.712** |
| 60 | 0.495 | 0.377 | 0.651 |

Three things fall out of this:

**1. The confound test comes back clean — emphatically.** Within-bird AMI (0.712) is *higher*
than global (0.547), not lower. This is not a bird clustering in disguise. It also says
something extra: between-bird variation is acting as *noise* in the global number. The same
call type produced by different individuals lands in slightly different places, so holding the
bird fixed sharpens the agreement rather than destroying it.

**2. Identity is encoded, but it is not what dominates.** AMI vs bird reaches 0.273 — half the
call-type figure — and it climbs steadily with k (0.026 at k=2 to 0.377 at k=60). Extra
clusters get spent carving out individuals. Call type instead peaks and then declines, which
is the signature of a real optimum rather than a metric artifact.

**3. The natural granularity is ~20 clusters, not 8.** The model was never told there are 8
call types, and it does not choose 8. Splitting finer *increases* agreement with the taxonomy,
which is what you would expect if call types contain sub-structure — variants, contexts,
intensity gradations — that the 8 labels collapse together. Those extra clusters are worth
listening to; they are the most interesting thing in this notebook.

Taken with notebook 2, the supervised and unsupervised results agree: call type is not merely
*decodable* from this representation, it is one of the axes the representation is organized
around. That is a strictly stronger statement than the probe alone could support."""),

    ("md", """## Chance level for these numbers"""),

    ("code", """rng = np.random.default_rng(0)
c_best = bk[3]
null = [ami(rng.permutation(y), c_best) for _ in range(200)]
print(f"observed AMI (k={bk[0]}) : {bk[1]:.3f}")
print(f"label-shuffled null     : mean {np.mean(null):+.4f}, 95th pct {np.percentile(null,95):.4f}")
print("\\n(The null mean sitting at ~0 is a sanity check on the metric, not a result --")
print(" that is what 'adjusted' in AMI means. The 95th percentile is the bar to clear.)")"""),

    ("md", """## Layer by layer, without labels

Notebook 2 found call type is readable from every layer at about the same accuracy. Does
unsupervised structure follow the same flat profile?"""),

    ("code", """lay = []
for L in range(emb.shape[1]):
    Xl = normalize(emb[:, L, :])
    c  = KMeans(n_clusters=bk[0], n_init=10, random_state=0).fit_predict(Xl)
    wb, _ = within_bird_ami(c, y, birds)
    lay.append((L, ami(y, c), ami(birds, c), wb))
    print(f"layer {L:2d} | call type {lay[-1][1]:.3f} | bird ID {lay[-1][2]:.3f} | within-bird {wb:.3f}")

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot([r[0] for r in lay], [r[1] for r in lay], "o-", label="AMI vs call type")
ax.plot([r[0] for r in lay], [r[2] for r in lay], "s--", label="AMI vs bird ID")
ax.plot([r[0] for r in lay], [r[3] for r in lay], "^:", label="AMI within bird")
ax.set_xlabel("layer"); ax.set_ylabel("AMI"); ax.set_xticks(range(12))
ax.legend(); ax.grid(alpha=.3); ax.set_title(f"unsupervised structure by layer (k={bk[0]})")
plt.tight_layout(); plt.show()"""),

    ("md", """## UMAP

Same embeddings in 2-D. Colour by call type, then by bird. Comparing the two panels is the
visual version of the control above: if the right panel looks more organized than the left,
the representation is more about *who* is calling than *what* the call is."""),

    ("code", """import umap
Z = umap.UMAP(n_neighbors=25, min_dist=0.1, metric="cosine", random_state=0).fit_transform(X)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for i, ci in enumerate(classes):
    m = y == i
    axes[0].scatter(Z[m, 0], Z[m, 1], s=5, alpha=.65, label=ci)
axes[0].legend(markerscale=3, fontsize=9); axes[0].set_title(f"layer {LAYER} - coloured by call type")

for b in bird_ids:
    m = birds == b
    axes[1].scatter(Z[m, 0], Z[m, 1], s=5, alpha=.65)
axes[1].set_title(f"layer {LAYER} - coloured by individual bird ({len(bird_ids)} birds)")
for a in axes: a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.show()"""),

    ("md", """The clustering profile favours shallow-to-middle layers a little more clearly than the
probe's essentially flat one (0.547 at layer 3 down to 0.489 at layer 9), and the within-bird
column tracks it. Bird identity, by contrast, is roughly constant across depth -- the model
does not concentrate voice information anywhere in particular."""),

    ("md", """## Things worth trying from here

- Other layers, and `metric="euclidean"` with z-scoring instead of L2 -- the choice of
  normalization can change the partition materially.
- A **random-initialized** encoder as a baseline: same architecture, no pretrained weights.
  Untrained deep nets are surprisingly decent feature extractors, so if trained ~ random then
  pretraining contributed nothing. This is the single most informative baseline and it is not
  in this notebook because it needs the model, not just the cached embeddings.
- A **duration** baseline: call types differ in length, so a clustering that merely sorts clips
  by duration can look impressive while encoding nothing acoustic.
- Frame-level rather than clip-level clustering, to find sub-call structure."""),
])

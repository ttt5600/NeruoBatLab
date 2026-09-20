# Zebra finch HuBERT — available to try

A HuBERT BASE speech model trained from scratch on ~100 hours of unlabelled colony recordings.
Training was fully self-supervised — no call-type labels were involved at any point, the model
learned by predicting masked pieces of audio. It converts recordings into embeddings that can
be classified, clustered, or visualized.

**What it does:**

- **Finds vocalizations in raw recordings** — AUC 0.921, compared to 0.666 for a standard
  energy VAD. The task is call vs. real colony background, not call vs. silence, and the ground
  truth is constructed by cross-correlation rather than hand-annotated (see `README.md`).
- **Distinguishes the 8 call types** — 81% accuracy with the classifier trained and tested on
  disjoint sets of birds (2814 clips, 26 individuals). Chance is 21%.
- **Recovers call-type structure with no labels at all** — Ward clustering on the embeddings
  matches the annotated call types at AMI 0.591, against a shuffled-label null of 0.002, with
  the number of clusters chosen without ever looking at the labels.

**On Savio** (already unpacked, nothing to download):

```
~jonathanswang/zf_hubert_run11/
```

Open it through OnDemand → Jupyter Server, or copy it somewhere convenient:

```bash
cp -r /global/home/users/jonathanswang/zf_hubert_run11 ~/
```

**Off Savio:**

```bash
rsync -avP hpc.brc.berkeley.edu:/global/home/users/jonathanswang/zf_hubert_run11 .
```

**Start with `notebooks/01_quickstart.ipynb`.** All three notebooks have their outputs saved,
so they're readable as-is without running anything. Notebooks 2 and 3 need no GPU and no audio
files — the embeddings come precomputed.

Using the model directly takes two lines:

```python
from zf_hubert import load_encoder, embed_file
emb = embed_file(load_encoder("weights/zf_hubert_run11_encoder.pt"), "my_clip.wav")
```

Dependencies are `torch`, `torchaudio`, `numpy`. It runs on a laptop CPU.

**Important caveat on the numbers.** Every bird in these evaluations was in the pretraining
corpus — the curated clips were cut from the same 120 colony recordings the model was
pretrained on, and no individual is held out. Pretraining was label-free, so call-type labels
can't have been memorized, but the model has heard all of these voices. The "disjoint sets of
birds" above refers to the classifier, not the encoder.

Read the 81% as an upper bound on what to expect from a genuinely new bird. Comparisons between
layers and between models are unaffected, since they all see identical clips. A pretraining
holdout would settle it and hasn't been run. Full details in the README.

# Pretraining holdout — LblRed0613

Every number in the main README was measured on birds the encoder heard during pretraining.
`leave-birds-out` constrains the linear probe, not the model. This directory removes that
caveat for exactly one bird — the only one it is possible to remove it for.

## Why only one bird

Holding a bird out means dropping the recordings it appears in. The 120 pretraining files are
**colony** recordings with several birds audible at once: for 41 of 62 dates there are more
birds with curated clips than there are recording files that day, which proves by pigeonhole
that at least one file carries multiple birds (81% of clips are on such dates). See
`../holdout_feasibility.py`.

`LblRed0613` is the **only** one of 26 birds that never shares a recording date. It has 241
clips spanning all 8 call types across 11 dates, and no other bird has curated clips on those
dates — so dropping them removes the bird without touching the evaluation set.

This is necessary, not sufficient: LblRed0613 could still be audible but uncurated in
recordings that are kept. The result is a lower bound on the exposure effect.

## The comparison

Identical probe protocol both times — 8-way logistic regression trained on the other 25 birds,
tested on LblRed0613's 241 clips (majority baseline 0.261). Only the encoder differs.

| encoder | heard LblRed0613? | layer 3 | layer 8 |
|---|---|---|---|
| run11 | yes | 0.846 | **0.867** |
| holdout | no | ? | ? |

The run11 row is already measured — no GPU needed, it comes straight from the cached
embeddings.

## Running it

```bash
bash 00_build_tsv.sh                      # login node, seconds
sbatch 01_preprocess_holdout.sh           # features + k-means refit + labels
sbatch 02_train_holdout.sh                # 93750 updates, 4x L40, ~12 h
python 03_eval_holdout.py \
    --ckpt /global/scratch/users/jonathanswang/temp_train_holdout_lblred0613/checkpoints_*/last.ckpt \
    --clip-dir /global/scratch/users/jonathanswang/adultvoc_16k \
    --labels /global/home/users/jonathanswang/zf_hubert_run11/data/calltype_labels.csv
```

`01` uses `--start-from features`, not `--start-from labels`, because the k-means codebook has
to be refit on the reduced corpus. Reusing run11's codebook would leak the held-out bird back
in through the pretraining target vocabulary.

`02` copies run11's recipe exactly (lr 1e-4, 93750 updates, 3125 warmup, feature_weight 0,
`HUBERT_NORMALIZE_INPUT=0`, k=100, virtual-chunk 20 s) so the corpus is the only variable.

## The confound you must not skip

The holdout corpus is **100 recordings, not 120** — 16.7% less pretraining audio. So a drop in
accuracy has two possible causes and this experiment alone cannot separate them:

1. the encoder never heard the bird (what we want to measure), or
2. the encoder saw 16.7% less data (a nuisance).

If the drop is small, it doesn't matter. **If the drop is large, run the control**: retrain
dropping 20 *random* recordings that contain no LblRed0613 audio, and compare holdout against
that control rather than against run11. Only the holdout-vs-control gap isolates exposure.

That control is a second 12-hour job, which is why it is worth running the holdout first and
looking at the size of the effect before paying for it.

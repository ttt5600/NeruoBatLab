# zfeval

A reusable evaluation suite for zebra finch vocalization models. Point it at a checkpoint and a
dataset registry; get back AUC, accuracy, embedding geometry, clustering, UMAP, onset/offset
scores, baselines, controls, and a regression comparison against a previous run.

It exists so that **retraining produces a comparable record instead of a fresh pile of numbers.**

```bash
# 1. GPU: load the checkpoint, cut features for every declared dataset
python run_eval.py extract --config config/datasets.yaml --ckpt <ckpt> --out runs/run14 \
       --num-classes 500 --hubert-dir ~/pytorchAudio/examples/hubert

# 2. CPU: probes, baselines, controls, geometry, events, report
python run_eval.py analyze --out runs/run14 --name run14

# 3. what moved since the last model
python run_eval.py compare --run runs/run14 --baseline runs/run11
```

Outputs land in the run directory: `report.json` (everything, machine-readable), `SUMMARY.md`
(human-readable), `COMPARISON.md` (deltas vs a baseline), plus the cached feature `.npz` files so
`analyze` can be re-run without touching a GPU.

## What it computes

| section | contents |
|---|---|
| `scores` | per-layer AUC / AP / accuracy for every dataset and every train→test pair, each carrying its split description and majority-class rate |
| `baselines` | log-mel (384-d) and log-energy through the identical probe and splits |
| `operating_points` | at 1/5/10/20% false alarms, how many calls are recovered |
| `controls` | shuffled-label, per-recording spread, loudness-stratified AUC, loudness-matched pairs |
| `geometry` | per-layer linear AUC, kNN-10 purity, silhouette, Fisher ratio, PCA dim@90%, k-means AMI |
| `dominant_structure` | what unsupervised clusters actually track: label, recording, or loudness |
| `loudness_dependence` | what removing the loudness-correlated direction costs |
| `events` | onset/offset F1 at several onset tolerances, error taxonomy, recall by call duration |

## Why several things are mandatory rather than optional

Each of these is here because its absence produced a wrong number on this project.

- **A span nobody listened to is never scored.** `RecordingSet.load` truncates audio to the
  annotated span. Scoring 50 unheard minutes of `111021-000` as negative dropped frame AUC from
  0.968 to 0.849, and it was caught only because a printed voiced-fraction looked wrong.
- **`labeled=0` means nobody listened.** Those rows are excluded, never defaulted to negative.
- **A checkpoint that does not fully load aborts.** `strict=False` is needed for the unused final
  projection, but it would equally tolerate a wholesale encoder mismatch and return plausible
  numbers from a randomly initialised model. Every non-`final_proj` key is checked and a weight
  fingerprint goes into the report.
- **Every metric carries its split.** `metrics.Score` cannot be constructed without one, and
  `splits.choose_cv` returns a description that says when a split is *not* a recording holdout.
- **Every comparison carries an interval.** `metrics.paired_bootstrap` resamples the unit that is
  actually independent — recordings for grouped CV, moving blocks for tiled windows and frames.
- **Every control declares its own null and raises if the null is wrong.** A broken control is
  worse than none: the first loudness-matched pair matcher scanned in one direction, paired
  positives with quieter negatives, and gave energy a 0.926 win rate on pairs that were supposed
  to be equal-loudness.
- **Baselines run every time.** If the model does not beat log-mel, the pretraining bought nothing
  and the suite should say so on its own.
- **Grid-edge tuning is flagged.** If a decoder knob lands on the edge of its grid, the reported
  score is a floor, not a maximum. That happened twice here.

## Declaring data

See `config/datasets.example.yaml`. Two dataset kinds:

- `windows` — a CSV of labeled fixed-length windows plus an audio directory.
- `recording` — one continuous wav plus onset/offset annotations, used for frame and event level.

Set `role: holdout` on anything the **encoder** has never heard. That distinction is the whole
point when you retrain: a probe holdout and an encoder holdout are different claims, and the suite
labels which one each number is.

> Channel warning for BirdPark (Zenodo 20608098): wav channels 0–1 are backpack **accelerometers**
> (spectral centroid ~500 Hz, 99% of energy below 4 kHz), not microphones. The mics are channels
> 2–5. Annotated regions exceed the gaps by 18.9 dB on the mic mix and only 2.0 dB on the
> accelerometer mix.

## Adding a new model

Nothing about the suite is run11-specific except `features.load_encoder`, which builds a
torchaudio HuBERT module. To evaluate a different architecture, provide a function returning
`(model, meta)` where the model exposes `wav2vec2.extract_features(waveform, None) -> (list_of_layers, _)`,
and pass it in place of `load_encoder`.

## Tests

```bash
python -m pytest tests/ -q      # 23 tests
```

They check the guards actually fire — NaN annotations, unheard spans, biased pair matching,
grid-edge detection, and that the fast interval matcher agrees with brute-force greedy.

## Reference run

`runs/run11/` is the stored baseline: the scored run11 features, so `compare` has something to
diff against from day one.

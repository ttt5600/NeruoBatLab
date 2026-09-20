# Orientation: how this project is built and how to operate it

Companion to `CONTEXT.md`. That file is auto-generated from `findings/` and tells you **what we
have learned**. This file is hand-maintained and tells you **where things are and how to run
them**. It deliberately contains no experimental results — those go stale; this does not.

Last verified 2026-09-19.

---

## 1. The model

`run11` and every AVES checkpoint are the **same architecture**, down to the parameter. This is
load-bearing: the corpus and the label recipe are the only things that ever differ, so there is no
architecture confound to argue about.

| | |
|---|---|
| constructor | `torchaudio.models.hubert_pretrain_model` (base config, built explicitly in `lightning_modules.py`) |
| encoder params | **94,370,944** — this is what `--init-weights` transplants |
| full pretrain model | 94,594,176 (adds the label head / logit generator) |
| conv feature extractor | **4,200,448** (4.4 %) — worth +0.244 detection AUC on its own |
| transformer | 12 layers × 768 dim × 12 heads, ff 3072 |
| extractor mode | `group_norm`, `conv_bias=False` |
| final dim | 256 |

**`--feature-type` is a LABEL-pipeline knob, not a model knob.** Its choices are
`spectrogram` / `hubert`, and it selects what the *k-means targets* are built from. It does not
change the network. This has confused people (including me) more than once.

---

## 2. The label pipeline (this is where the interesting choices live)

HuBERT learns by predicting cluster ids, so **the clusters are the supervision**. Changing them
changes what the model is asked to learn.

Features for clustering, per `utils/feature_utils.extract_feature_spectrogram`:

- `soundsig.sound.spectrogram`, a Gaussian STFT tuned for birdsong
- `spec_sample_rate=1000` (1 ms resolution), `freq_spacing=50` Hz → ~160 frequency bins
- windowed at **25 ms / 20 ms hop**, flattened → **~4000-D per frame**
- log10 power, computed in numpy (`torch.log10` segfaults here on the macOS env)

The 20 ms stride **is** HuBERT's native frame rate, so there is no decimation step — the legacy
MFCC branch `label = label[::2]` never fires. "We removed MFCC and use spectrogram at native
20 ms" is an accurate description of this pipeline.

Frame-count convention everywhere: `(n_samples - 400) // 320 + 1`. The extractor emits **one
extra trailing frame on some files** and always has; the invariant that matters is
`len(labels) >= (n_samples // 320000) * 999` for 20 s virtual chunks. Do not "fix" the surplus.

### preprocess.py stages

`tsv → resample → chunk → vad → features → kmeans → labels → eval`, resumable with
`--start-from <stage>` (monotonic — it runs that stage and everything after).

Two flags worth knowing:

- **`--km-model-dir <dir>`** reuses an existing k-means instead of fitting a new one. Use it
  whenever a new corpus must speak an existing model's vocabulary; refitting silently changes the
  baseline's labels and breaks every comparison.
- **`--start-from features`** is mandatory for already-16 kHz audio: the `resample` stage
  hardcodes `orig_freq=44100` and would pitch-shift 16 kHz input.

---

## 3. Training

`pytorchAudio/examples/hubert/train.py`, PyTorch Lightning, one process.

Flags that are non-obvious:

| flag | why |
|---|---|
| `--virtual-chunk-seconds 20` | slices long files into 20 s chunks via TSV offsets — no per-chunk wavs on disk |
| `HUBERT_NORMALIZE_INPUT=0` | env var, raw waveform in. AVES trained this way too (`normalize: False`) |
| `--feature-weight 0` | the conv L2 penalty. torchaudio's default of 10 drove the extractor output to ~0 (feature collapse) |
| `--init-weights` | **weights only.** Fresh optimizer, fresh schedule, step 0. For transfer |
| `--resume-checkpoint` | restores optimizer moments, LR position, global step. For preemption recovery. Mutually exclusive with the above |
| `--freeze-feature-extractor` | holds the conv front end fixed; requires `--init-weights` |
| `--checkpoint-every-n-steps` | Lightning writes these as `stepstep=15000.ckpt` (yes, doubled) |

`hubert_loss(logit_m, logit_u, feature_penalty, masked_weight=1.0, unmasked_weight=0.0,
feature_weight=10.0, reduction="sum")` — **`feature_weight` is the sixth positional arg.** Pass it
by keyword or you will silently set `masked_weight` and zero the cross-entropy. Chance for
`num_classes=k` is `ln(k+1)`.

---

## 4. Repository layout

```
/Users/jonathanwang/Desktop/vocalizations_lab
├── pytorchAudio/examples/hubert/     the training codebase
│   ├── train.py  preprocess.py  lightning_modules.py
│   ├── utils/        feature_utils, kmeans, common_utils, chunk_audio
│   ├── dataset/      hubert_dataset.py  (virtual chunking, bucketized sampler)
│   ├── slurm/        one script per experiment, each with a full rationale header
│   ├── scripts/      corpus builders (replay, merge, assemble)
│   └── tests/        run these before pushing; they are fast and CPU-only
├── zfeval/experiments/               evaluation, one script per question
├── knowledge/                        findings/*.yaml + CONTEXT.md + build.py
├── notebooks/                        build_*.py generate the .ipynb; never hand-edit cells
├── paper/                            results registry, number audit, figure pipeline
├── release/zf_hubert_run11/          shareable bundle
└── docs/                             roadmap.html, method notes
```

Local data lives **outside** the repo, in `~/zf_labelset/`:

```
audio/111021-000.wav                  the annotated colony recording
zf_detection_dataset_v1/features/     cached frame features (detvar/ is ~32 GB)
zf_detection_dataset_v1/analysis/     the JSON every notebook reads
external/aves|birdpark|dapt|chick/    checkpoints and holdout audio
```

**Never hand-type a number into a notebook or the paper.** Notebooks read
`analysis/*.json` at run time; `paper/` has a registry and an audit script for the same reason.

---

## 5. Savio (the cluster)

### Connecting

`~/.ssh/config` defines `savio-login` → `hpc.brc.berkeley.edu`, user `jonathanswang`, with
`ControlMaster auto`, `ControlPath ~/.ssh/cm-%r@%h:%p`, `ControlPersist 4h`.

```bash
ssh -MNf savio-login      # establishes the shared master; prompts once
```

- **The password prompt wants PIN and the 6-digit OTP concatenated, no space.**
- The master lasts 4 h. After that every call fails until it is re-established.
- **Do not run many SSH channels at once.** Layering a Monitor + background waiters + foreground
  calls over one master exhausts the server's `MaxSessions` and produces
  `Session open refused by peer` followed by `Permission denied` — which looks exactly like an
  auth failure and is not. Keep to one channel at a time.
- A failed auth can leave a dead socket that poisons later calls:
  `ssh -O exit savio-login; rm -f ~/.ssh/cm-*` then re-establish. `ssh -O check savio-login`
  tells you whether a master is alive.
- **`savio-dtn` needs its own separate auth.** Route transfers through the login node instead:
  `SAVIO_HOST=savio-login bash pytorchAudio/scp_to_savio.sh`.

### Filesystem

| path | note |
|---|---|
| `/global/home/users/jonathanswang` | code lives here; the push script targets `pytorchAudio/examples/hubert` |
| `/global/scratch/users/jonathanswang` | all data, checkpoints, corpora. Large and not backed up |
| `/tmp` on the **login node** | only 7.8 GB and shared with every other user — it has been 100 % full. Never write large fixtures there; clean up anything you do write |

### Environment

```bash
source /global/software/rocky-8.x86_64/manual/modules/langs/anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate hubert_env
```

Contents as last verified: torch 2.2.2, torchaudio 2.2.2, lightning 2.6.1. Re-check rather than
trust — version skew against the local machine has bitten before:

```bash
python -c "import torch,torchaudio,lightning
print(torch.__version__, torchaudio.__version__, lightning.__version__)"
```

Any script launched via `nohup` or `sbatch` **must activate conda itself** — a detached shell
inherits no conda state, and the failure (`python: command not found`) surfaces only after the
expensive part has already run.

Locally: `/Library/Frameworks/Python.framework/Versions/3.10/bin/python3` (torch 2.8.0, MPS).
Version skew to remember: torch ≥2.3 dropped `LRScheduler(verbose=...)`, which Savio's 2.2.2 still
accepts. The shipped code is correct for the cluster — shim locally, do not "fix" it.

### SLURM

Account `fc_birdpow`. Partitions available to it:

| partition | QOS | use |
|---|---|---|
| `savio3` | `savio_normal` | CPU work — feature extraction, corpus builds |
| `savio4_gpu` | `savio_lowprio` | L40 GPUs, `--gres=gpu:L40:N` |
| `savio3_gpu` | `savio_lowprio` | older GPUs |

**`--ntasks=1` means you get ONE GPU regardless of what `--gres` asks for**, because `srun`
launches a single task and Lightning initialises `world_size 1`. Asking for 4 and using 1 wasted
three L40s of scheduling priority for months. Derive realised resources from checkpoints and the
TSV, never from the request.

---

## 6. Pushing code

```bash
SAVIO_HOST=savio-login bash pytorchAudio/scp_to_savio.sh
```

One rsync, one auth. **The `FILES` array inside that script is explicit** — a new file is not
pushed until you add it there, and a job running against stale code is a silent, expensive
failure. Add the file in the same edit that creates it.

---

## 7. Standing rules that have each been paid for

- **A success message is not evidence the work ran.** SLURM has reported `COMPLETED 0:0` for a job
  that died at argparse; `nohup` has printed "launched" for a script that did not exist. Verify the
  artifact — the checkpoint, the file size, the process — not the launcher.
- **Never batch variable-length audio through these encoders.** Group norm runs over time, so
  padding corrupts short clips by >100×. `lengths=` does not save you. Encode one clip at a time.
- **Check a difference against the instrument's resolution before calling it a trend.** BirdPark is
  four independent 30 s blocks; its own bootstrap calls 0.049 AP "not distinguishable". Several
  confident claims in this project's history were scatter.
- **Pre-commit the arm.** Layer and normalisation are chosen by out-of-fold *in-distribution* AUC
  and frozen before any holdout is scored. Post-hoc maxima are never quoted.
- **Validate inputs at the boundary.** Zero-length wavs, NaN features and all-silence files have
  each reached the cluster and broken a later stage.
- **Diagnose before fixing on a remote.** Write the script that prints the actual failure mode
  first; each speculative round trip costs 5–15 minutes plus a queue wait.
- **Quote your heredoc delimiter** (`<<'EOF'`) when the body contains backticks or `$`. An
  unquoted one will run them as command substitution and silently blank the line.

---

## 8. Where to start reading

1. `knowledge/CONTEXT.md` — the findings digest, including the refutations. Read those; several
   refuted ideas look obviously correct and have each cost a day.
2. `docs/roadmap.html` — the experiment design space: what is done, running, planned.
3. `zfeval/experiments/detection_variants.py` — the canonical evaluation. Its docstring states the
   four ways this comparison can be faked and what it does about each.
4. `pytorchAudio/examples/hubert/slurm/` — every experiment's rationale is in its script header,
   including the ones that failed and why.

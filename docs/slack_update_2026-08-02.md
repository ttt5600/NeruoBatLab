# Slack update — draft to post

---

**Vocalization detector: our answer key was wrong, and I found the one real weakness**

I hand-labelled 300 windows from the detection eval set, blind to the model's scores (shuffled,
score hidden). Summary:

**1. The model was being punished for being right.** Of 150 windows the model scored highest among
those our ground truth called "not a call", **149 contained real calls.** One was ambiguous. Zero
were clean false positives.

**2. Why.** Our ground truth comes from localizing the curated `AdultVocalizations` clips inside
the recordings — anything not matching a curated clip was labelled "no call." But the curated set
was built by picking the *best-sounding* calls, so every call that wasn't picked became a
negative. In a uniform sample of the eval set, **77% of windows actually contain a call, against
the 33% our labels claimed.**

**3. Corrected numbers.** Precision goes from 0.785 to **0.977** (counting any human-verified call
as correct; 0.85–0.97 under stricter scopes). AUC barely moves: 0.921 → 0.878 when scored against
human labels instead of the curated ones. That asymmetry is the lesson — **ranking survives label
noise, fixed-threshold precision/recall/F1 do not.** We should not quote accuracy/P/R/F1 from the
old table. Sanity check on how bad the old framing was: "always say call" scored 0.767 accuracy,
beating the model's 0.513 at threshold 0.5.

**4. The one genuine weakness.** Calls buried in noise or overlapping another bird: **AUC 0.724**,
versus 0.921 on clean calls. ~22% of all calls in the sample are masked like this. This looks
taught rather than inherent — every training positive was a hand-picked clean call, and masked
calls were explicitly labelled negative.

**5. Next.** I built a synthesizer that mixes curated calls into human-verified call-free
background at controlled SNR, so masked calls come with exact onset/offset/SNR ground truth and
need no new labelling (same construction as DESED's synthetic partition in the DCASE benchmarks).
Validated locally: an energy-only baseline drops 0.97 → 0.51 across the sweep, so the difficulty
gradient is real. Running it on Savio next, then testing whether noise augmentation closes the
masked-call gap — and using the curve to back out what SNR the real masked calls sit at.

Worth knowing there's precedent: de Wolff et al. 2024 hit exactly this on manatees — 357 "false
positives" were re-checked by an expert and turned out to be genuine vocalizations
(arxiv.org/abs/2407.18083).

---

# The labelling tool (`pytorchAudio/examples/hubert/label_review.py`)

Not the same as **soundsep2**, which is the lab's tool for browsing recordings and cutting out
calls — that's what built `AdultVocalizations` in the first place. This one does a narrower job:
audit an existing set of machine labels without the auditor being able to cheat.

**What it is.** A single Python file that starts a local web server and serves one window at a
time: spectrogram on top, audio below, one keystroke to label. It writes to `review.csv` on every
keystroke, so quitting mid-session loses nothing.

**The design constraints, which are most of the point:**

- **Blind.** The model's score and which sampling stratum a window came from are never shown. The
  windows are shuffled with a seeded permutation, so the highest-scoring negatives are
  interleaved with random ones. If you can see that the model was confident, you can no longer
  answer "is there a call here" independently — and the entire result rests on that independence.
- **Scored span marked and audible.** Each clip is 1 s but only the middle 0.5 s counts (it's the
  detector's analysis window); context on either side is dimmed on the spectrogram. A playhead
  runs over the spectrogram in white and turns cyan while inside the scored span, and there are
  short clicks at each boundary during full-clip playback. Before that existed it was genuinely
  hard to tell whether a call you just heard was inside the window or in the context — which
  would have silently corrupted the labels.
- **Playback stops where it should.** The first version stopped the audio on the `timeupdate`
  event, which browsers fire only about every 250 ms — on a 500 ms span that overshot by up to
  half its length, so "play the scored span" was leaking context audio. It's driven by
  `requestAnimationFrame` (~16 ms) now.
- **Progress panel that doesn't break the blinding.** `s` shows how many windows you've done and
  your label distribution, but *not* which stratum they came from. Knowing you've called 14 of
  your last 16 windows "call" is useful self-knowledge; knowing they were all model-flagged is
  exactly the prime the blinding exists to prevent.
- **Peeking is recorded.** `m` reveals the model's score, and the row is stamped so any window
  you looked at can be excluded from analysis.

**Label set:** `voc` (clear call), `voc_noise` (call present but masked/overlapping), `noise`
(no call, audible background), `quiet` (no call, near-silent), `unsure`.

`voc_noise` is what surfaced the masked-call finding, and `noise`/`quiet` turned out to be the
most valuable output of the whole exercise: they are the only audio in this project *certified by
a human* to contain no call, which is what makes the SNR synthesis trustworthy.

**Usage:**

```bash
python3 pytorchAudio/examples/hubert/label_review.py --dir voc_detect_review_36020417
python3 pytorchAudio/examples/hubert/label_review.py --dir <d> --summary --by-set   # progress
```

One trap worth repeating, because it produced a confidently wrong answer: Python's `csv` module
writes CRLF line endings, so `awk '{print $NF}' | grep -c .` counts the stray `\r` as a value.
It once reported 300/300 labelled on a file with zero labels. `--summary` reads the file with
`csv.DictReader` for that reason.

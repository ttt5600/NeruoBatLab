#!/usr/bin/env python
"""Regenerate the 4900 UNLABELED negative windows of the detection benchmark, for hand-labeling.

Run on Savio (the recordings live there), then download the bundle and render it locally.

WHY THIS EXISTS
---------------
The published detection numbers used 7350 windows: 2450 positives (curated calls localized back
into their source recordings by cross-correlation -- these are ground truth, nobody needs to
relabel them) and 4900 negatives. A "negative" only ever meant "not one of the 2450 curated
calls". Nobody listened to them. Uncurated calls therefore sit in the negative class, which makes
precision a lower bound and moves AUC in an unknown direction. Labeling these 4900 is what turns
the constructed benchmark into a real one.

EXACT REPRODUCTION, NOT A FRESH DRAW
------------------------------------
A fresh random draw of negatives would be statistically equivalent but would NOT let us re-score
the published numbers: the labels have to attach to the windows that were actually scored. The
sampler in eval_voc_detection.py draws from a single np.random.default_rng(seed) stream consumed
across recordings in sorted order, and every rejected candidate consumes a draw too. So the
replay has to walk the same recordings in the same order and burn the same rejections.

Rather than paraphrase that loop, this script imports the real module's own helpers and mirrors
the loop line for line. It then PROVES the replay by checking the regenerated window positions
against the review bundle's recorded center_sample values, and refuses to export on a mismatch.
Getting this silently wrong would produce labels attached to the wrong audio, which is worse than
no labels at all.

USAGE
-----
  python export_negatives_for_labeling.py \\
      --rec-dir  $BASE/temp_files/run5-4-26-full/data/spectrogram/preprocessed_audio \\
      --clip-dir $BASE/adultvoc_16k \\
      --intervals-cache $BASE/voc_intervals_sr4000_p065.json \\
      --verify-against $BASE/voc_detect_review_run11_36020417/review.csv \\
      --out neg_bundle
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

HUBERT_DIR = "/global/home/users/jonathanswang/pytorchAudio/examples/hubert"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec-dir", required=True)
    ap.add_argument("--clip-dir", required=True)
    ap.add_argument("--intervals-cache", required=True)
    ap.add_argument("--verify-against", default=None,
                    help="review.csv from the scored run; the replay must reproduce its "
                         "negative center_sample values exactly or this script aborts")
    ap.add_argument("--out", default="neg_bundle")
    # these must match the scored run (slurm/eval_voc_detection.sh)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--win-sec", type=float, default=0.5)
    ap.add_argument("--neg-per-pos", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pad", type=float, default=0.5,
                    help="seconds of real surrounding audio each side, so the labeler can see "
                         "and hear context around the 0.5 s window being judged")
    ap.add_argument("--replay-only", action="store_true",
                    help="verify the replay and print stats, write no audio")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    sys.path.insert(0, HUBERT_DIR)
    try:
        from eval_voc_detection import date_of, load_mono          # noqa: F401
    except Exception as ex:                                        # noqa: BLE001
        sys.exit(f"cannot import eval_voc_detection from {HUBERT_DIR}: {ex}")
    import torchaudio

    sr, win = a.sample_rate, int(a.win_sec * a.sample_rate)

    # ---- same clip/date/recording enumeration as the scored run -------------------------
    clip_dir, rec_dir = Path(a.clip_dir), Path(a.rec_dir)
    all_clips = sorted(p.name for p in clip_dir.glob("*.wav") if not p.name.startswith("._"))
    clips_by_date = {}
    for c in all_clips:
        clips_by_date.setdefault(date_of(c), []).append(c)
    recordings = sorted(p.stem for p in rec_dir.glob("*.wav")
                        if p.stem.split("-")[0] in clips_by_date)
    print(f"{len(all_clips)} clips / {len(clips_by_date)} dates / {len(recordings)} recordings")

    import json
    blob = json.loads(Path(a.intervals_cache).read_text())
    cached = {k: [tuple(t) for t in v] for k, v in blob["intervals"].items()}
    print(f"[cache] intervals for {len(cached)} recordings  (key={blob.get('key')})")

    # ---- recording lengths ---------------------------------------------------------------
    # load_mono() returns len == num_frames when the file is already at the target rate, which
    # lets us skip decoding ~100 h of audio just to learn its length. Anything at another rate
    # is decoded properly rather than guessed at, because len(rec) sets the sampling range and
    # a wrong length would desynchronize the whole RNG stream.
    lengths, resampled = {}, []
    for rb in recordings:
        p = rec_dir / f"{rb}.wav"
        try:
            info = torchaudio.info(str(p))
        except Exception as ex:                                    # noqa: BLE001
            print(f"{rb}: SKIP (info failed: {ex})")
            lengths[rb] = None
            continue
        if info.sample_rate == sr:
            lengths[rb] = info.num_frames
        else:
            resampled.append(rb)
            lengths[rb] = None
    if resampled:
        print(f"{len(resampled)} recordings not at {sr} Hz; decoding those to get exact length")
        for rb in resampled:
            try:
                lengths[rb] = len(load_mono(rec_dir / f"{rb}.wav", sr))
            except Exception as ex:                                # noqa: BLE001
                print(f"{rb}: SKIP (load failed: {ex})")
                lengths[rb] = None

    # ---- replay the sampler --------------------------------------------------------------
    rng = np.random.default_rng(a.seed)
    negs, npos_total, short = [], 0, []
    for rb in recordings:
        n = lengths.get(rb)
        if n is None:                       # load failure -> `continue` before any rng use
            continue
        intervals = cached.get(rb)
        if not intervals:                   # `if ... or not intervals: continue` -- no rng used
            continue

        voc = np.zeros(n, dtype=bool)
        for s0, e0 in intervals:
            voc[max(0, s0):min(n, e0)] = True

        pos_centers = [(s0 + e0) // 2 for s0, e0 in intervals]
        npos_total += len(pos_centers)
        guard = win
        neg_starts = []
        need = int(len(pos_centers) * a.neg_per_pos)
        tries = 0
        while len(neg_starts) < need and tries < need * 50:
            tries += 1
            s = int(rng.integers(0, n - win))
            if not voc[max(0, s - guard):min(n, s + win + guard)].any():
                neg_starts.append(s)
        if len(neg_starts) < need:
            short.append((rb, len(neg_starts), need))
        for s in neg_starts:
            s = int(np.clip(s, 0, n - win))
            negs.append((rb, s, s + win // 2))

    print(f"\nreplayed {npos_total} positives / {len(negs)} negatives "
          f"({npos_total + len(negs)} windows total)")
    if short:
        print(f"  {len(short)} recordings yielded fewer negatives than requested "
              f"(too dense for the guard band): {short[:5]}")

    # ---- prove it ------------------------------------------------------------------------
    if a.verify_against:
        rows = list(csv.DictReader(open(a.verify_against)))
        want = [(r["recording"], int(r["center_sample"])) for r in rows
                if r["set"] in ("flagged_neg", "random_neg")]
        have = {(rb, c) for rb, _, c in negs}
        miss = [w for w in want if w not in have]
        print(f"\nVERIFY against {os.path.basename(a.verify_against)}: "
              f"{len(want) - len(miss)}/{len(want)} known negative windows reproduced")
        if miss:
            print(f"  MISMATCH on {len(miss)}, e.g. {miss[:5]}")
            sys.exit("ABORT: the replay does not reproduce the scored windows. Labels attached "
                     "to these would point at the wrong audio. Do not export.")
        print("  exact match -- these are the windows the published AUCs were computed on")
    else:
        print("\nWARNING: no --verify-against given, so the replay is UNPROVEN.")

    if a.replay_only:
        return

    # ---- export: one wav + offset table --------------------------------------------------
    import soundfile as sf
    if a.limit:
        negs = negs[:a.limit]
    order = np.random.default_rng(12345).permutation(len(negs))   # label order != recording order

    os.makedirs(a.out, exist_ok=True)
    pad = int(a.pad * sr)
    chunks, meta, cursor = [], [], 0
    for k, i in enumerate(order):
        rb, s, c = negs[i]
        n = lengths[rb]
        cs, ce = max(0, s - pad), min(n, s + win + pad)
        data, _ = sf.read(str(rec_dir / f"{rb}.wav"), start=cs, stop=ce,
                          dtype="float32", always_2d=True)
        seg = data.mean(axis=1)
        meta.append(dict(id=f"n{k:05d}", path="windows.wav",
                         start=round((cursor + (s - cs)) / sr, 4), dur=round(win / sr, 4),
                         ctx_start=round(cursor / sr, 4), ctx_dur=round(len(seg) / sr, 4),
                         kind="window", orig_path=f"{rb}.wav", orig_start=round(s / sr, 4),
                         constructed_label="0", recording=rb, center_sample=c, human_label=""))
        chunks.append(seg)
        cursor += len(seg)
        if (k + 1) % 500 == 0:
            print(f"  cut {k+1}/{len(negs)} ({cursor/sr/60:.1f} min)")

    sf.write(os.path.join(a.out, "windows.wav"), np.concatenate(chunks), sr, subtype="PCM_16")
    mb = os.path.getsize(os.path.join(a.out, "windows.wav")) / 1e6
    print(f"\nwrote windows.wav: {len(chunks)} windows, {cursor/sr/60:.1f} min, {mb:.0f} MB")

    cols = ["id", "path", "start", "dur", "ctx_start", "ctx_dur", "kind", "orig_path",
            "orig_start", "constructed_label", "recording", "center_sample", "human_label"]
    with open(os.path.join(a.out, "INDEX.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(meta)

    FIG, APS, BUDGET = 50_000, 43_000, 18_000_000
    batches, cur, cb = [], [], 0
    for m in meta:
        cost = FIG + m["ctx_dur"] * APS
        if cur and cb + cost > BUDGET:
            batches.append(cur); cur, cb = [], 0
        cur.append(m); cb += cost
    if cur:
        batches.append(cur)
    for b, rs in enumerate(batches):
        with open(os.path.join(a.out, f"batch_{b:03d}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rs)
    print(f"{len(batches)} batches of ~{len(batches[0])} windows each")
    print(f"\n  tar cf - {a.out}/ | ...   # ~{mb:.0f} MB")


if __name__ == "__main__":
    main()

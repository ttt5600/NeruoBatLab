#!/usr/bin/env python
"""Export the 7350 benchmark windows as ONE audio file + an offset table, ready to hand-label.

Run this on Savio (where the recordings live), scp the two output files down, then render the
labeling pages locally with local_labeler.py. No need to move the ~100 h corpus.

WHY ONE FILE INSTEAD OF 7350 WAVS
---------------------------------
7350 one-second files is 7350 inodes on a shared filesystem for a throwaway export. Instead
every window is concatenated into a single wav and the CSV records byte offsets into it -- the
same virtualized-chunking pattern used elsewhere in this project. One file, identical contents,
and `local_labeler.py` reads windows out of it with `start`/`dur` exactly as it would from a
recording.

Size: 7350 s at 16 kHz mono 16-bit is about 235 MB. That is the whole download.

WHAT IS DELIBERATELY NOT SHOWN
------------------------------
The manifest's constructed label (and the model's score, if present) is carried into the CSV but
never rendered by the labeler. If you can see what the pipeline already decided, your labels stop
being independent evidence and cannot be used to evaluate it. The whole point of this exercise is
to get a judgement that does not depend on the thing being judged.

USAGE
-----
  # on Savio
  python export_windows_for_labeling.py --manifest windows.csv --out label_bundle
  tar czf label_bundle.tar.gz label_bundle/
  # then locally
  python local_labeler.py --batch-dir label_bundle/
"""
import argparse
import csv
import os
import sys

import numpy as np

try:
    import soundfile as sf
except ImportError:
    sys.exit("needs soundfile:  pip install soundfile")

SR = 16000
FIG_BYTES = 50_000
AUDIO_BYTES_PER_SEC = 43_000
BATCH_BYTES = 18_000_000


def read_window(path, start, end):
    info = sf.info(path)
    a = max(0, int(start * info.samplerate))
    b = min(info.frames, int(end * info.samplerate))
    if b <= a:
        return None
    x, sr = sf.read(path, start=a, stop=b, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR and len(x):
        n = int(round(len(x) * SR / sr))
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype("float32")
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="CSV with columns path,start,end,label,recording -- the same window "
                         "list the published AUCs were computed on")
    ap.add_argument("--out", default="label_bundle")
    ap.add_argument("--limit", type=int, default=0, help="export only the first N windows")
    ap.add_argument("--shuffle", action="store_true", default=True,
                    help="randomize labeling order (default on; fatigue becomes noise rather "
                         "than landing systematically on one recording)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--context-pad", type=float, default=0.0,
                    help="seconds of REAL surrounding audio to export around each window so the "
                         "context view is genuine. 0 (default) exports windows only and the "
                         "labeler shows no context, because in a concatenated bundle the "
                         "neighbours are unrelated windows. 1.0 triples the download.")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.manifest)))
    if not rows:
        sys.exit(f"{a.manifest} is empty")
    need = {"path", "start", "end"}
    if not need.issubset(rows[0]):
        sys.exit(f"manifest needs columns {need}, found {list(rows[0])}")
    if a.limit:
        rows = rows[:a.limit]
    print(f"{len(rows)} windows from {a.manifest}")

    order = list(range(len(rows)))
    if a.shuffle:
        np.random.default_rng(a.seed).shuffle(order)

    os.makedirs(a.out, exist_ok=True)
    wav_path = os.path.join(a.out, "windows.wav")

    P = max(0.0, a.context_pad)
    chunks, meta, cursor, bad = [], [], 0.0, 0
    for n, i in enumerate(order):
        r = rows[i]
        s, e = float(r["start"]), float(r["end"])
        try:
            if P > 0:
                # Export real surrounding audio so the context view is genuine. Note the left
                # pad shrinks at the start of a recording, so the window's offset inside the
                # exported segment is measured, not assumed.
                cs = max(0.0, s - P)
                seg = read_window(r["path"], cs, e + P)
                x = read_window(r["path"], s, e)
                lead = s - cs
            else:
                seg = x = read_window(r["path"], s, e)
                lead = 0.0
        except Exception as ex:                           # noqa: BLE001
            print(f"  SKIP {r['path']} @{r['start']}: {ex}")
            seg = x = None
        if seg is None or x is None or len(x) < int(0.05 * SR):
            bad += 1
            continue
        dur = len(x) / SR
        m = dict(id=f"w{n:06d}", path="windows.wav",
                 start=round(cursor + lead, 4), dur=round(dur, 4), kind="window",
                 orig_path=r["path"], orig_start=r["start"],
                 constructed_label=r.get("label", ""),
                 recording=r.get("recording", ""), human_label="")
        if P > 0:
            m["ctx_start"] = round(cursor, 4)
            m["ctx_dur"] = round(len(seg) / SR, 4)
        meta.append(m)
        chunks.append(seg)
        cursor += len(seg) / SR
        if (n + 1) % 500 == 0:
            print(f"  {n+1}/{len(rows)}  ({cursor/60:.1f} min so far)")

    if not chunks:
        sys.exit("nothing exported -- check that the manifest paths resolve on this machine")
    sf.write(wav_path, np.concatenate(chunks), SR, subtype="PCM_16")
    mb = os.path.getsize(wav_path) / 1e6
    print(f"\nwrote {wav_path}: {len(chunks)} windows, {cursor/60:.1f} min, {mb:.0f} MB"
          + (f"  ({bad} skipped)" if bad else ""))

    # batch the offset table the same way the local builder does, so pages stay openable
    cols = ["id", "path", "start", "dur", "kind", "orig_path", "orig_start",
            "constructed_label", "recording", "human_label"]
    if P > 0:
        cols += ["ctx_start", "ctx_dur"]
    with open(os.path.join(a.out, "INDEX.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader(); w.writerows(meta)

    batches, cur, cur_bytes = [], [], 0
    for m in meta:
        # with --context-pad the page embeds the padded segment, not just the window
        cost = FIG_BYTES + m.get("ctx_dur", m["dur"]) * AUDIO_BYTES_PER_SEC
        if cur and cur_bytes + cost > BATCH_BYTES:
            batches.append(cur); cur, cur_bytes = [], 0
        cur.append(m); cur_bytes += cost
    if cur:
        batches.append(cur)

    for b, rs in enumerate(batches):
        p = os.path.join(a.out, f"batch_{b:03d}.csv")
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader(); w.writerows(rs)
    print(f"{len(batches)} batches of ~{len(batches[0])} windows each")

    pos = sum(1 for m in meta if str(m["constructed_label"]) == "1")
    print(f"constructed labels (HIDDEN while labeling): {pos} call / {len(meta)-pos} background")
    print(f"""
Next:
    tar czf {a.out}.tar.gz {a.out}/          # ~{mb:.0f} MB
    # download, then locally:
    python local_labeler.py --batch-dir {a.out}/
    open {a.out}/batch_000.html

Paths in the CSVs are relative ("windows.wav"), so run the labeler from inside the folder or
keep windows.wav beside the batch files.""")


if __name__ == "__main__":
    main()

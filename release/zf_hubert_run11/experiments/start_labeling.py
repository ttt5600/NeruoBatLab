#!/usr/bin/env python
"""One command: find the audio, build the labeling set, render it, open it.

Run this and start labeling. It discovers the corpus itself rather than needing paths, because
the directory layout varies between machines and nobody remembers where the recordings went.

  python start_labeling.py                       # searches the usual places
  python start_labeling.py --root ~/some/dir     # or point it somewhere specific
  python start_labeling.py --n-windows 400       # smaller first pass

HOW IT DECIDES WHAT IS WHAT
---------------------------
It groups wav files by directory and looks at the median duration:

  median < 30 s   -> curated CLIPS      -> the POSITIVE class (already ground truth)
  median >= 30 s  -> continuous RECORDINGS -> random 1 s windows for you to label

That heuristic is stated out loud in the output so you can override it if it guesses wrong.
Nothing is written outside --out.
"""
import argparse
import glob
import os
import subprocess
import sys

try:
    import soundfile as sf
except ImportError:
    sys.exit("needs soundfile:  pip install soundfile")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOTS = [
    "~/Desktop/vocalizations_lab",
    "~/Documents", "~/Downloads", "~/data", "~/audio",
]
LONG_S = 30.0          # a file this long is a continuous recording, not a curated clip
MIN_FILES = 8          # ignore directories with only a handful of wavs


def survey(root, cap=40000):
    """Directory -> (count, median duration). Header reads only; no audio is decoded."""
    import statistics
    by_dir = {}
    n = 0
    for p in glob.iglob(os.path.join(root, "**", "*.wav"), recursive=True):
        if os.path.basename(p).startswith("._"):
            continue
        n += 1
        if n > cap:
            print(f"  (stopped after {cap} files)")
            break
        try:
            d = sf.info(p).duration
        except Exception:                                 # noqa: BLE001
            continue
        by_dir.setdefault(os.path.dirname(p), []).append(d)
    return {k: (len(v), statistics.median(v)) for k, v in by_dir.items() if len(v) >= MIN_FILES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", default=None)
    ap.add_argument("--n-windows", type=int, default=1200)
    ap.add_argument("--audit", type=int, default=25)
    ap.add_argument("--out", default="~/labelset")
    ap.add_argument("--clips", help="skip discovery, use this clips directory")
    ap.add_argument("--recordings", help="skip discovery, use this recordings directory")
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()

    out = os.path.expanduser(a.out)
    clips, recs = a.clips, a.recordings

    if not (clips and recs):
        roots = [os.path.expanduser(r) for r in (a.root or DEFAULT_ROOTS)]
        found = {}
        for r in roots:
            if not os.path.isdir(r):
                continue
            print(f"searching {r} ...")
            try:
                found.update(survey(r))
            except PermissionError:
                print(f"  PERMISSION DENIED on {r}")
                print("  -> System Settings > Privacy & Security > Full Disk Access,")
                print("     add your terminal, restart it, and run this again.")
        if not found:
            sys.exit("\nNo .wav directories found. Pass --root, or --clips/--recordings "
                     "explicitly.")

        print(f"\n{'directory':60s} {'files':>7s} {'median':>9s}  guess")
        cand_clips, cand_recs = [], []
        for d, (n, med) in sorted(found.items(), key=lambda kv: -kv[1][0]):
            kind = "RECORDINGS" if med >= LONG_S else "clips"
            (cand_recs if med >= LONG_S else cand_clips).append((n, d))
            print(f"{d[-60:]:60s} {n:7d} {med:8.2f}s  {kind}")

        if not clips and cand_clips:
            clips = max(cand_clips)[1]
        if not recs and cand_recs:
            recs = max(cand_recs)[1]

    print(f"\nclips      : {clips or 'NONE FOUND'}")
    print(f"recordings : {recs or 'NONE FOUND'}")
    if not clips and not recs:
        sys.exit("nothing usable found")
    if not recs:
        print("\nWARNING: no continuous recordings found, so there are NO windows to label -- "
              "only\nthe already-known positives. The windows are the whole point; find the long "
              "files\nor pass --recordings explicitly.")

    cmd = [sys.executable, os.path.join(HERE, "build_local_labelset.py"),
           "--out", out, "--audit", str(a.audit)]
    if clips:
        cmd += ["--clips", clips]
    if recs:
        cmd += ["--recordings", recs, "--n-windows", str(a.n_windows)]
    print("\n$ " + " ".join(cmd))
    if subprocess.call(cmd) != 0:
        sys.exit("build failed")

    cmd = [sys.executable, os.path.join(HERE, "local_labeler.py"), "--batch-dir", out]
    print("\n$ " + " ".join(cmd))
    if subprocess.call(cmd) != 0:
        sys.exit("render failed")

    first = sorted(glob.glob(os.path.join(out, "batch_*.html")))
    if not first:
        sys.exit("no pages rendered")
    print(f"\n{len(first)} page(s) in {out}")
    if not a.no_open:
        subprocess.call(["open", first[0]])
        print(f"opened {first[0]}")
    print(f"\nWhen done with a page, click 'download CSV', move the file into {out}, then:\n"
          f"  python {os.path.join(HERE, 'build_local_labelset.py')} --merge {out}")


if __name__ == "__main__":
    main()

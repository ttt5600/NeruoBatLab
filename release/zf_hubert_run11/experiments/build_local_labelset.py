#!/usr/bin/env python
"""Build a hand-labeled detection test set locally, with no Savio and no cross-correlation.

WHY THIS EXISTS
---------------
The published detection benchmark had to *construct* its ground truth, by localizing curated
clips back into the continuous recordings. That was only necessary because no human labels
existed. It left two problems that no amount of cleverness fixes:

  * negatives mean "not a CURATED call", so uncurated calls sit in the negative class and the
    detector is punished for correctly firing on them;
  * positives are curated calls only -- the clean, isolated, good-SNR ones -- so recall is
    measured on the easy subset.

If a human labels windows directly, both problems disappear. The labels are the ground truth.
No localization, no contamination, no lower bounds. That is what this script prepares.

WHAT IT PRODUCES
----------------
CSV batches, which `local_labeler.py` turns into standalone browser pages. Two kinds of item:

  * every curated clip you point it at (`--clips`), included in full;
  * windows cut from continuous recordings (`--recordings`) -- either contiguous back-to-back
    coverage of whole recordings (`--tile`, the default choice for building a test set) or
    scattered random samples. The windows are where the genuinely new information is: they are
    what fixes the negative class.

Batches are sized by estimated PAGE WEIGHT, not item count, because clip durations span
0.03-11.7 s here and a fixed count can silently produce a page the browser cannot open.

A NOTE ON SPLITTING, WHICH IS EASY TO GET WRONG
-----------------------------------------------
Tiled windows are contiguous, so window t and t+1 overlap in content and are near-duplicates.
A random train/test split puts one in each and the evaluation becomes meaningless. Always split
by RECORDING. `--tile` covers whole recordings precisely so that this stays possible.

USAGE
-----
  # contiguous coverage of 6 recordings, plus every curated clip
  python build_local_labelset.py --clips DIR --recordings DIR --tile --tile-recordings 6 \
      --max-windows 3000 --out labelset/
  python local_labeler.py --batch-dir labelset/     # -> batch_XXX.html, label in a browser
  python build_local_labelset.py --merge labelset/  # collect the downloaded CSVs
"""
import argparse
import csv
import glob
import os
import random
import sys

try:
    import soundfile as sf
except ImportError:
    sys.exit("needs soundfile:  pip install soundfile")

WIN = 1.0            # seconds, matching the published benchmark's window
# Measured page cost per item: a fixed JPEG spectrogram figure plus base64 audio that scales
# with duration. Batching on bytes rather than item count matters because clip durations span
# 0.03-11.7 s here, so a fixed count can silently produce a page the browser cannot open.
FIG_BYTES = 50_000
AUDIO_BYTES_PER_SEC = 43_000
BATCH_BYTES = 18_000_000   # ~18 MB per page; stays responsive in a browser


def scan_clips(d):
    """Curated clips -> POSITIVE cases, standardized to a 1 s window centred on the loudest part.

    Why crop instead of using the clip as-is: these clips run 0.03-11.7 s while the negatives are
    1 s windows cut from continuous audio. Feed both to a classifier unchanged and it can
    separate them on duration and clip-boundary artefacts without hearing a vocalization at all.
    Forcing every item to the same 1 s format removes that shortcut.

    Residual confound worth knowing about: these clips were *extracted* by a curator, so they sit
    centred on a clean call, whereas a random window catches whatever is there. Perfectly matched
    positives would require localizing each clip back into its source recording -- which is the
    machinery this whole approach exists to avoid. Flagged rather than solved.
    """
    import numpy as np
    out = []
    for p in sorted(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True)):
        if os.path.basename(p).startswith("._"):      # AppleDouble sidecars are not audio
            continue
        try:
            info = sf.info(p)
            x, sr = sf.read(p, dtype="float32", always_2d=True)
        except Exception as e:                        # noqa: BLE001
            print(f"  unreadable, skipped: {p} ({e})")
            continue
        x = x.mean(axis=1)
        if len(x) == 0:
            continue
        bird = date = ctype = ""
        name = os.path.basename(p)[:-4]
        if "_" in name and "-" in name:
            bird, rest = name.split("_", 1)
            parts = rest.split("-")
            date = parts[0]
            if len(parts) > 1:
                ctype = parts[1]

        if info.duration <= WIN:
            start, dur = 0.0, round(info.duration, 3)   # shorter than a window; labeler pads
        else:
            # centre on the loudest 25 ms frame rather than the midpoint: long clips (Song runs
            # to 11.7 s) would otherwise be cropped to whatever happens to sit in the middle.
            hop = max(1, int(0.010 * sr))
            fr = max(1, int(0.025 * sr))
            n = (len(x) - fr) // hop + 1
            if n > 1:
                rms = np.sqrt(np.array([(x[i*hop:i*hop+fr] ** 2).mean() for i in range(n)]))
                c = (int(rms.argmax()) * hop + fr // 2) / sr
            else:
                c = info.duration / 2
            start = round(min(max(0.0, c - WIN / 2), info.duration - WIN), 3)
            dur = WIN
        out.append(dict(path=p, start=start, dur=dur, kind="clip",
                        bird=bird, date=date, prior=ctype, human_label="call"))
    return out


def sample_windows(d, n, seed=0, min_rms=None):
    rng = random.Random(seed)
    recs = []
    for p in sorted(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True)):
        if os.path.basename(p).startswith("._"):
            continue
        try:
            info = sf.info(p)
        except Exception:                             # noqa: BLE001
            continue
        if info.duration > WIN * 2:
            recs.append((p, info.duration))
    if not recs:
        print(f"  no usable recordings under {d}")
        return []

    total = sum(r[1] for r in recs)
    print(f"  {len(recs)} recordings, {total/3600:.1f} h total")
    out, tries = [], 0
    seen = set()
    while len(out) < n and tries < n * 40:
        tries += 1
        # pick a recording in proportion to its duration, so sampling is uniform over TIME
        x = rng.uniform(0, total)
        acc = 0.0
        for p, dur in recs:
            acc += dur
            if acc >= x:
                break
        start = round(rng.uniform(0, dur - WIN), 3)
        key = (p, int(start * 4))                     # dedupe at 250 ms resolution
        if key in seen:
            continue
        seen.add(key)
        if min_rms is not None:
            try:
                a = int(start * sf.info(p).samplerate)
                x_, _ = sf.read(p, start=a, stop=a + int(WIN * sf.info(p).samplerate),
                                dtype="float32", always_2d=True)
                if float((x_.mean(axis=1) ** 2).mean() ** 0.5) < min_rms:
                    continue
            except Exception:                         # noqa: BLE001
                continue
        out.append(dict(path=p, start=start, dur=WIN, kind="window",
                        bird="", date="", prior=""))
    if len(out) < n:
        print(f"  WARNING: only produced {len(out)} of {n} requested windows")
    return out


def tile_recordings(d, n_rec, max_windows, seed=0):
    """Contiguous back-to-back 1 s windows covering whole recordings.

    Whole recordings, not scattered windows, because adjacent windows are near-duplicates: a
    random train/test split over tiled windows puts window t in train and t+1 in test and the
    evaluation becomes meaningless. Covering entire recordings means you can always split by
    RECORDING later, which is the only safe unit here.
    """
    rng = random.Random(seed)
    recs = []
    for p in sorted(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True)):
        if os.path.basename(p).startswith("._"):
            continue
        try:
            info = sf.info(p)
        except Exception:                             # noqa: BLE001
            continue
        if info.duration > WIN * 10:
            recs.append((p, info.duration))
    if not recs:
        print(f"  no usable recordings under {d}")
        return []
    rng.shuffle(recs)
    recs = recs[:n_rec]

    out = []
    for p, dur in recs:
        n_here = 0
        t = 0.0
        while t + WIN <= dur and len(out) < max_windows:
            out.append(dict(path=p, start=round(t, 3), dur=WIN, kind="window",
                            bird="", date=os.path.basename(p)[:-4], prior=""))
            t += WIN
            n_here += 1
        print(f"  {os.path.basename(p):32s} {dur/60:6.1f} min -> {n_here} windows")
        if len(out) >= max_windows:
            print(f"  stopped at --max-windows={max_windows}")
            break
    return out


def write_batches(items, outdir, seed=0, shuffle=True, audit=0):
    """Split into (a) known positives needing no work and (b) windows to hand-label.

    The curated clips already carry ground truth -- relabeling 2450 of them would burn hours to
    re-derive what the corpus already states. The unknown windows are where all the new
    information is, so they are the only thing that goes into the labeling queue.
    """
    os.makedirs(outdir, exist_ok=True)
    rng = random.Random(seed)
    cols = ["id", "path", "start", "dur", "kind", "bird", "date", "prior", "human_label"]

    for i, it in enumerate(items):
        it["id"] = f"item{i:06d}"
    pos = [it for it in items if it.get("human_label")]
    todo = [it for it in items if not it.get("human_label")]

    # Salt a few known positives into the queue as blind controls. They look identical to any
    # other item while labeling, so agreement on them measures YOUR consistency -- if a known
    # call gets called noise, that is a drift signal you want before trusting the rest.
    audits = []
    if audit and pos:
        picked = rng.sample(pos, min(audit, len(pos)))
        for it in picked:
            a = dict(it); a["human_label"] = ""; a["kind"] = "audit"
            a["audit_truth"] = it["human_label"]
            a["id"] = it["id"] + "_audit"     # distinct id, else it overwrites its own original
            audits.append(a)                  # in the index and the positive is lost
        todo = todo + audits

    if shuffle:
        rng.shuffle(todo)
    else:
        todo.sort(key=lambda it: (it["path"], float(it["start"] or 0)))

    batches, cur, cur_bytes = [], [], 0
    for it in todo:
        cost = FIG_BYTES + float(it["dur"]) * AUDIO_BYTES_PER_SEC
        if cur and cur_bytes + cost > BATCH_BYTES:
            batches.append(cur); cur, cur_bytes = [], 0
        cur.append(it); cur_bytes += cost
    if cur:
        batches.append(cur)

    with open(os.path.join(outdir, "positives.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in pos:
            w.writerow({c: r.get(c, "") for c in cols})

    idx_path = os.path.join(outdir, "INDEX.csv")
    with open(idx_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols + ["batch", "audit_truth"])
        w.writeheader()
        for r in pos:
            w.writerow({**{c: r.get(c, "") for c in cols}, "batch": "", "audit_truth": ""})
        for b, rows in enumerate(batches):
            for r in rows:
                w.writerow({**{c: r.get(c, "") for c in cols}, "batch": b,
                            "audit_truth": r.get("audit_truth", "")})

    for b, rows in enumerate(batches):
        p = os.path.join(outdir, f"batch_{b:03d}.csv")
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})
        mb = sum(FIG_BYTES + float(r["dur"]) * AUDIO_BYTES_PER_SEC for r in rows) / 1e6
        print(f"  batch_{b:03d}.csv  {len(rows):4d} items  ~{mb:4.1f} MB")

    ctypes = sorted({r["prior"] for r in pos if r["prior"]})
    print(f"\nPOSITIVES (no labeling needed): {len(pos)} curated calls -> positives.csv")
    if ctypes:
        from collections import Counter
        cc = Counter(r["prior"] for r in pos if r["prior"])
        print(f"  call types: {dict(cc.most_common())}")
    print(f"TO LABEL: {len(todo)} random windows"
          + (f" (including {len(audits)} blind controls)" if audits else "")
          + f" in {len(batches)} batches")
    print(f"index: {idx_path}")
    order = "TIME ORDER (back to back)" if not shuffle else "SHUFFLED"
    print(f"""
Items are in {order}.

Next -- render the browser labeling pages:

    python local_labeler.py --batch-dir {outdir}

Open each batch_XXX.html, label with keys 1-4 (call / call+noise / noise / silence), click
"download CSV", put the downloaded files back in {outdir}, then:

    python {os.path.basename(__file__)} --merge {outdir}

NOTE: neither the `kind` nor the `prior` column is shown while labeling. If you could see which
items were already-curated calls you would label them differently, and the set would stop being
independent evidence about the model.""")


POS_LABELS = {"call", "call+noise", "voc"}       # "voc" accepted from the older vocabulary


def merge(outdir):
    idx = {r["id"]: r for r in csv.DictReader(open(os.path.join(outdir, "INDEX.csv")))}
    skip = {"INDEX.csv", "labels_merged.csv", "detection_dataset.csv"}
    got = {}
    for p in sorted(glob.glob(os.path.join(outdir, "*.csv"))):
        if os.path.basename(p) in skip:
            continue
        for r in csv.DictReader(open(p)):
            lab = (r.get("human_label") or "").strip().lower()
            if lab and r["id"] in idx:
                got[r["id"]] = lab
    if not got:
        sys.exit(f"no filled human_label found in {outdir}/*.csv -- download the labeled CSVs "
                 f"out of the browser first and put them in {outdir}")

    # blind controls: known calls the labeler saw without knowing what they were
    aud = [(i, got[i], idx[i]["audit_truth"]) for i in got
           if idx[i].get("audit_truth")]
    if aud:
        ok = sum(1 for _, g, t in aud if (g in POS_LABELS) == (t in POS_LABELS))
        print(f"BLIND CONTROLS: {ok}/{len(aud)} known calls labeled as calls "
              f"({ok/len(aud):.2f})")
        if ok < len(aud):
            print("  missed:", [i for i, g, t in aud
                                if (g in POS_LABELS) != (t in POS_LABELS)][:8])
            print("  A known call labeled 'noise' is a drift signal -- if this rate is high,")
            print("  the window crop may be cutting calls, or fatigue is setting in.")

    out = os.path.join(outdir, "detection_dataset.csv")
    n_written = 0
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "path", "start", "dur", "label", "human_label", "call_type",
                    "bird", "recording", "source"])
        for i, r in sorted(idx.items()):
            lab = got.get(i) or (r["human_label"] or "").strip().lower()
            if not lab or r["kind"] == "audit":       # audits are duplicates of positives
                continue
            # Group key for splitting. For a curated clip that is its SOURCE RECORDING (the
            # datecode in the filename), not the clip file -- otherwise every clip is its own
            # "recording" and clips from one session scatter across train and test.
            rec = (r["date"] if r["kind"] == "clip" and r["date"]
                   else os.path.basename(r["path"])[:-4])
            w.writerow([i, r["path"], r["start"], r["dur"],
                        1 if lab in POS_LABELS else 0, lab, r["prior"], r["bird"],
                        rec, r["kind"]])
            n_written += 1
    print(f"\nwrote {out}: {n_written} rows (label 1=call, 0=not)")
    print("  Split by the `recording` column, never randomly -- see the module docstring.")

    from collections import Counter
    unknown = set(got.values()) - POS_LABELS - {"noise", "silence", "quiet", "unsure"}
    if unknown:
        print(f"WARNING: unrecognized labels {sorted(unknown)} -- not counted as calls")

    rows = list(csv.DictReader(open(out)))
    npos = sum(1 for r in rows if r["label"] == "1")
    print(f"\nDATASET: {len(rows)} items, {npos} call / {len(rows)-npos} not "
          f"(prevalence {npos/max(1,len(rows)):.3f})")
    print("  by source:", dict(Counter(r["source"] for r in rows)))
    print("  by label :", dict(Counter(r["human_label"] for r in rows)))

    wins = [r for r in rows if r["source"] == "window"]
    if wins:
        v = sum(1 for r in wins if r["label"] == "1")
        rate = v / len(wins)
        se = (rate * (1 - rate) / len(wins)) ** 0.5
        print(f"\nCALL PREVALENCE in random windows: {v}/{len(wins)} = {rate:.3f} "
              f"(95% CI +/-{1.96*se:.3f})")
        print("  This is the real base rate in continuous audio. The published benchmark used")
        print("  0.333 by construction, so its precision does not transfer to deployment.")
        clean = sum(1 for r in wins if r["human_label"] == "call")
        noisy = sum(1 for r in wins if r["human_label"] == "call+noise")
        if clean + noisy:
            print(f"  of those: {clean} clean / {noisy} over background "
                  f"({noisy/(clean+noisy):.2f} noisy)")
        sil = sum(1 for r in wins if r["human_label"] in ("silence", "quiet"))
        neg = len(wins) - v
        print(f"\nNEGATIVES that are silence rather than sound: {sil}/{neg} "
              f"({sil/max(1,neg):.2f})")
        print("  Watch this. A negative class dominated by silence is what made the retracted")
        print("  0.984 trivially easy -- report detection against `noise` specifically too.")

    recs = Counter(r["recording"] for r in rows)
    print(f"\n{len(recs)} distinct recordings. Split on this column, never randomly.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", help="directory of curated call clips (included in full)")
    ap.add_argument("--recordings", help="directory of continuous recordings to sample from")
    ap.add_argument("--n-windows", type=int, default=1000)
    ap.add_argument("--min-rms", type=float, default=None,
                    help="drop windows quieter than this. CHANGES PREVALENCE -- see the docstring; "
                         "silence-heavy negatives are what got the old 0.984 retracted, but "
                         "all-silence windows also waste labeling time. Leave unset unless you "
                         "know why you want it.")
    ap.add_argument("--out", default="labelset")
    ap.add_argument("--merge")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tile", action="store_true",
                    help="contiguous back-to-back windows over whole recordings, in time order, "
                         "instead of scattered random samples")
    ap.add_argument("--tile-recordings", type=int, default=6)
    ap.add_argument("--max-windows", type=int, default=3000)
    ap.add_argument("--audit", type=int, default=25,
                    help="known positives salted into the labeling queue as blind consistency "
                         "controls (default 25; 0 disables)")
    a = ap.parse_args()

    if a.merge:
        merge(a.merge)
        return
    if not (a.clips or a.recordings):
        ap.error("give --clips and/or --recordings (or --merge)")

    items = []
    if a.clips:
        print(f"scanning clips: {a.clips}")
        items += scan_clips(a.clips)
        print(f"  found {len(items)} clips")
    if a.recordings:
        if a.tile:
            print(f"tiling {a.tile_recordings} recordings from {a.recordings}")
            items += tile_recordings(a.recordings, a.tile_recordings, a.max_windows, a.seed)
        else:
            print(f"sampling {a.n_windows} windows: {a.recordings}")
            items += sample_windows(a.recordings, a.n_windows, a.seed, a.min_rms)
    if not items:
        sys.exit("found nothing -- check the paths")
    write_batches(items, a.out, a.seed, shuffle=not a.tile, audit=a.audit)


if __name__ == "__main__":
    main()

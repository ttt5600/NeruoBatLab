"""Is a pretraining holdout structurally possible?

You can only hold a bird out of pretraining by dropping the recordings it appears in. If the
120 files are colony recordings with several birds audible at once, dropping recordings cannot
isolate one bird, and no amount of retraining fixes the leakage.

Cheap decisive test: per recording DATE, compare the number of recording files against the
number of distinct birds with curated clips from that date. If birds > files for a date, then
at least one file on that date carries more than one bird -- by pigeonhole, no localization
run required.
"""
import collections
import os

import numpy as np

REL = "/Users/jonathanwang/Desktop/vocalizations_lab/release/zf_hubert_run11"
recs = [l.strip().split("/")[-1].replace(".wav", "")
        for l in open(os.environ["CLAUDE_JOB_DIR"] + "/tmp/pretrain_files.txt") if l.strip()]

d = np.load(f"{REL}/data/run11_layersweep.npz", allow_pickle=True)
names, birds_raw = d["names"], d["birds"]
birds = np.array([b.lower() for b in birds_raw])
named = ~np.char.startswith(birds, "unknown")
names, birds = names[named], birds[named]

files_by_date = collections.defaultdict(set)
for r in recs:
    files_by_date[r.split("-")[0]].add(r)
birds_by_date = collections.defaultdict(set)
clips_by_date = collections.Counter()
for n, b in zip(names, birds):
    dt = n.split("_", 1)[1].split("-")[0]
    birds_by_date[dt].add(b)
    clips_by_date[dt] += 1

print(f"{len(recs)} pretraining recordings across {len(files_by_date)} dates")
print(f"{len(names)} named clips across {len(birds_by_date)} dates\n")

print(f"{'date':>9} {'files':>6} {'birds':>6} {'clips':>6}  verdict")
forced, checked, clips_forced = 0, 0, 0
for dt in sorted(birds_by_date):
    nf, nb = len(files_by_date.get(dt, ())), len(birds_by_date[dt])
    if nf == 0:
        continue
    checked += 1
    bad = nb > nf
    forced += bad
    clips_forced += clips_by_date[dt] if bad else 0
    if bad or nb >= 4:
        print(f"{dt:>9} {nf:6d} {nb:6d} {clips_by_date[dt]:6d}  "
              f"{'>=1 file holds MULTIPLE birds (pigeonhole)' if bad else 'possible 1:1'}")

print(f"\n{forced} of {checked} dates PROVE a shared recording, covering {clips_forced} clips "
      f"({100*clips_forced/len(names):.0f}% of the named set)")

# Per bird: how many dates does it share with another bird?
print("\nper bird -- dates it shares with >=1 other bird:")
rows = []
for b in sorted(set(birds)):
    dts = {n.split('_', 1)[1].split('-')[0] for n, bb in zip(names, birds) if bb == b}
    shared = sum(1 for dt in dts if len(birds_by_date[dt]) > 1)
    rows.append((b, len(dts), shared))
clean = [r for r in rows if r[2] == 0]
for b, nd, sh in sorted(rows, key=lambda r: r[2]):
    print(f"   {b:13s} {nd:3d} dates, {sh:3d} shared" + ("   <-- NEVER shares a date" if sh == 0 else ""))
print(f"\nbirds that never share a recording date: {len(clean)} of {len(rows)}")
if clean:
    print("   ->", [c[0] for c in clean], "are candidate holdout birds")

#!/bin/bash
# Step 0 (login node, seconds): build the held-out pretraining manifest.
#
# Drops the 20 recordings on the 11 dates where LblRed0613 has curated clips. That bird is the
# ONLY one of 26 that never shares a recording date with another bird, which is what makes a
# holdout possible at all -- see experiments/holdout_feasibility.py. No other bird has curated
# clips on those dates, so the evaluation set is untouched.
#
# Costs 20 of 120 recordings (16.7% of pretraining audio). That is a real confound: a drop in
# accuracy could be "never heard this bird" OR "16.7% less data". See 03_eval for the control.
set -euo pipefail

SRC=/global/scratch/users/jonathanswang/temp_files/run5-4-26-full/data/spectrogram/tsv/ZF_test_pipeline_train.tsv
DST_DIR=/global/scratch/users/jonathanswang/temp_files/holdout_lblred0613/data/spectrogram/tsv
DROP_DATES="110920 110921 110922 110923 111004 111006 111013 111018 111121 111202 111208"

mkdir -p "$DST_DIR"

python3 - "$SRC" "$DST_DIR/ZF_test_pipeline_train.tsv" "$DROP_DATES" <<'PY'
import sys
src, dst, drop_dates = sys.argv[1], sys.argv[2], set(sys.argv[3].split())
lines = open(src).read().splitlines()
root, rows = lines[0], lines[1:]
keep, dropped = [], []
for r in rows:
    if not r.strip():
        continue
    base = r.split("\t")[0].split("/")[-1].replace(".wav", "")
    (dropped if base.split("-")[0] in drop_dates else keep).append(base)
out = [root] + [r for r in rows if r.strip() and
                r.split("\t")[0].split("/")[-1].replace(".wav", "").split("-")[0] not in drop_dates]
open(dst, "w").write("\n".join(out) + "\n")
print(f"kept {len(keep)} recordings, dropped {len(dropped)}")
print("dropped:", sorted(dropped))
assert len(dropped) == 20, f"expected to drop 20 recordings, dropped {len(dropped)}"
assert len(keep) == 100, f"expected to keep 100 recordings, kept {len(keep)}"
PY

# an empty valid tsv, matching the original run
printf '%s\n' "$(head -1 "$SRC")" > "$DST_DIR/ZF_test_pipeline_valid.tsv"
echo "wrote $DST_DIR/ZF_test_pipeline_train.tsv"
wc -l "$DST_DIR"/*.tsv

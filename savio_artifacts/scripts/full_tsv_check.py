import os, soundfile as sf
from collections import Counter
P = "/global/scratch/users/jonathanswang/temp_files/combined_zf_fsd/data/spectrogram/tsv/ZF_test_pipeline_train.tsv"
lines = open(P).read().splitlines()
root, rows = lines[0], [l.split("\t") for l in lines[1:]]
pre = Counter(r[0].split("/")[0] for r in rows)
print("[prefixes] " + ", ".join("{}={}".format(k, v) for k, v in pre.most_common()))
missing, mismatch, ok = [], [], 0
for rel, n in rows:
    p = os.path.join(root, rel)
    if not os.path.exists(p):
        missing.append(rel); continue
    try:
        f = sf.info(p).frames
    except Exception as e:
        missing.append(rel + " (" + str(e)[:40] + ")"); continue
    if f != int(n):
        mismatch.append((rel, int(n), f))
    else:
        ok += 1
print("[resolve]  {} of {} rows OK   missing={}   frame-count mismatch={}".format(ok, len(rows), len(missing), len(mismatch)))
for m in missing[:8]: print("   MISSING  " + m)
for r, a, b in mismatch[:8]: print("   MISMATCH {}  tsv={} actual={}".format(r, a, b))

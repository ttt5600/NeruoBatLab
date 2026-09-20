import csv, json, numpy as np, sys
from pathlib import Path

SEG = "/Users/jonathanwang/Downloads/Segments Data.csv"
SR  = 44100
WIN = 1.0
OUT = Path.home()/"zf_labelset"/"annotated_111021"

# --- read onsets/offsets ---
segs = []
with open(SEG) as f:
    for r in csv.DictReader(f):
        a, b = int(r["StartIndex"]), int(r["StopIndex"])
        if b > a:
            segs.append((a, b))
segs.sort()
# merge overlaps
merged = []
for a, b in segs:
    if merged and a <= merged[-1][1]:
        merged[-1][1] = max(merged[-1][1], b)
    else:
        merged.append([a, b])
seg = np.array(merged, dtype=np.int64)

end = int(seg[:, 1].max())
win = int(WIN * SR)
nwin = end // win          # whole 1 s increments covering the annotated file

rows, npos = [], 0
for i in range(nwin):
    lo, hi = i*win, (i+1)*win
    # any overlap with any segment?
    j = np.searchsorted(seg[:, 1], lo, side="right")
    hit = j < len(seg) and seg[j, 0] < hi
    lab = "voc" if hit else "noise"
    npos += hit
    rows.append(dict(id="a%06d" % i, recording="111021-000",
                     start_sample=lo, stop_sample=hi,
                     start_s=round(lo/SR, 3), dur_s=WIN,
                     label=lab, y=int(hit)))

OUT.mkdir(parents=True, exist_ok=True)
with open(OUT/"windows_1s.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)

meta = dict(segments_csv=SEG, recording="111021-000.wav", src_sr=SR,
            n_segments=len(seg), file_end_sample=end,
            duration_min=round(end/SR/60, 3), win_sec=WIN,
            n_windows=nwin, n_voc=int(npos), n_noise=nwin-int(npos),
            prevalence=round(npos/nwin, 4),
            voiced_frac=round(float((seg[:,1]-seg[:,0]).sum())/end, 4))
(OUT/"meta_1s.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))

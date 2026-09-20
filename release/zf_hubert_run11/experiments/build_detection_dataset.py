"""Assemble the full ZF vocalization-detection dataset from every hand/expert label we have."""
import csv, json, collections
from pathlib import Path

NEG  = Path.home()/"zf_labelset"/"neg_bundle"
ANN  = Path.home()/"zf_labelset"/"annotated_111021"
OUT  = Path.home()/"zf_labelset"/"zf_detection_dataset_v1"
OUT.mkdir(parents=True, exist_ok=True)

# call / call+noise -> vocalization present; noise / silence -> absent
Y = {"call": 1, "call+noise": 1, "noise": 0, "silence": 0}

rows = []

# ---- source 1: the 4900 benchmark "negatives", hand-labeled ----
idx = {r["id"]: r for r in csv.DictReader(open(NEG/"INDEX.csv"))}
lab = {r["id"]: r for r in csv.DictReader(open(NEG/"labels.csv")) if r["id"] != "__selftest__"}
jee = {r["id"] for r in csv.DictReader(open("/Users/jonathanwang/Downloads/sample_labels_JEE (1).csv"))}

for wid, r in sorted(idx.items()):
    h = lab.get(wid, {}).get("human_label", "")
    rows.append(dict(
        id=wid, source="benchmark_neg_pool", recording=r["recording"],
        start_sample=int(r["center_sample"]) - 4000,   # 0.5 s @ 16 kHz centered
        dur_s=0.5, sr=16000,
        human_label=h, y=Y[h] if h else "",
        labeled=int(bool(h)),
        annotator="julie_elie" if wid in jee else ("jonathan_wang" if h else ""),
        batch=lab.get(wid, {}).get("batch", ""),
        orig_benchmark_y=0,
    ))

# ---- source 2: 111021-000, exhaustively annotated in SoundSep ----
for r in csv.DictReader(open(ANN/"windows_1s.csv")):
    rows.append(dict(
        id=r["id"], source="soundsep_111021", recording="111021-000",
        start_sample=int(r["start_sample"]), dur_s=1.0, sr=44100,
        human_label="voc" if r["y"] == "1" else "noise", y=int(r["y"]),
        labeled=1, annotator="bhavna_malladi", batch="",
        orig_benchmark_y="",
    ))

fields = list(rows[0])
with open(OUT/"dataset.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

# ---- summary ----
def tally(pred):
    sub = [r for r in rows if pred(r)]
    return len(sub), collections.Counter(r["human_label"] for r in sub)

meta = {}
for src in ("benchmark_neg_pool", "soundsep_111021"):
    sub = [r for r in rows if r["source"] == src]
    lb  = [r for r in sub if r["labeled"]]
    meta[src] = dict(
        n_windows=len(sub), n_labeled=len(lb), n_unlabeled=len(sub)-len(lb),
        label_counts=dict(collections.Counter(r["human_label"] for r in lb)),
        n_vocalization=sum(1 for r in lb if r["y"] == 1),
        n_no_vocalization=sum(1 for r in lb if r["y"] == 0),
    )
meta["total_rows"] = len(rows)
meta["total_labeled"] = sum(r["labeled"] for r in rows)
meta["expert_reviewed"] = len(jee)
meta["missing"] = ("2450 curated positives of the original benchmark are NOT in this file -- "
                   "their manifest lives on Savio, not locally")
(OUT/"meta.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))

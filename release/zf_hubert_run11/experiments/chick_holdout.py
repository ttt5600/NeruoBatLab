"""Chick holdout: does the encoder generalize to individuals it has never heard?

This is the closest thing to a true holdout available without retraining. The 18 chicks in
ChickVocalizations.zip share no individual with the 26 adults, and 14 of them were recorded on
dates absent from the 120-recording pretraining manifest. Those 14 are used here.

What it CANNOT do: reproduce the 8-way adult call-type number. Chicks produce only Be (begging)
and LT (long tonal), neither of which is one of the adult 8. So this is a different, easier task
(2-way, chance 0.50) and its accuracy is NOT comparable to the adult 0.811.

What it CAN do: ask whether call-type and identity structure survive in a population the
encoder never heard, and whether that structure is anything more than clip duration.
"""
import collections
import os
import zipfile
from pathlib import Path

import numpy as np
import torch
import torchaudio
import warnings

warnings.filterwarnings("ignore")

TMP = Path("/Users/jonathanwang/.claude/jobs/63c218d9/tmp")
REL = Path("/Users/jonathanwang/Desktop/vocalizations_lab/release/zf_hubert_run11")
MANIFEST = Path(os.environ["CLAUDE_JOB_DIR"]) / "tmp/pretrain_files.txt"
OUT = TMP / "chick_embeddings.npz"

pt_dates = {l.strip().split("/")[-1].split("-")[0] for l in open(MANIFEST) if l.strip()}

if not OUT.exists():
    ex = TMP / "chick"
    if not ex.exists():
        zipfile.ZipFile(TMP / "ChickVocalizations.zip").extractall(ex)
    wavs = sorted(p for p in ex.rglob("*.wav") if not p.name.startswith("._"))

    obj = torch.load(REL / "weights/zf_hubert_run11_encoder.pt", map_location="cpu",
                     weights_only=False)
    model = torchaudio.models.wav2vec2_model(aux_num_out=None, **obj["encoder_config"])
    model.load_state_dict(obj["state_dict"], strict=True)
    model.eval()

    rows, birds, types, dates, durs, names = [], [], [], [], [], []
    for i, p in enumerate(wavs):
        pre, rest = p.name.split("_", 1)
        dt, vt = rest.split("-")[0], rest.split("-")[1][:2]
        if dt in pt_dates:                      # keep only genuinely unseen dates
            continue
        w, sr = torchaudio.load(str(p))
        if sr != 16000:
            w = torchaudio.transforms.Resample(sr, 16000)(w)
        w = w.mean(0, keepdim=True)
        if w.shape[1] < 800:                    # conv extractor needs ~0.05 s
            continue
        with torch.no_grad():
            feats, _ = model.extract_features(w, None)
        rows.append(torch.stack([f.squeeze(0).mean(0) for f in feats]).numpy())
        birds.append(pre); types.append(vt); dates.append(dt)
        durs.append(w.shape[1] / 16000); names.append(p.name)
        if len(rows) % 100 == 0:
            print(f"  embedded {len(rows)}")
    np.savez(OUT, emb=np.stack(rows).astype(np.float32), birds=np.array(birds),
             types=np.array(types), dates=np.array(dates), durs=np.array(durs),
             names=np.array(names))
    print(f"wrote {OUT}")

d = np.load(OUT, allow_pickle=True)
emb, birds, types, durs = d["emb"], d["birds"], d["types"], d["durs"]
y = (types == "LT").astype(int)                 # 0 = Be, 1 = LT
print(f"\n{len(y)} clips | {len(np.unique(birds))} chicks | "
      f"Be={int((y==0).sum())} LT={int(y.sum())}")
print(f"majority baseline {max(np.mean(y), 1-np.mean(y)):.3f}")
print("clips per chick:", dict(collections.Counter(birds.tolist()).most_common()))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as ami

usable = [b for b in np.unique(birds)
          if (birds == b).sum() >= 5 and len(np.unique(y[birds == b])) == 2]
m = np.isin(birds, usable)
print(f"\nchicks with both call types and >=5 clips: {len(usable)} ({m.sum()} clips)")
E, Y, G, D = emb[m], y[m], birds[m], durs[m]
n_sp = min(5, len(usable))

print(f"\n=== Be vs LT, leave-chicks-out ({n_sp}-fold), chance 0.50 ===")
print(f"{'features':22s} {'acc':>7s} {'AUC':>7s}")
Xd = StandardScaler().fit_transform(np.c_[D, np.log(D)])
p = cross_val_predict(LogisticRegression(max_iter=1000), Xd, Y, groups=G,
                      cv=StratifiedGroupKFold(n_sp), method="predict_proba")[:, 1]
print(f"{'DURATION baseline':22s} {accuracy_score(Y, p > .5):7.3f} {roc_auc_score(Y, p):7.3f}")
best = (None, -1)
for L in range(E.shape[1]):
    X = StandardScaler().fit_transform(E[:, L, :])
    p = cross_val_predict(LogisticRegression(max_iter=2000), X, Y, groups=G,
                          cv=StratifiedGroupKFold(n_sp), method="predict_proba")[:, 1]
    a = roc_auc_score(Y, p)
    print(f"{'layer ' + str(L):22s} {accuracy_score(Y, p > .5):7.3f} {a:7.3f}")
    if a > best[1]:
        best = (L, a)
print(f"best layer {best[0]}  AUC {best[1]:.3f}")

print("\n=== unsupervised: k-means on layer 3, no labels ===")
X = normalize(emb[:, 3, :])
for k in [2, 4, 8, 14, 20]:
    c = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    print(f"k={k:3d} | AMI vs call type {ami(y, c):.3f} | AMI vs chick ID {ami(birds, c):.3f}")

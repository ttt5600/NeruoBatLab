#!/usr/bin/env python
"""Were run11's pretraining TARGETS the problem, rather than its corpus?

HuBERT learns by predicting cluster ids. The clusters ARE the supervision, so their quality bounds
what the model can learn. run11 and AVES differ here in a way that has been invisible until now:

    run11   iteration 1, k-means k=100 over LOG-MEL SPECTROGRAM frames
    AVES    iteration 2, k-means k=200 over a trained HuBERT's LAYER 6 frames

That is two changes at once -- the feature the clusters are built from, and how many there are -- and
both are confounded with the corpus difference the run11-vs-AVES comparison was supposed to measure.
Retraining to settle it needs a cluster. Measuring the TARGETS themselves does not, and it tests the
premise: if layer-6 k=200 targets are barely more informative than log-mel k=100 targets on this
audio, then the recipe is not what is holding run11 back and a rerun would be wasted compute.

Targets are scored against two references on the same frames:

  voc/noise     does a cluster know whether it is inside a call at all
  call id       do clusters respect CALL BOUNDARIES -- frames of one call landing in one cluster.
                This is the finer and more HuBERT-like question, and it is the analogue of the
                phone-purity measure used to validate speech HuBERT targets.

AMI is used rather than NMI or purity because all three references have very different cluster counts
and NMI/purity rise monotonically with k -- they would hand k=200 a win for free.

The k=200-vs-k=100 axis is separated from the feature axis by also clustering log-mel at k=200.
"""
from __future__ import annotations
import csv, gc, json, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zfeval import features as fxx                                           # noqa: E402

SR, HOP, RF = 16000, 320, 400
SPAN = int(round(79434253 * SR / 44100))
AUDIO = Path.home() / "zf_labelset/audio/111021-000.wav"
SEGMENTS = Path.home() / "Downloads/Segments Data.csv"
FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
SEED = 0


def call_index_reference(centers):
    """Per frame: which annotated call it falls in, or -1 for background.

    Clusters that respect call boundaries should score high against this; clusters that merely track
    loudness should not.
    """
    rows = list(csv.DictReader(open(SEGMENTS)))
    iv = np.array([[float(r["StartIndex"]), float(r["StopIndex"])] for r in rows])
    iv = iv[~np.isnan(iv).any(1)]
    iv = iv[iv[:, 1] > iv[:, 0]] * SR / 44100
    iv = np.clip(iv, 0, SPAN)
    iv = iv[np.argsort(iv[:, 0])]
    ref = np.full(len(centers), -1, dtype=int)
    for i, (a, b) in enumerate(iv):
        ref[(centers >= a) & (centers <= b)] = i
    return ref


def main():
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import adjusted_mutual_info_score as ami
    from sklearn.preprocessing import StandardScaler

    FR = np.load(FEAT / "frames_30min.npz", allow_pickle=True)
    y, centers = FR["y"], FR["centers"]
    nF = len(y)
    print(f"[frames] {nF:,}  voc fraction {y.mean():.4f}", flush=True)

    ref_call = call_index_reference(centers)
    print(f"[reference] call-id: {len(set(ref_call[ref_call >= 0]))} distinct calls, "
          f"{(ref_call >= 0).mean():.4f} of frames inside a call", flush=True)

    # ---- log-mel frames, the feature run11's iteration-1 targets were built from
    p_mel = FEAT / "mel_frames_30min.npy"
    if p_mel.exists():
        M = np.load(p_mel)
        print(f"[skip] cached log-mel {M.shape}")
    else:
        rec, _ = sf.read(AUDIO, dtype="float32", frames=min(SPAN + 20 * SR, sf.info(AUDIO).frames))
        M = np.zeros((nF, 384), dtype=np.float32)
        t0 = time.time()
        half = int(0.025 * SR / 2)
        for i in range(nF):
            c = int(centers[i])
            lo = max(0, c - half); hi = min(len(rec), lo + 2 * half); lo = max(0, hi - 2 * half)
            M[i] = fxx.mel_features(rec[lo:hi])
            if i % 20000 == 0:
                print(f"    mel {i}/{nF}  {time.time()-t0:.0f}s", flush=True)
        np.save(p_mel, M)
        del rec; gc.collect()

    AV = np.load(FEAT / "aves_frames_matched.npz", allow_pickle=True)
    sources = {
        "logmel_k100":   (M, 100, "run11's actual recipe: iteration-1 log-mel, k=100"),
        "logmel_k200":   (M, 200, "isolates k: same feature as run11, AVES's cluster count"),
        "run11_L6_k200": (FR["F6"].astype(np.float32), 200,
                          "AVES's recipe run on OUR encoder: iteration-2 from layer 6, k=200"),
        "aves_L6_k200":  (AV["F6"].astype(np.float32), 200,
                          "AVES's actual recipe and encoder"),
    }

    out = {"n_frames": int(nF), "voc_fraction": float(y.mean()),
           "n_calls": int(len(set(ref_call[ref_call >= 0]))),
           "note": "AMI is chance-corrected; NMI/purity would favour k=200 for free.",
           "targets": {}}

    print(f"\n{'target':16s} {'k':>5s} {'AMI(voc/noise)':>15s} {'AMI(call id)':>14s} "
          f"{'used clusters':>14s}")
    for name, (X, k, note) in sources.items():
        Z = StandardScaler().fit_transform(X)
        km = MiniBatchKMeans(n_clusters=k, random_state=SEED, n_init=5, batch_size=4096,
                             max_iter=300).fit(Z)
        lab = km.labels_
        a_voc = float(ami(y, lab))
        a_call = float(ami(ref_call, lab))
        used = int(len(set(lab)))
        out["targets"][name] = dict(k=k, ami_voc=a_voc, ami_call=a_call, used_clusters=used,
                                    note=note, inertia=float(km.inertia_))
        print(f"{name:16s} {k:5d} {a_voc:15.4f} {a_call:14.4f} {used:14d}", flush=True)
        del Z, km; gc.collect()

    t = out["targets"]
    d_feature = t["run11_L6_k200"]["ami_call"] - t["logmel_k200"]["ami_call"]
    d_k = t["logmel_k200"]["ami_call"] - t["logmel_k100"]["ami_call"]
    d_total = t["aves_L6_k200"]["ami_call"] - t["logmel_k100"]["ami_call"]
    out["decomposition_ami_call"] = dict(
        effect_of_k_100_to_200=d_k,
        effect_of_feature_logmel_to_layer6=d_feature,
        total_run11_recipe_to_aves_recipe=d_total)
    print(f"\ndecomposition on AMI(call id):")
    print(f"  k=100 -> k=200, feature held at log-mel      : {d_k:+.4f}")
    print(f"  log-mel -> layer 6, k held at 200            : {d_feature:+.4f}")
    print(f"  run11's whole recipe -> AVES's whole recipe  : {d_total:+.4f}")

    (ANA / "target_quality.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ANA/'target_quality.json'}")


if __name__ == "__main__":
    main()

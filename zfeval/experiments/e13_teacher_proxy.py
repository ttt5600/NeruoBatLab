#!/usr/bin/env python
"""E13 teacher proxy -- which teacher's k-means labels carry the most call-type information?

E13 asks whether labels from a stronger teacher than AVES-bio make a better student. A student costs a relabel job plus
~6 h per seed on 4 GPUs, and the teacher is a free choice of model AND layer (bioxn-large has 24). This scores
candidate LABELS without training anything:

  1. fit k-means k=200 per teacher feature on frames of one corpus recording (111021-000, 80.6 min, every 4th 20 ms
     frame), as the real relabel fits k=200 on corpus frames;
  2. encode each of the 3,412 11-class call-type clips alone on CPU (finding 076) and assign its frames;
  3. score the labels against the clips' call types two ways:
       ami   AMI between a frame's cluster and its clip's call type (chance-corrected, so k cannot win for free)
       bag   the board's leave-birds-out probe (cv_acc) on each clip's normalised 200-bin cluster histogram: how much
             call type the discrete targets carry at the clip level.

The proxy is only trusted if it reproduces what training already showed: students on AVES-bio layer-6 labels (run19,
two seeds) score above students on run16 layer-6 (run18), MFCC (run23) and spectrogram labels (run16, three seeds).
The older call-boundary proxy (target_quality.py) ranked AVES labels BELOW log-mel -- the wrong way round.

Layers follow the relabel's --layer-index: 1 = first transformer layer output (the board's L0), 6 = the board's L5.
Caveats: the real k-means is fit on 763 files, half FSD50K; this one on one zebra finch recording. Clips are encoded
in isolation, the corpus in independent 20 s windows (the relabel uses 20 s windows too).

    python zfeval/experiments/e13_teacher_proxy.py extract --teacher aves-base-bio
    python zfeval/experiments/e13_teacher_proxy.py score            -> analysis/e13_teacher_proxy.json
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "pytorchAudio/examples/hubert"))
from calltype11 import collect, KEEP11                                     # noqa: E402
from aves_calltype import cv_acc, load_audio                               # noqa: E402
from detection_variants import spec, load_model                           # noqa: E402

SR, WIN, K = 16000, 20 * 16000, 200
REC = Path.home() / "zf_labelset/audio/111021-000.wav"
FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features/e13proxy"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
TEACHERS = {
    "spectrogram": [0],                       # run16's own targets: soundsig 25 ms x 50 Hz log spectrogram
    "mfcc": [0],                              # run23's targets: fairseq's 39-d Kaldi MFCC
    "run16_compute4x": [6],                   # run18's teacher
    "aves-base-bio": [3, 6, 9],               # run19's teacher is layer 6
    "birdaves-biox-base": [3, 6, 9],
    "birdaves-bioxn-large": [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24],
    # the two model teachers AS THE RELABEL RAN THEM: build_teacher left them in train mode (dropout 0.1) until
    # 2026-10-27 (teacher_dropout_diagnostic.py), so these are the labels run18 and run19 actually learned from
    "aves-base-bio@train": [6],
    "run16_compute4x@train": [6],
}
# students already trained on four of these label sources (tags in calltype_cpu_rescore.json)
KNOWN = {("spectrogram", 0): ["run16_compute4x", "run16_seed2", "run16_seed3"],
         ("mfcc", 0): ["run23_mfcc"],
         ("run16_compute4x@train", 6): ["run18_iter2"],
         ("aves-base-bio@train", 6): ["run19_avesteacher", "run19_seed2"]}


def spec_frames(w: np.ndarray) -> np.ndarray:
    """extract_feature_spectrogram's frames for one short array: log10 power, 25 ms windows every 20 ms."""
    from soundsig.sound import spectrogram
    _, _, tf, _ = spectrogram(w.astype(np.float64), SR, spec_sample_rate=1000, freq_spacing=50, cmplx=False)
    ls = np.log10(tf.astype(np.float32) ** 2 + np.float32(1e-8))
    n = (ls.shape[1] - 25) // 20 + 1
    if n < 1:
        raise RuntimeError(f"spectrogram has {ls.shape[1]} ms, need 25")
    idx = np.arange(n)[:, None] * 20 + np.arange(25)[None, :]
    return ls[:, idx].transpose(1, 0, 2).reshape(n, -1)


def mfcc_frames(w: np.ndarray) -> np.ndarray:
    """extract_feature_mfcc's frames for one array: 13 ceps + deltas, every other 10 ms frame (the 20 ms grid)."""
    import torchaudio
    from utils.feature_utils import _kaldi_mfcc
    x = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)).view(1, -1)
    c = _kaldi_mfcc(x, SR).transpose(0, 1)
    d1 = torchaudio.functional.compute_deltas(c)
    d2 = torchaudio.functional.compute_deltas(d1)
    return torch.cat([c, d1, d2], 0).transpose(0, 1)[::2].numpy()


def frames_fn(teacher: str):
    """(1-D float32 array at 16 kHz) -> {layer: (n, D) float32} for every layer this teacher is scored at."""
    layers = TEACHERS[teacher]
    if teacher == "spectrogram":
        return lambda w: {0: spec_frames(w)}
    if teacher == "mfcc":
        return lambda w: {0: mfcc_frames(w)}
    if teacher.endswith("@train"):
        from utils.feature_utils import build_teacher
        from teacher_dropout_diagnostic import bare
        model = build_teacher(bare(spec(teacher[:-6])["weights"]), torch.device("cpu")).wav2vec2.train()
        torch.manual_seed(0)
    else:
        model, _ = load_model(teacher, spec(teacher), "cpu")

    @torch.no_grad()
    def f(w):
        feats, _ = model.extract_features(torch.from_numpy(w).view(1, -1), None)
        return {l: feats[l - 1].squeeze(0).numpy() for l in layers}
    return f


def pad400(w):
    return np.pad(w, (0, 400 - len(w))) if len(w) < 400 else w


def extract(teacher: str):
    FEAT.mkdir(parents=True, exist_ok=True)
    out = FEAT / f"{teacher}.npz"
    torch.set_num_threads(8)
    f = frames_fn(teacher)
    t0 = time.time()
    rec, sr = sf.read(str(REC), dtype="float32")
    assert sr == SR and rec.ndim == 1
    corpus = {l: [] for l in TEACHERS[teacher]}
    for s in range(0, len(rec) - WIN + 1, WIN):
        for l, x in f(rec[s:s + WIN]).items():
            assert x.shape[0] == 999, (teacher, l, x.shape)
            corpus[l].append(x[::4].astype(np.float16))
    print(f"  {teacher}: corpus {len(corpus[TEACHERS[teacher][0]])} windows, {time.time()-t0:.0f}s", flush=True)
    rows = [r for r in collect() if r[3] in KEEP11]
    clips = {l: [] for l in TEACHERS[teacher]}
    off = [0]
    for i, r in enumerate(rows):
        w = pad400(load_audio(r[0]).squeeze(0).numpy().astype(np.float32))
        fr = f(w)
        n = {x.shape[0] for x in fr.values()}
        assert len(n) == 1, (r[0], n)
        for l, x in fr.items():
            clips[l].append(x.astype(np.float16))
        off.append(off[-1] + n.pop())
        if (i + 1) % 500 == 0:
            print(f"    {teacher} clips {i+1}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)
    arrs = {f"corpus_L{l}": np.concatenate(v) for l, v in corpus.items()}
    arrs.update({f"clip_L{l}": np.concatenate(v) for l, v in clips.items()})
    for k, v in arrs.items():
        if not np.isfinite(v.astype(np.float32)).all():
            raise RuntimeError(f"{teacher} {k}: non-finite values (fp16 overflow?)")
    np.savez(out, off=np.array(off), **arrs)
    print(f"  wrote {out} ({sum(v.nbytes for v in arrs.values())/1e9:.2f} GB) in {time.time()-t0:.0f}s", flush=True)


def score(seeds=(0, 1)):
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import adjusted_mutual_info_score as ami
    rows = [r for r in collect() if r[3] in KEEP11]
    types = sorted(KEEP11)
    y = np.array([types.index(r[3]) for r in rows])
    groups = np.array([r[1] for r in rows])
    cpu = json.loads((ANA / "calltype_cpu_rescore.json").read_text())["models"]
    ss = json.loads((ANA / "calltype_split_sensitivity_cpu.json").read_text())["models"]
    res = {}
    for teacher, layers in TEACHERS.items():
        p = FEAT / f"{teacher}.npz"
        if not p.exists():
            print(f"  {teacher}: not extracted, skipped")
            continue
        z = np.load(p)
        off = z["off"]
        assert len(off) == len(rows) + 1
        yf = np.repeat(y, np.diff(off))
        for l in layers:
            C, X = z[f"corpus_L{l}"].astype(np.float32), z[f"clip_L{l}"].astype(np.float32)
            a, b, used = [], [], []
            for s in seeds:
                km = MiniBatchKMeans(n_clusters=K, batch_size=8192, n_init=3, random_state=s, max_iter=200).fit(C)
                lab = km.predict(X)
                a.append(float(ami(yf, lab)))
                H = np.zeros((len(rows), K), dtype=np.float32)
                np.add.at(H, (np.repeat(np.arange(len(rows)), np.diff(off)), lab), 1.0)
                H /= H.sum(1, keepdims=True)
                b.append(cv_acc(H, y, groups)[0])
                used.append(int(len(np.unique(km.labels_))))
            key = f"{teacher}@L{l}"
            st = KNOWN.get((teacher, l))
            res[key] = dict(teacher=teacher, layer=l, ami=float(np.mean(a)), ami_seeds=a, bag=float(np.mean(b)),
                            bag_seeds=b, clusters_used=used, corpus_frames=int(len(C)), clip_frames=int(len(X)),
                            students=st,
                            student_best=None if not st else float(np.mean([cpu[t]["acc11_cpu"] for t in st])),
                            student_10split=None if not st else float(np.mean([ss[t]["mean"] for t in st])))
            print(f"  {key:28s} ami {np.mean(a):.4f} {a}  bag {np.mean(b):.4f} {b}"
                  + (f"   students {st}: best {res[key]['student_best']:.4f}" if st else ""), flush=True)
    out = ANA / "e13_teacher_proxy.json"
    out.write_text(json.dumps(dict(k=K, seeds=list(seeds), recording=str(REC), results=res), indent=1))
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "score"])
    ap.add_argument("--teacher", choices=list(TEACHERS))
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.teacher)
    else:
        score()

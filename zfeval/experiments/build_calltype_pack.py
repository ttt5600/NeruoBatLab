#!/usr/bin/env python
"""Pack the 11-class hand-labelled call-type set into ONE file for the Savio eval job (eval_model.py).

Contents: every clip's waveform exactly as the board feeds it to an encoder (aves_calltype.load_audio: torchaudio
load, cached Resample to 16 kHz, mono mean, no normalisation), concatenated float32 with offsets; per clip its file
name, bird, call type and source folder; and the board's FOLD assignment (StratifiedGroupKFold(5, shuffle, seed 0)
from this Mac's sklearn 1.7.2 -- sklearn 1.8 splits differently, finding 062), so the Savio job reproduces the
board's folds whatever sklearn it has. One file, so nothing on Savio's Lustre holds thousands of tiny wavs.

    python zfeval/experiments/build_calltype_pack.py  -> ~/zf_labelset/eval_pack/calltype11_pack.npz
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from calltype11 import collect, KEEP11                                 # noqa: E402
from aves_calltype import load_audio, NFOLD, SEED                       # noqa: E402
import sklearn                                                          # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold                # noqa: E402

OUT = Path.home() / "zf_labelset/eval_pack/calltype11_pack.npz"
rows = [r for r in collect() if r[3] in KEEP11]
birds = np.array([r[1].lower() for r in rows]); tt = np.array([r[3] for r in rows]); classes = sorted(set(tt))
y = np.array([classes.index(t) for t in tt])
assert len(y) == 3412 and len(classes) == 11 and len(set(birds)) == 48
fold = np.full(len(y), -1)
for f, (_, te) in enumerate(StratifiedGroupKFold(n_splits=NFOLD, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y, birds)):
    fold[te] = f
assert (fold >= 0).all()
waves = [load_audio(r[0]).squeeze(0).numpy().astype(np.float32) for r in rows]
lens = np.array([len(w) for w in waves]); off = np.concatenate([[0], np.cumsum(lens)])
np.savez(OUT, audio=np.concatenate(waves), offsets=off, files=np.array([r[0].name for r in rows]), birds=birds,
         types=tt, classes=np.array(classes), y=y, source=np.array([r[4] for r in rows]), fold=fold,
         sample_rate=np.array(16000), sklearn_version=np.array(sklearn.__version__),
         note=np.array("Elie & Theunissen zebra finch call clips (CC BY 4.0), 11 types; board folds"))
print(f"{len(y)} clips, {lens.sum() / 16000:.0f} s audio, folds {np.bincount(fold).tolist()}, sklearn {sklearn.__version__} -> {OUT} "
      f"({OUT.stat().st_size / 1e6:.0f} MB)")

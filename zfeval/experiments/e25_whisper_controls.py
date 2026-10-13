#!/usr/bin/env python
"""Controls for the Whisper call-type result (finding 072): is it clip length, or Whisper's input features alone?

    ~/zf_labelset/envs/e25/bin/python e25_whisper_controls.py   -> analysis/e25_whisper_controls.json

Same cohort, probe and folds as the board (aves_calltype.cv_acc). Features: log clip duration (1-d); Whisper-base's
own log-mel input (80 bins, its feature extractor) mean and std over the real frames only (160-d); both together.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
from calltype11 import collect, KEEP11                     # noqa: E402
from aves_calltype import load_audio, cv_acc               # noqa: E402
from transformers import AutoFeatureExtractor              # noqa: E402

OUT = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis/e25_whisper_controls.json"
rows = [r for r in collect() if r[3] in KEEP11]
birds = np.array([r[1].lower() for r in rows]); cl = sorted(set(r[3] for r in rows)); y = np.array([cl.index(r[3]) for r in rows])
fe = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
dur = np.zeros((len(rows), 1)); lm = np.zeros((len(rows), 160))
for i, r in enumerate(rows):
    w = load_audio(r[0]).squeeze(0).numpy(); dur[i, 0] = np.log(len(w) / 16000)
    f = fe(w, sampling_rate=16000, return_tensors="np").input_features[0]
    g = f[:, :max(1, int(np.ceil(len(w) / 160)))]
    lm[i, :80] = g.mean(1); lm[i, 80:] = g.std(1)
res = dict(duration_only=cv_acc(dur, y, birds)[0], whisper_logmel_meanstd=cv_acc(lm, y, birds)[0],
           logmel_plus_duration=cv_acc(np.hstack([lm, dur]), y, birds)[0], n_clips=len(y),
           split="leave-birds-out StratifiedGroupKFold(5), seed 0")
OUT.write_text(json.dumps({k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in res.items()}, indent=2))
print(res, "->", OUT)

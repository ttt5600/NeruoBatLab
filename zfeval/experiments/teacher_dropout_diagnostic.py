#!/usr/bin/env python
"""How much did dropout change the iteration-2 / teacher labels?

utils.feature_utils.build_teacher never calls .eval(), so every relabel from a model teacher (run12, run13, run18,
run19, run22, the DAPT round 3-4 targets, E14, E27) read the teacher's layer with dropout ON: torchaudio's
hubert_pretrain_base defaults to 0.1 projection, 0.1 attention and 0.1 encoder dropout. (Layer drop is not applied on
the extract_features path.) fairseq's dump_hubert_feature.py puts the teacher in eval mode.

This loads each teacher through the SAME build_teacher the relabel uses and reads the SAME 20 s windows of one corpus
recording three times -- train mode twice (two dropout draws) and eval mode once -- then fits k-means k=200 on the
train-mode pass A, as the relabel did, and counts how often a frame's label changes:

    A vs B   label noise of the relabel as it ran: two dropout draws, same codebook
    A vs E   what eval mode would have given instead, same codebook
plus AMI for each pair, so the change is not read off a raw agreement rate alone.

    python zfeval/experiments/teacher_dropout_diagnostic.py      -> analysis/teacher_dropout_diagnostic.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "pytorchAudio/examples/hubert"))
from utils.feature_utils import build_teacher                            # noqa: E402

SR, WIN, LOOK, K, LAYER = 16000, 20 * 16000, 80, 200, 6
REC = Path.home() / "zf_labelset/audio/111021-000.wav"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
EXT = Path.home() / "zf_labelset/external"
TEACHERS = {"aves-base-bio (run19, run22 teacher)": EXT / "aves/aves-base-bio.torchaudio.pt",
            "run16 (run18 teacher)": EXT / "dapt/run16_compute4x.pt"}
N_WIN = 80                                        # 80 windows = 26.7 min of the recording, ~80k frames


def bare(ck: Path) -> Path:
    """The local run16 file is an export ({encoder_config, state_dict, meta} with bare encoder keys), not the Lightning
    checkpoint Savio's relabel read. Write its encoder to a temp file so build_teacher loads it into the SAME
    hubert_pretrain_base (same dropout) the relabel built -- through its strict bare-state_dict branch."""
    import tempfile
    obj = torch.load(str(ck), map_location="cpu", weights_only=False)
    sd = obj.get("state_dict") if isinstance(obj, dict) else None
    if sd is None or any(k.startswith("model.") for k in sd):
        return ck
    out = Path(tempfile.gettempdir()) / f"{ck.stem}.bare_encoder.pt"
    torch.save(dict(sd), out)
    return out


def read(model, rec, train: bool, seed: int):
    model.train(train)
    torch.manual_seed(seed)
    out = []
    with torch.inference_mode():
        for s in range(0, N_WIN * WIN, WIN):
            seg = torch.from_numpy(rec[s:s + WIN + LOOK]).view(1, -1)
            out.append(model.wav2vec2.extract_features(seg, num_layers=LAYER)[0][-1][0].numpy())
    return np.concatenate(out)


def main():
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import adjusted_mutual_info_score as ami
    torch.set_num_threads(4)
    rec, sr = sf.read(str(REC), dtype="float32")
    assert sr == SR and len(rec) >= N_WIN * WIN + LOOK
    res = {}
    for name, ck in TEACHERS.items():
        t0 = time.time()
        m = build_teacher(bare(ck), torch.device("cpu"))
        A, B, E = read(m, rec, True, 0), read(m, rec, True, 1), read(m, rec, False, 0)
        km = MiniBatchKMeans(n_clusters=K, batch_size=8192, n_init=3, random_state=0, max_iter=200).fit(A)
        la, lb, le = km.predict(A), km.predict(B), km.predict(E)
        cos = lambda x, y: float(np.mean(np.sum(x * y, 1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))))
        r = dict(frames=int(len(A)), layer=LAYER, k=K,
                 changed_A_vs_B=float(np.mean(la != lb)), changed_A_vs_E=float(np.mean(la != le)),
                 ami_A_B=float(ami(la, lb)), ami_A_E=float(ami(la, le)),
                 cos_A_B=cos(A, B), cos_A_E=cos(A, E))
        # the eval-mode labels with their own codebook, for comparison with A's own codebook
        km_e = MiniBatchKMeans(n_clusters=K, batch_size=8192, n_init=3, random_state=0, max_iter=200).fit(E)
        r["ami_A_vs_E_own_codebooks"] = float(ami(la, km_e.predict(E)))
        km_a2 = MiniBatchKMeans(n_clusters=K, batch_size=8192, n_init=3, random_state=1, max_iter=200).fit(A)
        r["ami_A_codebook_seed0_vs_seed1"] = float(ami(la, km_a2.predict(A)))
        res[name] = r
        print(f"{name}: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})}"
              f"  ({time.time()-t0:.0f}s)", flush=True)
    out = ANA / "teacher_dropout_diagnostic.json"
    out.write_text(json.dumps(dict(recording=str(REC), windows=N_WIN, results=res), indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()

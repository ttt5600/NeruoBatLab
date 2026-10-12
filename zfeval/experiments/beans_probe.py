#!/usr/bin/env python
"""E10: frozen-encoder linear probe on the BEANS classification tasks.

    python beans_probe.py --task bats                     # default model list
    python beans_probe.py --task watkins --models run16_compute4x aves-base-bio

Protocol, and how it differs from AVES's published BEANS row:
  * splits: BEANS's own train/valid/test CSVs, made by BEANS's scripts (beans_fetch.sh) and checked
    against BEANS's MD5s, so the clips are exactly theirs.
  * audio: mono channel mean, resampled to 16 kHz, truncated to the task's max_duration as BEANS does.
    NOT zero-padded up to max_duration as BEANS does: padding corrupts short clips through these
    encoders' group-norm (the 2023 AVES bug, finding 034), so each clip is encoded at its own length.
  * representation: mean over frames, every layer, FROZEN. AVES's published numbers instead
    fine-tune the whole network per task (lr sweep x 50 epochs), so ours are not expected to match
    them; they compare all models under one identical, cheap protocol.
  * probe: StandardScaler + LogisticRegression(max_iter=4000) fit on train; the layer is chosen on
    VALID accuracy, and test accuracy is reported at that layer (never choosing on test).
  * interval: paired bootstrap over test clips of (model - baseline) at each model's own chosen
    layer; the observed difference is quoted, with the 2.5/97.5 percentiles.
"""
from __future__ import annotations
import argparse, gc, json, sys, time
from pathlib import Path
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore", message=".*torchcodec.*")
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from detection_variants import spec, load_model                            # noqa: E402

BEANS = Path.home() / "zf_labelset/external/beans"
FEAT = BEANS / "features"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
SR = 16_000
DEFAULT_MODELS = ["run11", "run16_compute4x", "run16_seed2", "run19_avesteacher", "run20_fsdonly",
                  "aves-base-core", "aves-base-bio", "aves-base-all",
                  "birdaves-biox-base", "birdaves-biox-large", "birdaves-bioxn-large"]
# AVES's fine-tuned aves-base-bio, arXiv 2210.14493 Table 2 -- context only, not a target
AVES_FINETUNED = dict(watkins=0.879, bats=0.748, cbi=0.598, humbugdb=0.810, dogs=0.950,
                      esc50=0.773, **{"speech-commands": 0.964})

_resamplers: dict = {}


def load_clip(path: Path, max_s: float) -> torch.Tensor:
    import torchaudio
    w, sr = torchaudio.load(str(path))
    w = w.mean(0, keepdim=True)
    if sr != SR:
        if sr not in _resamplers:
            _resamplers[sr] = torchaudio.transforms.Resample(sr, SR)
        w = _resamplers[sr](w)
    w = w[:, : int(max_s * SR)]
    if w.shape[-1] < 400:                                   # one receptive field
        w = torch.nn.functional.pad(w, (0, 400 - w.shape[-1]))
    return w


@torch.no_grad()
def embed(model, paths, max_s, device, n_layers, dim, tag):
    out = np.zeros((len(paths), n_layers, dim), dtype=np.float16)
    t0 = time.time()
    for i, p in enumerate(paths):
        feats, _ = model.extract_features(load_clip(p, max_s).to(device), None)
        assert len(feats) == n_layers, f"{tag}: {len(feats)} layers, expected {n_layers}"
        for l in range(n_layers):
            out[i, l] = feats[l].squeeze(0).mean(0).float().cpu().numpy()
        if (i + 1) % 250 == 0:
            r = (i + 1) / (time.time() - t0)
            print(f"    {tag} {i+1}/{len(paths)}  {r:.1f} clips/s  ~{(len(paths)-i-1)/r/60:.0f} min left",
                  flush=True)
    print(f"    {tag} embedded {len(paths)} clips in {time.time()-t0:.0f}s", flush=True)
    return out


def probe(Etr, ytr, Eva, yva, Ete, yte):
    """Per layer: fit on train, score valid and test. Returns (valid accs, test accs, test preds)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    va, te, preds = [], [], []
    for l in range(Etr.shape[1]):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000))
        clf.fit(Etr[:, l].astype(np.float32), ytr)
        va.append(float((clf.predict(Eva[:, l].astype(np.float32)) == yva).mean()))
        p = clf.predict(Ete[:, l].astype(np.float32))
        te.append(float((p == yte).mean())); preds.append(p)
    return va, te, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--baseline", default="run16_compute4x")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    cfg = {d["name"]: d for d in yaml.safe_load(open(BEANS / "datasets.yml"))}[a.task]
    assert cfg["type"] == "classification", f"{a.task} is {cfg['type']}; this script is classification only"
    import pandas as pd
    split = {}
    for s in ("train", "valid", "test"):
        df = pd.read_csv(BEANS / cfg[f"{s}_data"])
        labels = [str(x) for x in cfg["labels"]]
        y = np.array([labels.index(str(v)) for v in df["label"]])
        split[s] = ([BEANS / p for p in df["path"]], y)
    n = {s: len(split[s][1]) for s in split}
    maj = float(np.bincount(split["test"][1]).max() / n["test"])
    print(f"[{a.task}] train {n['train']}  valid {n['valid']}  test {n['test']}  classes "
          f"{cfg['num_labels']}  max {cfg['max_duration']} s  test majority {maj:.4f}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    FEAT.mkdir(parents=True, exist_ok=True); ANA.mkdir(parents=True, exist_ok=True)
    out_p = ANA / f"beans_{a.task}.json"
    out = json.loads(out_p.read_text()) if out_p.exists() else {}
    out.update(task=a.task, n=n, test_majority=maj, max_duration=cfg["max_duration"],
               aves_bio_finetuned_published=AVES_FINETUNED.get(a.task),
               protocol="frozen mean-pooled layers, own-length clips truncated to max_duration, "
                        "StandardScaler+LogisticRegression(max_iter=4000) on train, layer chosen on "
                        "valid, test accuracy at that layer")
    out.setdefault("models", {})

    preds = {}
    for tag in a.models:
        s = spec(tag)
        E, model = {}, None
        for sp in ("train", "valid", "test"):
            c = FEAT / f"{a.task}_{sp}_{tag}.npy"
            if c.exists():
                E[sp] = np.load(c)
                continue
            if model is None:
                model, _ = load_model(tag, s, device)
            E[sp] = embed(model, split[sp][0], cfg["max_duration"], device, s["n_layers"], s["dim"],
                          f"{tag}/{sp}")
            np.save(c, E[sp])
        del model; gc.collect()
        t0 = time.time()
        va, te, P = probe(E["train"], split["train"][1], E["valid"], split["valid"][1],
                          E["test"], split["test"][1])
        b = int(np.argmax(va))
        preds[tag] = P[b]
        prev = out["models"].get(tag, {})            # a rerun against another --baseline must reproduce
        if "test" in prev:                           # the stored score, and keeps its earlier vs_* keys
            assert prev["layer"] == b and abs(prev["test"] - te[b]) < 1e-6, \
                f"{tag}: rescored layer {b} test {te[b]} != stored layer {prev['layer']} test {prev['test']}"
        out["models"].setdefault(tag, {}).update(dict(layer=b, valid=va[b], test=te[b], valid_by_layer=va,
                                                      test_by_layer=te, best_test_any_layer=max(te)))
        print(f"  {tag:22s} layer {b:2d}  valid {va[b]:.4f}  TEST {te[b]:.4f}   "
              f"(best test at any layer {max(te):.4f}; probe {time.time()-t0:.0f}s)", flush=True)
        out_p.write_text(json.dumps(out, indent=1))

    yte = split["test"][1]
    if a.baseline in preds:
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(yte), size=(a.n_boot, len(yte)))
        cb = (preds[a.baseline] == yte)
        for tag in preds:
            if tag == a.baseline:
                continue
            c = (preds[tag] == yte)
            obs = float(c.mean() - cb.mean())
            d = c[idx].mean(1) - cb[idx].mean(1)
            lo, hi = np.percentile(d, [2.5, 97.5])
            out["models"][tag]["vs_" + a.baseline] = dict(observed=obs, ci95=[float(lo), float(hi)],
                                                         resolved=bool(lo > 0 or hi < 0))
            print(f"  {tag:22s} - {a.baseline}: {obs:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                  f"{'  *' if lo > 0 or hi < 0 else ''}", flush=True)
        out_p.write_text(json.dumps(out, indent=1))
    print(f"[out] {out_p}")


if __name__ == "__main__":
    main()

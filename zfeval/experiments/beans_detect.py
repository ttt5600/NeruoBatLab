#!/usr/bin/env python
"""E10: frozen-encoder linear probe on the BEANS detection tasks (dcase, enabirds, hiceas, gibbons).

    python beans_detect.py --task dcase                   # default model list (beans_probe.py's)
    python beans_detect.py --task hiceas --models run16_compute4x aves-base-bio

Protocol (beans_probe.py's, adapted to BEANS's detection setup):
  * windows and labels: BEANS's RecognitionDataset exactly -- each 60-s chunk is cut into windows of
    window_width every window_shift seconds; a window is positive for a class by BEANS's overlap rule
    (copied below, including the unknown-label mapping).
  * audio: the chunk resampled to 16 kHz, each window encoded ON ITS OWN, as BEANS feeds it. All
    windows of a task have the same length, so they are batched with no padding.
  * representation: mean over frames, every layer, frozen.
  * probe: per class, L2 logistic regression with sklearn's default objective (C=1: mean log-loss +
    |w|^2 / 2n, intercept free), fit for all classes at once with L-BFGS in torch -- one-vs-rest
    sklearn would take hours per model here. Features standardised on train.
  * metric: BEANS's mean average precision (metrics.AveragePrecision, reimplemented in numpy;
    a class with no positives counts as AP 0, as in BEANS). Layer chosen on valid mAP; test at it.
  * interval: paired bootstrap over test CHUNKS (overlapping windows of one chunk are not
    independent), observed difference quoted with 2.5/97.5 percentiles.
"""
from __future__ import annotations
import argparse, gc, json, sys, time, warnings
from pathlib import Path
import numpy as np
import torch
import yaml
warnings.filterwarnings("ignore", message=".*torchcodec.*")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from detection_variants import spec, load_model                            # noqa: E402
from beans_probe import BEANS, FEAT, ANA, SR, DEFAULT_MODELS               # noqa: E402

AVES_FINETUNED = dict(dcase=0.392, enabirds=0.555, hiceas=0.629, rfcx=0.130,
                      **{"hainan-gibbons": 0.284})                         # arXiv 2210.14493 Table 2
_resamplers: dict = {}


def windows(jsonl, labels, unknown, width, shift):
    """BEANS RecognitionDataset.__init__, verbatim logic. Returns [(chunk, path, st, ed)], Y."""
    label_to_id = {l: i for i, l in enumerate(labels)}
    xs, ys = [], []
    for c, line in enumerate(open(jsonl)):
        d = json.loads(line)
        for w in range(int((d["length"] - width) / shift) + 1):
            st, ed = w * shift, w * shift + width
            xs.append((c, BEANS / d["path"], st, ed))
            y = np.zeros(len(labels), dtype=np.int8)
            for a in d["annotations"]:
                if a["label"] not in label_to_id:
                    if unknown is None:
                        raise KeyError(f"Unknown label: {a['label']}")
                    k = label_to_id[unknown]
                else:
                    k = label_to_id[a["label"]]
                if (st <= a["st"] <= ed) or (st <= a["ed"] <= ed):
                    denom = min(ed - st, a["ed"] - a["st"])
                    if denom == 0:
                        continue
                    if (min(ed, a["ed"]) - max(st, a["st"])) / denom > .2:
                        y[k] = 1
                if a["st"] <= st and ed <= a["ed"]:
                    y[k] = 1
            ys.append(y)
    return xs, np.stack(ys)


def load_16k(path):
    import torchaudio
    w, sr = torchaudio.load(str(path))
    w = w.mean(0, keepdim=True)
    if sr != SR:
        if sr not in _resamplers:
            _resamplers[sr] = torchaudio.transforms.Resample(sr, SR)
        w = _resamplers[sr](w)
    return w[0]


@torch.no_grad()
def embed(model, xs, width, device, n_layers, dim, tag, batch=16):
    n_samp = int(round(width * SR))
    out = np.zeros((len(xs), n_layers, dim), dtype=np.float16)
    t0, cur, wav, buf, idx = time.time(), None, None, [], []

    def flush():
        feats, _ = model.extract_features(torch.stack(buf).to(device), None)
        assert len(feats) == n_layers, f"{tag}: {len(feats)} layers, expected {n_layers}"
        for l in range(n_layers):
            out[idx, l] = feats[l].mean(1).float().cpu().numpy()
        buf.clear(); idx.clear()

    for i, (_, path, st, ed) in enumerate(xs):
        if path != cur:
            cur, wav = path, load_16k(path)
        seg = wav[int(round(st * SR)): int(round(st * SR)) + n_samp]
        if seg.shape[0] < n_samp:                         # rounding at the chunk end: a few samples
            assert n_samp - seg.shape[0] <= SR // 100, f"{path} window {st}-{ed} short by {n_samp-seg.shape[0]}"
            seg = torch.nn.functional.pad(seg, (0, n_samp - seg.shape[0]))
        buf.append(seg); idx.append(i)
        if len(buf) == batch:
            flush()
        if (i + 1) % 5000 == 0:
            r = (i + 1) / (time.time() - t0)
            print(f"    {tag} {i+1}/{len(xs)}  {r:.0f} win/s  ~{(len(xs)-i-1)/r/60:.0f} min left", flush=True)
    if buf:
        flush()
    print(f"    {tag} embedded {len(xs)} windows in {time.time()-t0:.0f}s", flush=True)
    return out


def mean_ap(scores, Y):
    """BEANS metrics.AveragePrecision -> mean over classes; no positives -> AP 0."""
    aps = []
    for k in range(Y.shape[1]):
        o = np.argsort(-scores[:, k], kind="stable")
        t = Y[o, k].astype(np.float64)
        prec = np.cumsum(t) / np.arange(1, len(t) + 1)
        aps.append(prec[t > 0].sum() / max(t.sum(), 1))
    return float(np.mean(aps))


def fit_predict(Xtr, Ytr, Xev_list):
    """Per-class L2 logistic regression, sklearn C=1 objective, all classes jointly via L-BFGS."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    X = torch.from_numpy((Xtr - mu) / sd).double()
    Y = torch.from_numpy(Ytr).double()
    n, d = X.shape
    W = torch.zeros(d, Y.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(Y.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=500, tolerance_grad=1e-6, tolerance_change=1e-10,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = X @ W + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, Y, reduction="sum") / n \
            + (W ** 2).sum() / (2 * n)
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return [(torch.from_numpy((Xe - mu) / sd).double() @ W + b).numpy() for Xe in Xev_list]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--baseline", default="run16_compute4x")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16, help="windows per forward pass (all equal length)")
    a = ap.parse_args()

    cfg = {d["name"]: d for d in yaml.safe_load(open(BEANS / "datasets.yml"))}[a.task]
    assert cfg["type"] == "detection", f"{a.task} is {cfg['type']}"
    W_, S_ = cfg["window_width"], cfg["window_shift"]
    sp = {}
    for s in ("train", "valid", "test"):
        sp[s] = windows(BEANS / cfg[f"{s}_data"], cfg["labels"], cfg["unknown_label"], W_, S_)
    n = {s: len(sp[s][0]) for s in sp}
    pos = {s: sp[s][1].mean(0).round(4).tolist() for s in sp}
    print(f"[{a.task}] windows train {n['train']}  valid {n['valid']}  test {n['test']}  classes "
          f"{cfg['num_labels']}  width {W_} s shift {S_} s", flush=True)
    print(f"  test positives per class: {sp['test'][1].sum(0).tolist()}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    FEAT.mkdir(parents=True, exist_ok=True)
    fs = a.task.replace("-", "_")
    out_p = ANA / f"beans_{fs}.json"
    out = json.loads(out_p.read_text()) if out_p.exists() else {}
    out.update(task=a.task, n_windows=n, positive_rate=pos, aves_bio_finetuned_published=AVES_FINETUNED.get(a.task),
               protocol="frozen mean-pooled layers, each BEANS window encoded alone at 16 kHz; per-class L2 "
                        "logistic regression (C=1) on train; layer chosen on valid mAP; BEANS mAP on test")
    out.setdefault("models", {})

    scores = {}
    for tag in a.models:
        s = spec(tag)
        E, model = {}, None
        for split in ("train", "valid", "test"):
            c = FEAT / f"{fs}_{split}_{tag}.npy"
            if c.exists():
                E[split] = np.load(c); continue
            if model is None:
                model, _ = load_model(tag, s, device)
            E[split] = embed(model, sp[split][0], W_, device, s["n_layers"], s["dim"], f"{tag}/{split}",
                             batch=a.batch)
            np.save(c, E[split])
        del model; gc.collect()
        t0 = time.time()
        va, te, S = [], [], []
        for l in range(s["n_layers"]):
            sv, st_ = fit_predict(E["train"][:, l].astype(np.float32), sp["train"][1],
                                  [E["valid"][:, l].astype(np.float32), E["test"][:, l].astype(np.float32)])
            va.append(mean_ap(sv, sp["valid"][1])); te.append(mean_ap(st_, sp["test"][1])); S.append(st_)
        b = int(np.argmax(va)); scores[tag] = S[b]
        prev = out["models"].get(tag, {})            # a rerun against another --baseline must reproduce
        if "test" in prev:                           # the stored score, and keeps its earlier vs_* keys
            assert prev["layer"] == b and abs(prev["test"] - te[b]) < 1e-6, \
                f"{tag}: rescored layer {b} test {te[b]} != stored layer {prev['layer']} test {prev['test']}"
        out["models"].setdefault(tag, {}).update(dict(layer=b, valid=va[b], test=te[b], valid_by_layer=va,
                                                      test_by_layer=te, best_test_any_layer=max(te)))
        print(f"  {tag:22s} layer {b:2d}  valid mAP {va[b]:.4f}  TEST mAP {te[b]:.4f}   "
              f"(best test at any layer {max(te):.4f}; probe {time.time()-t0:.0f}s)", flush=True)
        out_p.write_text(json.dumps(out, indent=1))

    if a.baseline in scores:
        Yte = sp["test"][1]
        chunk = np.array([x[0] for x in sp["test"][0]])
        ids = np.unique(chunk); members = {c: np.where(chunk == c)[0] for c in ids}
        rng = np.random.default_rng(0)
        boots = [np.concatenate([members[c] for c in rng.choice(ids, len(ids))]) for _ in range(a.n_boot)]
        base = mean_ap(scores[a.baseline], Yte)
        bb = np.array([mean_ap(scores[a.baseline][i], Yte[i]) for i in boots])
        for tag in scores:
            if tag == a.baseline:
                continue
            obs = mean_ap(scores[tag], Yte) - base
            d = np.array([mean_ap(scores[tag][i], Yte[i]) for i in boots]) - bb
            lo, hi = np.percentile(d, [2.5, 97.5])
            out["models"][tag]["vs_" + a.baseline] = dict(observed=float(obs), ci95=[float(lo), float(hi)],
                                                         resolved=bool(lo > 0 or hi < 0), unit="test chunk")
            print(f"  {tag:22s} - {a.baseline}: {obs:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                  f"{'  *' if lo > 0 or hi < 0 else ''}", flush=True)
        out_p.write_text(json.dumps(out, indent=1))
    print(f"[out] {out_p}")


if __name__ == "__main__":
    main()

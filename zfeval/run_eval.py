#!/usr/bin/env python3
"""zfeval — run the whole evaluation suite against a checkpoint.

Three stages, because only the first needs a GPU:

  extract   load the checkpoint (aborting if it does not fully load), cut every dataset's
            features, and cache them with a provenance record          [GPU]
  analyze   probes, baselines, controls, geometry, clustering, UMAP,
            event-level onset/offset, and the report                    [CPU]
  compare   diff two runs, so retraining produces a regression view rather than a surprise

  python run_eval.py extract --config config/datasets.yaml --ckpt <ckpt> --out runs/run14
  python run_eval.py analyze --out runs/run14
  python run_eval.py compare --run runs/run14 --baseline runs/run11
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from zfeval import datasets as ds, features as ft, metrics as mx, splits as sp
from zfeval import controls as ctl, embed as emb, events as ev, report as rp, validate as va


# ------------------------------------------------------------------ extract (GPU)
def cmd_extract(a):
    import torch, soundfile as sf
    reg, settings = ds.load_registry(a.config)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt_meta = ft.load_encoder(a.ckpt, dev, a.num_classes, a.hubert_dir)
    print(f"[ckpt] epoch={ckpt_meta['epoch']} step={ckpt_meta['global_step']} "
          f"fingerprint={ckpt_meta['encoder_fingerprint']}", flush=True)
    (out / "checkpoint.json").write_text(json.dumps(ckpt_meta, indent=2))
    layers = settings.get("layers", list(range(12)))
    flayers = settings.get("frame_layers", [0, 3, 6, 9])
    scope = settings.get("normalize_scope", "context")
    ctx = settings.get("normalize_context_sec", 20)

    for name, d in reg.items():
        if isinstance(d, ds.WindowSet):
            rows, meta = d.load()
            print(f"[{name}] {meta['n']} labeled windows "
                  f"({meta['dropped_unlabeled']} unlabeled excluded)", flush=True)
            by = {}
            for r in rows:
                by.setdefault(r[d.group_field], []).append(r)
            X, Xen, Xmel, y, grp, ids, starts_out = [], [], [], [], [], [], []
            for k, (g, rs) in enumerate(sorted(by.items())):
                wav, sr = sf.read(str(Path(d.audio_dir).expanduser() / f"{g}.wav"),
                                  dtype="float32", always_2d=True)
                rec = wav.mean(1)
                for r in rs:
                    f = ft.SR / float(r[d.sr_field])
                    s = int(round(int(r[d.start_field]) * f))
                    win = int(round(float(r[d.dur_field]) * ft.SR))
                    s = int(np.clip(s, 0, len(rec) - win))
                    xw, en = ft.window_features(model, rec, [s], win, dev, scope, ctx, layers)
                    X.append(xw[0]); Xen.append(en[0])
                    Xmel.append(ft.mel_features(rec[s:s + win]))
                    y.append(int(r[d.label_field])); grp.append(g); ids.append(r[d.id_field])
                    starts_out.append(s)
                print(f"  [{name}] {k+1}/{len(by)} {g}", flush=True)
            np.savez_compressed(out / f"windows_{name}.npz",
                                X=np.stack(X), Xen=np.array(Xen)[:, None],
                                Xmel=np.stack(Xmel), y=np.array(y),
                                grp=np.array(grp), ids=np.array(ids), layers=np.array(layers),
                                starts=np.array(starts_out))
            (out / f"windows_{name}.meta.json").write_text(json.dumps(meta, indent=2))
        else:
            rec, merged, meta = d.load()
            print(f"[{name}] {meta['span_sec']:.1f}s, {meta['n_merged']} events "
                  f"(dropped {meta['n_nan']} NaN, {meta['n_nonpositive']} non-positive)", flush=True)
            F, en, nF = ft.frame_features(model, rec, dev, flayers)
            voiced = np.zeros(len(rec), bool)
            for s0, e0 in merged:
                voiced[int(s0 * ft.SR):int(e0 * ft.SR)] = True
            yf = np.array([voiced[i * ft.HOP:(i + 1) * ft.HOP].any() for i in range(nF)])
            blocks = sp.contiguous_blocks(nF, int(d.block_sec * ft.SR / ft.HOP),
                                          max(2, int(meta["span_sec"] / d.block_sec)))
            print(f"  [{name}] {nF} frames, {yf.mean()*100:.1f}% voiced, "
                  f"{len(np.unique(blocks))} blocks", flush=True)
            np.savez_compressed(out / f"frames_{name}.npz", yf=yf, energy=en, blocks=blocks,
                                merged=merged, layers=np.array(flayers),
                                **{f"L{l}": F[l] for l in flayers})
            (out / f"frames_{name}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] features in {out}", flush=True)


# ------------------------------------------------------------------ analyze (CPU)
def cmd_analyze(a):
    warnings.filterwarnings("ignore")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    out = Path(a.out)
    ckpt_meta = json.loads((out / "checkpoint.json").read_text()) if (out / "checkpoint.json").exists() else {}
    settings = ds.load_registry(a.config)[1] if a.config else {}
    n_splits = settings.get("n_splits", 5); seed = settings.get("seed", 0)
    R = rp.Report(a.name or out.name, out, ckpt_meta)
    scores, baselines, ops, controls, preds = [], {}, {}, {}, {}

    wsets = sorted(out.glob("windows_*.npz"))
    loaded = {p.stem.replace("windows_", ""): np.load(p, allow_pickle=True) for p in wsets}

    _UNSET = object()

    def fit(Xtr, ytr, gtr, Xte=None, C=1.0, cv=None, cvgroups=_UNSET):
        # `cvgroups is _UNSET` not `==`: cvgroups is usually an array, and `array == str`
        # returns an array, which then explodes on truthiness.
        cv = cv or sp.make_cv("grouped", n_splits, seed)
        g = gtr if cvgroups is _UNSET else cvgroups
        p_in = cross_val_predict(LogisticRegression(max_iter=4000, C=C), Xtr, ytr,
                                 groups=g, cv=cv, method="predict_proba")[:, 1]
        p_out = (LogisticRegression(max_iter=4000, C=C).fit(Xtr, ytr)
                 .predict_proba(Xte)[:, 1]) if Xte is not None else None
        return p_in, p_out

    for name, d in loaded.items():
        X, y, grp, Xen, Xmel = d["X"], d["y"], d["grp"].astype(str), d["Xen"][:, 0], d["Xmel"]
        layers = list(d["layers"])
        va.finite(X, f"{name} features")
        starts = d["starts"] if "starts" in d.files else None
        cv, cvg, cv_desc = sp.choose_cv(grp, starts, n_splits, seed)
        print(f"[{name}] split: {cv_desc}", flush=True)
        best_l, best_auc, best_p = None, -1, None
        for li, l in enumerate(layers):
            p, _ = fit(X[:, li], y, grp, cv=cv, cvgroups=cvg)
            s = mx.score(y, p, split=f"{name} [{cv_desc}]", layer=f"L{l}")
            scores.append(s)
            if s.auc > best_auc:
                best_l, best_auc, best_p = li, s.auc, p
        preds[name] = dict(p=best_p, y=y, grp=grp, layer=layers[best_l], en=Xen)
        print(f"[{name}] best layer L{layers[best_l]} AUC {best_auc:.4f}", flush=True)

        p_mel, _ = fit(Xmel, y, grp, C=0.03, cv=cv, cvgroups=cvg)
        p_en, _ = fit(Xen[:, None], y, grp, cv=cv, cvgroups=cvg)
        baselines[name] = dict(model=best_auc,
                               logmel=mx.score(y, p_mel, name).auc,
                               energy=mx.score(y, p_en, name).auc)
        ops[name] = mx.operating_points(y, best_p)
        try:
            controls[f"{name}:shuffled_label"] = ctl.shuffled_label(
                X[:, best_l], y, cvg if cvg is not None else grp, cv)
        except ctl.ControlFailed as e:
            controls[f"{name}:shuffled_label"] = dict(passed=False, error=str(e))
        controls[f"{name}:per_group"] = ctl.per_group(y, best_p, grp)
        controls[f"{name}:loudness_stratified"] = ctl.loudness_stratified(y, best_p, Xen)
        try:
            controls[f"{name}:loudness_matched_pairs"] = ctl.loudness_matched_pairs(y, best_p, Xen)
        except ctl.ControlFailed as e:
            controls[f"{name}:loudness_matched_pairs"] = dict(matching_unbiased=False, error=str(e))

    # cross-dataset: train on each, test on every other
    for tr_name, tr in loaded.items():
        for te_name, te in loaded.items():
            if tr_name == te_name:
                continue
            li = int(np.where(np.array(list(tr["layers"])) == preds[tr_name]["layer"])[0][0])
            tcv, tcvg, _ = sp.choose_cv(tr["grp"].astype(str),
                                        tr["starts"] if "starts" in tr.files else None,
                                        n_splits, seed)
            _, p_out = fit(tr["X"][:, li], tr["y"], tr["grp"].astype(str), te["X"][:, li],
                           cv=tcv, cvgroups=tcvg)
            scores.append(mx.score(te["y"], p_out, split=f"{tr_name} -> {te_name} (held out)",
                                   layer=f"L{preds[tr_name]['layer']}"))
            ops[f"{tr_name} -> {te_name}"] = mx.operating_points(te["y"], p_out)

    # geometry + clustering on the largest window set
    big = max(loaded, key=lambda k: len(loaded[k]["y"]))
    d = loaded[big]
    cv, _, _ = sp.choose_cv(d["grp"].astype(str),
                            d["starts"] if "starts" in d.files else None, n_splits, seed)
    R.add("geometry", emb.layer_geometry(d["X"], d["y"], d["grp"].astype(str), cv,
                                         layers=range(d["X"].shape[1])))
    en = d["Xen"][:, 0]
    R.add("dominant_structure", emb.dominant_structure(
        d["X"][:, 0], dict(label=d["y"], recording=d["grp"].astype(str),
                           loudness=np.digitize(en, np.percentile(en, [20, 40, 60, 80])))))
    R.add("loudness_dependence", emb.loudness_dependence(d["X"][:, 0], d["y"], en,
                                                         d["grp"].astype(str), cv))

    # frame / event level
    GRID = [dict(thr=t, min_dur=md, merge_gap=mg, smooth=sm, thr_low=tl)
            for t in (0.2, 0.3, 0.5, 0.7) for md in (1, 2, 3, 4)
            for mg in (0, 1, 2, 4) for sm in (1, 3, 5) for tl in (None, 0.2, 0.3)]
    evres = {}
    for p in sorted(out.glob("frames_*.npz")):
        nm = p.stem.replace("frames_", "")
        f = np.load(p, allow_pickle=True)
        yf, blocks, merged = f["yf"], f["blocks"], f["merged"]
        true = (merged * ft.SR / ft.HOP).astype(int)
        true = np.stack([true[:, 0], np.maximum(true[:, 1], true[:, 0] + 1)], 1)
        feats = {f"L{l}": f[f"L{l}"] for l in f["layers"]}
        feats["energy"] = ft.frame_energy_features(f["energy"])
        for k, M in feats.items():
            pp = np.zeros(len(yf))
            for b in np.unique(blocks):
                te = blocks == b
                pp[te] = (LogisticRegression(max_iter=3000)
                          .fit(M[~te].astype(np.float32), yf[~te])
                          .predict_proba(M[te].astype(np.float32))[:, 1])
            scores.append(mx.score(yf, pp, split=f"{nm} frames", layer=k))
            evres[f"{nm}:{k}"] = ev.tune_decoder(pp, true, blocks, GRID)
            evres[f"{nm}:{k}"]["tolerance"] = ev.tolerance_sweep(
                ev.decode(pp, **evres[f"{nm}:{k}"]["per_block"][0]["params"]), true)
            print(f"[{nm}:{k}] collar-50 F1 {evres[f'{nm}:{k}']['overall']['collar_f1']:.3f}",
                  flush=True)
    if evres:
        R.add("events", evres)

    R.add("scores", [s.to_dict() for s in scores]).add("baselines", baselines)
    R.add("operating_points", ops).add("controls", controls)
    p = R.save()
    print(f"\n[done] {p}\n[done] {out/'SUMMARY.md'}", flush=True)
    print(R.markdown()[:2500])


def cmd_compare(a):
    print(rp.compare(Path(a.run) / "report.json", Path(a.baseline) / "report.json",
                     Path(a.run) / "COMPARISON.md"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.set_defaults(fn=cmd_extract)
    e.add_argument("--config", required=True); e.add_argument("--ckpt", required=True)
    e.add_argument("--out", required=True); e.add_argument("--num-classes", type=int, default=100)
    e.add_argument("--hubert-dir", default=None,
                   help="directory containing lightning_modules.py")
    n = sub.add_parser("analyze"); n.set_defaults(fn=cmd_analyze)
    n.add_argument("--out", required=True); n.add_argument("--config", default=None)
    n.add_argument("--name", default=None)
    c = sub.add_parser("compare"); c.set_defaults(fn=cmd_compare)
    c.add_argument("--run", required=True); c.add_argument("--baseline", required=True)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Integrity audit of the 11-class call-type cohort, and whether its defects move any board score.

Cohort under audit: calltype11.collect() over datasets/11905533/{Adult,Chick}Vocalizations, filtered to KEEP11
-> 3,412 clips, 48 birds (lower-cased ids). Probe: aves_calltype.cv_acc (leave-birds-out StratifiedGroupKFold(5,
shuffle, seed 0), StandardScaler + LogisticRegression). Embeddings: CPU only (features/ct11_<tag>_cpuemb.npy; the
MPS caches are wrong for 2-8 frame clips). Board layers: analysis/calltype_cpu_rescore.json best_layer_cpu.

What it checks, on every .wav in both folders (3,433):
  * duplicates by md5 of the file bytes AND by md5 of the decoded samples (raw int16 frames, and the channel mean the
    encoder sees), so audio-identical-but-not-byte-identical copies cannot hide behind a different header;
  * label conflicts (one recording, two call types), and placeholder-bird duplicates that the bird-grouped split can
    put in different folds (train/test leakage);
  * files the filename parser drops or maps to a type outside KEEP11, names carrying two call types, bird-id
    spellings that differ only by case, and chick bird ids filed in the adult folder;
  * crops: a clip whose samples sit inside another clip (rolling-hash sweep with 5 probe windows per file, every hit
    verified sample-by-sample on both channels; a sweep, not a proof).

Variant cohorts (folds are RECOMPUTED by StratifiedGroupKFold on each new cohort unless the name says fixedfolds):
  board          the 3,412 as scored today (reproduction guard: must equal calltype_cpu_rescore.json exactly)
  i_dedup        one copy per byte-identical group (chick-folder copy preferred, then first by name); both members of
                 the conflicting-label recording dropped
  i_fixedfolds   i_dedup's rows scored on the BOARD's fold assignment -- isolates removing rows from re-folding
  ii_merge       i_dedup with every Unknown* placeholder id merged into one group (primary variant; see below)
  ii_drop        i_dedup with every Unknown* clip dropped (the released 8-class evaluation's convention)
  iii_nocrops    ii_merge minus the shorter member of every verified crop/overlap pair (sensitivity only)

Fold-seed sweep: every variant above except i_fixedfolds re-folds, and a re-fold alone moves a model by ~0.01, so
board vs variant on one split mixes the cohort change with split noise. The sweep scores board / i_dedup / ii_merge /
ii_drop on StratifiedGroupKFold seeds 0..N-1 at the board layer and L3 (seed 0 == cv_acc, checked) and reports means,
per-seed bird-bootstrap contrasts, and ranking flips of the seed-mean.

    python zfeval/experiments/cohort_audit.py [--workers 6] [--seeds 10] [--no-crops] [--no-probe-dropped]
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys, time, warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")
from calltype11 import DIRS, KEEP11, CANON, parse, collect                       # noqa: E402
from aves_calltype import cv_acc, bird_bootstrap                                  # noqa: E402

FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
OUT = ANA / "cohort_audit.json"
RESCORE = ANA / "calltype_cpu_rescore.json"
MODELS = ["aves-base-bio", "birdaves-biox-base", "aves-base-all", "birdaves-bioxn-large", "run16_compute4x",
          "run16_seed2", "run16_seed3", "run19_avesteacher", "daptv4two_2e4s2_step15000",
          "daptv3aves_2e4s2_step15000", "run11"]
CONTRASTS = [("E20 - aves-base-bio", "daptv3aves_2e4s2_step15000", "aves-base-bio"),
             ("E23 - aves-base-bio", "daptv4two_2e4s2_step15000", "aves-base-bio"),
             ("bioxn-large - biox-base", "birdaves-bioxn-large", "birdaves-biox-base")]
NFOLD, SEED = 5, 0
# word stems of the 11 types as they appear in VT tokens; a token matching two types is ambiguous
STEMS = {"Ag": r"Ag", "Be": r"Beg", "DC": r"DC", "Di": r"Dis", "LT": r"LT", "Ne": r"Ne",
         "So": r"So", "Te": r"Te", "Th": r"Th", "Tu": r"Tu", "Wh": r"Wh"}


# ------------------------------------------------------------------------------------------------ inventory
def read_pcm(path):
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="int16", always_2d=True)
    return x, sr


def inventory():
    files = []
    for d in DIRS:
        for f in sorted(d.iterdir()):
            if f.suffix.lower() != ".wav":
                continue
            b = f.read_bytes()
            x, sr = read_pcm(f)
            mix = np.round(x.astype(np.float64).mean(1) * 2).astype(np.int64)    # exact channel mean x2
            p = parse(f.name)
            t = CANON.get(p[2].lower(), p[2]) if p else None
            files.append(dict(dir=d.name, name=f.name, path=str(f), size=len(b),
                              md5=hashlib.md5(b).hexdigest(),
                              pcm=hashlib.md5(x.tobytes() + str(sr).encode()).hexdigest(),
                              mix=hashlib.md5(mix.tobytes()).hexdigest(),
                              sr=int(sr), ch=int(x.shape[1]), n=int(x.shape[0]),
                              bird=(p[0] if p else None), date=(p[1] if p else None), type=t))
    return files


def vt_token(name):
    m = re.match(r"^[^_-]+[_-]\d+[_-]?(.*)\.wav$", name)
    return re.sub(r"[-_]\d+[A-Za-z_\d]*$", "", m.group(1)) if m else None


# ------------------------------------------------------------------------------------------------ crops
def crop_sweep(files, W=32, n_probe=5, min_overlap=1000):
    """Pairs where a clip's samples sit inside another clip (or the two overlap), verified on both channels."""
    X = [read_pcm(f["path"])[0] for f in files]
    pw = np.array([pow(1000003, W - 1 - k, 2 ** 64) for k in range(W)], dtype=np.uint64)

    def hashes(c):
        v = np.lib.stride_tricks.sliding_window_view((c.astype(np.int64) + 40000).astype(np.uint64), W)
        return (v * pw).sum(1)                       # wraps mod 2^64

    probes = {}                                       # hash -> list of (file, pos)
    for i, x in enumerate(X):
        c = x[:, 0]
        if len(c) < 4 * W:
            continue
        h = hashes(c)
        for seg in np.array_split(np.arange(len(h)), n_probe):
            p = int(seg[np.argmax(np.abs(c[seg].astype(np.int32)))])
            if np.ptp(c[p:p + W]) == 0:              # silence or a clipped run matches everything
                continue
            probes.setdefault(int(h[p]), []).append((i, p))
    keys = np.array(sorted(probes), dtype=np.uint64)
    cand = defaultdict(set)
    for j, x in enumerate(X):
        if len(x) < W:
            continue
        h = hashes(x[:, 0])
        for q in np.where(np.isin(h, keys))[0]:
            for i, p in probes[int(h[q])]:
                if i != j and files[i]["md5"] != files[j]["md5"]:
                    cand[(i, j)].add(int(q) - p)     # x_j[t + s] == x_i[t]
    pairs = {}
    for (i, j), shifts in cand.items():
        a, b = X[i], X[j]
        for s in shifts:
            lo, hi = max(0, -s), min(len(a), len(b) - s)
            if hi - lo < min_overlap:
                continue
            seg_a, seg_b = a[lo:hi], b[lo + s:hi + s]
            if not np.array_equal(seg_a, seg_b) or not np.any(seg_a):
                continue
            k = tuple(sorted((i, j)))
            if k not in pairs or pairs[k]["overlap"] < hi - lo:
                pairs[k] = dict(a=files[k[0]]["dir"][:5] + "/" + files[k[0]]["name"],
                                b=files[k[1]]["dir"][:5] + "/" + files[k[1]]["name"],
                                ia=k[0], ib=k[1], n_a=len(X[k[0]]), n_b=len(X[k[1]]), overlap=int(hi - lo),
                                contained=bool(hi - lo == min(len(a), len(b))))
    return sorted(pairs.values(), key=lambda d: (d["a"], d["b"]))


# ------------------------------------------------------------------------------------------------ scoring
def cv_fixed(X, y, fold, n_classes):
    """cv_acc's estimator on a GIVEN fold assignment (for the fixed-fold control)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    P = np.zeros((len(y), n_classes))
    for k in np.unique(fold):
        te, tr = fold == k, fold != k
        est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000)).fit(X[tr], y[tr])
        P[np.ix_(te, est.classes_)] = est.predict_proba(X[te])
    return float((P.argmax(1) == y).mean()), P


def score_task(args):
    tag, cname, rows, y, groups, fold, keep_layers = args
    E = np.load(FEAT / f"ct11_{tag}_cpuemb.npy", mmap_mode="r")
    accs, Ps = [], {}
    for l in range(E.shape[1]):
        X = np.ascontiguousarray(E[rows, l])
        if fold is None:
            a, _, P = cv_acc(X, y, groups, return_proba=True)
        else:
            a, P = cv_fixed(X, y, fold, int(y.max()) + 1)
        accs.append(a)
        Ps[l] = P
    b = int(np.argmax(accs))
    keep = {l: Ps[l] for l in set(keep_layers) | {b}}
    return tag, cname, accs, keep


def board_folds(y, groups, seed=SEED):
    from sklearn.model_selection import StratifiedGroupKFold
    fold = np.full(len(y), -1)
    for k, (_, te) in enumerate(StratifiedGroupKFold(NFOLD, shuffle=True, random_state=seed)
                                .split(np.zeros((len(y), 1)), y, groups)):
        fold[te] = k
    return fold


def sweep_task(args):
    """One model/cohort/layer over several fold seeds (cv_acc's estimator on StratifiedGroupKFold(seed))."""
    tag, cname, kind, layer, rows, y, groups, seeds, keepP = args
    E = np.load(FEAT / f"ct11_{tag}_cpuemb.npy", mmap_mode="r")
    X = np.ascontiguousarray(E[rows, layer])
    accs, Ps = [], []
    for sd in seeds:
        acc, P = cv_fixed(X, y, board_folds(y, groups, sd), int(y.max()) + 1)
        accs.append(acc)
        if keepP:
            Ps.append(P)
    return tag, cname, kind, accs, Ps


def observed_bootstrap(y, Pa, Pb, groups):
    r = bird_bootstrap(y, Pa, Pb, groups, n=2000, seed=0)
    obs = float((Pa.argmax(1) == y).mean() - (Pb.argmax(1) == y).mean())
    return dict(observed=obs, lo=r["lo"], hi=r["hi"], resample_mean=r["delta"], verdict=r["verdict"],
                n_groups=int(len(np.unique(groups))), n_clips=int(len(y)))


# ------------------------------------------------------------------------------------------------ dropped clips
def probe_dropped(files, coh_rows, y, groups, classes, best, paths, tags=("aves-base-bio", "run16_compute4x")):
    """Embed the parser-dropped non-duplicate clips and ask a probe trained WITHOUT their bird what they are."""
    import torch
    from detection_variants import spec, load_model
    from aves_variants_calltype import embed_all
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    torch.set_num_threads(8)
    res = {}
    for tag in tags:
        s = spec(tag)
        model, _ = load_model(tag, s, "cpu"); model.eval()
        with torch.no_grad():
            Ed = embed_all(model, [Path(p) for p in paths], "cpu", s["n_layers"], s["dim"], tag=tag)
        del model
        E = np.load(FEAT / f"ct11_{tag}_cpuemb.npy", mmap_mode="r")
        l = best[tag]
        X = np.ascontiguousarray(E[coh_rows, l])
        out = []
        for p, e in zip(paths, Ed[:, l]):
            bird = parse(Path(p).name)
            bird = bird[0].lower() if bird else re.match(r"^([A-Za-z]+\d+[A-Za-z]*)", Path(p).name).group(1).lower()
            tr = groups != bird
            est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000)).fit(X[tr], y[tr])
            pr = est.predict_proba(e[None])[0]
            o = np.argsort(pr)[::-1][:3]
            out.append(dict(file=Path(p).name, held_out_bird=bird,
                            top3=[(str(classes[est.classes_[k]]), round(float(pr[k]), 3)) for k in o]))
        res[f"{tag}_L{l}"] = out
    return res


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--no-crops", action="store_true")
    ap.add_argument("--no-probe-dropped", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    out = dict(script="zfeval/experiments/cohort_audit.py", date=time.strftime("%Y-%m-%d"),
               protocol="cv_acc leave-birds-out StratifiedGroupKFold(5, shuffle, seed 0); StandardScaler + "
                        "LogisticRegression(max_iter=4000); CPU embeddings; folds RECOMPUTED on every variant "
                        "cohort except i_fixedfolds",
               sklearn=__import__("sklearn").__version__)

    # ---------------- inventory
    files = inventory()
    by_path = {f["path"]: i for i, f in enumerate(files)}
    rows = [r for r in collect() if r[3] in KEEP11]
    coh = [by_path[str(r[0])] for r in rows]                     # cohort row -> file index
    birds = np.array([r[1].lower() for r in rows])
    tt = np.array([r[3] for r in rows]); classes = sorted(set(tt))
    y = np.array([classes.index(t) for t in tt])
    assert len(y) == 3412 and len(set(birds)) == 48
    fold0 = board_folds(y, birds)
    pos = {fi: k for k, fi in enumerate(coh)}
    print(f"[inventory] {len(files)} wav files; cohort {len(y)} clips, {len(set(birds))} birds "
          f"({time.time()-t0:.0f}s)", flush=True)

    dup = {}
    for key in ("md5", "pcm", "mix"):
        g = defaultdict(list)
        for i, f in enumerate(files):
            g[f[key]].append(i)
        dup[key] = sorted([sorted(v) for v in g.values() if len(v) > 1])
    assert dup["md5"] == dup["pcm"] == dup["mix"], "decoded-audio duplicates differ from byte duplicates"
    groups_all = dup["md5"]

    def desc(i):
        f = files[i]
        k = pos.get(i)
        return dict(dir=f["dir"], name=f["name"], bird=f["bird"], type=f["type"], in_cohort=k is not None,
                    board_fold=(int(fold0[k]) if k is not None else None))

    dgroups = []
    for v in groups_all:
        m = [desc(i) for i in v]
        inc = [d for d in m if d["in_cohort"]]
        dgroups.append(dict(
            members=m, n_in_cohort=len(inc),
            labels=sorted({d["type"] for d in inc}),
            birds=sorted({d["bird"].lower() for d in inc}),
            label_conflict=len({d["type"] for d in inc}) > 1,
            cross_bird=len({d["bird"].lower() for d in inc}) > 1,
            cross_fold=len({d["board_fold"] for d in inc}) > 1,
            kind=("adult/chick folder copy" if {d["dir"] for d in m} == set(d.name for d in DIRS) else
                  "parser-dropped twin" if len(inc) < 2 else "same folder")))
    in_groups = [g for g in dgroups if g["n_in_cohort"] > 1]
    extra = sum(g["n_in_cohort"] - 1 for g in in_groups)
    out["duplicates"] = dict(
        hashes_agree="md5(bytes) == md5(int16 frames + sr) == md5(channel mean): identical grouping",
        n_files=len(files), n_distinct_files=len(files) - sum(len(v) - 1 for v in groups_all),
        n_groups_all=len(groups_all), n_groups_in_cohort=len(in_groups), n_extra_copies_in_cohort=extra,
        n_distinct_in_cohort=int(len(y) - extra),
        n_cross_fold_groups=sum(g["cross_fold"] for g in in_groups),
        n_cross_bird_groups=sum(g["cross_bird"] for g in in_groups),
        n_label_conflicts=sum(g["label_conflict"] for g in in_groups),
        by_kind={k: sum(g["kind"] == k for g in dgroups) for k in sorted({g["kind"] for g in dgroups})},
        groups=dgroups)
    print(f"[dups] {len(groups_all)} byte-identical groups over all files; {len(in_groups)} inside the cohort "
          f"-> {extra} extra copies, {len(y)-extra} distinct; cross-fold {out['duplicates']['n_cross_fold_groups']}"
          f", label conflicts {out['duplicates']['n_label_conflicts']}", flush=True)

    # ---------------- parser drops / label anomalies
    dropped, ambiguous, tokens = [], [], defaultdict(lambda: defaultdict(int))
    md5_in_cohort = {files[i]["md5"] for i in coh}
    for i, f in enumerate(files):
        tok = vt_token(f["name"])
        if f["type"] is None:
            dropped.append(dict(dir=f["dir"], name=f["name"], reason="filename does not parse (bird id has no digits)",
                                byte_copy_in_cohort=f["md5"] in md5_in_cohort))
        elif f["type"] not in KEEP11:
            dropped.append(dict(dir=f["dir"], name=f["name"], reason=f"parsed type '{f['type']}' not in KEEP11",
                                date_field=re.match(r"^[^_-]+[_-](\d+)", f["name"]).group(1),
                                byte_copy_in_cohort=f["md5"] in md5_in_cohort))
        if tok:
            tokens[tok][f["type"]] += 1
            hit = sorted({t for t, s in STEMS.items() if re.search(s, tok)})
            if len(hit) > 1:
                ambiguous.append(dict(dir=f["dir"], name=f["name"], token=tok, types_in_name=hit,
                                      assigned=f["type"], in_cohort=i in pos))
    spell = defaultdict(set)
    for f in files:
        if f["bird"]:
            spell[f["bird"].lower()].add(f["bird"])
    chick_birds = {f["bird"].lower() for f in files if f["bird"] and f["dir"] == "ChickVocalizations"}
    chick_in_adult = defaultdict(lambda: defaultdict(int))
    for f in files:
        if f["dir"] == "AdultVocalizations" and f["bird"] and f["bird"].lower() in chick_birds:
            chick_in_adult[f["bird"]][f["type"]] += 1
    out["parser"] = dict(
        dropped=dropped, n_dropped=len(dropped),
        n_dropped_not_copies=sum(not d["byte_copy_in_cohort"] for d in dropped),
        ambiguous_names=ambiguous,
        token_to_type={k: dict(v) for k, v in sorted(tokens.items())},
        bird_id_case_variants={k: sorted(v) for k, v in spell.items() if len(v) > 1},
        placeholder_birds={b: int((birds == b).sum()) for b in sorted(set(birds)) if b.startswith("unknown")},
        chick_ids_filed_in_adult_folder={k: dict(v) for k, v in chick_in_adult.items()},
        wc_note="'WC' is not one of the 11 codes in datasets/11905533/Library_notes.pdf (Ag Be DC Di LT Ne So Te Th "
                "Tu Wh); the same bird (LblBla4548) uses 'Wh' for whines on the same date, so WC's type is unresolved")
    print(f"[parser] dropped {len(dropped)} files ({out['parser']['n_dropped_not_copies']} are not copies of a "
          f"cohort clip); ambiguous names {[d['name'] for d in ambiguous]}", flush=True)

    # ---------------- crops
    crops = [] if a.no_crops else crop_sweep(files)
    for c in crops:
        c["in_cohort"] = bool(c["ia"] in pos and c["ib"] in pos)
        if c["in_cohort"]:
            ka, kb = pos[c["ia"]], pos[c["ib"]]
            c["same_bird"] = bool(birds[ka] == birds[kb]); c["same_label"] = bool(tt[ka] == tt[kb])
            c["cross_fold"] = bool(fold0[ka] != fold0[kb])
    out["crops"] = dict(method="rolling hash of 32-sample ch0 windows, 5 loudest non-constant probes per file; "
                               "every hit verified sample-exact on both channels over >= 1000 samples of non-zero "
                               "audio; a sweep, not an exhaustive proof", pairs=crops,
                        n_pairs=len(crops), n_pairs_in_cohort=sum(c["in_cohort"] for c in crops),
                        n_cross_bird=sum(c.get("same_bird") is False for c in crops),
                        n_cross_fold=sum(c.get("cross_fold") is True for c in crops))
    print(f"[crops] {len(crops)} verified overlap pairs, {out['crops']['n_pairs_in_cohort']} in cohort, "
          f"cross-bird {out['crops']['n_cross_bird']}, cross-fold {out['crops']['n_cross_fold']} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---------------- variant cohorts
    drop = set()
    for g in in_groups:
        ks = sorted((pos[by_path[str(Path(DIRS[0].parent) / m["dir"] / m["name"])]] for m in g["members"]
                     if m["in_cohort"]),
                    key=lambda k: (rows[k][4] != "ChickVocalizations", rows[k][0].name))
        drop.update(ks if g["label_conflict"] else ks[1:])
    keep_i = np.array([k for k in range(len(y)) if k not in drop])
    unk = np.array([b.startswith("unknown") for b in birds])
    g_merge = np.where(unk, "unknown*", birds)
    cohorts = {"board": (np.arange(len(y)), birds, None),
               "i_dedup": (keep_i, birds, None),
               "i_fixedfolds": (keep_i, birds, fold0),
               "ii_merge": (keep_i, g_merge, None),
               "ii_drop": (keep_i[~unk[keep_i]], birds, None)}
    if crops:
        k2 = set(keep_i.tolist()); child = set()
        for c in sorted(crops, key=lambda c: -c["overlap"]):
            if c["in_cohort"]:
                ka, kb = pos[c["ia"]], pos[c["ib"]]
                if ka in k2 and kb in k2 and ka not in child and kb not in child:
                    child.add(ka if c["n_a"] < c["n_b"] else kb)
        cohorts["iii_nocrops"] = (np.array(sorted(k2 - child)), g_merge, None)
    out["cohorts"] = {}
    for n, (r, g, f) in cohorts.items():
        out["cohorts"][n] = dict(n_clips=int(len(r)), n_groups=int(len(np.unique(g[r]))),
                                 class_counts={c: int((tt[r] == c).sum()) for c in classes},
                                 folds=("board (fixed)" if f is not None else "recomputed"),
                                 dropped_vs_board=[rows[k][4][:5] + "/" + rows[k][0].name
                                                   for k in sorted(set(range(len(y))) - set(r.tolist()))])
        print(f"[cohort] {n:13s} {len(r)} clips, {len(np.unique(g[r]))} groups", flush=True)

    # ---------------- scoring
    rs = json.loads(RESCORE.read_text())["models"]
    best = {m: rs[m]["best_layer_cpu"] for m in MODELS}
    tasks = []
    for m in MODELS:
        for n, (r, g, f) in cohorts.items():
            tasks.append((m, n, r, y[r], g[r], (f[r] if f is not None else None), [best[m], 3]))
    res = {}
    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "2"                          # inherited by the spawned workers
    with ProcessPoolExecutor(a.workers) as ex:
        for tag, n, accs, P in ex.map(score_task, tasks):
            res[(tag, n)] = (accs, P)
            print(f"  {tag:28s} {n:13s} L{best[tag]} {accs[best[tag]]:.4f}  L3 {accs[3]:.4f}  "
                  f"reselected L{int(np.argmax(accs))} {max(accs):.4f}  ({time.time()-t0:.0f}s)", flush=True)

    # reproduction guard
    guard = {m: float(np.abs(np.array(res[(m, "board")][0]) - np.array(rs[m]["per_layer_acc11_cpu"])).max())
             for m in MODELS}
    out["reproduction_max_abs_diff_vs_cpu_rescore"] = guard
    print(f"[guard] max |board - calltype_cpu_rescore| over all layers: {max(guard.values()):.6f}", flush=True)
    if max(guard.values()) > 0:
        print(f"  WARNING: board cohort does not reproduce calltype_cpu_rescore.json exactly: {guard}")

    out["scores"] = {}
    for m in MODELS:
        e = dict(board_best_layer=best[m])
        for n in cohorts:
            accs = res[(m, n)][0]
            e[n] = dict(at_board_layer=accs[best[m]], L3=accs[3], reselected_layer=int(np.argmax(accs)),
                        reselected_acc=float(max(accs)), per_layer=accs)
        out["scores"][m] = e

    # rankings
    def flips(n, mode):
        key = {"board_layer": "at_board_layer", "L3": "L3", "reselected": "reselected_acc"}[mode]
        fl = []
        for i, p in enumerate(MODELS):
            for q in MODELS[i + 1:]:
                d0 = out["scores"][p]["board"][key] - out["scores"][q]["board"][key]
                d1 = out["scores"][p][n][key] - out["scores"][q][n][key]
                if np.sign(d0) != np.sign(d1):
                    fl.append(dict(a=p, b=q, board_diff=round(d0, 4), new_diff=round(d1, 4)))
        return fl
    out["ranking_flips"] = {n: {mode: flips(n, mode) for mode in ("board_layer", "L3", "reselected")}
                            for n in cohorts if n != "board"}

    # leakage on the board split: test clips whose byte-identical twin sat in a training fold
    leak = [pos[by_path[str(Path(DIRS[0].parent) / mm["dir"] / mm["name"])]]
            for g in in_groups if g["cross_fold"] for mm in g["members"] if mm["in_cohort"]]
    out["board_leakage"] = dict(test_clips_with_train_twin=[rows[k][0].name for k in leak],
                                max_effect_on_acc=len(leak) / len(y),
                                correct_at_board_layer={m: int((res[(m, "board")][1][best[m]][leak].argmax(1)
                                                                == y[leak]).sum()) for m in MODELS})

    # bird bootstrap, observed difference, at the board's CPU best layer
    out["bootstrap"] = {}
    for n in [c for c in ("board", "i_dedup", "i_fixedfolds", "ii_merge", "ii_drop", "iii_nocrops") if c in cohorts]:
        r, g, _ = cohorts[n]
        out["bootstrap"][n] = {}
        for lab, pa, pb in CONTRASTS:
            out["bootstrap"][n][lab] = observed_bootstrap(y[r], res[(pa, n)][1][best[pa]],
                                                          res[(pb, n)][1][best[pb]], g[r])
            b = out["bootstrap"][n][lab]
            print(f"[boot] {n:9s} {lab:24s} {b['observed']:+.4f} [{b['lo']:+.4f}, {b['hi']:+.4f}] "
                  f"{b['verdict']}", flush=True)

    # ---------------- fold-seed sweep: separate the cohort change from re-folding
    seeds = list(range(a.seeds))
    sw_coh = ["board", "i_dedup", "ii_merge", "ii_drop"]
    cm = {m for _, p_, q_ in CONTRASTS for m in (p_, q_)}
    tasks = [(m, n, kind, L, cohorts[n][0], y[cohorts[n][0]], cohorts[n][1][cohorts[n][0]], seeds,
              kind == "board_layer" and m in cm)
             for m in MODELS for n in sw_coh for kind, L in (("board_layer", best[m]), ("L3", 3))]
    sw, swP = {}, {}
    with ProcessPoolExecutor(a.workers) as ex:
        for tag, n, kind, accs, Ps in ex.map(sweep_task, tasks):
            sw[(tag, n, kind)] = accs
            if Ps:
                swP[(tag, n)] = Ps
    chk = max(abs(sw[(m, n, k)][0] - out["scores"][m][n]["at_board_layer" if k == "board_layer" else "L3"])
              for m in MODELS for n in sw_coh for k in ("board_layer", "L3"))
    print(f"[sweep] seed-0 check vs cv_acc: max |diff| {chk:.6f} ({time.time()-t0:.0f}s)", flush=True)
    S = dict(seeds=seeds, seed0_max_abs_diff_vs_cv_acc=chk, acc={}, contrasts={}, ranking_flips_of_seed_mean={})
    for n in sw_coh:
        S["acc"][n] = {m: {k: dict(per_seed=sw[(m, n, k)], mean=float(np.mean(sw[(m, n, k)])),
                                  sd=float(np.std(sw[(m, n, k)], ddof=1)))
                          for k in ("board_layer", "L3")} for m in MODELS}
        r, g, _ = cohorts[n]
        S["contrasts"][n] = {}
        for lab, pa, pb in CONTRASTS:
            bs = [observed_bootstrap(y[r], swP[(pa, n)][i], swP[(pb, n)][i], g[r]) for i in range(len(seeds))]
            obs = [b["observed"] for b in bs]
            S["contrasts"][n][lab] = dict(per_seed=bs, mean_observed=float(np.mean(obs)),
                                         min_observed=float(min(obs)), max_observed=float(max(obs)),
                                         n_a_better=sum(b["verdict"] == "a_better" for b in bs),
                                         n_b_better=sum(b["verdict"] == "b_better" for b in bs),
                                         n_positive=sum(o > 0 for o in obs))
            c = S["contrasts"][n][lab]
            print(f"[sweep] {n:9s} {lab:24s} mean {c['mean_observed']:+.4f} range [{c['min_observed']:+.4f}, "
                  f"{c['max_observed']:+.4f}] resolved a>b {c['n_a_better']}/{len(seeds)}, "
                  f"positive {c['n_positive']}/{len(seeds)}", flush=True)
        if n != "board":
            S["ranking_flips_of_seed_mean"][n] = {}
            for k in ("board_layer", "L3"):
                fl = []
                for i, p_ in enumerate(MODELS):
                    for q_ in MODELS[i + 1:]:
                        d0 = S["acc"]["board"][p_][k]["mean"] - S["acc"]["board"][q_][k]["mean"]
                        d1 = S["acc"][n][p_][k]["mean"] - S["acc"][n][q_][k]["mean"]
                        if np.sign(d0) != np.sign(d1):
                            fl.append(dict(a=p_, b=q_, board_diff=round(d0, 4), new_diff=round(d1, 4)))
                S["ranking_flips_of_seed_mean"][n][k] = fl
    out["seed_sweep"] = S

    # ---------------- dropped non-copy clips: what does a probe that never saw their bird call them?
    if not a.no_probe_dropped:
        paths = [files[by_path[str(Path(DIRS[0].parent) / d["dir"] / d["name"])]]["path"]
                 for d in out["parser"]["dropped"] if not d["byte_copy_in_cohort"]]
        r, g, _ = cohorts["ii_merge"]
        out["dropped_clip_probe"] = dict(
            note="hint only: a leave-that-bird-out probe on ii_merge at the model's board layer; not a label",
            preds=probe_dropped(files, r, y[r], g[r], classes, best, paths))
        for k, v in out["dropped_clip_probe"]["preds"].items():
            for d in v:
                print(f"[dropped] {k:24s} {d['file']:36s} {d['top3']}", flush=True)

    OUT.write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"-> {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

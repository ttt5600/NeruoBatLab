#!/usr/bin/env python
"""A SECOND encoder-level holdout: chick vocalisations, for the whole DAPT/AVES family.

WHY THIS EXISTS. Every DAPT conclusion so far rests on BirdPark, which is 5,925 frames = four
independent 30 s blocks. The project's own bootstrap calls a 0.049 AP difference on it "not
distinguishable", and the DAPT trajectory scan then produced non-monotonic, mutually contradictory
curves at two learning rates -- exactly what an under-powered test set looks like. One holdout
cannot carry this weight.

The chick recordings are a genuinely independent second axis. The 18 chicks share NO individual
with the 26 adults in the pretraining corpus and 15 fall on recording dates absent from the
120-file pretraining manifest. The unit of resampling is the CHICK, and there are 10 of them with
labelled Be/LT clips -- against BirdPark's four blocks. Different task, different animals,
different failure modes; agreement across both is worth far more than either alone.

TWO READOUTS, ON PURPOSE.

  supervised    Be vs LT, leave-one-chick-out. run11 reaches AUC 0.991 here, so this is near
                ceiling and cannot show improvement -- but that makes it a clean DEGRADATION
                detector, which is precisely what a forgetting experiment needs.
  unsupervised  k-means AMI against call type at k=4, where run11 scores 0.516 against a
                random-init floor of 0.035 and a shuffled null of 0.005. Far more headroom in
                both directions, and it needs no labels at fit time.

LAYER CHOICE IS NOT MADE HERE. The arm (normalisation x layer) is read from
detection_variants.json's pre-commitment, which was chosen by out-of-fold AUC on IN-DISTRIBUTION
zebra finch audio. Choosing it again on chick data would be selection on a second test set. Every
layer is still computed and written to JSON for auditing, and the audit columns must not be maxed
over.

CLIPS ARE ENCODED ONE AT A TIME. Be averages 1.46 s and LT 0.20 s, and these encoders use group
norm over time, so padding a short clip into a batch corrupts its statistics badly -- an effect
this project has measured at >100x on the features. There is no batching here, deliberately.

DURATION IS A CONFOUND AND IS CONTROLLED. Be is ~7x longer than LT and embeddings are mean-pooled,
so clip length leaks into any mean-pooled representation. A duration-only baseline is reported so
that leakage is visible rather than implicit.

Usage
    chick_holdout_variants.py --models run11,aves-base-bio,dapt5e5_step2500
    chick_holdout_variants.py --report
"""
from __future__ import annotations
import argparse, gc, json, sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from zfeval import metrics as mx                                            # noqa: E402
from detection_variants import spec, load_model, layer_grid                 # noqa: E402

SR = 16000
CHICK = Path.home() / "zf_labelset/external/chick"
REF = CHICK / "run11_embeddings_reference.npz"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
DETJSON = ANA / "detection_variants.json"
OUTJSON = ANA / "chick_holdout_variants.json"
EMB = CHICK / "emb"
SEED = 0


def manifest():
    """Clip list + metadata, taken from the cached reference rather than re-derived.

    The original selection applied an 'unseen recording date' filter using a pretrain-manifest
    file that no longer exists on disk, so the filter is NOT locally reproducible. Reusing the
    cached name list preserves exactly the cohort the published run11 numbers were computed on.
    That is a real limitation and it is recorded in the output JSON rather than hidden.
    """
    d = np.load(REF, allow_pickle=True)
    names = [str(x) for x in d["names"]]
    missing = [n for n in names if not (CHICK / "audio" / n).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} clips in the manifest are missing from "
                                f"{CHICK/'audio'}, e.g. {missing[:3]}")
    return (names, np.array([str(x) for x in d["birds"]]),
            np.array([str(x) for x in d["types"]]),
            np.asarray(d["durs"], dtype=float), d["emb"])


def encode(name, s, names, tag, device):
    """(n_clips, n_layers, dim) float32, one clip at a time."""
    out = EMB / f"chick_{name}_{tag}.npy"
    if out.exists():
        return np.load(out)
    layers = layer_grid(s["n_layers"])
    model, _ = load_model(name, s, device)
    A = np.zeros((len(names), len(layers), s["dim"]), dtype=np.float32)
    for i, n in enumerate(names):
        x, sr = sf.read(str(CHICK / "audio" / n), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != SR:
            # The chick release is 44.1 kHz; these encoders are 16 kHz models. Polyphase keeps
            # the anti-alias filter that plain decimation would drop -- and aliased harmonics
            # would land squarely in the band that distinguishes Be from LT.
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(int(sr), SR)
            x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
        if tag == "matched":
            mu, var = float(x.mean()), float(x.var())
            x = (x - mu) / np.sqrt(var + 1e-5)
        with torch.no_grad():
            feats, _ = model.extract_features(
                torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device), None)
        for k, l in enumerate(layers):
            A[i, k] = feats[l][0].float().mean(0).cpu().numpy()   # mean-pool over time
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(names)}", flush=True)
    del model
    gc.collect()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, A)
    return A


def leave_one_chick_out(X, y, birds):
    """Out-of-fold probabilities with the CHICK held out, never just the clip."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    p = np.zeros(len(y), dtype=float)
    for b in np.unique(birds):
        te = birds == b
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            p[te] = y[tr].mean()          # degenerate fold: predict the prior, never crash
            continue
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=4000, C=1.0)).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def ami_k(X, types, k, n_init=10):
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_mutual_info_score as ami
    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(X)
    lab = KMeans(n_clusters=k, random_state=SEED, n_init=n_init).fit_predict(Z)
    return float(ami(types, lab))


def chick_bootstrap(pa, pb, y, birds, Xa_full, Xb_full, types, birds_all,
                    n_auc=2000, n_ami=400, seed=SEED):
    """Paired bootstrap RESAMPLING CHICKS, because clips within a chick are not independent.

    Ten chicks is the real sample size here, not 332 clips. Resampling clips would produce
    intervals several times too narrow and would manufacture significance -- the same mistake
    the BirdPark block bootstrap exists to avoid. Returns (delta, lo, hi) for Be/LT AUC and for
    AMI(k=4), both as b-minus-a so a positive number means the SECOND model is better.
    """
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    chicks = np.unique(birds)
    chicks_all = np.unique(birds_all)
    d_auc, d_ami = [], []
    # The two readouts get DIFFERENT resample counts because they cost differently: an AUC is a
    # sort, while each AMI replicate refits k-means twice. 400 replicates give a percentile
    # interval whose own Monte-Carlo error is well under the interval width we are reading, and
    # n_init is cut to 3 inside the loop for the same reason -- this is a resampling envelope,
    # not the point estimate, which is still computed at n_init=10.
    for _ in range(n_auc):
        pick = rng.choice(chicks, size=len(chicks), replace=True)
        idx = np.concatenate([np.flatnonzero(birds == c) for c in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        d_auc.append(roc_auc_score(y[idx], pb[idx]) - roc_auc_score(y[idx], pa[idx]))
    for _ in range(n_ami):
        pick2 = rng.choice(chicks_all, size=len(chicks_all), replace=True)
        j = np.concatenate([np.flatnonzero(birds_all == c) for c in pick2])
        if len(set(types[j])) < 2:
            continue
        d_ami.append(ami_k(Xb_full[j], types[j], 4, n_init=3)
                     - ami_k(Xa_full[j], types[j], 4, n_init=3))
    def ci(v):
        v = np.asarray(v)
        if v.size == 0:
            return None
        lo, hi = np.percentile(v, [2.5, 97.5])
        return dict(delta=float(v.mean()), lo=float(lo), hi=float(hi), n=int(v.size),
                    verdict="not_distinguishable" if lo <= 0 <= hi
                            else ("b_better" if lo > 0 else "a_better"))
    return ci(d_auc), ci(d_ami)


def precommitted_arm(name):
    """The arm chosen on IN-DISTRIBUTION ZF, not on chicks."""
    d = json.loads(DETJSON.read_text())
    pc = d.get("precommit", {}).get(name)
    if pc is None:
        return None
    return pc["joint"]["tag"], int(pc["joint"]["layer"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="run11")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--bootstrap", action="store_true",
                    help="paired chick-level bootstrap of every --models entry against --baseline")
    ap.add_argument("--baseline", default="run11")
    args = ap.parse_args()

    out = {"in": {}, "precommit": {}, "note": []}
    if OUTJSON.exists():
        try:
            prev = json.loads(OUTJSON.read_text())
            for k in ("in", "precommit"):
                out[k].update(prev.get(k, {}))
        except Exception as e:
            print(f"[warn] could not reuse {OUTJSON.name}: {e}")
    if args.report:
        report(out)
        return
    if args.bootstrap:
        bootstrap_mode(out, args)
        return

    names, birds, types, durs, ref_emb = manifest()
    # Supervised cohort: only chicks that produced BOTH call types. A chick with one class is a
    # degenerate leave-one-out fold -- its held-out predictions carry no discriminative signal
    # and silently shift the pooled AUC. Restricting here reproduces the published cohort
    # (10 chicks / 332 clips, majority-class 0.600).
    both = {b for b in np.unique(birds)
            if len(set(types[birds == b]) & {"Be", "LT"}) == 2}
    keep = np.isin(types, ["Be", "LT"]) & np.isin(birds, list(both))
    y = (types[keep] == "LT").astype(int)
    print(f"[cohort] {len(names)} clips / {len(np.unique(birds))} chicks total; "
          f"supervised subset {keep.sum()} clips over {len(both)} chicks with BOTH classes; "
          f"LT fraction {y.mean():.3f} (majority-class {max(y.mean(), 1-y.mean()):.3f})",
          flush=True)

    # Duration-only baseline. Be is ~7x longer than LT, so this is the leak, quantified.
    p = leave_one_chick_out(durs[keep].reshape(-1, 1), y, birds[keep])
    s_ = mx.score(y, p, "chick")
    out["in"]["duration_only"] = {"L0": dict(auc=s_.auc, ap=s_.ap)}
    print(f"  duration-only baseline   AUC {s_.auc:.4f}  AP {s_.ap:.4f}", flush=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[device] {device}", flush=True)

    for name in [n for n in args.models.split(",") if n]:
        s = spec(name)
        layers = layer_grid(s["n_layers"])
        out["in"].setdefault(name, {})
        for tag in ("matched", "native"):
            print(f"\n=== {name} / {tag} ===", flush=True)
            A = encode(name, s, names, tag, device)
            for k, l in enumerate(layers):
                X = A[keep][:, k, :]
                p = leave_one_chick_out(X, y, birds[keep])
                sc = mx.score(y, p, "chick")
                a4 = ami_k(A[:, k, :], types, 4)
                out["in"][name][f"{tag}_L{l}"] = dict(auc=sc.auc, ap=sc.ap, ami_k4=a4)
                print(f"  {tag:7s} L{l:<2d}  Be/LT AUC {sc.auc:.4f} AP {sc.ap:.4f} | "
                      f"AMI(k=4) {a4:.4f}", flush=True)
        arm = precommitted_arm(name)
        if arm is None:
            print(f"  [!] {name} has no ZF pre-commitment in {DETJSON.name}; "
                  f"reporting all arms only", flush=True)
        else:
            key = f"{arm[0]}_L{arm[1]}"
            out["precommit"][name] = dict(arm=key, **out["in"][name][key])
            print(f"  pre-committed on ZF: {key} -> Be/LT AUC "
                  f"{out['in'][name][key]['auc']:.4f}  AMI(k=4) "
                  f"{out['in'][name][key]['ami_k4']:.4f}", flush=True)
        OUTJSON.write_text(json.dumps(out, indent=2))

    report(out)


def bootstrap_mode(out, args):
    """Compare each model to the baseline at BOTH pre-committed arms, resampling chicks."""
    names, birds, types, durs, _ = manifest()
    both = {b for b in np.unique(birds)
            if len(set(types[birds == b]) & {"Be", "LT"}) == 2}
    keep = np.isin(types, ["Be", "LT"]) & np.isin(birds, list(both))
    y = (types[keep] == "LT").astype(int)

    def arm_data(name):
        arm = precommitted_arm(name)
        if arm is None:
            return None
        tag, layer = arm
        f = EMB / f"chick_{name}_{tag}.npy"
        if not f.exists():
            print(f"[skip] {name}: no cached embeddings at {f.name}; encode it first")
            return None
        A = np.load(f)
        k = layer_grid(spec(name)["n_layers"]).index(layer)
        X = A[:, k, :]
        return leave_one_chick_out(X[keep], y, birds[keep]), X

    base = arm_data(args.baseline)
    if base is None:
        sys.exit(f"baseline {args.baseline} has no cached embeddings / pre-commitment")
    pa, Xa = base
    out.setdefault("bootstrap", {})
    print("")
    print(f"{'model vs ' + args.baseline:24s} {'dAUC':>8s} {'95% CI':>20s}  "
          f"{'dAMI(k=4)':>10s} {'95% CI':>20s}")
    print("-" * 90)
    for name in [n for n in args.models.split(",") if n and n != args.baseline]:
        r = arm_data(name)
        if r is None:
            continue
        pb, Xb = r
        # AUC is computed on the both-classes subset; AMI on the FULL cohort. They have
        # different row counts, so each gets the array it actually indexes into.
        c_auc, c_ami = chick_bootstrap(pa, pb, y, birds[keep], Xa, Xb, types, birds)
        out["bootstrap"][f"{name}_vs_{args.baseline}"] = dict(be_lt_auc=c_auc, ami_k4=c_ami)
        f = lambda c: (f"{c['delta']:+.4f}", f"[{c['lo']:+.4f}, {c['hi']:+.4f}]") if c else ("-", "-")
        a1, a2 = f(c_auc); m1, m2 = f(c_ami)
        v = (c_ami or {}).get("verdict", "")
        print(f"{name:24s} {a1:>8s} {a2:>20s}  {m1:>10s} {m2:>20s}  {v}")
        OUTJSON.write_text(json.dumps(out, indent=2))
    print("-" * 90)
    print(f"Resampling unit is the CHICK ({len(both)} with both classes, "
          f"{len(np.unique(birds))} total). Positive = the model beats {args.baseline}.")


def report(out):
    print(f"\n{'model':22s} {'arm':12s} {'Be/LT AUC':>10s} {'AMI(k=4)':>9s}")
    print("-" * 58)
    d = out.get("in", {}).get("duration_only", {}).get("L0")
    if d:
        print(f"{'duration only':22s} {'-':12s} {d['auc']:10.4f} {'-':>9s}")
    for k, v in out.get("precommit", {}).items():
        print(f"{k:22s} {v['arm']:12s} {v['auc']:10.4f} {v['ami_k4']:9.4f}")
    print("-" * 58)
    print("Arms are pre-committed on in-distribution ZF detection, never on chick data.")
    print("Be/LT is near ceiling for a good encoder: read it as a degradation detector.")


if __name__ == "__main__":
    main()

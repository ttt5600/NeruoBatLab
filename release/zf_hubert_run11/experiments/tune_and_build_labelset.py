#!/usr/bin/env python
"""Tune the detection operating point, and build a set of windows for human labeling.

Two problems with the published detection numbers, both fixable here:

1. **The operating point was never chosen.** acc/precision/recall/F1 were all reported at
   threshold 0.5, which is just the logistic default. With classes at 2:1 against the positives
   and no `class_weight="balanced"`, 0.5 sits where the class prior puts it, which suppresses
   recall. AUC 0.921 is threshold-free; recall 0.734 is one arbitrary point on that curve.

2. **The negative labels are unverified.** "Background" means "not one of the 2450 localized
   curated calls" -- nobody listened. Uncurated calls sit in the negative class, so the model is
   penalized for correctly firing on them. Precision and AUC are therefore LOWER BOUNDS.

Usage
-----
  # 1. see the whole precision/recall curve instead of one point
  python tune_and_build_labelset.py --npz pretrans_aucs.npz --tune

  # 2. emit windows to hand-label (default strategy is the statistically efficient one)
  python tune_and_build_labelset.py --npz pretrans_aucs.npz --build --n 500 --out labelset.csv

  # 3. after labeling, fold the labels back in and re-score on verified ground truth
  python tune_and_build_labelset.py --npz pretrans_aucs.npz --rescore labels_done.csv

Requires the npz to contain `probs` and `windows`, which the patched
eval_pretransformer_detection.py now saves. Older artifacts lack them.
"""
import argparse
import csv
import sys

import numpy as np


def load(npz):
    d = np.load(npz, allow_pickle=True)
    missing = [k for k in ("probs", "windows", "y", "groups") if k not in d]
    if missing:
        sys.exit(f"{npz} lacks {missing}. Re-run eval_pretransformer_detection.py -- the older "
                 f"artifact saved only the AUCs, not the per-window predictions.")
    return d


def curve(y, p):
    """Precision/recall/F1 across thresholds. Reported at the layer's own out-of-fold probs."""
    ts = np.unique(np.round(np.quantile(p, np.linspace(0.001, 0.999, 400)), 5))
    out = []
    for t in ts:
        pred = p >= t
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        if tp == 0:
            continue
        pr, rc = tp / (tp + fp), tp / (tp + fn)
        out.append((float(t), pr, rc, 2 * pr * rc / (pr + rc), tp, fp, fn))
    return out


def do_tune(d, layer):
    y, p = d["y"], d["probs"][:, layer]
    names = d["names"]
    print(f"tuning {names[layer]}  ({len(y)} windows, {int(y.sum())} calls, "
          f"prevalence {y.mean():.3f})\n")
    c = curve(y, p)

    at_half = min(c, key=lambda r: abs(r[0] - 0.5))
    best_f1 = max(c, key=lambda r: r[3])
    print(f"{'operating point':28s} {'thr':>6s} {'prec':>7s} {'recall':>7s} {'F1':>7s}")
    print(f"{'default 0.5 (as published)':28s} {at_half[0]:6.3f} {at_half[1]:7.3f} "
          f"{at_half[2]:7.3f} {at_half[3]:7.3f}")
    print(f"{'best F1':28s} {best_f1[0]:6.3f} {best_f1[1]:7.3f} {best_f1[2]:7.3f} "
          f"{best_f1[3]:7.3f}")
    for target in (0.80, 0.90, 0.95):
        ok = [r for r in c if r[2] >= target]
        if ok:
            b = max(ok, key=lambda r: r[1])
            print(f"{'recall >= ' + str(target):28s} {b[0]:6.3f} {b[1]:7.3f} {b[2]:7.3f} "
                  f"{b[3]:7.3f}")
        else:
            print(f"{'recall >= ' + str(target):28s} unreachable")

    print("\nCAVEATS -- read before quoting any of the above:")
    print("  PRECISION is a LOWER BOUND. Uncurated calls sit in the negative class, so some")
    print("    'false positives' are correct detections being punished. True precision is higher.")
    print("  RECALL is recall ON CURATED CALLS ONLY, which is not the same as recall on all")
    print("    vocalizations. Curation selected clean, isolated, good-SNR exemplars, so the")
    print("    uncurated remainder is enriched for hard cases the model is MORE likely to miss.")
    print("    Recall over all real calls is probably LOWER than the number above.")
    print("  AUC can move in EITHER direction once labels are fixed. A contaminating call the")
    print("    model scored high is currently a false positive (deflating AUC); one it scored")
    print("    low is currently counted as a correct rejection (inflating AUC). Which effect")
    print("    dominates is an empirical question -- that is what --rescore measures.")
    return best_f1


def do_build(d, layer, n, strategy, out):
    y, p, w = d["y"], d["probs"][:, layer], d["windows"]
    rng = np.random.default_rng(0)

    if strategy == "stratified":
        # Unbiased across the score range: bin by model score, sample evenly within bins, keep
        # both classes. Gives a clean estimate of precision AND recall from one labeled set.
        bins = np.clip((p * 10).astype(int), 0, 9)
        idx = []
        per = max(1, n // 10)
        for b in range(10):
            pool = np.flatnonzero(bins == b)
            if len(pool):
                idx.extend(rng.choice(pool, size=min(per, len(pool)), replace=False))
        idx = np.array(idx)
    elif strategy == "balanced":
        pos = np.flatnonzero(y == 1); neg = np.flatnonzero(y == 0)
        k = n // 2
        idx = np.concatenate([rng.choice(pos, min(k, len(pos)), replace=False),
                              rng.choice(neg, min(k, len(neg)), replace=False)])
    elif strategy == "audit":
        # Only the windows where model and label DISAGREE, plus a control sample. Labeling every
        # predicted-positive gives EXACT precision for the cost of just the flagged windows.
        thr = 0.5
        dis = np.flatnonzero((p >= thr) & (y == 0))
        ctl = rng.choice(np.flatnonzero((p < thr) & (y == 0)),
                         min(n // 4, int(((p < thr) & (y == 0)).sum())), replace=False)
        idx = np.concatenate([dis[:n - len(ctl)], ctl])
    else:
        sys.exit(f"unknown strategy {strategy}")

    rng.shuffle(idx)
    with open(out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["id", "path", "start", "dur", "current_label", "model_score", "recording",
                     "human_label"])
        for i in idx:
            path, start, end, rec = w[i]
            wr.writerow([f"win{i:06d}", path, float(start), round(float(end) - float(start), 3),
                         int(y[i]), round(float(p[i]), 4), rec, ""])

    npos = int((y[idx] == 1).sum())
    print(f"wrote {out}: {len(idx)} windows ({npos} currently-positive, "
          f"{len(idx) - npos} currently-background), strategy={strategy}")
    se = float(np.sqrt(0.2 * 0.8 / max(1, len(idx) - npos)))
    print(f"with {len(idx) - npos} background windows, a contamination rate near 20% is "
          f"estimated to about +/-{1.96 * se * 100:.1f} percentage points (95% CI).")
    print("\nLoad into ZF_Labeling_Tool.ipynb:")
    print("  items = [dict(id=r['id'], path=r['path'], start=float(r['start']), "
          "dur=float(r['dur']))\n           for r in csv.DictReader(open('%s'))]" % out)
    print("  labeler(items, labels=('voc','noise','quiet','unsure'))")
    print("\nKeep show_meta=False. The set was built from the model's own scores, so a labeler "
          "who\ncan see them would ratify the model and the result would be circular.")


def do_rescore(d, layer, labels_csv):
    y, p = d["y"].copy(), d["probs"][:, layer]
    rows = list(csv.DictReader(open(labels_csv)))
    done = [r for r in rows if r.get("human_label", "").strip()]
    if not done:
        sys.exit(f"{labels_csv} has no filled human_label values")

    flips, agree = 0, 0
    touched = []
    for r in done:
        i = int(r["id"].replace("win", ""))
        truth = 1 if r["human_label"].strip().lower() == "voc" else 0
        touched.append(i)
        if truth != y[i]:
            flips += 1
        else:
            agree += 1
        y[i] = truth

    touched = np.array(touched)
    orig = d["y"][touched]
    bg = touched[orig == 0]
    contaminated = int((y[bg] == 1).sum())
    print(f"{len(done)} labeled | {agree} agreed with the constructed label | {flips} corrected")
    if len(bg):
        rate = contaminated / len(bg)
        se = float(np.sqrt(rate * (1 - rate) / len(bg)))
        print(f"\nCONTAMINATION: {contaminated}/{len(bg)} = {rate:.3f} of 'background' windows "
              f"actually contain a call\n  95% CI +/-{1.96 * se:.3f}")
        print(f"  -> the published precision was penalized on ~{rate*100:.1f}% of its "
              f"'false positives'")

    # Where did the contamination sit on the score axis? This decides which way AUC moves.
    if len(bg):
        flipped = bg[y[bg] == 1]
        if len(flipped):
            print(f"\n  contaminating calls: median model score {np.median(p[flipped]):.3f} "
                  f"vs {np.median(p[bg[y[bg] == 0]]):.3f} for true background")
            hi = int((p[flipped] >= 0.5).sum())
            print(f"  {hi} of {len(flipped)} were already scored >=0.5 (model caught them; "
                  f"correcting these RAISES AUC)")
            print(f"  {len(flipped)-hi} were scored <0.5 (model missed them too; correcting "
                  f"these LOWERS AUC)")

    from sklearn.metrics import roc_auc_score
    sub = touched
    a_old = roc_auc_score(d["y"][sub], p[sub])
    a_new = roc_auc_score(y[sub], p[sub])
    print(f"\nOn the {len(sub)} verified windows only:")
    print(f"  AUC with constructed labels : {a_old:.4f}")
    print(f"  AUC with YOUR labels        : {a_new:.4f}   ({a_new - a_old:+.4f})")
    print("\nThe second number is the honest one. It can move EITHER WAY: contamination the model\n"
          "caught was deflating AUC, but contamination it also missed was inflating AUC by\n"
          "counting a real call as a correct rejection. Do not assume the correction is\n"
          "favourable -- report whichever direction it actually went.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--layer", type=int, default=1,
                    help="index into `names` (0=pre-transformer, 1=layer 0, the best detector)")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--rescore")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--strategy", default="stratified",
                    choices=["stratified", "balanced", "audit"])
    ap.add_argument("--out", default="labelset.csv")
    a = ap.parse_args()

    d = load(a.npz)
    if a.tune:
        do_tune(d, a.layer)
    if a.build:
        do_build(d, a.layer, a.n, a.strategy, a.out)
    if a.rescore:
        do_rescore(d, a.layer, a.rescore)
    if not (a.tune or a.build or a.rescore):
        ap.error("pick at least one of --tune / --build / --rescore")


if __name__ == "__main__":
    main()

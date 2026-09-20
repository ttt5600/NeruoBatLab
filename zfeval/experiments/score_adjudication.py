#!/usr/bin/env python
"""Score a finished blind adjudication against the answer key.

Input: the verdicts exported from the artifact's store (a JSON list or {id: record} map) plus
`adjudication_key.json`. Output: the rate comparison the blinding was built for, and what it
implies for the reported precision.

The read-out is NOT "how many of the 52 were called calls". It is that rate placed BETWEEN the two
control rates, because a human adjudicating 1 s windows of colony audio is not a perfect oracle --
the TP controls measure how often a real call is recognised here, and the TN controls how often a
non-call is mistaken for one. Without them a raw rate has no scale.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def boot_diff(k1, n1, k2, n2, n=20000, seed=0):
    """Interval on (rate1 - rate2) by resampling each arm's judgements."""
    rng = np.random.default_rng(seed)
    a = rng.binomial(n1, k1 / n1, n) / n1 if n1 else np.zeros(n)
    b = rng.binomial(n2, k2 / n2, n) / n2 if n2 else np.zeros(n)
    d = a - b
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", required=True, help="JSON exported from the artifact store")
    ap.add_argument("--key", default=str(ANA / "adjudication_key.json"))
    a = ap.parse_args()

    key = json.loads(Path(a.key).read_text())
    kind = {c["id"]: c for c in key["cards"]}
    raw = json.loads(Path(a.verdicts).read_text())
    rows = raw if isinstance(raw, list) else list(raw.values())
    got = {}
    for r in rows:
        cid = r.get("clip_id") or r.get("id")
        if cid in kind and r.get("verdict") in ("call", "nocall", "unsure"):
            got[cid] = r["verdict"]
    print(f"{len(got)}/{len(kind)} clips judged\n")

    arms = {}
    for k in ("high_conf_FP", "control_TP", "control_TN"):
        ids = [c["id"] for c in key["cards"] if c["kind"] == k]
        v = [got[i] for i in ids if i in got]
        dec = [x for x in v if x != "unsure"]              # unsure excluded from the RATE, counted separately
        arms[k] = dict(n_total=len(ids), n_judged=len(v), n_unsure=v.count("unsure"),
                       n_decided=len(dec), n_call=dec.count("call"),
                       rate=(dec.count("call") / len(dec)) if dec else float("nan"))
        lo, hi = wilson(arms[k]["n_call"], arms[k]["n_decided"])
        arms[k].update(ci_lo=lo, ci_hi=hi)
        print(f"{k:14s} judged {arms[k]['n_judged']:3d}/{arms[k]['n_total']:<3d} "
              f"unsure {arms[k]['n_unsure']:2d}  called-call {arms[k]['n_call']:3d}/"
              f"{arms[k]['n_decided']:<3d} = {arms[k]['rate']:.3f} [{lo:.3f}, {hi:.3f}]")

    if min(arms[k]["n_decided"] for k in arms) == 0:
        print("\nnot enough judgements yet for the comparison")
        (ANA / "adjudication_result.json").write_text(json.dumps(dict(arms=arms), indent=2))
        return

    fp, tp, tn = arms["high_conf_FP"], arms["control_TP"], arms["control_TN"]
    cmp_ = {}
    for nm, other in (("vs_TP_control", tp), ("vs_TN_control", tn)):
        d, lo, hi = boot_diff(fp["n_call"], fp["n_decided"], other["n_call"], other["n_decided"])
        cmp_[nm] = dict(delta=d, lo=lo, hi=hi, distinguishable=bool(lo > 0 or hi < 0))
        print(f"\nFP rate - {nm[3:]:12s} = {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
              f"{'distinguishable' if (lo>0 or hi<0) else 'NOT distinguishable'}")

    # Where does the FP rate sit on the line between the two controls? 1.0 = behaves exactly like a
    # known call, 0.0 = exactly like a known non-call.
    span = tp["rate"] - tn["rate"]
    pos = (fp["rate"] - tn["rate"]) / span if span > 1e-9 else float("nan")
    print(f"\nposition between controls: {pos:+.2f}  "
          f"(0 = indistinguishable from a known non-call, 1 = from a known call)")
    if span < 0.2:
        print("  WARNING: the controls are only " + f"{span:.2f}" +
              " apart, so they do not span a usable scale -- the adjudicator could not "
              "separate known calls from known non-calls either, and this position means little.")

    # Implication for the reported number, stated as a range rather than a point: only the FPs
    # actually judged "call" are reassigned, and the unjudged ones are left alone.
    n_reassigned = fp["n_call"]
    implied = dict(n_high_conf_fp=fp["n_total"], n_judged_call=n_reassigned,
                   note="eval-B FP count at threshold 0.5 was 169 of which 52 scored p>0.944; "
                        "reassigning the ones judged 'call' moves them from FP to TP",
                   fp_at_0p5_before=169, fp_at_0p5_after=169 - n_reassigned)
    print(f"\nif the {n_reassigned} judged 'call' are real calls, eval-B false positives at "
          f"threshold 0.5 go 169 -> {169 - n_reassigned}")

    out = dict(arms=arms, comparisons=cmp_, position_between_controls=pos,
               control_span=span, implied=implied, key_counts=key["counts"])
    (ANA / "adjudication_result.json").write_text(json.dumps(out, indent=2))
    print("\nwrote", ANA / "adjudication_result.json")


if __name__ == "__main__":
    main()

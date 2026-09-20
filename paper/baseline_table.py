#!/usr/bin/env python
"""One canonical baseline for the project: every model, on classification and on detection.

Results have accumulated across a dozen analysis files with different cohorts, splits and metrics,
which makes it far too easy to quote two numbers side by side that were never comparable. This
assembles the whole picture in one place and, for every row, prints the split and the chance level
alongside the score -- because an accuracy without its majority rate, or an AUC without its split,
is not a result.

Rows are marked MISSING rather than omitted when an analysis has not been run, so the gaps in the
baseline are visible instead of silently absent.

  python paper/baseline_table.py            # print
  python paper/baseline_table.py --md       # emit markdown for the manuscript
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"


def load(n):
    p = ANA / n
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def g(d, *path, default=None):
    cur = d
    for k in path:
        if cur is None:
            return default
        cur = cur.get(k) if isinstance(cur, dict) else None
    return default if cur is None else cur


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    ct11, ct8, cta = load("calltype11.json"), load("aves_calltype.json"), load("calltype_analysis.json")
    ident, var = load("identity_probe.json"), load("aves_variants_calltype.json")
    fb, av, ah = load("frame_baselines.json"), load("aves_baseline.json"), load("aves_holdout.json")
    ens, dv = load("ensemble_detect.json"), load("detection_variants.json")
    ri, mb = load("randinit_control.json"), load("mel_baseline.json")
    ft = load("finetune_calltype.json")

    rows = []          # (section, task, split, chance, model, metric, value)

    def add(sec, task, split, chance, model, metric, val):
        rows.append((sec, task, split, chance, model, metric,
                     "MISSING" if val is None else f"{val:.4f}"))

    # ---------------- CLASSIFICATION
    s, t, sp = "CLASSIFICATION", "call type, 11-way", "leave-birds-out"
    ch = g(ct11, "majority")
    add(s, t, sp, ch, "run11", "acc", g(ct11, "run11", "best", "acc"))
    add(s, t, sp, ch, "AVES base-bio", "acc", g(ct11, "aves", "best", "acc"))
    if var:
        for k, v in (var.get("models") or {}).items():
            if k == "run11":
                continue
            add(s, t, sp, ch, k, "acc",
                g(v, "best_11", "acc"))
    if ft:
        for k, v in (ft.get("results") or {}).items():
            add(s, t + " (fine-tuned)", sp, ch, k, "acc", g(v, "mean") or g(v, "acc"))

    t, ch = "call type, 8-way (adults)", g(ct11, "adults_8class", "majority")
    add(s, t, sp, ch, "run11", "acc", g(ct11, "adults_8class", "run11", "best", "acc"))
    add(s, t, sp, ch, "AVES base-bio", "acc", g(ct11, "adults_8class", "aves", "best", "acc"))

    t, sp, ch = "bird identity, 31-way", "leave-session-out", g(ident, "majority")
    add(s, t, sp, ch, "run11", "acc", g(ident, "best", "run11", "acc"))
    add(s, t, sp, ch, "AVES base-bio", "acc", g(ident, "best", "aves", "acc"))

    # ---------------- DETECTION
    s = "DETECTION"
    t, sp, ch = "frame, 20 ms, in-distribution", "contiguous 60 s blocks", g(fb, "prevalence")
    add(s, t, sp, ch, "run11 L0", "AUC", g(fb, "full", "hubert_L0", "auc"))
    add(s, t, sp, ch, "run11 L0", "AP", g(fb, "full", "hubert_L0", "ap"))
    add(s, t, sp, ch, "AVES best", "AUC", g(av, "best_aves", "auc"))
    add(s, t, sp, ch, "AVES best", "AP", g(av, "best_aves", "ap"))
    add(s, t, sp, ch, "run11+AVES mean", "AUC", g(ens, "in_distribution", "mean", "auc"))
    add(s, t, sp, ch, "run11+AVES mean", "AP", g(ens, "in_distribution", "mean", "ap"))
    add(s, t, sp, ch, "log-mel 100 ms", "AUC", g(fb, "full", "logmel_100ms", "auc"))
    add(s, t, sp, ch, "log-energy", "AUC", g(fb, "full", "logenergy", "auc"))
    if dv:
        pc = dv.get("precommit", {})
        for k in (dv.get("in_distribution") or {}):
            if k in ("run11", "logenergy"):
                continue
            add(s, t, sp, ch, k, "AUC", g(pc, k, "joint", "zf", "auc"))
            add(s, t, sp, ch, k, "AP", g(pc, k, "joint", "zf", "ap"))

    t, sp, ch = "frame, ZF -> BirdPark holdout", "unseen recording, other lab", g(ah, "birdpark", "prevalence")
    add(s, t, sp, ch, "run11 L0", "AUC", g(ah, "zf_to_bp", "run11_L0", "auc"))
    add(s, t, sp, ch, "AVES L6 (pre-committed)", "AUC", g(ah, "zf_to_bp", "aves_L6", "auc"))
    add(s, t, sp, ch, "run11+AVES mean", "AUC", g(ens, "zf_to_bp", "mean", "auc"))
    add(s, t, sp, ch, "log-energy", "AUC", g(ah, "zf_to_bp", "logenergy", "auc"))
    if dv:
        pc = dv.get("precommit", {})
        for k in (dv.get("zf_to_bp") or {}):
            if k in ("run11", "logenergy"):
                continue
            add(s, t, sp, ch, k, "AUC", g(pc, k, "joint", "bp", "auc"))

    t, sp, ch = "1 s windows, held-out recording", "eval B", None
    add(s, t, sp, ch, "run11 L0", "AUC", g(mb, "hubert_L0", "B_auc"))
    add(s, t, sp, ch, "log-mel (tuned)", "AUC", g(mb, "logmel_C0.03", "B_auc"))
    add(s, t, sp, ch, "log-energy", "AUC", g(mb, "energy_only", "B_auc"))

    t, sp, ch = "1 s windows, pretraining control", "contiguous 60 s blocks", g(ri, "prevalence")
    add(s, t, sp, ch, "run11 CNN", "AUC", g(ri, "layer_table", "cnn", "pretrained"))
    add(s, t, sp, ch, "random init CNN", "AUC", g(ri, "layer_table", "cnn", "rand_mean"))
    add(s, t, sp, ch, "log-mel", "AUC", g(ri, "baselines", "logmel"))

    # ---------------- render
    w = [max(len(str(r[i])) for r in rows + [("SECTION", "task", "split", "chance", "model",
                                              "metric", "value")]) for i in range(7)]
    hdr = ("section", "task", "split", "chance", "model", "metric", "value")
    if a.md:
        print("| " + " | ".join(hdr) + " |")
        print("|" + "|".join("---" for _ in hdr) + "|")
        for r in rows:
            c = f"{r[3]:.4f}" if isinstance(r[3], float) else "-"
            print("| " + " | ".join([r[0], r[1], r[2], c, r[4], r[5], r[6]]) + " |")
    else:
        last = None
        for r in rows:
            if r[0] != last:
                print(f"\n{'='*len(r[0])}\n{r[0]}\n{'='*len(r[0])}")
                last = r[0]
            c = f"{r[3]:.4f}" if isinstance(r[3], float) else "  -   "
            flag = "  <-- MISSING" if r[6] == "MISSING" else ""
            print(f"  {r[1]:<34s} {r[4]:<24s} {r[5]:<5s} {r[6]:>9s}   chance {c}"
                  f"   [{r[2]}]{flag}")
    miss = sum(1 for r in rows if r[6] == "MISSING")
    print(f"\n{len(rows)} rows, {miss} missing")


if __name__ == "__main__":
    main()

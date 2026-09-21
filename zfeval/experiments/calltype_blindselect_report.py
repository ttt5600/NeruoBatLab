#!/usr/bin/env python
"""Turn the blind-selection curves into the three answers.

Q1  how much accuracy does each label-free rule cost against the oracle k?
Q2  does the rule rank the five encoders the way ground truth does?
Q3  can the rule tell a real embedding from a column-shuffled one?

Two families of rule are scored, and the distinction matters more than any single number:

  ARGMAX     k_hat = argmax_k criterion(k).        Presumes a "natural" number of groups.
  THRESHOLD  k_hat = largest k still above a bar.  Presumes more groups are better until they
                                                   stop being reproducible.

The threshold bars are chosen so they need no tuning on this corpus and so they transfer to a
species with no labels at all:
  ps80      largest k with prediction strength >= 0.8      (Tibshirani & Walther's own rule)
  vs_null   largest k where bird-held-out stability still beats the SHUFFLED null at that k
  half      largest k where bird-held-out stability is >= half its value at the smallest k
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

D = Path(__file__).resolve().parents[2] / "paper" / "figures" / "calltype"
R = json.loads((D / "calltype_blindselect.json").read_text())
KS = R["ks"]
NULL = "NULL-shuffled"
REAL = [e for e in R["encoders"] if e != NULL]
ARGMAX = ["silhouette", "calinski_harabasz", "neg_davies_bouldin", "gap",
          "stab_clip", "stab_bird"]


def curve(enc, crit):
    return np.array([R["encoders"][enc]["criteria"][crit][str(k)] for k in KS])


def truth(enc, key):
    return np.array([R["encoders"][enc]["truth"][key][str(k)] for k in KS])


def pick_argmax(enc, crit):
    return KS[int(curve(enc, crit).argmax())]


def pick_threshold(enc, rule):
    if rule == "ps80":
        c = curve(enc, "prediction_strength")
        ok = [k for k, v in zip(KS, c) if v >= R["ps_threshold"]]
    elif rule == "vs_null":
        c, n = curve(enc, "stab_bird"), curve(NULL, "stab_bird")
        ok = [k for k, v, u in zip(KS, c, n) if v > u]
    elif rule == "half":
        c = curve(enc, "stab_bird")
        ok = [k for k, v in zip(KS, c) if v >= 0.5 * c[0]]
    else:
        raise ValueError(rule)
    return max(ok) if ok else KS[0]


RULES = [("argmax", c) for c in ARGMAX] + [("threshold", r) for r in ("ps80", "vs_null", "half")]


def main():
    lines = []
    def p(s=""):
        print(s); lines.append(s)

    p(f"cohort {R['n_clips']} clips / {len(R['classes'])} classes / {R['n_birds']} birds "
      f"| layer {R['layer']} | {R['algo']} | B_stab={R['b_stability']}")
    p()

    # ---- Q1: cost of a blind k, per encoder -------------------------------
    p("Q1  COST OF CHOOSING k BLIND   (cluster-vote accuracy, leave-birds-out)")
    p()
    hdr = f"{'rule':<22}" + "".join(f"{e[:13]:>15}" for e in REAL)
    p(hdr); p("-" * len(hdr))
    oracle_k = {e: KS[int(truth(e, 'vote_acc').argmax())] for e in REAL}
    oracle_a = {e: float(truth(e, 'vote_acc').max()) for e in REAL}
    p(f"{'ORACLE k (uses labels)':<22}" +
      "".join(f"{oracle_a[e]:.3f}@{oracle_k[e]:<9d}" for e in REAL))
    picks, costs = {}, {}
    for fam, name in RULES:
        row, ks_, cs_ = [], {}, {}
        for e in REAL:
            k = pick_argmax(e, name) if fam == "argmax" else pick_threshold(e, name)
            a = float(truth(e, "vote_acc")[KS.index(k)])
            ks_[e], cs_[e] = k, oracle_a[e] - a
            row.append(f"{a:.3f}@{k:<9d}")
        picks[name], costs[name] = ks_, cs_
        p(f"{name:<22}" + "".join(row))
    p()
    p(f"{'mean cost vs oracle':<22}" +
      "".join("" for _ in REAL))
    for fam, name in RULES:
        m = np.mean([costs[name][e] for e in REAL])
        p(f"   {name:<19} {-m:+.4f}")
    p()

    # ---- Q2: can a blind criterion rank the encoders? ---------------------
    p("Q2  RANKING THE ENCODERS BLIND   (Spearman rho vs true vote accuracy, n=5 encoders)")
    p()
    p("   true vote-acc @ oracle k:  " +
      "  ".join(f"{e}={oracle_a[e]:.3f}" for e in sorted(REAL, key=lambda x: -oracle_a[x])))
    p()
    for kfix in (11, 30, 70):
        a = np.array([truth(e, "vote_acc")[KS.index(kfix)] for e in REAL])
        p(f"   -- scored at matched k={kfix} --   true acc " +
          " ".join(f"{e[:10]}={v:.3f}" for e, v in zip(REAL, a)))
        for crit in ARGMAX + ["prediction_strength"]:
            c = np.array([curve(e, crit)[KS.index(kfix)] for e in REAL])
            rho, pv = spearmanr(c, a)
            p(f"      {crit:<22} rho {rho:+.3f}  (p={pv:.3f})")
        amis = np.array([truth(e, "ami")[KS.index(kfix)] for e in REAL])
        rho, pv = spearmanr(amis, a)
        p(f"      {'[AMI: uses labels]':<22} rho {rho:+.3f}  (p={pv:.3f})")
        p()
    a = np.array([oracle_a[e] for e in REAL])
    p("   -- each encoder scored at its OWN blind-chosen k --")
    for fam, name in RULES:
        if fam == "argmax":
            c = np.array([curve(e, name)[KS.index(picks[name][e])] for e in REAL])
        else:
            c = np.array([curve(e, "stab_bird")[KS.index(picks[name][e])] for e in REAL])
        rho, pv = spearmanr(c, a)
        p(f"      {name:<22} rho {rho:+.3f}  (p={pv:.3f})  ks={[picks[name][e] for e in REAL]}")
    p()

    # ---- Q3: does the criterion notice destroyed structure? ---------------
    p("Q3  NULL CONTROL   (column-shuffled AVES: same marginals, no joint structure)")
    p()
    p(f"   true vote-acc  real median {np.median([oracle_a[e] for e in REAL]):.3f}"
      f"   null {float(truth(NULL,'vote_acc').max()):.3f}"
      f"   null AMI@best {float(truth(NULL,'ami').max()):.3f}")
    p()
    p(f"   {'criterion':<22}{'k':>4}{'real(med)':>12}{'null':>10}{'ratio':>9}  verdict")
    for crit in ARGMAX + ["prediction_strength"]:
        for kfix in (11, 30):
            r = np.median([curve(e, crit)[KS.index(kfix)] for e in REAL])
            n = curve(NULL, crit)[KS.index(kfix)]
            ratio = (r / n) if abs(n) > 1e-9 else np.inf
            ok = "SEPARATES" if (r - n) > 0 and (abs(n) < 1e-6 or ratio > 1.5 or ratio < 0.5) \
                else "blind" if abs(r - n) < 1e-6 else "weak"
            p(f"   {crit:<22}{kfix:>4}{r:>12.4f}{n:>10.4f}{ratio:>9.2f}  {ok}")
    p()
    (D / "blindselect_report.txt").write_text("\n".join(lines))
    json.dump({"picks": picks, "costs": costs, "oracle_k": oracle_k, "oracle_acc": oracle_a},
              open(D / "blindselect_summary.json", "w"), indent=2)
    print(f"\n-> {D/'blindselect_report.txt'}")


if __name__ == "__main__":
    main()

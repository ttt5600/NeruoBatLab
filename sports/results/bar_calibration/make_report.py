"""Tables for notes/bar_calibration.md from fixed_n.csv / sprt.csv / n80.csv (run after sim.py).
../.venv_quant/bin/python results/bar_calibration/make_report.py > results/bar_calibration/report.txt"""
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parent
f = pd.read_csv(D / "fixed_n.csv")
s = pd.read_csv(D / "sprt.csv")
n80 = pd.read_csv(D / "n80.csv")
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 1000)


def fmt(r):
    return f"{r.rate:.4f} [{r.lo:.4f},{r.hi:.4f}]"


print("== PLACEBO false-pass (planted 0), rate [95% CP], nominal; flag = CP lower > nominal (excess) / CP upper > 1.5x")
p = f[(f.planted_roi == 0) & (f.stage != "check")].copy()
p["cell"] = p.apply(fmt, axis=1)
p["excess"] = p.lo > p.nominal
p["over_1.5x"] = p.hi > 1.5 * p.nominal
print(p[["scenario", "variant", "rule", "stage", "n", "nominal", "cell", "reps", "excess", "over_1.5x"]].to_string(index=False))

print("\n== CURRENT rule check-level rates (which check binds), base variant")
c = f[(f.stage == "check") & (f.variant == "base")]
print(c.pivot_table(index=["scenario", "n", "rule"], columns="planted_roi", values="rate").round(4).to_string())

print("\n== POWER (base), rate by planted ROI")
w = f[(f.variant == "base") & (f.stage != "check") & f.rule.isin(["current_pass", "current_pass_or_audit", "proposed",
                                                                   "proposed+halves", "proposed_stat_only"])]
print(w.pivot_table(index=["scenario", "rule", "stage", "n"], columns="planted_roi", values="rate").round(3).to_string())
print("\n== true ROI actually planted (mean over reps; capped where 1 - c(1+r) < 0)")
print(f.groupby(["scenario", "planted_roi"]).true_roi_mean.mean().round(4).to_string())

print("\n== STRESS variants at planted 0 / +2% (proposed vs current)")
st = f[(f.variant != "base") & (f.stage != "check") & f.rule.isin(["current_pass", "current_pass_or_audit", "proposed"])]
print(st.pivot_table(index=["scenario", "variant", "rule", "stage", "n"], columns="planted_roi", values="rate").round(4).to_string())

print("\n== SEQUENTIAL (forward, proposed): SPRT H0 L=1-c vs H1 +2%, alpha .05 beta .2, horizon 36 months")
cols = ["scenario", "variant", "planted_roi", "reps", "p_H1", "p_H1_lo", "p_H1_hi", "p_H0", "p_none", "p_H1_by_100",
        "p_H1_by_300", "p_H1_by_1000", "p_H1_by_6mo", "p_H1_by_12mo", "p_H1_by_24mo", "mean_fills_to_decision",
        "mean_months_to_decision", "median_months_to_decision"]
print(s[cols].round(4).to_string(index=False))

print("\n== n for 80% power at +2% on the grid (fixed-n rules) and months at volume")
print(n80.round(3).to_string(index=False))

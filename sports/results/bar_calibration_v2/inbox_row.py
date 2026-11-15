"""Append the bar_calibration_v2 result row to notes/findings_inbox.jsonl (run once from sports/)."""
import json
from pathlib import Path

row = {"ts": "2026-11-15T16:00:00Z", "agent": "bar_calibration_v2", "kind": "result", "edge": "bar_review",
       "verdict": "weakened",
       "headline": "PARTIAL: month-level loss clustering (not night) makes a nominal-.05 SPRT false-accept .077-.087; "
                   "calibrated config rho1 .70/alpha .005/beta .10 holds placebo .034-.045 under 3x stress; E1-size edge "
                   "kill .10-.17 but H1 by 36 mo only .52-.62 (median ~21-25 mo at 55 fills/mo); sports ROI SPRT cannot "
                   "promote +2% within a year (H1 .27-.30 in 10 in-season yrs at 60/mo); sealed-pass gate -> .011 "
                   "false accepts/yr vs .55 with no charge",
       "numbers": {"placebo_nominal05_base": [0.077, 0.087], "chosen": {"rho1": 0.7, "alpha": 0.005, "beta": 0.1},
                   "valid_placebo_stress_e1": [0.0446, 0.0418, 0.0476],
                   "valid_placebo_stress_live": [0.0337, 0.0312, 0.0363],
                   "e1edge_kill_stress_e1_live": [0.173, 0.162], "e1edge_kill_base": 0.096,
                   "e1edge_H1_36m_e1_live": [0.615, 0.546], "e1edge_H1_24m_e1_live": [0.394, 0.309],
                   "plus1pct_kill_vs_H1": [0.43, 0.29], "sp_plus2_H1_60permo": [0.27, 0.30],
                   "sp_plus5_200permo_median_months": 14,
                   "E_FA_per_yr": {"no_charge": 0.55, "family_year": 0.25, "sealed_gate": 0.011},
                   "forward_trials_since_1102": 11, "sealed_weather_family_total_alpha": 0.104},
       "files": ["sports/notes/bar_calibration_v2.md", "sports/results/bar_calibration_v2/sim2.py",
                 "sports/results/bar_calibration_v2/nc_grid.csv", "sports/results/bar_calibration_v2/nc_valid.csv",
                 "sports/results/bar_calibration_v2/nc_tune.csv", "sports/results/bar_calibration_v2/sp_grid.csv",
                 "sports/results/bar_calibration_v2/family.json", "sports/results/bar_calibration_v2/diag.py"],
       "lesson": "Placebo-calibrate sequential tests with month-level shocks: a lucky month crosses the boundary; "
                 "night ICC alone hid a ~1.7x false-accept inflation.",
       "memory": "bar v2 11-15: forward SPRT (rho1 .7, a .005, b .1) calibrated under 3x clustering, E1-size edge kill "
                 ".10-.17 but ~2 yr to accept at 55 fills/mo; sports forward ROI can't promote +2% in a year; gate "
                 "forward behind sealed pass (.011 FA/yr); sealed .05/j sums to .05 H(k)."}
with open(Path(__file__).resolve().parents[2] / "notes/findings_inbox.jsonl", "a") as f:
    f.write(json.dumps(row) + "\n")
print("appended")

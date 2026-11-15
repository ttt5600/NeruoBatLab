"""Append the bar_calibration result row to notes/findings_inbox.jsonl (run once from sports/)."""
import datetime
import json
from pathlib import Path

SPORTS = Path(__file__).resolve().parents[2]
row = {"ts": datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), "agent": "bar_calibration",
       "kind": "result", "edge": "harness", "verdict": "holds",
       "headline": "Exp A (seed 20261115): current rule 5 false-passes E4-like placebos 5-6x its bar (pass|audit .0040 "
                   "[.0017,.0079] n1000 vs .00085) yet passes real +2% only .20 (Sharpe guard -> audit) and sports +2% "
                   ".025 at n2000; proposed staged rule calibrated in all base regimes but sports dev n100 (1.85x); power "
                   "E4 sealed .998 n1000, sports .799 n300; E4 loss-SPRT 96% in mean 3.5 mo. PARTIAL pass of prereg; "
                   "net-CLV invalid if Kalshi close biased +0.5c (false-pass 17-71%)",
       "numbers": {"cur_e4_placebo_pass_or_audit_n1000": [0.0040, 0.0017, 0.0079], "cur_e4_plus2_pass_n1000": 0.201,
                   "cur_e4_plus2_pass_or_audit_n1000": 0.994, "cur_sp_plus2_pass_n2000": 0.025,
                   "prop_e4_sealed_plus2_n1000": 0.998, "prop_e4_sealed_plus2_n300": 0.485,
                   "prop_sp_sealed_plus2_n300": 0.799, "prop_sp_dev_placebo_n100": [0.0019, 0.0013, 0.0025],
                   "sprt_e4_placebo": [0.0535, 0.0492, 0.0581], "sprt_e4_plus2_H1": 0.960, "sprt_e4_mean_months": 3.5,
                   "sprt_sp_plus2_H1": 0.842, "sprt_sp_placebo": [0.0431, 0.0392, 0.0473],
                   "sp_bias05_sealed_falsepass_n100_n1000": [0.165, 0.707], "x1_prop_dev_plus2_n1000": 0.062},
       "files": ["sports/notes/bar_calibration.md", "sports/results/bar_calibration/sim.py",
                 "sports/results/bar_calibration/fixed_n.csv", "sports/results/bar_calibration/sprt.csv",
                 "sports/results/bar_calibration/report.txt"],
       "lesson": "Calibrate the harness on placebos in the regime you trade: rule 5 is too loose (bootstrap) and too "
                 "strict (Sharpe guard, share cap) at once.",
       "memory": "bar_calibration 11-15: edgelab rule 5 false-passes 96-98c placebos 5-6x its bar but passes real +2% "
                 "edges ~20% (Sharpe guard); proposed staged rule (sealed .05/#sealed, forward loss-SPRT, net CLV, "
                 "leave-top-week-out) calibrated except sports dev n100; needs exp B (close bias << .25c) + city-month "
                 "design effect. GOAL amendment = user's call."}
with open(SPORTS / "notes" / "findings_inbox.jsonl", "a") as fh:
    fh.write(json.dumps(row) + "\n")
print("appended")

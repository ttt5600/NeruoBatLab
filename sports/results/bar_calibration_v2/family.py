"""Family control for forward SPRTs at our real trial rate (read-only on lab/edgelab_registry.jsonl).
Run from sports/: ../.venv_quant/bin/python results/bar_calibration_v2/family.py  -> family.json"""
import collections
import json
from pathlib import Path

SPORTS = Path(__file__).resolve().parents[2]
ev = [json.loads(line) for line in open(SPORTS / "lab/edgelab_registry.jsonl")]
since = "2026-11-02"
fw = [e for e in ev if e.get("event") == "evaluate" and e.get("split") == "forward" and e["ts"] >= since]
se = [e for e in ev if e.get("event") == "evaluate" and e.get("split") == "sealed" and e["ts"] >= since]
fw_trials = {e["trial_id"]: e["family"] for e in fw}
fw_fam = collections.Counter(fw_trials.values())
se_fam = collections.Counter(e["family"] for e in se)
se_pass = sorted({e["trial_id"] for e in se if e.get("verdict") == "pass"})
in_fwd_paper = sorted({e["trial_id"] for e in ev if e.get("event") == "transition" and e.get("to") == "forward_paper"})
reg_since = sum(1 for e in ev if e.get("event") == "register" and e["ts"] >= since)
fw_after_import = sorted({(e["trial_id"], e["ts"][:10]) for e in fw if e["ts"][:10] > since})

a_f = 0.05
H = lambda k: sum(1 / j for j in range(1, k + 1))  # noqa: E731
out = {
    "window": f"{since}..(registry end)", "forward_eval_events": len(fw), "forward_trials": len(fw_trials),
    "forward_trials_by_family": dict(fw_fam), "sealed_evals": len(se), "sealed_by_family": dict(se_fam),
    "sealed_passes": se_pass, "ever_forward_paper": in_fwd_paper, "registered_since": reg_since,
    "forward_evals_after_11-02_import": fw_after_import,
    "note": "Nearly all 11-02 rows are the retroactive import of legacy trials; treat 11 forward starts/yr as the "
            "planning rate (upper: the import), not 11 per 13 days.",
    # expected false accepts per year if EVERY trial is null (upper bounds; SPRT false-accept taken at its nominal a_f)
    "E_FA_no_charge": a_f * len(fw_trials),
    "E_FA_per_family_year_budget": a_f * len(fw_fam),
    # sealed-pass gate: a null trial must first pass sealed at .05/j (j-th sealed shot in the family) -> sum .05*H(k)
    "sealed_family_FWER_harmonic": {f: 0.05 * H(k) for f, k in se_fam.items()},
    "E_FA_sealed_gate": a_f * sum(0.05 * H(k) for k in se_fam.values()),
    "E_FA_sealed_gate_bonferroni_k": a_f * 0.05 * len(se_fam),
}
(Path(__file__).resolve().parent / "family.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))

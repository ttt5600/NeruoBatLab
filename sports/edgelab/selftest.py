"""GOAL.md rule 8 -- the harness tests itself. Everything runs in SANDBOX registries (temp files): the real registry's
trial counts, accesses and unseals are never touched.

(a) placebo: >= 200 no-information strategies on real DEV data through the full pipeline (register -> vault -> rule ->
    outcome resolution -> PIT -> rule 5 at the family bar). Two batches:
      real  : 70 NFL random-side + 70 NFL shuffled-wind-signal (closing odds incl. vig) + 70 Kalshi coin-flip brackets
              (22:00 ask + taker fee)
      fair  : the same 210 NFL-style placebos at the proportional NO-VIG price (zero edge by construction -- the case
              where only the statistics stand between a lucky placebo and a "pass")
    All placebos of a batch are registered BEFORE any is evaluated, so each is judged at alpha / (batch size), the bar
    rule 2 gives a family of that size. Reported: false-pass rate with the correction, and the same rows re-judged at
    an uncorrected p < alpha (other rule-5 checks unchanged), and p < alpha alone.
(b) golden: a sandbox carrying the full imported history (so the family bars are the real ones) replays E1's spent
    sealed rows (must be verdict pass) and L2 T1's (must be fail).
(c) cheaters: a rule that peeks at the post-decision tape, one that takes a later price, one that reads the hidden
    outcome, one that sets its own outcome; a spec amended after data access; a rule module edited after data access;
    a second unseal; a sealed load without dev pass; evaluate() trying to promote to live_small. Every one must be
    refused.
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
import tempfile
import textwrap
from pathlib import Path

from . import ALPHA, ladder, paths, run, schema, vault
from .registry import Registry, RegistryError, sha256_file

OUT = paths.LAB / "edgelab_selftest.json"


def _placebo_spec(i: int, batch: str, dataset: str, kind: str, fair: bool, seed: int) -> dict:
    rule = "edgelab.rules.placebo:nfl" if dataset == "nfl_total_lab" else "edgelab.rules.placebo:kalshi"
    return {"id": f"placebo_{batch}_{i:03d}", "family": f"placebo_{batch}", "hypothesis": f"placebo {kind}",
            "dataset": dataset, "rule": rule, "params": {"kind": kind, "seed": seed, "fair": fair},
            "decision_time": "kickoff - 10 min" if dataset == "nfl_total_lab" else "22:00 local",
            "price_model": "book_quote" if dataset == "nfl_total_lab" else "taker", "split": f"vault:{dataset}",
            "min_n": 50, "kill": {"kind": "roi_ci_below_zero", "after_n": 100},
            "fee": {"rounding": "exact"}}


def placebo(n_each: int = 70, log=print) -> dict:
    out = {}
    plan = {
        "real": [("nfl_total_lab", "random_side", False), ("nfl_total_lab", "shuffled_signal", False),
                 ("kalshi_temp_all", "coin_flip", False)],
        "fair": [("nfl_total_lab", "random_side", True), ("nfl_total_lab", "shuffled_signal", True),
                 ("nfl_total_lab", "coin_flip", True)],
    }
    for batch, kinds in plan.items():
        with tempfile.TemporaryDirectory() as td:
            reg = Registry(Path(td) / "reg.jsonl")
            tids = []
            for j, (ds, kind, fair) in enumerate(kinds):
                for k in range(n_each):
                    i = j * n_each + k
                    tids.append(reg.register(_placebo_spec(i, batch, ds, kind, fair, seed=1000 * (j + 1) + k)))
            bar, n_fam = reg.family_bar(f"placebo_{batch}")
            res = []
            for tid in tids:
                r = run.evaluate(tid, "dev", reg, save_rows=False)
                res.append(r)
            ok = [r for r in res if r.get("verdict") in ("pass", "fail", "needs_audit")]
            others = lambda r: all(v for k, v in r["checks"].items() if k != "a_p_below_bar")
            no_e = lambda r: all(v for k, v in r["checks"].items() if k not in ("a_p_below_bar", "e_clv_nonnegative"))
            p = [r["p"] for r in ok if r.get("p") is not None]
            out[batch] = {
                "placebos": len(tids), "evaluated": len(ok), "family_trials": n_fam, "bar": bar,
                "false_pass_corrected": sum(r["verdict"] == "pass" for r in ok) / max(len(ok), 1),
                "false_pass_uncorrected": sum((r["p"] < ALPHA) and others(r) for r in ok) / max(len(ok), 1),
                "false_pass_uncorrected_ignoring_clv": sum((r["p"] < ALPHA) and no_e(r) for r in ok) / max(len(ok), 1),
                "p_below_alpha_only": sum(x < ALPHA for x in p) / max(len(p), 1),
                "any_pass_corrected": any(r["verdict"] == "pass" for r in ok),
                "any_pass_uncorrected": any((r["p"] < ALPHA) and others(r) for r in ok),
                "median_roi": float(sorted(r["net_roi"] for r in ok)[len(ok) // 2]) if ok else None,
                "by_kind": {f"{ds}:{kind}": sum(1 for r in ok[j * n_each:(j + 1) * n_each] if r["p"] < ALPHA)
                            for j, (ds, kind, _) in enumerate(kinds)},
            }
            log(f"placebo[{batch}] {json.dumps(out[batch])}")
    rates = [out[b]["false_pass_corrected"] for b in out]
    out["false_pass_rate"] = max(rates)
    out["ok"] = out["false_pass_rate"] <= ALPHA
    return out


def _history_sandbox(td: Path) -> Registry:
    from . import migrate
    reg = Registry(td / "reg.jsonl")
    migrate.migrate(reg, evaluate_live=False, log=lambda *a: None)
    return reg


def golden(log=print) -> dict:
    from . import migrate
    with tempfile.TemporaryDirectory() as td:
        reg = _history_sandbox(Path(td))
        ids = {t["spec"]["id"]: tid for tid, t in reg.trials().items()}
        e1 = run.evaluate(ids["e1_taker"], "sealed", reg, save_rows=False)
        l2 = run.evaluate(ids["l2_t1_ml_gap05"], "sealed", reg, save_rows=False)
    out = {"e1_sealed": {k: e1.get(k) for k in ("verdict", "n", "net_roi", "ci", "p", "bar", "n_trials_family",
                                                 "half1", "half2", "reasons")},
           "l2_t1": {k: l2.get(k) for k in ("verdict", "n", "net_roi", "ci", "p", "bar", "n_trials_family", "reasons")}}
    out["e1_concentration"] = (e1.get("concentration") or {}).get("share")
    out["e1_clv"] = (e1.get("clv") or {}).get("mean")
    out["ok"] = e1.get("verdict") == "pass" and l2.get("verdict") == "fail"
    log(f"golden {json.dumps(out, default=str)}")
    return out


def cheaters(log=print) -> dict:
    res = {}
    base = {"family": "cheat", "hypothesis": "cheater", "dataset": "kalshi_temp_e1", "decision_time": "22:00",
            "price_model": "taker", "split": "vault:kalshi_temp_e1", "min_n": 10,
            "kill": {"kind": "roi_ci_below_zero", "after_n": 100}, "fee": {"rounding": "exact"}}
    with tempfile.TemporaryDirectory() as td:
        reg = Registry(Path(td) / "reg.jsonl")
        for name in ("peek_tape", "late_price"):
            tid = reg.register({**base, "id": f"cheat_{name}", "rule": f"edgelab.rules.cheater:{name}"})
            r = run.evaluate(tid, "dev", reg, save_rows=False)
            res[name] = {"refused": r.get("verdict") == "pit_rejected", "pit_violations": r.get("pit_violations")}
        for name, exc in (("read_outcome", KeyError), ("set_outcome", vault.VaultError)):
            tid = reg.register({**base, "id": f"cheat_{name}", "rule": f"edgelab.rules.cheater:{name}"})
            try:
                run.evaluate(tid, "dev", reg, save_rows=False)
                res[name] = {"refused": False}
            except exc as e:
                res[name] = {"refused": True, "error": f"{type(e).__name__}: {str(e)[:120]}"}
        # spec amended after access
        tid = reg.register({**base, "id": "honest_e1_copy", "rule": "edgelab.rules.kalshi_wx:taker"})
        run.evaluate(tid, "dev", reg, save_rows=False)
        try:
            reg.amend(tid, {**reg.trial(tid)["spec"], "min_n": 5})
            res["amend_after_access"] = {"refused": False}
        except RegistryError as e:
            res["amend_after_access"] = {"refused": True, "error": str(e)[:160]}
        # amend BEFORE access is allowed (sanity: the refusal is about access, not about amending)
        t2 = reg.register({**base, "id": "amend_before_access", "rule": "edgelab.rules.kalshi_wx:taker"})
        reg.amend(t2, {**reg.trial(t2)["spec"], "min_n": 20})
        res["amend_before_access_allowed"] = {"ok": reg.trial(t2)["spec"]["min_n"] == 20}
        # sealed without dev pass (a non-spent spec id that never ran dev)
        try:
            run.evaluate(t2, "sealed", reg, save_rows=False)
            res["sealed_without_dev_pass"] = {"refused": False}
        except vault.VaultError as e:
            res["sealed_without_dev_pass"] = {"refused": True, "error": str(e)[:160]}
        # rule module edited after access
        mdir = Path(td) / "mods"
        mdir.mkdir()
        (mdir / "elab_tmp_rule.py").write_text("from edgelab.rules.kalshi_wx import taker as rule\n")
        sys.path.insert(0, str(mdir))
        try:
            tid3 = reg.register({**base, "id": "edited_rule", "rule": "elab_tmp_rule:rule"})
            run.evaluate(tid3, "dev", reg, save_rows=False)
            (mdir / "elab_tmp_rule.py").write_text("from edgelab.rules.kalshi_wx import taker as rule\nMARGIN = 2.0\n")
            importlib.invalidate_caches()
            sys.modules.pop("elab_tmp_rule", None)
            try:
                run.evaluate(tid3, "dev", reg, save_rows=False)
                res["rule_edited_after_access"] = {"refused": False}
            except RegistryError as e:
                res["rule_edited_after_access"] = {"refused": True, "error": str(e)[:160]}
            tid4 = reg.register({**base, "id": "edited_rule", "rule": "elab_tmp_rule:rule"})
            res["edited_rule_is_new_trial"] = {"ok": tid4 != tid3, "old": tid3, "new": tid4,
                                               "family_count": reg.family_counts()["cheat"]}
        finally:
            sys.path.remove(str(mdir))
        # evaluate() cannot promote to live_small; user transition without approval refused
        for label, args in (("evaluate_to_live_small", ("forward_paper", "live_small", "evaluate", None)),
                            ("user_without_approval", ("forward_paper", "live_small", "user", None))):
            try:
                ladder.check(*args)
                res[label] = {"refused": False}
            except ladder.LadderError as e:
                res[label] = {"refused": True, "error": str(e)[:120]}
    with tempfile.TemporaryDirectory() as td:
        reg = _history_sandbox(Path(td))
        ids = {t["spec"]["id"]: tid for tid, t in reg.trials().items()}
        run.evaluate(ids["e1_taker"], "sealed", reg, save_rows=False)
        try:
            run.evaluate(ids["e1_taker"], "sealed", reg, save_rows=False)
            res["second_unseal"] = {"refused": False}
        except vault.VaultError as e:
            res["second_unseal"] = {"refused": True, "error": str(e)[:160]}
        # tamper evidence: edit one historical line -> hash chain breaks
        lines = reg.path.read_text().splitlines()
        lines[5] = lines[5].replace('"dead"', '"dev_pass"') if '"dead"' in lines[5] else lines[5] + " "
        reg.path.write_text("\n".join(lines) + "\n")
        res["tampered_registry_detected"] = {"refused": not reg.verify()["ok"], "verify": reg.verify()}
    res["ok"] = all(v.get("refused", v.get("ok", False)) for v in res.values() if isinstance(v, dict))
    log(f"cheaters {json.dumps(res, default=str)}")
    return res


def selftest(n_each: int = 70, log=print) -> dict:
    out = {"ts": __import__("edgelab.registry", fromlist=["now"]).now(), "git_head": paths.git_head(),
           "alpha": ALPHA, "placebo": placebo(n_each, log), "golden": golden(log), "cheaters": cheaters(log)}
    out["placebo_false_pass_rate"] = out["placebo"]["false_pass_rate"]
    out["golden_cases_reproduce"] = bool(out["golden"]["ok"])
    out["ok"] = bool(out["placebo"]["ok"] and out["golden"]["ok"] and out["cheaters"]["ok"])
    OUT.write_text(json.dumps(out, indent=1, default=str))
    return out

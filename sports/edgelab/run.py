"""evaluate(): the ONLY path from data to a verdict and to an automatic ladder move.

    registry.check_unchanged (rule 1) -> vault.load (rule 3, logged) -> rule(frame, params) -> vault.resolve_outcomes
    -> schema.enforce (rule 4) -> evaluator.score at the family bar (rules 2, 5) -> kill rule (7) -> ladder (6)

Every call appends an `evaluate` event (also on a PIT rejection) and writes the bet rows to lab/edgelab_runs/.
"""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

from . import evaluator, ladder, paths, schema, vault
from .registry import Registry, sha256_file

RUNS = paths.LAB / "edgelab_runs"


def _rule(spec: dict):
    mod, _, fn = spec["rule"].partition(":")
    return getattr(importlib.import_module(mod), fn)


def kill_check(kill: dict, res: dict) -> dict:
    kind = kill.get("kind")
    n = res.get("n") or 0
    if kind == "clv_negative_significant":
        after = int(kill.get("after_n", 100))
        p = (res.get("clv") or {}).get("p_negative")
        trig = bool(n >= after and p is not None and p < float(kill.get("p", 0.05)))
        return {"kind": kind, "triggered": trig, "armed": n >= after, "p_clv_negative": p}
    if kind == "roi_ci_below_zero":
        after = int(kill.get("after_n", 100))
        hi = (res.get("ci") or [None, None])[1]
        return {"kind": kind, "triggered": bool(n >= after and hi is not None and hi < 0), "armed": n >= after}
    if kind == "losses_in_first":
        return {"kind": kind, "triggered": bool(res.get("losses", 0) > int(kill["max_losses"])
                                                 and n <= int(kill["n"])), "armed": True}
    if kind == "none_preset":
        return {"kind": kind, "triggered": None, "armed": False, "missing": True}
    raise ValueError(f"unknown kill kind {kind!r}")


def _slim(res: dict) -> dict:
    return {k: v for k, v in res.items() if k not in ("cluster_pnl",)}


def evaluate(trial_id: str, split: str, reg: Registry | None = None, save_rows: bool = True) -> dict:
    reg = reg or Registry()
    t = reg.trial(trial_id)
    spec = t["spec"]
    reg.check_unchanged(trial_id)
    frame, hidden, access = vault.load(spec["dataset"], split, trial_id, reg)
    bets = _rule(spec)(frame, spec.get("params") or {})
    bets = vault.resolve_outcomes(spec["dataset"], bets, hidden, frame)
    bar, n_family = reg.family_bar(spec["family"])
    base = {"event": "evaluate", "trial_id": trial_id, "split": split, "family": spec["family"],
            "n_trials_family": n_family, "bar": bar, "replay": access.get("replay", False),
            "access_ts": access["ts"], "sources": access["sources"], "vault_sha256": access["vault_sha256"],
            "rule_sha256": t["rule_sha256"], "spec_sha256": t["spec_sha256"]}
    try:
        rows = schema.enforce(bets)
    except schema.PITViolation as e:
        ev = reg.append({**base, "verdict": "pit_rejected", "pit_violations": e.counts,
                         "pit_violation_rows": int(len(e.examples))})
        return {**ev, "error": str(e)}
    except schema.SchemaError as e:
        ev = reg.append({**base, "verdict": "schema_rejected", "schema_error": str(e)})
        return {**ev, "error": str(e)}
    res = evaluator.score(rows, bar, int(spec["min_n"]), spec.get("fee") or {})
    out = {**base, **_slim(res), "pit_violations": {}}
    if save_rows and len(rows):
        RUNS.mkdir(parents=True, exist_ok=True)
        p = RUNS / f"{trial_id}_{split}.parquet"
        rows.to_parquet(p, index=False)
        out["rows_file"] = {"path": str(p), "sha256": sha256_file(p)}
    if split == "forward":
        out["kill"] = kill_check(spec["kill"], res)
        out["forward_eligible_for_user"] = bool(res["verdict"] == "pass" and res["n"] >= ladder.FORWARD_MIN_N)
    reg.append(out)
    # ---- ladder moves (evaluate may only make EVALUATE_EDGES moves)
    st = reg.state(trial_id)
    move = None
    if split == "dev" and st == "registered":
        move = "dev_pass" if res["verdict"] == "pass" else ("dead" if res["verdict"] == "fail" else None)
    elif split == "sealed" and st in ("dev_pass", "registered") and access.get("gate_ok", True):
        move = "sealed_pass" if res["verdict"] == "pass" else ("dead" if res["verdict"] == "fail" else None)
    elif split == "forward":
        if st == "sealed_pass":
            move = "forward_paper"
        if out["kill"].get("triggered") and st in ("forward_paper", "sealed_pass"):
            move = "dead"
    if move:
        if move == "dead" and st == "sealed_pass":
            ladder.transition(reg, trial_id, "forward_paper", "evaluate", reason="forward evaluation started")
        ladder.transition(reg, trial_id, move, "evaluate", split=split, verdict=res["verdict"])
    out["state"] = reg.state(trial_id)
    return out


def score_only(spec: dict, split: str, bar: float | None, fee: dict | None = None) -> dict:
    """Sensitivity runs that are NOT verdicts (no registry, no ladder): loads through the loader directly, dev split
    only, and says so in its output."""
    if split != "dev":
        raise vault.VaultError("score_only is dev-only")
    cfg = vault.config()[spec["dataset"]]
    df, _ = __import__("edgelab.loaders", fromlist=["LOADERS"]).LOADERS[cfg["loader"]]()
    rows = vault._split_rows(df, vault.window(spec["dataset"], "dev"))
    hidden_cols = [c for c in cfg.get("hidden", []) if c in rows.columns]
    key = cfg["outcome"]["key"]
    keys = [key] if isinstance(key, str) else list(key)
    hidden = rows[[c for c in keys if c in rows.columns] + hidden_cols]
    vis = rows.drop(columns=hidden_cols)
    vis.attrs = dict(rows.attrs)
    bets = vault.resolve_outcomes(spec["dataset"], _rule(spec)(vis, spec.get("params") or {}), hidden, vis)
    rows = schema.enforce(bets)
    return {"NOT_A_VERDICT": True, **_slim(evaluator.score(rows, bar, int(spec["min_n"]), fee or spec.get("fee") or {}))}

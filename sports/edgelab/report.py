"""`status` table and the hub-verifier `evidence` JSON. Read-only over the registry."""
from __future__ import annotations

import json
from pathlib import Path

from . import ALPHA, paths, vault
from .registry import HISTORICAL_ORIGINS, Registry, RegistryError, sha256_file

SELFTEST = paths.LAB / "edgelab_selftest.json"


def runnable(reg: Registry) -> list[dict]:
    return [t for t in reg.trials().values() if t.get("origin") not in HISTORICAL_ORIGINS
            and not t["family"].startswith(("placebo", "cheat"))]


def resolve(reg: Registry, name: str) -> dict:
    ts = reg.trials()
    if name in ts:
        return ts[name]
    c = [t for t in ts.values() if t["spec"]["id"] == name]
    if not c:
        raise RegistryError(f"no trial or spec id {name!r}")
    return sorted(c, key=lambda t: t["ts"])[-1]          # the latest registration of that spec id


def latest(reg: Registry, tid: str, split: str) -> dict | None:
    ev = reg.evaluations(tid, split)
    return ev[-1] if ev else None


def legacy(reg: Registry, tid: str) -> dict | None:
    ev = [e for e in reg.events() if e["event"] == "legacy_note" and e["trial_id"] == tid]
    return ev[-1] if ev else None


def next_gate(reg: Registry, t: dict) -> str:
    st = reg.state(t["trial_id"])
    sp = vault.config()[t["spec"]["dataset"]]["splits"]
    if st == "registered":
        if sp.get("dev") is not None and not reg.evaluations(t["trial_id"], "dev"):
            return "run dev"
        if sp.get("dev") is not None:
            e = latest(reg, t["trial_id"], "dev")
            return f"dev {e.get('verdict')}: fix and register a NEW trial"
        if sp.get("sealed") is None:
            return "no dev/sealed window: user must pre-register a sealed forward window in vault.yaml"
        return "sealed (prospective, once)"
    if st == "dev_pass":
        return "sealed: one shot"
    if st == "sealed_pass":
        return "start forward paper (evaluate --split forward)"
    if st == "forward_paper":
        e = latest(reg, t["trial_id"], "forward") or {}
        pk = reg.preset_kill(t["trial_id"])
        kl = ("kill " + "+".join(r["kind"] for r in pk["rules"]) + " (preset)") if pk else \
            f"kill {t['spec']['kill'].get('kind')} (spec; no setkill yet)"
        return f"forward n {e.get('n', 0)}/100 passing rule 5 -> USER decides live_small; {kl}"
    if st in ("live_small", "live"):
        return "user-sized"
    last = [e for e in reg.events() if e["event"] == "transition" and e["trial_id"] == t["trial_id"]]
    return f"dead ({last[-1].get('split', '')} {last[-1].get('verdict', '')})" if last else "dead"


SHORT = {"a_p_below_bar": "5a p>=bar", "b_both_halves_positive": "5b half<=0", "c_concentration_ok": "5c conc>30%",
         "d_min_n": "5d n<min", "e_clv_nonnegative": "5e CLV<0/null", "too_good_guard": "too-good->audit"}


def _fmt(x, f="{:+.2%}"):
    return "-" if x is None else f.format(x)


def status_rows(reg: Registry) -> list[dict]:
    out = []
    counts = reg.family_counts()
    for t in sorted(runnable(reg), key=lambda t: (t["family"], t["spec"]["id"])):
        tid = t["trial_id"]
        tr = [x for x in reg.events() if x["event"] == "transition" and x["trial_id"] == tid and x.get("split")]
        e = None
        if reg.state(tid) == "dead" and tr:
            e = latest(reg, tid, tr[-1]["split"])
        for sp in ("forward", "sealed", "dev"):
            x = latest(reg, tid, sp)
            if e is None and x and x.get("n"):
                e = x
        e = e or latest(reg, tid, "dev") or latest(reg, tid, "forward") or {}
        ci = e.get("ci") or [None, None]
        lg = legacy(reg, tid) or {}
        out.append({
            "strategy": t["spec"]["id"], "trial_id": tid, "family": t["family"],
            "family_trials": counts.get(t["family"]), "bar": ALPHA / counts.get(t["family"], 1),
            "state": reg.state(tid), "split": e.get("split"), "verdict": e.get("verdict"), "n": e.get("n"),
            "net_roi": e.get("net_roi"), "ci": ci, "p": e.get("p"),
            "clv": (e.get("clv") or {}).get("mean"), "reasons": e.get("reasons") or (
                list((e.get("pit_violations") or {}).keys()) if e.get("verdict") == "pit_rejected" else []),
            "next_gate": next_gate(reg, t), "legacy": lg.get("note"),
            "sealed_replay": {k: (latest(reg, tid, "sealed") or {}).get(k) for k in ("verdict", "n", "net_roi", "p")},
        })
    return out


def status_text(reg: Registry) -> str:
    rows = status_rows(reg)
    h = f"{'strategy':20s} {'family (trials, bar)':30s} {'state':13s} {'split':7s} {'n':>5s} {'net ROI [95% CI]':26s} " \
        f"{'p':>8s} {'CLV':>7s}  next gate / why"
    lines = [h, "-" * len(h)]
    for r in rows:
        ci = r["ci"]
        roi = "-" if r["net_roi"] is None else f"{r['net_roi']:+.2%} [{ci[0]:+.2%},{ci[1]:+.2%}]"
        fam = f"{r['family']} ({r['family_trials']}, {r['bar']:.1e})"
        why = r["next_gate"]
        if r["reasons"] and r["state"] in ("dead", "registered"):
            why += ": " + ",".join(SHORT.get(x, x) for x in r["reasons"])
        lines.append(f"{r['strategy']:20s} {fam:30s} {r['state']:13s} {str(r['split'] or '-'):7s} "
                     f"{str(r['n'] if r['n'] is not None else '-'):>5s} {roi:26s} {_fmt(r['p'], '{:.1e}'):>8s} "
                     f"{_fmt(r['clv'], '{:+.4f}'):>7s}  {why}")
        sr = r["sealed_replay"]
        if r["split"] != "sealed" and sr.get("verdict"):
            lines.append(f"{'':20s}   sealed replay: {sr['verdict']} n {sr['n']} ROI {_fmt(sr['net_roi'])} p {_fmt(sr['p'], '{:.1e}')}")
        if r["legacy"]:
            lines.append(f"{'':20s}   legacy (not a verdict): {r['legacy']}")
    v = reg.verify()
    lines.append(f"registry {reg.path} lines {v['lines']} hash chain {'OK' if v['ok'] else 'BROKEN at line ' + str(v['first_break'])}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ evidence
EVIDENCE_FIELDS = ["ladder_state", "sealed_unseal_count", "sealed_pass", "forward_p_one_sided_net_roi",
                   "n_trials_family_all_time", "forward_roi_half1", "forward_roi_half2",
                   "forward_max_cluster_profit_share", "forward_n_meets_preset_min", "forward_n", "forward_mean_clv",
                   "forward_net_roi", "needs_audit", "pit_violations", "placebo_false_pass_rate",
                   "golden_cases_reproduce", "clv_negative_significant", "preset_kill_triggered", "kill_file_present",
                   "spec_registered_at", "first_data_access_at", "spec_hash_matches_registered"]


def evidence(reg: Registry, name: str) -> dict:
    from . import migrate
    t = resolve(reg, name)
    tid, spec = t["trial_id"], t["spec"]
    fwd = latest(reg, tid, "forward")
    fwd_scored = fwd if (fwd and fwd.get("n")) else None      # n = 0 / rejected -> forward numbers are null
    sealed = latest(reg, tid, "sealed")
    accesses = reg.accesses(tid)
    cfg = vault.config().get(spec.get("dataset"), {})
    spent = [s for s in cfg.get("spent_unseals") or [] if s["spec_id"] == spec["id"]]
    unseals = len(spent) + sum(1 for a in accesses if a["split"] == "sealed" and not a.get("replay"))
    st = json.loads(SELFTEST.read_text()) if SELFTEST.exists() else {}
    try:
        reg.check_unchanged(tid)
        unchanged = True
    except Exception:
        unchanged = False
    hist = migrate.first_looks().get(spec["id"])
    first_access = hist if hist else (accesses[0]["ts"] if accesses else None)
    pit = None
    lat = [x for x in (latest(reg, tid, s) for s in ("dev", "sealed", "forward")) if x]
    if lat:
        pit = sum(sum(v for k, v in (x.get("pit_violations") or {}).items() if not k.startswith("_")) for x in lat)
    kill = (fwd or {}).get("kill") or {}
    pk = reg.preset_kill(tid)
    if pk and fwd_scored and kill.get("source") != "preset":
        raise RegistryError(f"{tid}: latest scored forward evaluation did not use the pre-set kill; re-evaluate")
    conc = (fwd_scored or {}).get("concentration") or {}
    clv = (fwd_scored or {}).get("clv") or {}
    kfile = paths.KALSHI_KILL.exists() if str(spec.get("dataset", "")).startswith("kalshi") else None
    ev = {
        "ladder_state": reg.state(tid),
        "sealed_unseal_count": unseals,
        "sealed_pass": (sealed.get("verdict") == "pass") if sealed else None,
        "forward_p_one_sided_net_roi": (fwd_scored or {}).get("p"),
        "n_trials_family_all_time": reg.family_counts().get(t["family"]),
        "forward_roi_half1": (fwd_scored or {}).get("half1"),
        "forward_roi_half2": (fwd_scored or {}).get("half2"),
        "forward_max_cluster_profit_share": conc.get("share"),
        "forward_n_meets_preset_min": (fwd_scored["n"] >= int(spec["min_n"])) if fwd_scored else None,
        "forward_n": fwd.get("n") if fwd else None,
        "forward_mean_clv": clv.get("mean"),
        "forward_net_roi": (fwd_scored or {}).get("net_roi"),
        "needs_audit": (fwd_scored or {}).get("needs_audit") if fwd_scored else None,
        "pit_violations": pit,
        "placebo_false_pass_rate": st.get("placebo_false_pass_rate"),
        "golden_cases_reproduce": st.get("golden_cases_reproduce"),
        "clv_negative_significant": (bool(clv["p_negative"] < 0.05) if clv.get("p_negative") is not None
                                     and (fwd_scored or {}).get("n", 0) >= 100 else None),
        "preset_kill_triggered": kill.get("triggered") if fwd_scored else None,
        "kill_file_present": kfile,
        "spec_registered_at": t["ts"],
        "first_data_access_at": first_access,
        "spec_hash_matches_registered": unchanged,
    }
    assert list(ev) == EVIDENCE_FIELDS
    srcs = {str(reg.path), str(paths.VAULT_YAML), str(SELFTEST), str(paths.GOAL), str(paths.LAB / "ledger.jsonl")}
    for x in lat:
        for s in x.get("sources") or []:
            srcs.add(s["path"])
        if x.get("rows_file"):
            srcs.add(x["rows_file"]["path"])
    if t.get("rule_path"):
        srcs.add(t["rule_path"])
    ev["_preset_kill"] = ({"source": "preset", "set_at": pk["ts"], "rules": pk["rules"]} if pk else
                          {"source": "spec", "set_at": t["ts"], "rules": [spec.get("kill")],
                           "note": "no `edgelab setkill` rule; the spec's registration-time kill applies"})
    ev["_sources"] = [{"path": p, "sha256": sha256_file(p)} for p in sorted(srcs)]
    ev["_generated_by"] = {"module": "edgelab.report.evidence", "git_sha": paths.git_head(), "trial_id": tid}
    return ev

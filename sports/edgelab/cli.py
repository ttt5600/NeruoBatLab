"""python -m edgelab <command>   (run from sports/ with ../.venv_quant/bin/python)

  status                               table: strategy, family, state, n, net ROI [CI], CLV, next gate
  register SPEC.yaml [--amend TRIAL]   register a trial (or amend one that has not touched data)
  evaluate STRATEGY --split S          dev | sealed | forward (sealed: once, after dev pass)
  selftest [--n-each 70]               placebo + golden + cheater proofs (sandbox registries)
  evidence STRATEGY --out f.json       hub-verifier evidence (null where unknown; never defaulted)
  approve STRATEGY --to live_small|live --user-approved --by NAME [--size X]   USER ONLY
  retire STRATEGY --reason TEXT        -> dead (anyone may turn a strategy off)
  migrate                              import history + evaluate the live/forward strategies (idempotent)
  verify                               check the registry hash chain
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from . import ladder, migrate, report, run, selftest
from .registry import Registry


def _print(x):
    print(json.dumps(x, indent=1, default=str))


def cmd_status(a, reg):
    print(report.status_text(reg))
    c = reg.family_counts()
    print("\nfamily bars (alpha 0.05 / all-time trials):")
    for f, n in sorted(c.items(), key=lambda x: -x[1]):
        if not f.startswith(("placebo", "cheat")):
            print(f"  {f:26s} {n:5d}  p < {0.05 / n:.2e}")
    return 0


def cmd_register(a, reg):
    spec = yaml.safe_load(Path(a.spec).read_text())
    tid = reg.amend(a.amend, spec) if a.amend else reg.register(spec)
    fam_bar, n = reg.family_bar(spec["family"])
    print(f"{tid}  family {spec['family']}: {n} trials all-time -> bar p < {fam_bar:.2e}")
    return 0


def cmd_evaluate(a, reg):
    t = report.resolve(reg, a.strategy)
    out = run.evaluate(t["trial_id"], a.split, reg)
    keep = ["trial_id", "split", "verdict", "n", "net_roi", "ci", "p", "bar", "n_trials_family", "half1", "half2",
            "concentration", "clv", "sharpe_ann", "needs_audit", "reasons", "kill", "state", "error", "pit_violations"]
    _print({k: out.get(k) for k in keep if k in out})
    return 0


def cmd_selftest(a, reg):
    out = selftest.selftest(a.n_each, log=lambda *x: None)
    _print({"placebo": out["placebo"], "golden": out["golden"],
            "cheaters": {k: (v.get("refused", v.get("ok")) if isinstance(v, dict) else v)
                         for k, v in out["cheaters"].items()},
            "ok": out["ok"], "written": str(selftest.OUT)})
    return 0 if out["ok"] else 1


def cmd_evidence(a, reg):
    ev = report.evidence(reg, a.strategy)
    Path(a.out).write_text(json.dumps(ev, indent=1, default=str))
    print(f"wrote {a.out}")
    return 0


def cmd_approve(a, reg):
    if not a.user_approved:
        print("refused: user-only transition; pass --user-approved --by NAME (recorded with a timestamp)")
        return 1
    t = report.resolve(reg, a.strategy)
    ap = ladder.UserApproval.make(a.by, note=a.note or "")
    if a.size is not None and not a.to:
        reg.append({"event": "user_approval", "trial_id": t["trial_id"], "size": a.size,
                    "approval": {"by": ap.by, "when": ap.when, "note": ap.note}})
        print(f"size {a.size} recorded for {t['trial_id']} by {ap.by} at {ap.when}")
        return 0
    if a.to == "live_small":
        f = report.latest(reg, t["trial_id"], "forward") or {}
        if not f.get("forward_eligible_for_user"):
            print(f"refused: needs >= {ladder.FORWARD_MIN_N} forward bets passing rule 5 on forward data alone "
                  f"(latest forward: n {f.get('n')}, verdict {f.get('verdict')})")
            return 1
    ladder.transition(reg, t["trial_id"], a.to, "user", ap, size=a.size)
    print(f"{t['trial_id']} -> {a.to} (by {ap.by} at {ap.when})")
    return 0


def cmd_retire(a, reg):
    t = report.resolve(reg, a.strategy)
    ladder.transition(reg, t["trial_id"], "dead", "retire", reason=a.reason)
    print(f"{t['trial_id']} -> dead ({a.reason})")
    return 0


def cmd_migrate(a, reg):
    migrate.migrate(reg)
    return 0


def cmd_verify(a, reg):
    _print(reg.verify())
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="edgelab", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", help="registry path (default lab/edgelab_registry.jsonl)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    p = sub.add_parser("register"); p.add_argument("spec"); p.add_argument("--amend"); p.set_defaults(fn=cmd_register)
    p = sub.add_parser("evaluate"); p.add_argument("strategy")
    p.add_argument("--split", required=True, choices=["dev", "sealed", "forward"]); p.set_defaults(fn=cmd_evaluate)
    p = sub.add_parser("selftest"); p.add_argument("--n-each", type=int, default=70); p.set_defaults(fn=cmd_selftest)
    p = sub.add_parser("evidence"); p.add_argument("strategy"); p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_evidence)
    p = sub.add_parser("approve"); p.add_argument("strategy"); p.add_argument("--to", choices=["live_small", "live"])
    p.add_argument("--user-approved", action="store_true"); p.add_argument("--by", default="")
    p.add_argument("--size", type=float); p.add_argument("--note"); p.set_defaults(fn=cmd_approve)
    p = sub.add_parser("retire"); p.add_argument("strategy"); p.add_argument("--reason", required=True)
    p.set_defaults(fn=cmd_retire)
    sub.add_parser("migrate").set_defaults(fn=cmd_migrate)
    sub.add_parser("verify").set_defaults(fn=cmd_verify)
    a = ap.parse_args(argv)
    reg = Registry(a.registry) if a.registry else Registry()
    return a.fn(a, reg)


if __name__ == "__main__":
    sys.exit(main())

"""Sandbox re-evaluation of every trial with a forward evaluation (or at sealed_pass/forward_paper), under BAR_VERSION 1
and 2, on COPIES of lab/edgelab_registry.jsonl (the real registry is never written). Run from sports/:
  ../.venv_quant/bin/python results/ops_goal_amendment/sandbox_forward.py  -> sandbox_forward.json
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

SPORTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SPORTS))
from edgelab import paths, run  # noqa: E402
from edgelab.registry import Registry, sha256_file  # noqa: E402

OUT = Path(__file__).resolve().parent
real = paths.REGISTRY
sha_before = sha256_file(real)
base = Registry(real)
tids = sorted({e["trial_id"] for e in base.events() if e["event"] == "evaluate" and e["split"] == "forward"} |
              {t for t in base.trials() if base.state(t) in ("sealed_pass", "forward_paper", "live_small", "live")})
res = {}
for bv in (1, 2):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "reg.jsonl"
        shutil.copy(real, p)
        reg = Registry(p)
        for tid in tids:
            st0 = reg.state(tid)
            try:
                o = run.evaluate(tid, "forward", reg, save_rows=False, bar_version=bv)
                ft = o.get("forward_test") or {}
                r = {"state_before": st0, "state_after": reg.state(tid), "verdict": o.get("verdict"), "n": o.get("n"),
                     "net_roi": o.get("net_roi"), "kill": (o.get("kill") or {}).get("triggered"),
                     "kill_rules": [(k["kind"], k.get("triggered")) for k in (o.get("kill") or {}).get("rules", [])],
                     "eligible": o.get("forward_eligible_for_user"), "needs_audit": o.get("needs_audit"),
                     "reasons": o.get("reasons"),
                     "ft": {k: ft.get(k) for k in ("kind", "status", "llr", "A", "B", "n_used", "gate_ok", "months_elapsed",
                                                   "mean_cost", "why")} if ft else None}
            except Exception as e:
                r = {"state_before": st0, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            res.setdefault(tid, {})[f"v{bv}"] = r
assert sha256_file(real) == sha_before, "real registry changed!"
changes = {t: (v["v1"].get("state_after"), v["v2"].get("state_after")) for t, v in res.items()
           if v["v1"].get("state_after") != v["v2"].get("state_after")}
promoted = [t for t, v in res.items() if v["v2"].get("eligible")]
out = {"registry_sha256": sha_before, "trials": res, "v1_vs_v2_state_changes": changes, "v2_eligible_for_user": promoted}
(OUT / "sandbox_forward.json").write_text(json.dumps(out, indent=1, default=str))
for t, v in res.items():
    a, b = v["v1"], v["v2"]
    print(f"{t:34s} {a.get('state_before', '-'):13s} v1->{a.get('state_after', a.get('error', ''))!s:13s} "
          f"v2->{b.get('state_after', b.get('error', ''))!s:13s} n={b.get('n')} ft={(b.get('ft') or {}).get('kind')}/"
          f"{(b.get('ft') or {}).get('status')} llr={(b.get('ft') or {}).get('llr')} elig={b.get('eligible')}")
print("state changes v1->v2:", changes, "| v2 eligible:", promoted)

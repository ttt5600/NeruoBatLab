"""Synthetic check of edgelab/adapters/e4p_fwd.py: dry bracket rows (re-dated to look pre-event) + fake settles -> bets ->
schema/PIT enforce, both variants. Run from sports/ (writes only to results/ops_e4p_forward/)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import edgelab.adapters.e4p_fwd as A  # noqa: E402
from edgelab import schema  # noqa: E402

out = Path("results/ops_e4p_forward/synthetic.jsonl")
dry = [json.loads(l) for l in open("results/ops_e4p_forward/dry_1116.jsonl")]
bids = [r for r in dry if r.get("kind") == "bracket" and r.get("status") == "bid"]
other = [r for r in dry if r.get("kind") == "bracket" and not r.get("sel_c")][:3]
rows = []
# (sel_cagree override, tt_fill, result, e4_live_final)
spec = [(True, True, "no", False), (False, True, "yes", False), (True, False, "no", False), (True, True, "no", True)]
for r, (ca, tt, res, late) in zip(bids, spec):
    p = {**r, "event_day": "2026-11-17", "decided_at": "2026-11-17T03:00:00+00:00", "sel_cagree": ca}
    p.pop("dry", None)
    rows.append(p)
    rows.append({"kind": "settle", "ticker": p["ticker"], "result": res, "close_time": "2026-11-18T07:59:00Z",
                 "tt_fill": tt, "fill_ts": "2026-11-17T09:00:00+00:00" if tt else None, "close_no": 0.98,
                 "e4_live_final": late})
rows += [{**r, "event_day": "2026-11-17", "decided_at": "2026-11-17T03:00:00+00:00"} for r in other]
out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
A.qs_or_repo = lambda rel: out
fr, src = A.load({})
print(fr.attrs["loader_report"])
for v, exp in (("C", 2), ("C_agree", 1)):
    b = A.rows(fr, {"variant": v})
    print(v, b.attrs["adapter_report"])
    assert len(b) == exp, (v, len(b))
    b = b.copy()
    b["outcome"] = [1.0 if fr.set_index("bet_id").loc[t, "result"] == "no" else 0.0 for t in b.bet_id]
    df = schema.enforce(b)
    print(df[["bet_id", "event_start", "decided_at", "price", "price_time", "fill_deadline"]].to_string())
    print("PIT ok:", schema.pit_check(df)[0])

"""Synthetic check of edgelab/adapters/crypto_barrier_v2_fwd (load + rows + schema.enforce) on a TEMP log (no real log)."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from edgelab.adapters import crypto_barrier_v2_fwd as A  # noqa: E402
from edgelab import schema  # noqa: E402

base = {"series": "KXBTCMAXMON", "event": "E", "asset": "BTC", "barrier": "max", "strike": 82500.0, "spot": 64000.0,
        "sigma": .43, "sigma_b": .55, "tau_d": 14, "d": .25, "p0": .02, "p1": .03, "p2": .01, "ps": .02,
        "yes_bid": .07, "yes_ask": .09, "quote_time": "2026-11-16T16:00:40+00:00",
        "decided_at": "2026-11-16T16:00:41+00:00", "window_end": "2026-12-01T04:59:59+00:00", "kind": "pick",
        "date": "2026-11-16", "side": "no", "ask": .93, "p_cons": None, "edge": .03, "fee": .0046,
        "inputs_as_of": "2026-11-16T16:00:30+00:00", "cluster": "crypto_month|2026-11", "late_min": 0.6}
rows = [dict(base, rule="CB3", ticker="A", status="bet"), dict(base, rule="CB3", ticker="B", status="off_time"),
        dict(base, rule="CB3", ticker="C", status="bet"), dict(base, rule="CB2", ticker="A", status="bet"),
        {"kind": "day", "uday": "2026-11-16"},
        {"kind": "settle", "ticker": "A", "result": "no"}, {"kind": "settle", "ticker": "B", "result": "yes"}]
tmp = Path(tempfile.mkdtemp()) / "crypto_barrier_v2.jsonl"
tmp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
A.qs_or_repo = lambda _rel: tmp                              # point the loader at the temp log only
for rule, expect in (("CB3", 1), ("CB2", 1)):
    fr, _ = A.load({"rule": rule})
    b = A.rows(fr, {"rule": rule})
    print(rule, b[["bet_id", "cluster", "price", "side"]].to_dict("records"), b.attrs["adapter_report"])
    b["outcome"] = 1.0
    schema.enforce(b)
    assert len(b) == expect, (rule, len(b))
print("PIT ok; CB3 1 bet (A; B off_time; C unsettled), CB2 1 bet (A) -- rules separated, shared settle")

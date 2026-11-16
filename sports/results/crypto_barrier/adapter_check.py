"""Synthetic check of edgelab/adapters/crypto_barrier_fwd.rows + schema.enforce (no real log touched)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from edgelab.adapters import crypto_barrier_fwd as A  # noqa: E402
from edgelab import schema  # noqa: E402

base = {"series": "KXBTCMINMON", "event": "E", "asset": "BTC", "barrier": "min", "strike": 60000.0, "spot": 64000.0,
        "sigma": .43, "tau_d": 14, "d": .06, "p0": .4, "p1": .3, "p2": .46, "yes_bid": .42, "yes_ask": .43,
        "quote_time": "2026-11-16T16:00:40+00:00", "decided_at": "2026-11-16T16:00:41+00:00",
        "window_end": "2026-12-01T04:59:59+00:00", "kind": "pick", "date": "2026-11-16", "side": "no", "ask": .58,
        "p_cons": .54, "edge": .05, "fee": .017, "inputs_as_of": "2026-11-16T16:00:30+00:00",
        "cluster": "crypto_month|2026-11", "late_min": 0.6}
rows = [dict(base, ticker="A", status="bet"), dict(base, ticker="B", status="off_time"),
        dict(base, ticker="C", status="bet")]
d = pd.DataFrame(rows)
d["bet_id"] = d.ticker
d["date"] = pd.to_datetime(d["date"])
d["result"] = d.ticker.map({"A": "no", "B": "yes"})
d["settled"] = d["result"].isin(["yes", "no"])
b = A.rows(d, {})
print(b[["bet_id", "cluster", "event_start", "decided_at", "inputs_as_of", "price", "price_time", "side"]].to_string())
print(b.attrs["adapter_report"])
b["outcome"] = 1.0
schema.enforce(b)
print("PIT ok; bets", len(b), "(expect 1: A bet+settled; B off_time; C unsettled)")

"""CB3 would-fire COUNT on the DEV frame (power only). Reads NO outcome column: only quote/model/timing columns."""
import sys

import pandas as pd

sys.path.insert(0, "scripts")
import crypto_barrier_v2_model as V  # noqa: E402

f = "results/crypto_barrier/frame.parquet"
cols = pd.read_parquet(f, columns=None).columns.tolist() if False else None
import pyarrow.parquet as pq  # noqa: E402

names = pq.read_schema(f).names
print("schema:", names)
need = [c for c in ("ticker", "event", "barrier", "kind", "yes_bid", "yes_ask", "bid", "ask", "p1", "tau_d", "date", "month")
        if c in names]
d = pd.read_parquet(f, columns=need)
kind = "barrier" if "barrier" in d else "kind"
bid = "yes_bid" if "yes_bid" in d else "bid"
ask = "yes_ask" if "yes_ask" in d else "ask"
fire = [bool(V.decide_cb3(r[kind], r[bid], r[ask], r["p1"], r["tau_d"])) for _, r in d.iterrows()]
d["fire"] = fire
first = d[d.fire].drop_duplicates("ticker")
print("rows", len(d), "fire rows", int(d.fire.sum()), "markets (first firing)", len(first),
      "events", first["event"].nunique() if "event" in first else None)
if "event" in first:
    print(first.groupby("event").size().to_string())

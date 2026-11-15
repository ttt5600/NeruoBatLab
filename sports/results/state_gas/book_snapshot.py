"""Read-only snapshot of the active KXAAAGASCD* books (top of book + depth). Pre-outcome data only."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from kx_throttle import kget  # noqa: E402

d = json.loads((ROOT / "results/state_gas/census_raw.json").read_text())
rows = []
for st, x in d.items():
    for m in x["markets"]:
        if m["status"] != "active":
            continue
        ob = (kget(f"/markets/{m['ticker']}/orderbook") or {}).get("orderbook_fp") or {}
        yes = ob.get("yes_dollars") or []
        no = ob.get("no_dollars") or []
        rows.append({"state": st, "ticker": m["ticker"], "bucket": m["ticker"].split("-")[-1],
                     "yes_bid": m["yes_bid_dollars"], "yes_ask": m["yes_ask_dollars"],
                     "yes_bid_size": m.get("yes_bid_size_fp"), "yes_ask_size": m.get("yes_ask_size_fp"),
                     "volume": m["volume_fp"], "oi": m["open_interest_fp"],
                     "yes_depth": sum(float(q) for _, q in yes), "no_depth": sum(float(q) for _, q in no),
                     "yes_levels": len(yes), "no_levels": len(no), "ts": time.time()})
(ROOT / "results/state_gas/book_snapshot_2026-11-15.json").write_text(json.dumps(rows))
print(len(rows))

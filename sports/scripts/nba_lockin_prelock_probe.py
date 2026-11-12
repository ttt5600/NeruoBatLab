"""Read-only one-shot probe (agent nba_lockin_fill): for every NBA game in progress, the leader's KXNBAGAME book now --
99c YES queue, best bid/ask -- with lead and clock. Appends to results/nba_lockin_fill/prelock_probe.jsonl.
Descriptive only (data from 2026-11-11 onward are NOT used to choose trial parameters)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import lock_queue_logger as LQ  # noqa: E402

OUT = ROOT / "results" / "nba_lockin_fill" / "prelock_probe.jsonl"


def main() -> None:
    now = time.time()
    rows = []
    for g in LQ.fetch_games(LQ.et_dates(now)[1:], LQ._get):
        if g["state"] != "in":
            continue
        m = g["hs"] - g["as"]
        if m == 0:
            continue
        lead = g["home"] if m > 0 else g["away"]
        tk = f"KXNBAGAME-{g['ev']}-{LQ.k(lead)}"
        try:
            b = LQ.book_summary(LQ._get(f"{LQ.KAPI}/markets/{tk}/orderbook", {"depth": 0}))
        except Exception as ex:  # noqa: BLE001
            b = {"error": str(ex)[:100]}
        r = dict(ts=now, ticker=tk, period=g["period"], clock=g["clock"], lead=abs(m), **b)
        rows.append(r)
        print(json.dumps({k: r.get(k) for k in ("ticker", "period", "clock", "lead", "yes_99", "best_bid_c", "best_bid_sz", "best_ask_c")}))
    with OUT.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()

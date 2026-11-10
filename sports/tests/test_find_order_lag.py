"""Regression for incident (a) of 2026-11-09 (notes/incident_find_order_miss.md): WHY the exit lookup missed fills.

Measured on prod (scripts/diag_find_order.py, read-only): every incident exit order IS listed now by GET
/portfolio/orders?ticker=T with status "executed" and fill_count_fp == count, single page, no status filter needed.
But in the live journal, the lookup made right after the POST did not list the new order on 27 of 28 live POSTs;
all 27 were found on a later tick. The old exit read only that immediate lookup and threw away the POST response,
whose `fill_count` (no _fp suffix on the create response) is the synchronous truth.

These tests model the listing lag and a stale positions read, and require the POST response to be honoured.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import lead_exec as lx  # noqa: E402
import order_guard as og  # noqa: E402
from test_lead_exec import FakeClient  # noqa: E402
from test_ingame_exec import NF, run  # noqa: E402
from test_order_guard import at, ex, filled_nflf, lagged, realistic, sells  # noqa: E402,F401


def test_resp_fill_prod_shapes():
    # the create response actually logged on prod 2026-11-10 (lead_exec.jsonl): strings, no _fp, no status
    assert og.resp_fill({"client_order_id": "c", "fill_count": "8.00", "order_id": "o", "remaining_count": "0.00",
                         "ts_ms": 1794278611048}) == 8.0
    assert og.resp_fill({"fill_count": "0.00"}) == 0.0
    assert og.resp_fill({"order": {"fill_count_fp": "3.00"}}) == 3.0
    assert og.resp_fill({"error": "x"}) == 0.0 and og.resp_fill(None) == 0.0 and og.resp_fill({"fill_count": "?"}) == 0.0


def test_listing_lag_alone_reproduces_the_old_miss(ex):
    """The old exit's lookup (find_order right after POST) returns None for an order that filled."""
    c = FakeClient(realistic(ex, hide_exits=True))
    ex.positions[NF] = 3
    og.safe_reduce(c, NF, "yes", 3, 0.43, "cid-lag", strategy="T", cap=5, on_trip=lambda w: None)
    assert ex.positions[NF] == 0                       # it filled on the exchange ...
    assert og._find(c, NF, "cid-lag") is None          # ... and the immediate lookup cannot see it


def test_post_response_counts_when_listing_and_positions_lag(ex):
    c = FakeClient(realistic(ex, hide_exits=True))
    ex.positions[NF] = 3
    lagged(ex, stale_reads=50)                         # every positions read after the POST is stale
    r = og.safe_reduce(c, NF, "yes", 3, 0.43, "cid-1", strategy="T", cap=5, on_trip=lambda w: None)
    assert r["filled"] == 3 and r["resp_fill"] == 3


def test_incident_replay_lagged_exchange_one_sell_booked_first_tick(ex, tmp_path):
    realistic(ex, hide_exits=True)
    sig = filled_nflf(ex, tmp_path)
    lagged(ex, stale_reads=50)
    run(ex, [], at=at(sig, 485))
    r = lx.fold(lx.journal(), "live")[sig["key"]]
    assert r["settled"] and r["exit_filled"] == 4 and r["result"] == "exited" and not r.get("exit_pending")
    for k in range(1, 6):
        run(ex, [], at=at(sig, 485 + 15 * k))
    assert len(sells(ex)) == 1 and ex.positions[NF] == 0

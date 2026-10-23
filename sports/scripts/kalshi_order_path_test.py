"""Zero-cost PROD order-path test for the Kalshi executor (2026-10-22).

    ~/.claude/quant-kalshi/.venv/bin/python scripts/kalshi_order_path_test.py list            # pick a market
    ~/.claude/quant-kalshi/.venv/bin/python scripts/kalshi_order_path_test.py run TICKER      # rest, dupe, read, cancel

Uses the DEPLOYED executor's client + .env (~/.claude/quant-kalshi), so it signs exactly as the launchd run does.
`run` rests ONE post-only buy-NO at $0.01 (= sell YES at 0.99) on a market whose best YES bid is far below 0.99, so it
cannot fill (post_only also rejects instead of crossing). Re-sends the same client_order_id (expects 409), reads the
order back, cancels it, confirms it is gone from resting orders, and journals one `order_path_test` row (keyed
`market`, not `ticker`, so the executor's fold/reconcile ignore it). Worst case if it somehow filled: $0.01 + fee.
Never prints the key or headers.
"""
from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

DEPLOY = Path.home() / ".claude" / "quant-kalshi"
sys.path.insert(0, str(DEPLOY / "scripts"))
import kalshi_exec as K  # noqa: E402

K.set_root(DEPLOY)


def client():
    e = K.env()
    cfg = K.config(e)
    return K.Kalshi(e, cfg["env"]), cfg


def cmd_list(series="KXHIGHNY"):
    c, cfg = client()
    code, out = c.request("GET", "/markets", params={"series_ticker": series, "status": "open", "limit": 50}, auth=False)
    print(f"[{cfg['env']}] HTTP {code}")
    for m in out.get("markets", []):
        print(m["ticker"], "yes_bid", m.get("yes_bid_dollars"), "yes_ask", m.get("yes_ask_dollars"),
              "vol", m.get("volume_fp") or m.get("volume"), "close", m.get("close_time"))


def cmd_run(t: str):
    c, cfg = client()
    if K.killed():
        print("KILL present -> refusing")
        return 1
    m = c.market(t) or {}
    yb = float(m.get("yes_bid_dollars") or 0)
    if m.get("status") not in ("active", "open") or yb >= 0.90:
        print(f"refusing: status {m.get('status')} yes_bid {yb} (need open market with yes_bid < 0.90)")
        return 1
    coid = f"pathtest-{uuid.uuid4()}"
    body = {"ticker": t, "side": "ask", "count": "1", "price": "0.9900", "time_in_force": "good_till_canceled",
            "post_only": True, "self_trade_prevention_type": "taker_at_cross", "client_order_id": coid}
    if "--expiration" in sys.argv:  # maker-mode body: rest until the market's close (field never sent on prod before 10-23)
        body["expiration_time"] = int(datetime.fromisoformat(m["close_time"].replace("Z", "+00:00")).timestamp())
    steps = []
    bal0 = c.balance()
    c1, o1 = c.create_order(body)
    print("create response fields:", {k: v for k, v in o1.items() if k != "client_order_id"})  # schema check
    oid = o1.get("order_id")
    steps.append(("create (post-only buy NO @ $0.01)", c1, f"order_id {'set' if oid else None} fill_count {o1.get('fill_count')}",
                  c1 in (200, 201) and float(o1.get("fill_count") or 0) == 0))
    c2, o2 = c.create_order(body)
    dup_oid = o2.get("order_id")
    steps.append(("duplicate client_order_id", c2, str(o2.get("error") or o2.get("code") or "")[:80], c2 == 409))
    if oid:
        c3, o3 = c.request("GET", f"/portfolio/orders/{oid}")
        o = o3.get("order", {})
        steps.append(("get order", c3, f"status {o.get('status')} outcome_side {o.get('outcome_side')} "
                      f"no_price {o.get('no_price_dollars')} side {o.get('side')}",
                      c3 == 200 and o.get("status") == "resting"))
        listed = [x for x in c.orders(ticker=t, status="resting") if x.get("order_id") == oid]
        steps.append(("listed in /portfolio/orders?status=resting", 200, f"{len(listed)} match", len(listed) == 1))
        bal1 = c.balance()
        steps.append(("balance reserved while resting", 200, f"${bal0:.2f} -> ${bal1:.2f}", bal1 <= bal0))
        c4, o4 = c.cancel(oid, t)
        steps.append(("cancel", c4, f"reduced_by {o4.get('reduced_by') or o4.get('reduced_by_fp')}", c4 == 200))
        if dup_oid and dup_oid != oid:  # dedupe failed: cancel the second order too
            c5, _ = c.cancel(dup_oid, t)
            steps.append(("cancel duplicate (dedupe FAILED)", c5, "", False))
        # 2026-10-22 first run: a read 0.3 s after a 200 cancel still said "resting" (read-after-write lag;
        # last_update_time showed it cancelled ~1.4 s after create). Poll up to ~6 s.
        import time
        for _ in range(6):
            time.sleep(1)
            still = [x for x in c.orders(ticker=t, status="resting") if x.get("client_order_id") == coid]
            c6, o6 = c.request("GET", f"/portfolio/orders/{oid}")
            if not still and o6.get("order", {}).get("status") == "canceled":
                break
        steps.append(("confirm cancelled", c6, f"status {o6.get('order', {}).get('status')}; resting matches {len(still)}",
                      not still and o6.get("order", {}).get("status") == "canceled"))
        pos = [p for p in c.positions(ticker=t) if float(p.get("position_fp") or 0) != 0]
        bal2 = c.balance()
        steps.append(("no position, balance restored", 200, f"positions {len(pos)}; balance ${bal2:.2f}",
                      not pos and abs(bal2 - bal0) < 1e-9))
    ok = True
    for name, code, info, good in steps:
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  {name:42s} HTTP {code}  {info}")
    K.append({"event": "order_path_test", "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "mode": "live-test", "env": cfg["env"], "market": t,
              "steps": [(s[0], s[1], s[3]) for s in steps], "pass": ok})
    print(f"\norder-path test on {cfg['env']} market {t}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def cmd_check(t: str):
    """Read-only: every order of ours on t, all resting orders, positions, balance."""
    c, cfg = client()
    xs = c.orders(ticker=t)
    if xs:
        print("order object fields:", sorted(xs[0]))
    for x in xs:
        print({k: x.get(k) for k in ("status", "client_order_id", "remaining_count_fp", "fill_count_fp",
                                     "initial_count_fp", "taker_fill_cost_dollars", "taker_fees_dollars",
                                     "created_time", "last_update_time")})
    print("resting (all tickers):", [(x.get("ticker"), x.get("client_order_id")) for x in c.orders(status="resting")])
    print("positions:", len(c.positions()), "balance", c.balance())


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["check"] and len(a) == 2:
        cmd_check(a[1])
    elif a[:1] == ["list"]:
        cmd_list(*a[1:2])
    elif a[:1] == ["run"] and len(a) >= 2:
        raise SystemExit(cmd_run(a[1]))
    else:
        print(__doc__)

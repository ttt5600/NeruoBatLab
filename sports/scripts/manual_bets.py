"""Manual Kalshi bets authorized by the user (2026-10-26: "use my $25 bonus on Kalshi to place the bets").

    ~/.claude/quant-kalshi/.venv/bin/python scripts/manual_bets.py balance
    ~/.claude/quant-kalshi/.venv/bin/python scripts/manual_bets.py quote TICKER
    ~/.claude/quant-kalshi/.venv/bin/python scripts/manual_bets.py place TICKER yes|no PRICE COUNT [--taker]
    ~/.claude/quant-kalshi/.venv/bin/python scripts/manual_bets.py orders | cancel ORDER_ID TICKER

Default = POST-ONLY maker limit (rests, never crosses; expires at the market's close), so no taker fee and no paying
the spread. PRICE is the price of the side you buy (YES price for yes, NO price for no). Hard cap: total manual
exposure <= MANUAL_CAP_USD (25). Journal: ~/.claude/quant-kalshi/data/manual_bets.jsonl (separate from the weather
executor; its place() skips tickers that already have an order/position, and these tickers are not weather series).
Uses the DEPLOYED executor's signed client + .env. Never prints the key.
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

DEPLOY = Path.home() / ".claude" / "quant-kalshi"
sys.path.insert(0, str(DEPLOY / "scripts"))
import kalshi_exec as K  # noqa: E402

K.set_root(DEPLOY)
JOURNAL = DEPLOY / "data" / "manual_bets.jsonl"
MANUAL_CAP_USD = 25.0


def client():
    e = K.env()
    cfg = K.config(e)
    return K.Kalshi(e, cfg["env"]), cfg


def journal() -> list[dict]:
    return [json.loads(l) for l in JOURNAL.read_text().splitlines() if l.strip()] if JOURNAL.exists() else []


def committed() -> float:
    return sum(r.get("max_cost", 0) for r in journal() if r.get("status") == "placed")


def cmd_balance() -> int:
    c, cfg = client()
    print(f"[{cfg['env']}] balance ${c.balance():.2f}; manual committed ${committed():.2f} / cap ${MANUAL_CAP_USD:.2f}")
    return 0


def cmd_quote(t: str) -> int:
    c, _ = client()
    m = c.market(t)
    if not m:
        print("no such market")
        return 1
    print(t, "|", m.get("title"), "|", m.get("yes_sub_title"), "| status", m.get("status"), "| close", m.get("close_time"))
    print("  yes bid/ask", m.get("yes_bid_dollars"), m.get("yes_ask_dollars"), " no bid/ask", m.get("no_bid_dollars"),
          m.get("no_ask_dollars"), " vol", m.get("volume_fp") or m.get("volume"))
    return 0


def cmd_place(t: str, side: str, price: float, count: int, taker: bool) -> int:
    c, cfg = client()
    m = c.market(t)
    if not m or m.get("status") not in ("active", "open"):
        print("market not active:", m and m.get("status"))
        return 1
    fee = K.fee_total(count, price) if taker else 0.0
    max_cost = round(count * price + fee, 4)
    if committed() + max_cost > MANUAL_CAP_USD + 1e-9:
        print(f"REFUSED: would exceed manual cap (${committed():.2f} + ${max_cost:.2f} > ${MANUAL_CAP_USD})")
        return 1
    yes_price = price if side == "yes" else round(1 - price, 2)
    coid = f"manual-{uuid.uuid4().hex[:16]}"
    body = {"ticker": t, "side": "bid" if side == "yes" else "ask", "count": str(count), "price": f"{yes_price:.2f}",
            "self_trade_prevention_type": "taker_at_cross", "client_order_id": coid}
    if taker:
        body["time_in_force"] = "immediate_or_cancel"
    else:
        # expire at the game's START (passed --expire ISO) so a resting order is never picked off in-play
        exp = EXPIRE or m.get("close_time")
        body.update(time_in_force="good_till_canceled", post_only=True,
                    expiration_time=int(datetime.fromisoformat(exp.replace("Z", "+00:00")).timestamp()))
    code, out = c.create_order(body)
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "ticker": t, "side": side, "price": price,
           "count": count, "taker": taker, "max_cost": max_cost, "client_order_id": coid, "http": code,
           "order_id": (out or {}).get("order_id"), "fill_count": (out or {}).get("fill_count"),
           "status": "placed" if code in (200, 201) else "rejected", "response": str(out)[:300]}
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"HTTP {code} {rec['status']} order {rec['order_id']} filled {rec['fill_count']} | {side.upper()} {count} @ {price} {t}")
    return 0 if rec["status"] == "placed" else 1


def cmd_orders() -> int:
    c, _ = client()
    mine = {r["client_order_id"] for r in journal()}
    for o in c.orders():
        if o.get("client_order_id") in mine:
            print(o.get("ticker"), o.get("status"), "yes_price", o.get("yes_price_dollars") or o.get("yes_price"),
                  "fill", o.get("fill_count_fp") or o.get("fill_count"), "remaining",
                  o.get("remaining_count_fp") or o.get("remaining_count"), o.get("order_id"))
    for p in c.positions():
        if any(p.get("ticker") == r["ticker"] for r in journal()):
            print("POSITION", p.get("ticker"), p.get("position_fp"), "exposure", p.get("market_exposure_dollars"))
    return 0


EXPIRE = None

if __name__ == "__main__":
    a = sys.argv[1:]
    if "--expire" in a:
        i = a.index("--expire")
        EXPIRE = a[i + 1]
        del a[i:i + 2]
    if not a:
        print(__doc__)
        raise SystemExit(1)
    if a[0] == "balance":
        raise SystemExit(cmd_balance())
    if a[0] == "quote":
        raise SystemExit(cmd_quote(a[1]))
    if a[0] == "place":
        raise SystemExit(cmd_place(a[1], a[2], float(a[3]), int(a[4]), "--taker" in a))
    if a[0] == "orders":
        raise SystemExit(cmd_orders())
    if a[0] == "cancel":
        c, _ = client()
        print(c.cancel(a[1], a[2]))
        raise SystemExit(0)
    print(__doc__)
    raise SystemExit(1)

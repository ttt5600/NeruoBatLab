"""Experiment-budget executor (user 2026-11-21: "$50 experiment budget", "be aggressive"). LIVE MONEY, small.

Runs from ~/.claude/quant-kalshi (its venv, .env, kalshi_exec client, order_guard). Deployed by scripts/install_kfg1_agent.sh.

    exp_exec.py kfg1      # place $2 IOC orders for fresh KFG1 'bet' rows (picks.jsonl), once per row
    exp_exec.py status    # total experiment spend vs the $50 cap, orders placed

Rules:
- Hard cap: total cost of ALL experiment orders (data/exp_budget.jsonl, any strategy, incl. the manual OpenRouter
  order of 2026-11-21) <= BUDGET. Max loss = spend, so this is also the -$50 stop.
- KFG1: one order per pick row, only if the row is <= 20 min old, stake $2, immediate-or-cancel at the logged price
  (+0 slippage): YES -> bid at cost; NO -> ask (sell YES) at 1 - cost. Never rests, never cancels anything else.
- Stops on data/LEAD_KILL, kalshi_exec KILL, or data/EXP_STOP.
"""
from __future__ import annotations

import json
import math
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kalshi_exec as kx  # noqa: E402
import order_guard as og  # noqa: E402

BUDGET = 50.00
STAKE = 2.00
MAX_AGE = timedelta(minutes=20)
LOG = kx.DATA / "exp_budget.jsonl"
DONE = kx.DATA / "exp_kfg1_done.json"
PICKS = Path.home() / ".claude/quant-sports/data/kfg1/picks.jsonl"


def fee_unit(p: float) -> float:
    return 0.07 * p * (1 - p)


def spent() -> float:
    tot = 0.0
    if LOG.exists():
        for line in LOG.open():
            r = json.loads(line)
            if r.get("event") == "response" and r.get("code") in (200, 201):
                o = r.get("out") or {}
                n = float(o.get("fill_count") or 0)
                tot += n * (float(o.get("average_fill_price") or 0) + float(o.get("average_fee_paid") or 0))
    return round(tot, 4)


def log(rec: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def stopped() -> str | None:
    for k in (kx.DATA / "LEAD_KILL", kx.killed(), kx.DATA / "EXP_STOP"):
        if k and Path(k).exists():
            return str(k)
    return None


def kfg1() -> int:
    if (k := stopped()):
        print(f"stop file present ({k}) -> nothing done")
        return 0
    if not PICKS.exists():
        return 0
    done = set(json.loads(DONE.read_text())) if DONE.exists() else set()
    now = datetime.now(timezone.utc)
    e = kx.env()
    client = None
    for line in PICKS.open():
        r = json.loads(line)
        key = f"{r['game_id']}|{r['team']}"
        if r.get("decision") != "bet" or key in done:
            continue
        done.add(key)                       # one attempt per row, placed or not
        if now - datetime.fromisoformat(r["ts"]) > MAX_AGE:
            log({"ts": now.isoformat(), "event": "skip_stale", "strategy": "KFG1", "key": key})
            continue
        b = r["best"]
        cost = float(b["cost"])
        n = math.floor(STAKE / (cost + fee_unit(cost)))
        room = BUDGET - spent()
        n = min(n, math.floor(room / (cost + fee_unit(cost) + 1e-9)))
        if n < 1:
            log({"ts": now.isoformat(), "event": "skip_budget", "strategy": "KFG1", "key": key, "room": round(room, 4)})
            continue
        side, price = ("bid", cost) if b["side"] == "yes" else ("ask", round(1 - cost, 4))
        cid = f"exp_kfg1_{uuid.uuid4().hex[:10]}"
        body = {"ticker": b["ticker"], "side": side, "count": str(n), "price": f"{price:.4f}",
                "time_in_force": "immediate_or_cancel", "self_trade_prevention_type": "taker_at_cross",
                "client_order_id": cid}
        log({"ts": now.isoformat(), "event": "submit", "strategy": "KFG1", "key": key, "body": body,
             "edge": b.get("edge"), "p_side": b.get("p_side")})
        client = client or kx.Kalshi(e, kx.config(e)["env"])
        try:
            code, out = og.guarded_create(client, body, strategy="EXP_KFG1", cap=STAKE + 0.5,
                                          on_trip=lambda why: (kx.DATA / "EXP_STOP").write_text(why), now=now)
        except Exception as ex:  # noqa: BLE001 -- record and stop this run; never retry blind
            log({"ts": datetime.now(timezone.utc).isoformat(), "event": "error", "cid": cid, "error": str(ex)[:300]})
            break
        log({"ts": datetime.now(timezone.utc).isoformat(), "event": "response", "strategy": "KFG1", "cid": cid,
             "code": code, "out": out})
        print(f"KFG1 {key} {b['ticker']} {b['side']} n{n} @ {cost}: code {code} filled {(out or {}).get('fill_count')}")
    DONE.write_text(json.dumps(sorted(done)))
    return 0


def status() -> int:
    print(f"experiment spend ${spent():.2f} / ${BUDGET:.2f}; stop file: {stopped()}")
    if LOG.exists():
        for line in LOG.open():
            r = json.loads(line)
            if r.get("event") == "response":
                o = r.get("out") or {}
                print(" ", r["ts"][:16], r.get("strategy", "manual"), r.get("cid"), "filled", o.get("fill_count"),
                      "@", o.get("average_fill_price"))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    raise SystemExit({"kfg1": kfg1, "status": status}[cmd]())

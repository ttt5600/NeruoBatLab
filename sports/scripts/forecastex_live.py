"""FX1-live: LIVE orders for the ForecastEx weather consensus-reject picks (edge E5) on the user's IBKR account.

    python scripts/forecastex_live.py tick     # launchd every 5 min: keep session alive, order new picks
    python scripts/forecastex_live.py dry      # what tick WOULD send now (no orders)
    python scripts/forecastex_live.py report   # orders sent, fills, spend

AUTHORISED by the user 2026-10-28 ("connect and trade ... same way you make kalshi trades"), overriding the
setup note's ">= 100 settled paper picks first" gate. Risk stays inside the existing client caps:
FX_MAX_ORDER_USD 5, FX_MAX_CONTRACTS 10, FX_MAX_DAILY_USD 25 (forecastex_client.check_order), the shared KILL file,
IBKR_MODE=live + FX_ALLOW_LIVE_ACCOUNT=1 + a non-paper account id in .env.

RULE (no new selection, only execution): every FX1-fwd `pick` row that forecastex_paper.py logged (frozen
consensus-reject rule) for an event day >= today (UTC), logged within the last 6 h, gets ONE decision:
  * side ask exists and ask + 0.01 < 1  -> BUY 1 at the ask (variant A, the pre-registered primary price);
  * else side bid exists                 -> BUY 1 resting at the best bid, post-only, DAY (variant MK, maker);
  * else                                 -> skip (no quote).
Book is re-read immediately before ordering (the logged quote can be minutes to hours old). 1 contract per pick,
in log order, until the daily $ cap refuses: spreads the cap over ~25 picks instead of stacking a few.
Idempotent: tag FX1L-<ticker>; the client refuses a reused tag, so re-running never double-orders.
Journal: data/forecastex_live.jsonl (decision rows) + the client's data/forecastex_orders.jsonl (wire rows).
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from venues.forecastex_client import ForecastEx  # noqa: E402

DEPLOY = Path.home() / ".claude" / "quant-kalshi"
PICKS = DEPLOY / "data" / "forecastex_paper.jsonl"
LOG = DEPLOY / "data" / "forecastex_live.jsonl"
LOGIN = DEPLOY / "ibkr" / "login.py"
FRESH_H = 6


def _rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def _log(rec: dict) -> None:
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def due_picks(now: datetime) -> list[dict]:
    done = {r["ticker"] for r in _rows(LOG) if r.get("kind") == "decision"}
    out = []
    for r in _rows(PICKS):
        if r.get("kind") != "pick" or r.get("trial") != "FX1-fwd" or r["ticker"] in done:
            continue
        ts = datetime.fromisoformat(r["ts"])
        if now - ts > timedelta(hours=FRESH_H) or r["event_date"] < now.strftime("%Y-%m-%d"):
            continue
        out.append(r)
    return out


def ensure_session(c: ForecastEx) -> bool:
    if c.session_ok(refresh=True):
        c._call("POST", "/tickle")          # keep the ~10 min SSO timer alive
        return True
    last = [r for r in _rows(LOG) if r.get("kind") == "relogin"]
    if last and datetime.now(timezone.utc) - datetime.fromisoformat(last[-1]["ts"]) < timedelta(hours=2):
        return False                        # at most one SMS-costing re-login per 2 h
    r = subprocess.run([str(DEPLOY / ".venv/bin/python"), str(LOGIN), "--live-session"],
                       capture_output=True, text=True, timeout=400)
    _log({"kind": "relogin", "ts": datetime.now(timezone.utc).isoformat(), "rc": r.returncode,
          "out": (r.stdout + r.stderr)[-300:]})
    return c.session_ok(refresh=True)


def decide(c: ForecastEx, pick: dict, dry: bool) -> dict:
    t, side = pick["ticker"], pick["side"].lower()
    b = c.book(t)
    bid, ask = b.best(side) if b else (None, None)
    if ask is not None and ask + 0.01 < 1:
        variant, price, post = "A", round(ask, 2), False
    elif bid is not None:
        variant, price, post = "MK", round(bid, 2), True
    else:
        return {"variant": None, "status": "skip", "detail": "no quote"}
    res = c.place_limit(t, side, price, 1, post_only=post, dry=dry, tag=f"FX1L-{t}")
    return {"variant": variant, "price": price, "status": res.status, "order_id": res.order_id,
            "detail": (res.detail or "")[:200]}


def tick(dry: bool = False) -> None:
    now = datetime.now(timezone.utc)
    c = ForecastEx(live=not dry)
    if not dry and not ensure_session(c):
        print("session down; re-login pending or failed (see data/forecastex_live.jsonl)")
        return
    for p in due_picks(now):
        d = decide(c, p, dry)
        rec = {"kind": "decision" if not dry else "dry", "ts": now.isoformat(), "ticker": p["ticker"],
               "side": p["side"], "pick_ts": p["ts"], **d}
        print(json.dumps(rec))
        if not dry:
            _log(rec)
            if d["status"] == "rejected" and "daily cap" in d.get("detail", ""):
                break                       # remaining picks wait; they expire from the 6 h window


def report() -> None:
    rows = [r for r in _rows(LOG) if r.get("kind") == "decision"]
    placed = [r for r in rows if r["status"] == "placed"]
    print(f"decisions {len(rows)}  placed {len(placed)}  "
          f"(A {sum(r['variant'] == 'A' for r in placed)}, MK {sum(r['variant'] == 'MK' for r in placed)})  "
          f"committed ${sum(r['price'] + 0.01 for r in placed):.2f}")
    for r in rows[-15:]:
        print(f"  {r['ts'][:16]} {r['ticker']:<18} {r['side']:<3} {r.get('variant') or '-':<2} "
              f"{r.get('price') or '':<5} {r['status']:<9} {r.get('detail', '')[:60]}")


if __name__ == "__main__":
    cmd = (sys.argv[1:] or ["report"])[0]
    {"tick": lambda: tick(False), "dry": lambda: tick(True), "report": report}[cmd]()

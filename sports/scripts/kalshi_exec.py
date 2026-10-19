"""Kalshi order executor for the ONE lockbox-validated rule (temperature consensus-reject). DRY-RUN BY DEFAULT.

    python scripts/kalshi_exec.py check          # read-only: verify the API key signs requests (GET balance)
    python scripts/kalshi_exec.py run            # dry-run: size today's paper picks, log what WOULD be ordered
    python scripts/kalshi_exec.py run --live     # real orders -- only if sports/.env also has EXEC_MODE=live
    python scripts/kalshi_exec.py report

Inputs: picks written by kalshi_paper.py (data/kalshi_paper.jsonl). Each unexecuted pick logged within the last
3 h is re-priced from the live book; we never chase: skip unless the NO ask now is <= the pick's NO price + 1c.

Safety (all enforced here, not by the caller):
  * live requires BOTH `--live` and EXEC_MODE=live in sports/.env (the user sets that; Claude never does)
  * kill switch: a file named KILL in sports/ or ~/.claude/quant-kalshi/ stops everything, live or dry
  * caps (sports/.env, small defaults): KALSHI_MAX_ORDER_USD=25, KALSHI_MAX_DAILY_USD=100, KALSHI_MAX_OPEN_USD=300
  * size <= half the resting depth within 1c; immediate-or-cancel limit orders (nothing rests overnight)
Credentials: KALSHI_KEY_ID and KALSHI_PRIVATE_KEY_PATH in sports/.env, put there by the user. This script only reads
them to sign requests; it never prints, logs or copies them. sports/.env and *.pem are gitignored.
VERIFY BEFORE THE FIRST LIVE ORDER (written 2026-10-19 from the v2 docs as remembered, not yet exercised):
signature = RSA-PSS/SHA256 over timestamp_ms + METHOD + path; order body fields (no_price in cents,
time_in_force). Run `check`, then one live order of 1 contract, before raising caps.
"""
from __future__ import annotations

import base64
import json
import math
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PICKS = DATA / "kalshi_paper.jsonl"
LOG = DATA / "kalshi_exec.jsonl"
ENV = ROOT / ".env"
API = "https://api.elections.kalshi.com"
BASE = "/trade-api/v2"
FEE = 0.07
KILLS = [ROOT / "KILL", Path.home() / ".claude" / "quant-kalshi" / "KILL"]


def env() -> dict:
    out = {}
    if ENV.exists():
        for line in ENV.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _sign(e: dict, method: str, path: str) -> dict:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    key = serialization.load_pem_private_key(Path(e["KALSHI_PRIVATE_KEY_PATH"]).expanduser().read_bytes(), password=None)
    ts = str(int(time.time() * 1000))
    sig = key.sign((ts + method + path).encode(), padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                                             salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
    return {"KALSHI-ACCESS-KEY": e["KALSHI_KEY_ID"], "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(), "Content-Type": "application/json"}


def authed(e, method, path, body=None):
    r = requests.request(method, API + BASE + path, headers=_sign(e, method, BASE + path),
                         data=json.dumps(body) if body is not None else None, timeout=20)
    return r.status_code, (r.json() if r.content else {})


def killed() -> bool:
    return any(k.exists() for k in KILLS)


def book_no(ticker: str) -> tuple[float, float]:
    """(best NO ask, contracts within 1c of it), from the public orderbook (NO ask = 1 - best YES bid)."""
    d = requests.get(f"{API}{BASE}/markets/{ticker}/orderbook", timeout=20).json()
    ob = d.get("orderbook_fp") or d.get("orderbook") or {}  # same parsing as kalshi_paper.book_depth
    bids = ob.get("yes_dollars") or ob.get("yes") or []
    lv = sorted(((float(p) if float(p) <= 1 else float(p) / 100, float(q)) for p, q in bids), reverse=True)
    if not lv:
        return math.nan, 0.0
    best = lv[0][0]
    return round(1 - best, 2), sum(q for p, q in lv if p >= best - 0.0101)


def logged() -> list[dict]:
    return [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()] if LOG.exists() else []


def cmd_check() -> int:
    e = env()
    if not {"KALSHI_KEY_ID", "KALSHI_PRIVATE_KEY_PATH"} <= set(e):
        print("no key configured: add KALSHI_KEY_ID and KALSHI_PRIVATE_KEY_PATH to sports/.env yourself")
        return 1
    code, body = authed(e, "GET", "/portfolio/balance")
    print(f"GET /portfolio/balance -> HTTP {code}; " + (f"balance ${body.get('balance', 0) / 100:,.2f}" if code == 200 else str(body)[:200]))
    return 0 if code == 200 else 1


def cmd_run() -> int:
    if killed():
        print("KILL file present -> nothing done")
        return 0
    e = env()
    live = "--live" in sys.argv and e.get("EXEC_MODE") == "live"
    if "--live" in sys.argv and not live:
        print("--live ignored: sports/.env does not have EXEC_MODE=live -> dry run")
    cap_order = float(e.get("KALSHI_MAX_ORDER_USD", 25))
    cap_day = float(e.get("KALSHI_MAX_DAILY_USD", 100))
    cap_open = float(e.get("KALSHI_MAX_OPEN_USD", 300))
    hist = logged()
    done = {h["ticker"] for h in hist}
    now = datetime.now(timezone.utc)
    today = now.date().isoformat()
    spent_today = sum(h["cost"] for h in hist if h["mode"] == ("live" if live else "dry") and h["ts"][:10] == today)
    open_cost = sum(h["cost"] for h in hist if h["mode"] == ("live" if live else "dry") and not h.get("settled"))
    picks = [json.loads(l) for l in PICKS.read_text().splitlines() if l.strip()] if PICKS.exists() else []
    fresh = [p for p in picks if p["ticker"] not in done and p.get("result") is None
             and now - datetime.fromisoformat(p["logged_at"]) < timedelta(hours=3)]
    for p in fresh:
        ask, depth = book_no(p["ticker"])
        rec = {"ts": now.isoformat(timespec="seconds"), "mode": "live" if live else "dry", "ticker": p["ticker"],
               "pick_no_price": p["no_price"], "no_ask_now": ask, "depth_1c": depth}
        budget = min(cap_order, cap_day - spent_today, cap_open - open_cost)
        if not (ask == ask) or ask > p["no_price"] + 0.01 + 1e-9:
            rec.update(status="skip_price_moved", count=0, cost=0.0)
        elif budget < ask:
            rec.update(status="skip_cap", count=0, cost=0.0)
        else:
            n = int(min(budget // (ask + FEE * ask * (1 - ask)), depth // 2))
            rec.update(count=n, cost=round(n * ask, 2), limit_cents=int(round(ask * 100)))
            if n < 1:
                rec.update(status="skip_depth", cost=0.0)
            elif live:
                code, body = authed(e, "POST", "/portfolio/orders", {
                    "ticker": p["ticker"], "action": "buy", "side": "no", "type": "limit", "count": n,
                    "no_price": rec["limit_cents"], "time_in_force": "immediate_or_cancel",
                    "client_order_id": str(uuid.uuid4())})
                rec.update(status=f"http_{code}", response=body.get("order", body))
            else:
                rec["status"] = "dry"
            spent_today += rec["cost"]
            open_cost += rec["cost"]
        with LOG.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{rec['mode']:4s} {p['ticker']:28s} NO ask {ask} (pick {p['no_price']}) depth {depth:.0f} -> "
              f"{rec['status']} {rec['count']} contracts ${rec['cost']:.2f}")
    if not fresh:
        print("no fresh unexecuted picks")
    return 0


def cmd_report() -> int:
    h = logged()
    picks = {json.loads(l)["ticker"]: json.loads(l) for l in PICKS.read_text().splitlines() if l.strip()} if PICKS.exists() else {}
    for mode in ("dry", "live"):
        rows = [x for x in h if x["mode"] == mode and x["count"] > 0]
        st = [(x, picks.get(x["ticker"], {}).get("result")) for x in rows]
        settled = [(x, r) for x, r in st if r in ("yes", "no")]
        pnl = sum(x["count"] * ((1 - x["no_ask_now"]) if r == "no" else -x["no_ask_now"]) - x["count"] * FEE * x["no_ask_now"] * (1 - x["no_ask_now"])
                  for x, r in settled)
        print(f"{mode}: {len(rows)} orders, ${sum(x['cost'] for x in rows):,.2f} staked, {len(settled)} settled, P&L ${pnl:+,.2f}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"check": cmd_check, "run": cmd_run, "report": cmd_report}.get(cmd, lambda: print(__doc__) or 1)())

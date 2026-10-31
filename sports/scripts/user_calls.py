"""User's judgment calls ("Mavs look low") -- stamp, grade, report. PAPER RECORD ONLY, no orders.

    ../.venv_quant/bin/python scripts/user_calls.py call DAL --game HOU@DAL [--date 2026-10-30] [--stake 2] [--note ".."]
                                                    [--price 0.33 --ts 2026-10-30T22:25:00Z]   # backfill a seen price
    ../.venv_quant/bin/python scripts/user_calls.py grade        # finals (ESPN) + closing price (last Kalshi trade pre-tip)
    ../.venv_quant/bin/python scripts/user_calls.py report       # record, ROI at call price, CLV with bootstrap CI

Why (user 10-30: "we have to quantify this... wish you could track my performance"): wins on a handful of games can't
separate skill from luck; CLV can. CLV = closing price of the picked side - price when called (cents). A caller who beats
the close on average has information the market hadn't priced yet; that shows up in far fewer games than ROI does.
Price at call = Kalshi KXNBAGAME yes ask of the picked team (what a taker would pay). Close = last Kalshi trade before
tip. Data: data/user_calls.jsonl (gitignored). Moneyline calls only (v1).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data" / "user_calls.jsonl"
K = "https://api.elections.kalshi.com/trade-api/v2"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_TO_KALSHI = {"WSH": "WAS", "NY": "NYK", "GS": "GSW", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "PHX": "PHX"}
MON = "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split()


def kc(t: str) -> str:
    return ESPN_TO_KALSHI.get(t.upper(), t.upper())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def rows() -> list[dict]:
    return [json.loads(x) for x in LOG.read_text().splitlines() if x.strip()] if LOG.exists() else []


def espn_games(date: str) -> dict:
    d = requests.get(ESPN, params={"dates": date.replace("-", "")}, timeout=20).json()
    out = {}
    for e in d.get("events", []):
        c = e["competitions"][0]
        t = {x["homeAway"]: x for x in c["competitors"]}
        a, h = t["away"]["team"]["abbreviation"], t["home"]["team"]["abbreviation"]
        out[f"{a}@{h}"] = {"start": e["date"], "final": c["status"]["type"]["name"] == "STATUS_FINAL",
                           "score": {a: int(t["away"].get("score") or 0), h: int(t["home"].get("score") or 0)}}
    return out


def ticker(date: str, game: str, pick: str) -> str:
    a, h = game.upper().split("@")
    y, m, d = date.split("-")
    return f"KXNBAGAME-{y[2:]}{MON[int(m) - 1]}{d}{kc(a)}{kc(h)}-{kc(pick)}"


def market(tk: str) -> dict:
    r = requests.get(f"{K}/markets/{tk}", timeout=20)
    return r.json().get("market", {}) if r.ok else {}


def close_price(tk: str, start_iso: str) -> float | None:
    ts = int(datetime.fromisoformat(start_iso.replace("Z", "+00:00")).timestamp())
    r = requests.get(f"{K}/markets/trades", params={"ticker": tk, "max_ts": ts, "limit": 1}, timeout=20)
    tr = r.json().get("trades", []) if r.ok else []
    if not tr:
        return None
    t = tr[0]
    v = t.get("yes_price_dollars") or t.get("yes_price")
    v = float(v)
    return v / 100 if v > 1 else v


def cmd_call(a) -> int:
    date = a.date or datetime.now().strftime("%Y-%m-%d")
    g = espn_games(date)
    game = a.game.upper()
    if game not in g:
        print(f"{game} not on ESPN {date}: {sorted(g)}")
        return 2
    tk = ticker(date, game, a.pick)
    if a.price is None:
        m = market(tk)
        if not m:
            print("no Kalshi market", tk)
            return 2
        price, src = float(m.get("yes_ask_dollars") or 0), "kalshi_ask_live"
    else:
        price, src = a.price, "backfill_seen_in_session"
    rec = {"id": f"{date}|{game}|{a.pick.upper()}", "ts": a.ts or now_iso(), "date": date, "game": game,
           "pick": a.pick.upper(), "ticker": tk, "price": price, "price_src": src, "stake": a.stake,
           "start": g[game]["start"], "note": a.note}
    if any(r["id"] == rec["id"] for r in rows()):
        print("already called", rec["id"])
        return 1
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"called {rec['id']} @ {price:.2f} ({src}) tip {rec['start']}")
    return 0


def cmd_grade(_a=None) -> int:
    rs, cache, changed = rows(), {}, 0
    for r in rs:
        if r.get("result"):
            continue
        g = cache.setdefault(r["date"], espn_games(r["date"])).get(r["game"])
        if not g or not g["final"]:
            continue
        a, h = r["game"].split("@")
        other = h if r["pick"] == a else a
        r["result"] = "win" if g["score"][r["pick"]] > g["score"][other] else "loss"
        r["close"] = close_price(r["ticker"], r["start"])
        r["clv_c"] = None if r["close"] is None else round(100 * (r["close"] - r["price"]), 1)
        r["roi"] = round((1 - r["price"]) / r["price"], 4) if r["result"] == "win" else -1.0
        changed += 1
        time.sleep(0.5)
    LOG.write_text("".join(json.dumps(r) + "\n" for r in rs))
    print(f"graded {changed}")
    return 0


def boot(v: list[float], n: int = 4000) -> tuple[float, float]:
    if len(v) < 2:
        return (float("nan"), float("nan"))
    rnd = random.Random(0)
    ms = sorted(sum(rnd.choice(v) for _ in v) / len(v) for _ in range(n))
    return ms[int(.025 * n)], ms[int(.975 * n)]


def cmd_report(_a=None) -> int:
    rs = rows()
    done = [r for r in rs if r.get("result")]
    for r in rs:
        print(f"{r['date']} {r['game']:9s} {r['pick']:4s} @ {r['price']:.2f}  close {r.get('close') or '-':>5}  "
              f"CLV {r.get('clv_c') if r.get('clv_c') is not None else '-':>5}c  {r.get('result') or 'open'}"
              f"  [{r['price_src']}]")
    if not done:
        print("no graded calls yet")
        return 0
    w = sum(r["result"] == "win" for r in done)
    roi = [r["roi"] for r in done]
    clv = [r["clv_c"] for r in done if r.get("clv_c") is not None]
    lo, hi = boot(roi)
    print(f"\ncalls {len(done)}: {w}-{len(done) - w}  ROI at call price {sum(roi) / len(roi):+.1%} [{lo:+.1%}, {hi:+.1%}]"
          f"  expected wins at your prices {sum(r['price'] for r in done):.1f}")
    if clv:
        clo, chi = boot(clv)
        print(f"CLV mean {sum(clv) / len(clv):+.1f}c [{clo:+.1f}, {chi:+.1f}] n {len(clv)}  "
              f"(> 0 consistently = you see things before the market prices them)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("call")
    c.add_argument("pick")
    c.add_argument("--game", required=True)
    c.add_argument("--date")
    c.add_argument("--stake", type=float)
    c.add_argument("--price", type=float)
    c.add_argument("--ts")
    c.add_argument("--note", default="")
    sp.add_parser("grade")
    sp.add_parser("report")
    a = p.parse_args()
    return {"call": cmd_call, "grade": cmd_grade, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

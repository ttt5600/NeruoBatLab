"""E4 06:00-D add-on maker paper logger (READ-ONLY: public Kalshi GETs via kx_throttle; no keys, no orders).

Pre-registration + forward judge: sports/notes/ops_e4_0600.md (lead from notes/e4_timing.md, agent ops_e4_0600).
edgelab trial: e4t_0600_addon_fwd (family kalshi_weather_temp), adapter edgelab/adapters/e4_0600_fwd.py.

    python scripts/e4_0600_paper.py tick            # decide for every orig-7 city whose local clock is in [06:00, 06:40) on D
    python scripts/e4_0600_paper.py settle          # trade-through fill + outcome for decided brackets whose market settled
    python scripts/e4_0600_paper.py status          # counts, fills, SPRT LLR, review progress
    python scripts/e4_0600_paper.py dry [--out F]   # decide NOW for today's picks ignoring the window -> scratch file

Inputs (read-only): ~/.claude/quant-kalshi/data/kalshi_paper.jsonl (E4 picks, cohort orig7) and kalshi_exec.jsonl (the live
executor's 22:00 maker order + hourly exchange reconcile snapshots = the exchange's fill state).
Output (append-only, never rewritten): ~/.claude/quant-sports/data/kx_forward/e4_0600.jsonl (override E4_0600_LOG):
  kind city_day  one per (city, D): picks found, decided / missed
  kind pick      one per (ticker, D): cohort (filled | partial | resting_unfilled | not_placed | rejected | other),
                 live 22:00 order fields, book at T, hypothetical q, status (bid | skip_* | missed)
  kind settle    one per bid once the market settled: tt_fill, fill_ts, n10, result, pnl_unit, close_no
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kx_throttle import kget  # noqa: E402

KD = Path(os.environ.get("E4_0600_KD", str(Path.home() / ".claude" / "quant-kalshi" / "data")))
LOG = Path(os.environ.get("E4_0600_LOG", str(Path.home() / ".claude" / "quant-sports" / "data" / "kx_forward" / "e4_0600.jsonl")))
# orig-7 series -> station tz (copied from sports/wx.py CITIES; the deployed copy has no sports package)
CITIES = {"KXHIGHNY": "America/New_York", "KXHIGHCHI": "America/Chicago", "KXHIGHMIA": "America/New_York",
          "KXHIGHAUS": "America/Chicago", "KXHIGHDEN": "America/Denver", "KXHIGHLAX": "America/Los_Angeles",
          "KXHIGHPHIL": "America/New_York"}
T_HOUR, WIN_MIN = 6, 40           # decide in [06:00, 06:40) station-local on D; later -> `missed`
MAX_IMPROVE = 0.05                # kalshi_exec MAKER_MAX_IMPROVE
FORWARD_FROM = "2026-11-16"       # first event day of the forward log (registration 2026-11-16 ~10:05Z)
CLOSE_H = 10                      # E4 CLV close window (edgelab e4_live CLOSE_H)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat(timespec="seconds")


def pts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def jl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def append(rows: list[dict], out: Path) -> None:
    if not rows:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def cent_floor(x: float) -> float:
    return math.floor(round(x * 100, 6)) / 100


def event_day(ticker: str) -> str:
    return datetime.strptime(ticker.split("-")[1], "%y%b%d").strftime("%Y-%m-%d")


# ------------------------------------------------------------------ inputs (read-only)
def orig7_picks(day: str) -> dict[str, list[dict]]:
    """city -> E4 orig-7 picks (kalshi_paper.jsonl) for event day `day` (one per ticker, first logged)."""
    out, seen = {}, set()
    for r in jl(KD / "kalshi_paper.jsonl"):
        if r.get("cohort") != "orig7" or r.get("city") not in CITIES or r["ticker"] in seen:
            continue
        if event_day(r["ticker"]) != day:
            continue
        seen.add(r["ticker"])
        out.setdefault(r["city"], []).append(r)
    return out


def live_state(ticker: str, t: datetime) -> dict:
    """The live 22:00 maker order on `ticker` as the executor journal knew it at time t (last snapshot <= t)."""
    ev = jl(KD / "kalshi_exec.jsonl")
    od = [e for e in ev if e.get("event") == "order" and e.get("mode") == "live" and e.get("ticker") == ticker]
    if not od:
        return {"cohort": "not_placed", "live_status": "no_order_row"}
    o = od[-1]
    base = {"live_order_ts": o["ts"], "live_status": o.get("status"), "live_exec_style": o.get("exec_style"),
            "live_q": o.get("maker_q"), "live_count": o.get("count")}
    if o.get("status") != "submitting":
        return {**base, "cohort": "not_placed"}
    up = sorted([e for e in ev if e.get("event") == "update" and e.get("mode") == "live" and e.get("ticker") == ticker
                 and e["ts"] >= o["ts"]], key=lambda e: e["ts"])
    post = [e for e in up if "http" in e]
    if not post or post[0].get("http") != 201:
        return {**base, "cohort": "rejected", "live_http": post[0].get("http") if post else None}
    oid = post[0].get("order_id")
    snaps = [e for e in up if e.get("order_id") == oid and pts(e["ts"]) <= t]
    if not snaps:
        return {**base, "cohort": "other", "live_snapshot": None}
    s = snaps[-1]
    filled, cnt = float(s.get("filled") or 0), float(o.get("count") or 0)
    st = s.get("status")
    if st == "filled" or (cnt > 0 and filled >= cnt):
        coh = "filled"
    elif filled > 0:
        coh = "partial"
    elif st == "resting":
        coh = "resting_unfilled"
    else:
        coh = "other"
    return {**base, "cohort": coh, "live_order_id": oid, "live_snapshot_ts": s["ts"], "live_snapshot_status": st,
            "live_filled": filled, "live_avg_no": s.get("avg_no")}


# ------------------------------------------------------------------ public market data
def book(ticker: str) -> dict:
    d = kget(f"/markets/{ticker}/orderbook") or {}
    ob = d.get("orderbook_fp") or d.get("orderbook") or {}
    lvl = lambda rows: sorted(((float(p) if float(p) <= 1 else float(p) / 100, float(q)) for p, q in (rows or [])),  # noqa: E731
                              reverse=True)
    ys = lvl(ob.get("yes_dollars") or ob.get("yes"))
    ns = lvl(ob.get("no_dollars") or ob.get("no"))
    out = {"yes_bids": ys[:5], "no_bids": ns[:5], "yes_depth_total": sum(q for _, q in ys),
           "no_depth_total": sum(q for _, q in ns)}
    if ys:
        out["no_ask"] = round(1 - ys[0][0], 4)
        out["no_ask_depth_1c"] = sum(q for p, q in ys if p >= ys[0][0] - 0.0101)
    if ns:
        out["no_bid"] = round(ns[0][0], 4)
        out["no_bid_size"] = ns[0][1]
    return out


def market(ticker: str) -> dict:
    return (kget(f"/markets/{ticker}") or {}).get("market") or {}


_FEE: dict = {}


def fee_type(series: str):
    if series not in _FEE:
        _FEE[series] = ((kget(f"/series/{series}") or {}).get("series") or {}).get("fee_type")
    return _FEE[series]


def tape(ticker: str, t0: int, t1: int) -> list[tuple[int, float, float, str]]:
    """Public prints in [t0, t1] -> (ts, no_price, count, taker_side), oldest first (/markets/trades, else /historical)."""
    out = []
    for p in ("/markets/trades", "/historical/trades"):
        cur = None
        while True:
            prm = {"ticker": ticker, "min_ts": int(t0), "max_ts": int(t1), "limit": 1000}
            if cur:
                prm["cursor"] = cur
            d = kget(p, prm)
            if not d:
                break
            out += d.get("trades", [])
            cur = d.get("cursor")
            if not cur or not d.get("trades"):
                break
        if out:
            break
    rows = []
    for r in out:
        npx = r.get("no_price_dollars")
        npx = float(npx) if npx is not None else float(r.get("no_price")) / 100
        cnt = r.get("count_fp")
        cnt = float(cnt) if cnt is not None else float(r.get("count", 0))
        rows.append((int(pts(r["created_time"]).timestamp()), npx, cnt, r.get("taker_side")))
    return sorted(set(rows))


# ------------------------------------------------------------------ decision
def decide(p: dict, t: datetime, status_override: str | None = None) -> dict:
    tk = p["ticker"]
    st = live_state(tk, t)
    row = {"kind": "pick", "ticker": tk, "city": p["city"], "event_day": event_day(tk), "decided_at": iso(t),
           "pick_logged_at": p["logged_at"], "pick_no_price": p.get("no_price"), "strike_type": p.get("strike_type"),
           "floor": p.get("floor"), "cap": p.get("cap"), "forecasts": p.get("forecasts"), **st}
    if status_override:
        return {**row, "status": status_override}
    m = market(tk)
    row.update({"market_status": m.get("status"), "close_time": m.get("close_time")})
    if m.get("status") not in ("active", "open"):
        return {**row, "status": "skip_market_not_active"}
    ft = fee_type(tk.split("-")[0])
    row["fee_type"] = ft
    b = book(tk)
    row["book"] = b
    row["book_ts"] = iso(now_utc())
    # did the 22:00 order's level already trade through by T (public tape; diagnostic only)
    if st.get("live_q") is not None and st.get("live_order_ts"):
        tp = tape(tk, int(pts(st["live_order_ts"]).timestamp()) + 1, int(t.timestamp()))
        row["live_q_tt_by_T"] = any(px < float(st["live_q"]) - 1e-9 for _, px, _, _ in tp)
        row["tape_prints_post22_to_T"] = len(tp)
    if "no_ask" not in b or "no_bid" not in b:
        return {**row, "status": "skip_no_book"}
    ask, bid = b["no_ask"], b["no_bid"]
    q = round(min(cent_floor((ask + bid) / 2), ask - 0.01), 4)
    row.update({"q": q, "no_ask": ask, "no_bid": bid, "spread": round(ask - bid, 4)})
    if ft != "quadratic":
        return {**row, "status": "skip_fee_type"}
    if q <= 0 or ask - q > MAX_IMPROVE + 1e-9:
        return {**row, "status": "skip_maker_spread"}
    return {**row, "status": "bid", "side": "no"}


def cmd_tick(a) -> int:
    t = now_utc()
    have = {(r["city"], r["event_day"]) for r in jl(LOG) if r.get("kind") == "city_day"}
    rows = []
    for city, tzn in CITIES.items():
        loc = t.astimezone(ZoneInfo(tzn))
        day = loc.strftime("%Y-%m-%d")
        if day < FORWARD_FROM or (city, day) in have:
            continue
        mins = (loc.hour - T_HOUR) * 60 + loc.minute
        if mins < 0:
            continue                                   # before 06:00 local: wait
        picks = orig7_picks(day).get(city, [])
        missed = mins >= WIN_MIN
        rows.append({"kind": "city_day", "city": city, "event_day": day, "ts": iso(t), "local": loc.isoformat(timespec="seconds"),
                     "n_picks": len(picks), "status": "missed" if missed else "decided"})
        for p in picks:
            rows.append(decide(p, t, "missed" if missed else None))
    append(rows, LOG)
    for r in rows:
        if r["kind"] == "pick":
            print(f"{r['ticker']} cohort={r.get('cohort')} status={r['status']} q={r.get('q')} ask={r.get('no_ask')} "
                  f"bid={r.get('no_bid')}")
        else:
            print(f"{r['city']} {r['event_day']} picks={r['n_picks']} {r['status']}")
    if not rows:
        print("idle: no city in its 06:00 window")
    return 0


def cmd_dry(a) -> int:
    """Decide NOW for every orig-7 pick of `--date` (default: today UTC), ignoring the window. Scratch output only."""
    t = now_utc()
    day = a.date or t.strftime("%Y-%m-%d")
    out = Path(a.out)
    rows = []
    for city, ps in orig7_picks(day).items():
        for p in ps:
            rows.append({**decide(p, t), "dry": True})
    append(rows, out)
    for r in rows:
        print(json.dumps({k: r.get(k) for k in ("ticker", "cohort", "live_snapshot_status", "live_filled", "live_q",
                                                 "status", "q", "no_ask", "no_bid", "close_time", "live_q_tt_by_T")}))
    print(f"{len(rows)} dry rows -> {out}")
    return 0


# ------------------------------------------------------------------ settle
def cmd_settle(a) -> int:
    rs = jl(LOG)
    done = {r["ticker"] for r in rs if r.get("kind") == "settle"}
    rows = []
    for p in rs:
        if p.get("kind") != "pick" or p.get("status") != "bid" or p["ticker"] in done:
            continue
        tk = p["ticker"]
        m = market(tk)
        res = m.get("result")
        if m.get("status") not in ("finalized", "settled") or res not in ("yes", "no"):
            continue
        t0 = int(pts(p["decided_at"]).timestamp())
        ct = m.get("close_time") or p.get("close_time")
        t1 = int(pts(ct).timestamp())
        tp = tape(tk, t0 + 1, t1)
        q = float(p["q"])
        tt = [(ts, px, n) for ts, px, n, _ in tp if px < q - 1e-9]
        cum, n10_ts = 0.0, None
        for ts, px, n, _ in tp:
            if px <= q + 1e-9:
                cum += n
                if cum >= 10 and n10_ts is None:
                    n10_ts = ts
        nos = [(ts, px) for ts, px, _, side in tp if side == "no" and ts <= t0 + CLOSE_H * 3600]
        win = res == "no"
        rows.append({"kind": "settle", "ticker": tk, "ts": iso(now_utc()), "result": res, "market_status": m.get("status"),
                     "close_time": ct, "tt_fill": bool(tt),
                     "fill_ts": iso(datetime.fromtimestamp(tt[0][0], timezone.utc)) if tt else None,
                     "tt_volume": sum(n for _, _, n in tt), "n10_fill": n10_ts is not None,
                     "n10_ts": iso(datetime.fromtimestamp(n10_ts, timezone.utc)) if n10_ts else None,
                     "prints": len(tp), "close_no": nos[-1][1] if nos else None,
                     "pnl_unit": round((1 - q) / q if win else -1.0, 6), "cohort": p.get("cohort")})
    append(rows, LOG)
    for r in rows:
        print(f"settle {r['ticker']} result={r['result']} tt_fill={r['tt_fill']} pnl_unit={r['pnl_unit']}")
    print(f"{len(rows)} settled")
    return 0


# ------------------------------------------------------------------ status / judge
def sprt(fills: list[tuple[float, bool]], nights: int) -> dict:
    """bar_calibration_v2 near-certain SPRT: H0 loss p0 = 1 - c, H1 = .7 p0; alpha .005, beta .10; LLR / (1 + m x .031)."""
    llr = 0.0
    for c, lost in fills:
        p0 = 1 - c
        llr += math.log(0.7) if lost else math.log((1 - 0.7 * p0) / (1 - p0))
    m = len(fills) / nights if nights else 0.0
    adj = llr / (1 + m * 0.031)
    A, B = math.log(0.9 / 0.005), math.log(0.1 / 0.995)
    return {"llr_adj": round(adj, 4), "A": round(A, 3), "B": round(B, 3),
            "state": "H1 (ask user; no self-promotion)" if adj >= A else "KILL" if adj <= B else "continue"}


def poisson_binom_le(p: list[float], k: int) -> float:
    dist = [1.0]
    for pi in p:
        nd = [0.0] * (len(dist) + 1)
        for j, v in enumerate(dist):
            nd[j] += v * (1 - pi)
            nd[j + 1] += v * pi
        dist = nd
    return float(sum(dist[: k + 1]))


def cmd_status(a) -> int:
    rs = jl(LOG)
    picks = [r for r in rs if r.get("kind") == "pick"]
    st = {r["ticker"]: r for r in rs if r.get("kind") == "settle"}
    from collections import Counter
    print(f"{LOG}: city_days {sum(r.get('kind') == 'city_day' for r in rs)}, picks {len(picks)}, settles {len(st)}")
    print("cohort x status:", dict(Counter((r.get("cohort"), r["status"]) for r in picks)))
    for coh in ("filled", "not_placed", "partial", "resting_unfilled"):
        g = [(r, st[r["ticker"]]) for r in picks if r.get("cohort") == coh and r["ticker"] in st]
        f = [(float(r["q"]), s["result"] != "no") for r, s in g if s["tt_fill"]]
        uf = [s["result"] != "no" for r, s in g if not s["tt_fill"]]
        if not g:
            continue
        pnl = sum((1 - c) / c if not lost else -1 for c, lost in f)
        print(f"  {coh}: settled bids {len(g)}, tt fills {len(f)} (losses {sum(l for _, l in f)}, units {pnl:+.2f}), "
              f"unfilled {len(uf)} (losses {sum(uf)})")
    prim = [(r, st[r["ticker"]]) for r in picks if r.get("cohort") == "filled" and r["ticker"] in st and st[r["ticker"]]["tt_fill"]]
    f = [(float(r["q"]), s["result"] != "no") for r, s in prim]
    nights = len({r["event_day"] for r, _ in prim})
    if f:
        L = sum(l for _, l in f)
        p0 = [1 - c for c, _ in f]
        print(f"PRIMARY fills {len(f)} / review at 60: losses {L}, expected under H0 {sum(p0):.2f}, "
              f"P(L_H0 <= L) = {poisson_binom_le(p0, L):.4f}; SPRT {sprt(f, nights)}")
    else:
        print("PRIMARY fills 0 / review at 60")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tick").set_defaults(fn=cmd_tick)
    sub.add_parser("settle").set_defaults(fn=cmd_settle)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    p = sub.add_parser("dry")
    p.add_argument("--date")
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "ops_e4_0600" / "dry.jsonl"))
    p.set_defaults(fn=cmd_dry)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
